"""Kilosort 2.5 CUDA mex functions as CuPy RawModule kernels.

Ports of `mexGetSpikes2.cu`, `mexSVDsmall2.cu`, `mexMPnu8.cu` and
`mexWtW2.cu` (branch `kilosort25`). The kernels keep the index arithmetic of
the originals, so every array is stored in MATLAB's (column-major) memory
order: a MATLAB array of shape (a, b, c) is a contiguous torch tensor of
shape (c, b, a). The Python wrappers take `Params` (the MATLAB double array,
0-based positions: [0]=NT, [1]=Nfilt, [2]=Th, [3]=nInnerIter, [4]=nt0,
[5]=Nnearest, [6]=Nrank, [7]=lam, [8]=pm, [9]=Nchan, [10]=NchanNear,
[11]=nt0min, [12]=feature flag, [13]=Nsum, [14]=NrankPC, [15]=Th for
average_snips, [16]=useStableMode) and torch CUDA tensors, and return torch
tensors on the same device. Work buffers are allocated per call (through
PyTorch's caching allocator).

Intentional differences from the MATLAB mex files are listed in
`.agent-notes/ks25_fable.md`; in short:

- the spikes of `get_spikes2` come back sorted by time (the mex order is the
  order of atomic increments), and its `bestFilter` resets the running
  maximum per time point (the original keeps it across grid-stride
  iterations, which copies the head of `err` into the unfiltered tail);
- `svd_small2` never materializes the blanked copy of dWU: the sums run over
  the NchanNear channels of the template (the other channels contribute
  exactly zero);
- `mpnu8` keeps the spike counter on the device (the kernels loop over the
  new spikes themselves) and only synchronizes with the host every
  `sync_every` iterations; with `early_stop=True` (default) the loop ends
  once an iteration finds no new spike, which is a fixed point of the
  iteration (same outputs as running all `nInnerIter` iterations). The
  subtraction only touches the NchanNear channels of the template (U is
  zero elsewhere), the three updates after a subtraction round are
  organized per spike instead of per template, and average_snips visits the
  spikes grouped by template; the sums are accumulated in the original
  order;
- with `matlab_tail=False` (default) the whole batch is scanned for spikes;
  `matlab_tail=True` reproduces the loop bounds of the mex files, which
  leave the last ~Nthreads+nt0 samples of a batch unscanned.
"""
import contextlib
import warnings

import numpy as np
import torch

_modules = {}
TPB = 1024  # Nthreads of the mex files
MAXSLOTS = 192  # channels of a spike's data window kept in shared memory (mpnu8)


def _function(src_name, name, defines=()):
    """Compiled kernel `name` from `_SOURCES[src_name]` (one module per
    distinct set of -D values and per device)."""
    import cupy as cp
    key = (src_name, tuple(defines), cp.cuda.Device().id)
    if key not in _modules:
        options = tuple(f'-D{k}={v}' for k, v in defines)
        _modules[key] = cp.RawModule(code=_SOURCES[src_name], options=options)
    return _modules[key].get_function(name)


def _device(t):
    """PyTorch's and CuPy's current device set to the device of tensor t."""
    import cupy as cp
    stack = contextlib.ExitStack()
    stack.enter_context(torch.cuda.device(t.device))
    stack.enter_context(cp.cuda.Device(t.device.index))
    return stack


def _stream():
    """CuPy view of PyTorch's current CUDA stream."""
    import cupy as cp
    stream = torch.cuda.current_stream()
    if hasattr(stream, '__cuda_stream__'):
        return cp.cuda.Stream.from_external(stream)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return cp.cuda.ExternalStream(stream.cuda_stream)


def _p(t):
    # Raw device pointers (declared as unsigned long long in the kernels).
    return np.uint64(t.data_ptr())


def _i32(*xs):
    return tuple(np.int32(x) for x in xs)


def _f32(*xs):
    return tuple(np.float32(x) for x in xs)


def _tensor(x, dtype, shape=None, name='array', device=None):
    """`x` as a contiguous torch tensor of `dtype` (and `shape` if given)."""
    t = torch.as_tensor(x, device=device)
    if t.dtype != dtype:
        t = t.to(dtype)
    t = t.contiguous()
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f'{name}: expected shape {tuple(shape)}, got {tuple(t.shape)}')
    return t


def _params(Params):
    p = np.asarray(Params, dtype=np.float64).flatten()
    if p.size < 17:
        p = np.concatenate([p, np.zeros(17 - p.size)])
    return p


def _tail(NT, nt0, matlab_tail):
    """End (exclusive) of the time range that the mex Conv1D / timeFilter
    loops write, i.e. the last tile start plus Nthreads, or the whole valid
    range when `matlab_tail` is False."""
    if not matlab_tail:
        return max(0, NT - nt0 + 1)
    bound = NT - TPB - nt0 + 1
    if bound <= 0:
        return 0
    return -(-bound // TPB) * TPB


def _cleanup_range(NT, lockout, matlab_tail, strict):
    """[lo, hi) of the peak positions examined by the mex cleanup_spikes."""
    hi = NT - lockout - (1 if strict else 0)
    if matlab_tail:
        bound = NT - TPB - lockout + 1
        niter = -(-bound // TPB) if bound > 0 else 0
        hi = min(hi, niter * TPB + lockout)
    return lockout, max(lockout, hi)


_SOURCES = {}

# --------------------------------------------------------------------------
# mexGetSpikes2
# --------------------------------------------------------------------------
_SOURCES['getspikes'] = r"""
typedef unsigned long long ptr_t;

// Conv1D: filter channel bid with the NRANK temporal templates
// (out[r, c, t] = sum_i W[r, i] * data[c, t + i] for t < Tend; the rest is
// left as it was, zero).
extern "C" __global__
void gs_conv1d(ptr_t pdata, ptr_t pW, ptr_t pout, int NT, int Nchan, int Tend)
{
    const float* data = (const float*)pdata;
    const float* W = (const float*)pW;
    float* out = (float*)pout;
    __shared__ float sW[NT0 * NRANK];
    __shared__ float sdata[TPB + NT0];
    const int tid = threadIdx.x, bid = blockIdx.x;
    for (int i = tid; i < NT0 * NRANK; i += TPB) sW[i] = W[i];
    __syncthreads();
    for (int tid0 = 0; tid0 < Tend; tid0 += TPB) {
        for (int i = tid; i < TPB + NT0; i += TPB) {
            int s = tid0 + i;
            sdata[i] = (s < NT) ? data[s + (long long)NT * bid] : 0.0f;
        }
        __syncthreads();
        int t = tid0 + tid;
        if (t < Tend) {
            for (int nid = 0; nid < NRANK; ++nid) {
                float y = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < NT0; ++i) y += sW[i + nid * NT0] * sdata[i + tid];
                out[t + (long long)NT * bid + (long long)nid * NT * Nchan] = y;
            }
        }
        __syncthreads();
    }
}

// sumChannels: cumulative sum over the Nsum nearest channels of each
// channel, squared and normalized, max over the number of channels and
// the templates.
extern "C" __global__
void gs_sum_channels(ptr_t pdata, ptr_t pdatasum, ptr_t piC,
                     int NT, int Nchan, int NchanNear, int Nsum)
{
    const float* data = (const float*)pdata;
    float* datasum = (float*)pdatasum;
    const int* iC = (const int*)piC;
    for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < NT; t += blockDim.x * gridDim.x) {
        for (int i = 0; i < Nchan; ++i) {
            float Cmax = 0.0f;
            for (int r = 0; r < NRANK; ++r) {
                float Cf = 0.0f;
                for (int j = 0; j < Nsum; ++j) {
                    int iChan = iC[j + NchanNear * i];
                    Cf += data[t + (long long)NT * iChan + (long long)r * NT * Nchan];
                    float c = Cf * Cf / (1 + j);
                    if (c > Cmax) Cmax = c;
                }
            }
            datasum[t + (long long)NT * i] = Cmax;
        }
    }
}

// bestFilter: best channel at each time point.
extern "C" __global__
void gs_best_filter(ptr_t pdatasum, ptr_t perr, ptr_t pftype, int NT, int Nchan)
{
    const float* datasum = (const float*)pdatasum;
    float* err = (float*)perr;
    int* ftype = (int*)pftype;
    for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < NT; t += blockDim.x * gridDim.x) {
        float Cbest = 0.0f;
        int ibest = 0;
        for (int i = 0; i < Nchan; ++i) {
            float Cf = datasum[t + (long long)NT * i];
            if (Cf > Cbest + 1e-6f) { Cbest = Cf; ibest = i; }
        }
        err[t] = Cbest;
        ftype[t] = ibest;
    }
}

// cleanup_spikes: mask[t0] = err[t0] > Th^2 and no larger value within
// +-lockout, for t0 in [Tlo, Thi).
extern "C" __global__
void gs_cleanup_spikes(ptr_t perr, ptr_t pmask, int lockout, int Tlo, int Thi, float Th)
{
    const float* err = (const float*)perr;
    bool* mask = (bool*)pmask;
    for (int t0 = Tlo + threadIdx.x + blockIdx.x * blockDim.x; t0 < Thi; t0 += blockDim.x * gridDim.x) {
        float err0 = err[t0];
        bool flag = err0 > Th * Th;
        if (flag) {
            for (int j = -lockout; j <= lockout; ++j)
                if (err[t0 + j] > err0) { flag = false; break; }
        }
        mask[t0] = flag;
    }
}

// cleanup_heights: drop a candidate if a larger one lies within 4 channel
// indices (anywhere in the batch).
extern "C" __global__
void gs_cleanup_heights(ptr_t px, ptr_t pid, ptr_t pkeep, int n)
{
    const float* x = (const float*)px;
    const int* id = (const int*)pid;
    bool* keep = (bool*)pkeep;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        float xmax = x[i];
        int id0 = id[i];
        bool flag = true;
        for (int t = 0; t < n; ++t) {
            int d = abs(id[t] - id0);
            if (d < 5 && xmax < x[t]) { flag = false; break; }
        }
        keep[i] = flag;
    }
}
"""


def get_spikes2(Params, X, wTEMP, iC, maxFR=5000, matlab_tail=False):
    """mexGetSpikes2: spatiotemporal clips of the threshold crossings of the
    residual, used to seed new templates.

    X (Nchan, NT) f32, wTEMP (NrankPC, nt0) f32, iC (Nchan, NchanNear) i32.
    Returns dWU (nspk, Nchan, nt0) f32 (clips starting at the detected
    template start, sorted by time) and dout (Nchan, NT) f32 (the summed,
    squared filter output per channel, MATLAB's second output).
    """
    p = _params(Params)
    NT, nt0 = int(p[0]), int(p[4])
    Nchan, NchanNear = int(p[9]), int(p[10])
    Nsum, Nrank = int(p[13]), int(p[14])
    Th = float(p[2])
    X = _tensor(X, torch.float32, (Nchan, NT), 'X')
    dev = X.device
    wTEMP = _tensor(wTEMP, torch.float32, (Nrank, nt0), 'wTEMP', dev)
    iC = _tensor(iC, torch.int32, (Nchan, NchanNear), 'iC', dev)
    if Nsum > NchanNear:
        raise ValueError('Nsum (Params[13]) must not exceed NchanNear (Params[10])')
    if nt0 * Nrank > TPB:
        raise ValueError('nt0 * NrankPC must not exceed 1024')
    dout = torch.zeros((Nchan, NT), device=dev)
    empty = torch.zeros((0, Nchan, nt0), device=dev)
    if NT < nt0 or Nchan == 0:
        return empty, dout
    with _device(X), _stream():
        defs = (('NT0', nt0), ('NRANK', Nrank), ('TPB', TPB))
        dfilt = torch.zeros((Nrank, Nchan, NT), device=dev)
        err = torch.zeros(NT, device=dev)
        ftype = torch.zeros(NT, dtype=torch.int32, device=dev)
        mask = torch.zeros(NT, dtype=torch.bool, device=dev)
        Tend = _tail(NT, nt0, matlab_tail)
        nblk = (NT + TPB - 1) // TPB
        _function('getspikes', 'gs_conv1d', defs)(
            (Nchan,), (TPB,), (_p(X), _p(wTEMP), _p(dfilt), *_i32(NT, Nchan, Tend)))
        _function('getspikes', 'gs_sum_channels', defs)(
            (nblk,), (TPB,), (_p(dfilt), _p(dout), _p(iC), *_i32(NT, Nchan, NchanNear, Nsum)))
        _function('getspikes', 'gs_best_filter', defs)(
            (nblk,), (TPB,), (_p(dout), _p(err), _p(ftype), *_i32(NT, Nchan)))
        lockout = nt0 - 1
        Tlo, Thi = _cleanup_range(NT, lockout, matlab_tail, strict=True)
        if Thi > Tlo:
            _function('getspikes', 'gs_cleanup_spikes', defs)(
                (nblk,), (TPB,), (_p(err), _p(mask), *_i32(lockout, Tlo, Thi), *_f32(Th)))
        st = mask.nonzero().flatten()[:maxFR].to(torch.int32).contiguous()
        n = st.numel()
        if n == 0:
            return empty, dout
        ids = ftype[st.long()].contiguous()
        x = err[st.long()].contiguous()
        keep = torch.zeros(n, dtype=torch.bool, device=dev)
        _function('getspikes', 'gs_cleanup_heights', defs)(
            ((n + 255) // 256,), (256,), (_p(x), _p(ids), _p(keep), *_i32(n)))
        st = st[keep][:maxFR]
        if st.numel() == 0:
            return empty, dout
        idx = st.long()[:, None] + torch.arange(nt0, device=dev)[None, :]
        dWU = X[:, idx].permute(1, 0, 2).contiguous()
    return dWU, dout


# --------------------------------------------------------------------------
# mexSVDsmall2
# --------------------------------------------------------------------------
_SOURCES['svd'] = r"""
typedef unsigned long long ptr_t;

// getwtw (with blankdWU folded in): nt0 x nt0 covariance of each template
// over its NchanNear channels.
extern "C" __global__
void svd_getwtw(ptr_t pdWU, ptr_t piC, ptr_t piW, ptr_t pwtw, int Nchan, int NchanNear)
{
    const double* dWU = (const double*)pdWU;
    const int* iC = (const int*)piC;
    const int* iW = (const int*)piW;
    double* wtw = (double*)pwtw;
    const int tidx = threadIdx.x, bid = blockIdx.x;
    const int* row = iC + NchanNear * iW[bid];
    const double* d = dWU + (long long)bid * Nchan * NT0;
    for (int tidy = threadIdx.y; tidy < NT0; tidy += blockDim.y) {
        double x = 0.0;
        for (int j = 0; j < NchanNear; ++j) {
            int c = row[j];
            x += d[tidx + c * NT0] * d[tidy + c * NT0];
        }
        wtw[tidx + tidy * NT0 + (long long)bid * NT0 * NT0] = x;
    }
}

// getW: NRANK power iterations with deflation, warm-started from W.
extern "C" __global__
void svd_getW(ptr_t pwtw, ptr_t pW, int Nfilt, int tmax)
{
    const double* wtw = (const double*)pwtw;
    double* W = (double*)pW;
    __shared__ double sW[NT0 * NRANK];
    __shared__ double swtw[NT0 * NT0];
    __shared__ double xN;
    const int tid = threadIdx.x, bid = blockIdx.x;
    for (int k = 0; k < NT0; ++k)
        swtw[tid + k * NT0] = wtw[tid + k * NT0 + (long long)bid * NT0 * NT0];
    for (int k = 0; k < NRANK; ++k)
        sW[tid + k * NT0] = W[tid + bid * NT0 + (long long)k * NT0 * Nfilt];
    __syncthreads();
    double x = 0.0;
    for (int k = 0; k < NRANK; ++k) {
        for (int i = 0; i < 100; ++i) {
            x = 0.0;
            for (int t = 0; t < NT0; ++t) x += swtw[tid + t * NT0] * sW[t + k * NT0];
            __syncthreads();
            if (i < 99) {
                sW[tid + k * NT0] = x;
                __syncthreads();
                if (tid == 0) {
                    double x0 = 0.00001f;
                    for (int t = 0; t < NT0; ++t) x0 += sW[t + k * NT0] * sW[t + k * NT0];
                    xN = sqrt(x0);
                }
                __syncthreads();
                sW[tid + k * NT0] = x / xN;
                __syncthreads();
            }
        }
        for (int t = 0; t < NT0; ++t) swtw[tid + t * NT0] -= sW[t + k * NT0] * x;
        __syncthreads();
    }
    double xmax = sW[tmax];
    __syncthreads();
    sW[tid] = -sW[tid] * copysign(1.0, xmax);
    for (int k = 0; k < NRANK; ++k)
        W[tid + bid * NT0 + (long long)k * NT0 * Nfilt] = sW[tid + k * NT0];
}

// getU: U = W' * dWU on the NchanNear channels of the template (U is zero
// elsewhere).
extern "C" __global__
void svd_getU(ptr_t pdWU, ptr_t pW, ptr_t pU, ptr_t piC, ptr_t piW, int Nfilt, int Nchan, int NchanNear)
{
    const double* dWU = (const double*)pdWU;
    const double* W = (const double*)pW;
    double* U = (double*)pU;
    const int* iC = (const int*)piC;
    const int* iW = (const int*)piW;
    const int tidx = threadIdx.x, bid = blockIdx.x;
    if (tidx >= NRANK) return;
    const int* row = iC + NchanNear * iW[bid];
    for (int j = threadIdx.y; j < NchanNear; j += blockDim.y) {
        int c = row[j];
        double x = 0.0;
        for (int k = 0; k < NT0; ++k)
            x += W[k + NT0 * bid + (long long)NT0 * Nfilt * tidx] * dWU[k + c * NT0 + (long long)bid * Nchan * NT0];
        U[c + Nchan * bid + (long long)Nchan * Nfilt * tidx] = x;
    }
}

// reNormalize: mu and unit-norm U, align the peak of W to tmax (integer
// shift, then sub-sample interpolation with the kriging kernels A, B).
extern "C" __global__
void svd_renormalize(ptr_t pA, ptr_t pB, ptr_t pW, ptr_t pU, ptr_t pmu, int Nfilt, int Nchan, int tmax)
{
    const double* A = (const double*)pA;
    const double* B = (const double*)pB;
    double* W = (double*)pW;
    double* U = (double*)pU;
    double* mu = (double*)pmu;
    __shared__ double sW[NRANK * NT0];
    __shared__ double sU[NCHAN * NRANK];
    __shared__ double sS[NRANK + 1];
    __shared__ double sWup[10];
    const int tid = threadIdx.x, bid = blockIdx.x;
    for (int k = 0; k < NRANK; ++k)
        sW[tid + k * NT0] = W[tid + bid * NT0 + (long long)k * Nfilt * NT0];
    for (int i = tid; i < Nchan * NRANK; i += blockDim.x)
        sU[i] = U[i % Nchan + bid * Nchan + (long long)(i / Nchan) * Nchan * Nfilt];
    __syncthreads();
    if (tid < NRANK) {
        double x = 0.0;
        for (int k = 0; k < Nchan; ++k) x += sU[k + tid * Nchan] * sU[k + tid * Nchan];
        sS[tid] = sqrt(x);
    }
    __syncthreads();
    if (tid == 0) {
        double x = 0.0000001f;
        for (int k = 0; k < NRANK; ++k) x += sS[k] * sS[k];
        sS[NRANK] = sqrt(x);
        mu[bid] = sqrt(x);
    }
    __syncthreads();
    for (int i = tid; i < Nchan * NRANK; i += blockDim.x)
        U[i % Nchan + bid * Nchan + (long long)(i / Nchan) * Nchan * Nfilt] = sU[i] / sS[NRANK];
    __syncthreads();
    // integer alignment of the largest |W| of the first component to tmax
    double xmax = 0.0;
    int imax = 0;
    for (int t = 0; t < NT0; ++t)
        if (fabs(sW[t]) > xmax) { xmax = fabs(sW[t]); imax = t; }
    for (int k = 0; k < NRANK; ++k) {
        int ishift = tid + (imax - tmax);
        ishift = (ishift % NT0 + NT0) % NT0;
        W[tid + bid * NT0 + (long long)k * NT0 * Nfilt] = sW[ishift + k * NT0];
    }
    __syncthreads();
    for (int k = 0; k < NRANK; ++k)
        sW[tid + k * NT0] = W[tid + bid * NT0 + (long long)k * NT0 * Nfilt];
    __syncthreads();
    // 10x upsampled peak of the first component
    if (tid < 10) {
        double s = 0.0;
        for (int t = 0; t < NT0; ++t) s += A[tid + t * 10] * sW[t];
        sWup[tid] = s;
    }
    __syncthreads();
    xmax = 0.0;
    imax = 0;
    double sgnmax = 1.0;
    for (int t = 0; t < 10; ++t)
        if (fabs(sWup[t]) > xmax) { xmax = fabs(sWup[t]); imax = t; sgnmax = copysign(1.0, sWup[t]); }
    for (int k = 0; k < NRANK; ++k) {
        double xshift = 0.0;
        for (int t = 0; t < NT0; ++t)
            xshift += B[tid + t * NT0 + (long long)NT0 * NT0 * imax] * sW[t + k * NT0];
        if (k == 0) xshift = -xshift * sgnmax;
        W[tid + bid * NT0 + (long long)k * NT0 * Nfilt] = xshift;
    }
}
"""


def svd_small2(Params, dWU, W, iC, iW, Ka, Kb):
    """mexSVDsmall2: rank-Nrank spatiotemporal decomposition of the templates.

    dWU (Nfilt, Nchan, nt0) f64, W (Nrank, Nfilt, nt0) f64 (warm start, not
    modified), iC (Nchan, NchanNear) i32, iW (Nfilt,) i32, Ka (nt0, 10) f64,
    Kb (10, nt0, nt0) f64 (Kb[j, b, a] = MATLAB Kb(a, b, j)).
    Returns W (Nrank, Nfilt, nt0) f64, U (Nrank, Nfilt, Nchan) f64, mu (Nfilt,) f64.
    """
    p = _params(Params)
    Nfilt, nt0, Nrank = int(p[1]), int(p[4]), int(p[6])
    Nchan, NchanNear, tmax = int(p[9]), int(p[10]), int(p[11])
    dWU = _tensor(dWU, torch.float64, (Nfilt, Nchan, nt0), 'dWU')
    dev = dWU.device
    W = _tensor(W, torch.float64, (Nrank, Nfilt, nt0), 'W', dev).clone()
    iC = _tensor(iC, torch.int32, (Nchan, NchanNear), 'iC', dev)
    iW = _tensor(iW, torch.int32, (Nfilt,), 'iW', dev)
    Ka = _tensor(Ka, torch.float64, (nt0, 10), 'Ka', dev)
    Kb = _tensor(Kb, torch.float64, (10, nt0, nt0), 'Kb', dev)
    if not 0 <= tmax < nt0:
        raise ValueError('nt0min (Params[11]) must be in [0, nt0)')
    if nt0 > TPB or Nrank > nt0:
        raise ValueError('unsupported nt0 / Nrank')
    U = torch.zeros((Nrank, Nfilt, Nchan), dtype=torch.float64, device=dev)
    mu = torch.zeros(Nfilt, dtype=torch.float64, device=dev)
    if Nfilt == 0:
        return W, U, mu
    with _device(dWU), _stream():
        defs = (('NT0', nt0), ('NRANK', Nrank), ('NCHAN', Nchan))
        wtw = torch.empty((Nfilt, nt0, nt0), dtype=torch.float64, device=dev)
        ty = max(1, min(16, TPB // nt0))
        _function('svd', 'svd_getwtw', defs)(
            (Nfilt,), (nt0, ty), (_p(dWU), _p(iC), _p(iW), _p(wtw), *_i32(Nchan, NchanNear)))
        _function('svd', 'svd_getW', defs)(
            (Nfilt,), (nt0,), (_p(wtw), _p(W), *_i32(Nfilt, tmax)))
        _function('svd', 'svd_getU', defs)(
            (Nfilt,), (Nrank, max(1, TPB // Nrank)),
            (_p(dWU), _p(W), _p(U), _p(iC), _p(iW), *_i32(Nfilt, Nchan, NchanNear)))
        _function('svd', 'svd_renormalize', defs)(
            (Nfilt,), (nt0,), (_p(Ka), _p(Kb), _p(W), _p(U), _p(mu), *_i32(Nfilt, Nchan, tmax)))
    return W, U, mu


# --------------------------------------------------------------------------
# mexWtW2
# --------------------------------------------------------------------------
_SOURCES['wtw'] = r"""
typedef unsigned long long ptr_t;
#define NB 32
#define NLAG (2 * NT0 - 1)

// crossFilter: WtW[lag, b, a] = UtU[b, a] * sum_t W1[a, t] W2[b, t + NT0 - lag - 1]
extern "C" __global__
void wtw_cross(ptr_t pW1, ptr_t pW2, ptr_t pUtU, ptr_t pWtW, int Nfilt)
{
    const float* W1 = (const float*)pW1;
    const float* W2 = (const float*)pW2;
    const float* UtU = (const float*)pUtU;
    float* WtW = (float*)pWtW;
    __shared__ float shW1[NB * NT0], shW2[NB * NT0];
    const int tidx = threadIdx.x, tidy = threadIdx.y;
    const int tid1 = tidx + blockIdx.x * NB, tid2 = tidy + blockIdx.y * NB;
    int r = tidy + blockIdx.x * NB;
    if (r < Nfilt)
        for (int t = tidx; t < NT0; t += NB) shW1[t + tidy * NT0] = W1[t + r * NT0];
    r = tidy + blockIdx.y * NB;
    if (r < Nfilt)
        for (int t = tidx; t < NT0; t += NB) shW2[t + tidy * NT0] = W2[t + r * NT0];
    __syncthreads();
    if (tid1 < Nfilt && tid2 < Nfilt) {
        float u = UtU[tid1 + tid2 * Nfilt];
        for (int i = 0; i < NLAG; ++i) {
            float x = 0.0f;
            if (i < NT0)
                for (int t = 0; t < i + 1; ++t) x += shW1[t + NT0 * tidx] * shW2[t + (NT0 - i - 1) + NT0 * tidy];
            else
                for (int t = i - NT0 + 1; t < NT0; ++t) x += shW1[t + NT0 * tidx] * shW2[t + (NT0 - i - 1) + NT0 * tidy];
            WtW[tid1 + tid2 * Nfilt + (long long)i * Nfilt * Nfilt] = x * u;
        }
    }
}

// Max over lags of the sum over the NRANK x NRANK component pairs:
// cc[b, a] = max_lag sum_{i,j} utu[i, j, b, a] * xcorr(W[i, a], W[j, b])[lag]
extern "C" __global__
void wtw_max(ptr_t pW, ptr_t putu, ptr_t pcc, int Nfilt)
{
    const float* W = (const float*)pW;
    const float* utu = (const float*)putu;
    float* cc = (float*)pcc;
    __shared__ float shW1[NRANK * TILE * NT0], shW2[NRANK * TILE * NT0];
    const int tidx = threadIdx.x, tidy = threadIdx.y;
    const int tid1 = tidx + blockIdx.x * TILE, tid2 = tidy + blockIdx.y * TILE;
    for (int k = 0; k < NRANK; ++k) {
        int r = tidy + blockIdx.x * TILE;
        if (r < Nfilt)
            for (int t = tidx; t < NT0; t += TILE)
                shW1[t + tidy * NT0 + k * TILE * NT0] = W[t + (long long)(r + k * Nfilt) * NT0];
        r = tidy + blockIdx.y * TILE;
        if (r < Nfilt)
            for (int t = tidx; t < NT0; t += TILE)
                shW2[t + tidy * NT0 + k * TILE * NT0] = W[t + (long long)(r + k * Nfilt) * NT0];
    }
    __syncthreads();
    if (tid1 < Nfilt && tid2 < Nfilt) {
        float u[NRANK * NRANK];
        for (int i = 0; i < NRANK; ++i)
            for (int j = 0; j < NRANK; ++j)
                u[i * NRANK + j] = utu[tid1 + tid2 * Nfilt + (long long)(i * NRANK + j) * Nfilt * Nfilt];
        float best = -3.0e38f;
        for (int lag = 0; lag < NLAG; ++lag) {
            float total = 0.0f;
            int t0 = (lag < NT0) ? 0 : lag - NT0 + 1;
            int t1 = (lag < NT0) ? lag + 1 : NT0;
            int off = NT0 - lag - 1;
            for (int i = 0; i < NRANK; ++i) {
                const float* a = shW1 + NT0 * tidx + i * TILE * NT0;
                for (int j = 0; j < NRANK; ++j) {
                    const float* b = shW2 + NT0 * tidy + j * TILE * NT0 + off;
                    float x = 0.0f;
                    for (int t = t0; t < t1; ++t) x += a[t] * b[t];
                    total += x * u[i * NRANK + j];
                }
            }
            best = fmaxf(best, total);
        }
        cc[tid1 + tid2 * Nfilt] = best;
    }
}
"""


def wtw2(W1, W2, UtU):
    """mexWtW2: cross-correlation of two sets of temporal components at all
    lags, weighted by the spatial products.

    W1, W2 (Nfilt, nt0) f32, UtU (Nfilt, Nfilt) f32 (UtU[b, a] = MATLAB
    utu0(a, b)). Returns WtW (2*nt0-1, Nfilt, Nfilt) f32 with
    WtW[lag, b, a] = MATLAB WtW(a, b, lag).
    """
    W1 = _tensor(W1, torch.float32, None, 'W1')
    Nfilt, nt0 = W1.shape
    dev = W1.device
    W2 = _tensor(W2, torch.float32, (Nfilt, nt0), 'W2', dev)
    UtU = _tensor(UtU, torch.float32, (Nfilt, Nfilt), 'UtU', dev)
    out = torch.empty((2 * nt0 - 1, Nfilt, Nfilt), device=dev)
    if Nfilt == 0:
        return out
    with _device(W1), _stream():
        nb = 32
        grid = ((Nfilt + nb - 1) // nb, (Nfilt + nb - 1) // nb)
        _function('wtw', 'wtw_cross', (('NT0', nt0), ('NRANK', 1), ('TILE', 16)))(
            grid, (nb, nb), (_p(W1), _p(W2), _p(UtU), _p(out), *_i32(Nfilt)))
    return out


def wtw_max(W, U):
    """max over lags of the full template correlation (MATLAB
    `max(getMeWtW(W, U), [], 3)`, i.e. simScore) without materializing the
    (2*nt0-1, Nfilt, Nfilt) array.

    W (Nrank, Nfilt, nt0), U (Nrank, Nfilt, Nchan) (any float dtype).
    Returns cc (Nfilt, Nfilt) f32 with cc[b, a] = MATLAB simScore(a, b).
    """
    W = _tensor(W, torch.float32, None, 'W')
    Nrank, Nfilt, nt0 = W.shape
    dev = W.device
    U = _tensor(U, torch.float32, None, 'U', dev)
    if U.shape[:2] != (Nrank, Nfilt):
        raise ValueError('U must be (Nrank, Nfilt, Nchan)')
    cc = torch.empty((Nfilt, Nfilt), device=dev)
    if Nfilt == 0:
        return cc
    # utu[i, j, b, a] = sum_c U[i, a, c] U[j, b, c] = MATLAB U(:,:,i)' * U(:,:,j) at (a, b)
    utu = torch.einsum('iac,jbc->ijba', U, U).contiguous()
    tile = 16
    with _device(W), _stream():
        grid = ((Nfilt + tile - 1) // tile, (Nfilt + tile - 1) // tile)
        _function('wtw', 'wtw_max', (('NT0', nt0), ('NRANK', Nrank), ('TILE', tile)))(
            grid, (tile, tile), (_p(W), _p(utu), _p(cc), *_i32(Nfilt)))
    return cc


# --------------------------------------------------------------------------
# mexMPnu8
# --------------------------------------------------------------------------
_SOURCES['mpnu'] = r"""
typedef unsigned long long ptr_t;
#define NLAG (2 * NT0 - 1)

// The NRANK projections of one time point of the NCHANU channel rows of the
// template (the sums run over the channels in ascending order, as in the
// original; the data value is loaded once for the NRANK components).
#define PROJECT(T, rows, t, x)                                                                  \
    {                                                                                           \
        _Pragma("unroll")                                                                       \
        for (int k = 0; k < NRANK; ++k) x[k] = 0.0f;                                            \
        _Pragma("unroll 4")                                                                     \
        for (int i = 0; i < NCHANU; ++i) {                                                      \
            float d = (float)(rows[i][t]);                                                      \
            _Pragma("unroll")                                                                   \
            for (int k = 0; k < NRANK; ++k) x[k] += sU[i + NCHANU * k] * d;                     \
        }                                                                                       \
    }

// spaceFilter: dprod[k, f, t] = sum_i U[k, f, iU_f[i]] * data[iU_f[i], t]
extern "C" __global__ void __launch_bounds__(TPB)
mp_space_filter(ptr_t pdata, ptr_t pU, ptr_t piC, ptr_t piW, ptr_t pdprod, int NT, int Nfilt, int Nchan)
{
    const float* data = (const float*)pdata;
    const float* U = (const float*)pU;
    const int* iC = (const int*)piC;
    const int* iW = (const int*)piW;
    float* dprod = (float*)pdprod;
    __shared__ float sU[NCHANU * NRANK];
    __shared__ const float* rows[NCHANU];
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid < NCHANU) {
        int c = iC[tid + NCHANU * iW[bid]];
        rows[tid] = data + (long long)NT * c;
        for (int k = 0; k < NRANK; ++k) sU[tid + NCHANU * k] = U[c + Nchan * bid + (long long)Nchan * Nfilt * k];
    }
    __syncthreads();
    for (int t = tid; t < NT; t += blockDim.x) {
        float x[NRANK];
        PROJECT(float, rows, t, x)
        #pragma unroll
        for (int k = 0; k < NRANK; ++k) dprod[t + (long long)NT * (bid + k * Nfilt)] = x[k];
    }
}

// The updates after a subtraction round. The originals
// (spaceFilterUpdate / timeFilterUpdate / bestFilterUpdate) use one block
// per template that walks the list of new spikes serially (a chain of
// dependent loads per spike) and re-reads the data window of a spike once
// per overlapping template. Here one block handles one new spike: it lists
// the templates that overlap it (UtU), and the spatial update gathers the
// data window of the union of their channels into shared memory once. The
// sums are accumulated in the same order as the original kernels.

// list of the templates overlapping template f: mlist[0..*nm)
__device__ __forceinline__ void overlapping(const bool* UtU, int f, int Nfilt, int* mlist, int* nm)
{
    for (int bid = threadIdx.x; bid < Nfilt; bid += blockDim.x)
        if (UtU[f + (long long)Nfilt * bid]) mlist[atomicAdd(nm, 1)] = bid;
}

// spaceFilterUpdate: dprod of the overlapping templates at [st, st + NT0)
// (the original recomputes [st - NT0, st + NT0); the residual only changes
// in [st, st + NT0), one time point of data per value).
// dynamic shared memory: win[MAXSLOTS * NT0] floats, slot[Nchan] ints,
// chan_of[MAXSLOTS] ints, mlist[Nfilt] ints, counters.
#define SPACE_FILTER_UPDATE(NAME, T)                                                            \
extern "C" __global__                                                                           \
void NAME(ptr_t pdata, ptr_t pU, ptr_t pUtU, ptr_t piC, ptr_t piW, ptr_t pdprod,                \
          ptr_t pst, ptr_t pid, ptr_t pcounter, int NT, int Nfilt, int Nchan, int maxFR)        \
{                                                                                               \
    const T* data = (const T*)pdata;                                                            \
    const float* U = (const float*)pU;                                                          \
    const bool* UtU = (const bool*)pUtU;                                                        \
    const int* iC = (const int*)piC;                                                            \
    const int* iW = (const int*)piW;                                                            \
    float* dprod = (float*)pdprod;                                                              \
    const int* st = (const int*)pst;                                                            \
    const int* id = (const int*)pid;                                                            \
    const int* counter = (const int*)pcounter;                                                  \
    extern __shared__ float smem[];                                                             \
    float* win = smem;                                                                          \
    int* slot = (int*)(smem + MAXSLOTS * NT0);                                                  \
    int* chan_of = slot + Nchan;                                                                \
    int* mlist = chan_of + MAXSLOTS;                                                            \
    int* nm = mlist + Nfilt;                                                                    \
    int* nslots = nm + 1;                                                                       \
    const int tid = threadIdx.x;                                                                \
    const int n1 = counter[1], n0 = min(counter[0], maxFR);                                     \
    for (int ind = n1 + blockIdx.x; ind < n0; ind += gridDim.x) {                               \
        const int s = st[ind], f = id[ind];                                                     \
        __syncthreads();                                                                        \
        for (int c = tid; c < Nchan; c += blockDim.x) slot[c] = -1;                             \
        if (tid == 0) { *nm = 0; *nslots = 0; }                                                 \
        __syncthreads();                                                                        \
        overlapping(UtU, f, Nfilt, mlist, nm);                                                  \
        __syncthreads();                                                                        \
        const int m = *nm;                                                                      \
        for (int i = tid; i < m * NCHANU; i += blockDim.x)                                      \
            slot[iC[i % NCHANU + NCHANU * iW[mlist[i / NCHANU]]]] = 0;                          \
        __syncthreads();                                                                        \
        for (int c = tid; c < Nchan; c += blockDim.x)                                           \
            if (slot[c] == 0) { int j = atomicAdd(nslots, 1); slot[c] = j; if (j < MAXSLOTS) chan_of[j] = c; } \
        __syncthreads();                                                                        \
        const bool gathered = *nslots <= MAXSLOTS;                                              \
        if (gathered) {                                                                         \
            for (int i = tid; i < *nslots * NT0; i += blockDim.x) {                             \
                int j = i / NT0, t = i - j * NT0;                                               \
                win[i] = (s + t < NT) ? (float)(data[s + t + (long long)NT * chan_of[j]]) : 0.0f; \
            }                                                                                   \
        }                                                                                       \
        __syncthreads();                                                                        \
        for (int i = tid; i < m * NT0; i += blockDim.x) {                                       \
            int j = i / NT0, t = i - j * NT0;                                                   \
            int bid = mlist[j];                                                                 \
            if (s + t >= NT) continue;                                                          \
            const int* row = iC + NCHANU * iW[bid];                                             \
            const float* u = U + Nchan * bid;                                                   \
            float x[NRANK];                                                                     \
            _Pragma("unroll")                                                                   \
            for (int k = 0; k < NRANK; ++k) x[k] = 0.0f;                                        \
            for (int i2 = 0; i2 < NCHANU; ++i2) {                                               \
                int c = row[i2];                                                                \
                float d = gathered ? win[slot[c] * NT0 + t] : (float)(data[s + t + (long long)NT * c]); \
                _Pragma("unroll")                                                               \
                for (int k = 0; k < NRANK; ++k) x[k] += u[c + (long long)Nchan * Nfilt * k] * d; \
            }                                                                                   \
            _Pragma("unroll")                                                                   \
            for (int k = 0; k < NRANK; ++k) dprod[s + t + (long long)NT * (bid + k * Nfilt)] = x[k]; \
        }                                                                                       \
    }                                                                                           \
}
SPACE_FILTER_UPDATE(mp_space_filter_update_f32, float)
SPACE_FILTER_UPDATE(mp_space_filter_update_f64, double)

// timeFilter: dout[f, t] = sum_k sum_i W[k, f, i] * dprod[k, f, t + i], t < Tend.
// Grid (tiles, Nfilt), 256 threads per tile of 256 * TRF outputs; each
// thread computes TRF consecutive outputs with a sliding register window
// of the template (one shared load of the data and of the template per
// step), the terms of each output in the original order.
#define TRF 4
#define TPF 256
#define TILEF (TPF * TRF)
extern "C" __global__ void __launch_bounds__(TPF)
mp_time_filter(ptr_t pdprod, ptr_t pW, ptr_t pdout, int NT, int Nfilt, int Tend)
{
    const float* dprod = (const float*)pdprod;
    const float* W = (const float*)pW;
    float* dout = (float*)pdout;
    __shared__ float sW[NT0 * NRANK];
    __shared__ float sdata[(TILEF + NT0) * NRANK];
    const int tid = threadIdx.x, bid = blockIdx.y, t0 = blockIdx.x * TILEF;
    if (t0 >= Tend) return;
    for (int i = tid; i < NT0 * NRANK; i += TPF) sW[i] = W[i % NT0 + (bid + (long long)Nfilt * (i / NT0)) * NT0];
    for (int k = 0; k < NRANK; ++k) {
        const float* src = dprod + (long long)NT * (bid + (long long)Nfilt * k);
        for (int i = tid; i < TILEF + NT0; i += TPF) {
            int s = t0 + i;
            sdata[i + k * (TILEF + NT0)] = (s < NT) ? src[s] : 0.0f;
        }
    }
    __syncthreads();
    const int base = tid * TRF;
    float x[TRF];
    #pragma unroll
    for (int j = 0; j < TRF; ++j) x[j] = 0.0f;
    for (int k = 0; k < NRANK; ++k) {
        const float* w = sW + k * NT0;
        const float* d = sdata + k * (TILEF + NT0) + base;
        float wv[TRF];
        #pragma unroll
        for (int j = 0; j < TRF; ++j) wv[j] = 0.0f;
        #pragma unroll
        for (int t = 0; t < NT0 + TRF - 1; ++t) {
            #pragma unroll
            for (int j = TRF - 1; j > 0; --j) wv[j] = wv[j - 1];
            wv[0] = (t < NT0) ? w[t] : 0.0f;
            float dv = d[t];
            #pragma unroll
            for (int j = 0; j < TRF; ++j)
                if (t - j >= 0 && t - j < NT0) x[j] += wv[j] * dv;
        }
    }
    #pragma unroll
    for (int j = 0; j < TRF; ++j) {
        int t = t0 + base + j;
        if (t < Tend) dout[t + (long long)NT * bid] = x[j];
    }
}

// timeFilterUpdate: dout of the overlapping templates at [st - NT0, st + NT0)
// (block per spike, one half-warp per template, each lane computes TR
// consecutive outputs with a sliding register window of the template: one
// load of the data row and one (broadcast) load of the template per step;
// the terms of each output are accumulated in the original order).
// dynamic shared memory: mlist[Nfilt] ints + counter.
#define TR 8
extern "C" __global__ void __launch_bounds__(TPB)
mp_time_filter_update(ptr_t pdprod, ptr_t pW, ptr_t pUtU, ptr_t pdout, ptr_t pst, ptr_t pid,
                      ptr_t pcounter, int NT, int Nfilt, int maxFR)
{
    const float* dprod = (const float*)pdprod;
    const float* W = (const float*)pW;
    const bool* UtU = (const bool*)pUtU;
    float* dout = (float*)pdout;
    const int* st = (const int*)pst;
    const int* id = (const int*)pid;
    const int* counter = (const int*)pcounter;
    extern __shared__ int mlist[];
    int* nm = mlist + Nfilt;
    const int tid = threadIdx.x, lane = tid & 15, half = tid >> 4, nhalf = blockDim.x >> 4;
    const int n1 = counter[1], n0 = min(counter[0], maxFR);
    for (int ind = n1 + blockIdx.x; ind < n0; ind += gridDim.x) {
        const int s = st[ind], f = id[ind];
        __syncthreads();
        if (tid == 0) *nm = 0;
        __syncthreads();
        overlapping(UtU, f, Nfilt, mlist, nm);
        __syncthreads();
        const int m = *nm;
        for (int j = half; j < m; j += nhalf) {
            const int bid = mlist[j];
            const int base = s - NT0 + lane * TR;  // outputs base .. base + TR - 1
            float x[TR];
            #pragma unroll
            for (int jj = 0; jj < TR; ++jj) x[jj] = 0.0f;
            for (int k = 0; k < NRANK; ++k) {
                const float* src = dprod + (long long)NT * (bid + (long long)Nfilt * k);
                const float* w = W + NT0 * (bid + (long long)Nfilt * k);
                float wv[TR];
                #pragma unroll
                for (int jj = 0; jj < TR; ++jj) wv[jj] = 0.0f;
                #pragma unroll
                for (int t = 0; t < NT0 + TR - 1; ++t) {
                    #pragma unroll
                    for (int jj = TR - 1; jj > 0; --jj) wv[jj] = wv[jj - 1];
                    wv[0] = (t < NT0) ? w[t] : 0.0f;
                    int tt = base + t;
                    float dv = (tt >= 0 && tt < NT) ? src[tt] : 0.0f;
                    #pragma unroll
                    for (int jj = 0; jj < TR; ++jj)
                        if (t - jj >= 0 && t - jj < NT0) x[jj] += wv[jj] * dv;
                }
            }
            #pragma unroll
            for (int jj = 0; jj < TR; ++jj) {
                int tid0 = base + jj;
                if (lane * TR + jj < NLAG && tid0 >= 0 && tid0 < NT - NT0)
                    dout[tid0 + (long long)NT * bid] = x[jj];
            }
        }
    }
}

__device__ __forceinline__ void best_at(const float* dout, const float* mu, int t, int NT, int Nfilt, float lam,
                                        float& Cbest, float& Cnextbest, int& ibest)
{
    Cbest = 0.0f;
    Cnextbest = 0.0f;
    ibest = 0;
    for (int i = 0; i < Nfilt; ++i) {
        float a = 1 + lam;
        float b = fmaxf(0.0f, dout[t + (long long)NT * i]) + lam * mu[i];
        float Cf = b * b / a - lam * mu[i] * mu[i];
        if (Cf > Cbest + 1e-6f) { Cnextbest = Cbest; Cbest = Cf; ibest = i; }
        else if (Cf > Cnextbest + 1e-6f) Cnextbest = Cf;
    }
}

// bestFilter: best template at each time point in [NT0, NT - NT0)
extern "C" __global__
void mp_best_filter(ptr_t pdout, ptr_t pmu, ptr_t perr, ptr_t peloss, ptr_t pftype, int NT, int Nfilt, float lam)
{
    const float* dout = (const float*)pdout;
    const float* mu = (const float*)pmu;
    float* err = (float*)perr;
    float* eloss = (float*)peloss;
    int* ftype = (int*)pftype;
    for (int t = NT0 + threadIdx.x + blockIdx.x * blockDim.x; t < NT - NT0; t += blockDim.x * gridDim.x) {
        float Cbest, Cnextbest;
        int ibest;
        best_at(dout, mu, t, NT, Nfilt, lam, Cbest, Cnextbest, ibest);
        err[t] = Cbest;
        eloss[t] = Cbest - Cnextbest;
        ftype[t] = ibest;
    }
}

// bestFilterUpdate: recompute err / ftype around the new spikes (eloss is
// not updated, as in the original). The original allows t < NT; its loop
// bounds never reach the tail, and err is only defined for t < NT - NT0.
//
// Block per spike (TPBF threads). Only the templates that overlap the
// spikes whose windows reach this window ("changed", a superset of the
// templates whose dout changed at a point) can have a new value at a point
// t; all the others still have the value of the last scan, whose winner is
// ftype[t]. If ftype[t] is not changed, the scan over the changed templates
// plus ftype[t], in index order, gives the same (err, ftype) as the scan
// over all templates: an unchanged template other than ftype[t] lost to it
// before and cannot win now (its value is unchanged and ftype[t]'s is too).
// With the 1e-6 tolerance this holds exactly for values above 8 (the float
// resolution there exceeds 1e-6), which covers everything above Th^2 >= 16,
// the only values that the detection reads. If ftype[t] is changed, the
// full scan is done.
#define TPBF 128
#define MAXOVL 32
extern "C" __global__ void __launch_bounds__(TPBF)
mp_best_filter_update(ptr_t pdout, ptr_t pmu, ptr_t perr, ptr_t pftype, ptr_t pst, ptr_t pid, ptr_t pcounter,
                      ptr_t pUtU, int NT, int Nfilt, float lam, int maxFR)
{
    const float* dout = (const float*)pdout;
    const float* mu = (const float*)pmu;
    float* err = (float*)perr;
    int* ftype = (int*)pftype;
    const int* st = (const int*)pst;
    const int* id = (const int*)pid;
    const int* counter = (const int*)pcounter;
    const bool* UtU = (const bool*)pUtU;
    extern __shared__ int ssh[];
    int* list = ssh;                 // sorted changed templates
    int* ovl_f = list + Nfilt;       // templates of the overlapping spikes
    int* novl = ovl_f + MAXOVL;
    int* warp_sum = novl + 1;        // TPBF / 32 entries
    int* nlist = warp_sum + TPBF / 32;
    unsigned char* mark = (unsigned char*)(nlist + 1);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n1 = counter[1], n0 = min(counter[0], maxFR);
    for (int ind = n1 + blockIdx.x; ind < n0; ind += gridDim.x) {
        const int s = st[ind], f = id[ind];
        __syncthreads();
        if (tid == 0) { *novl = 0; *nlist = 0; }
        for (int i = tid; i < Nfilt; i += TPBF) mark[i] = 0;
        __syncthreads();
        // the new spikes whose windows overlap this one (they are >= NT0 - 1 apart)
        for (int j = n1 + tid; j < n0; j += TPBF) {
            int d = st[j] - s;
            if (d > -2 * NT0 && d < 2 * NT0) {
                int q = atomicAdd(novl, 1);
                if (q < MAXOVL) ovl_f[q] = id[j];
            }
        }
        __syncthreads();
        const bool full = *novl > MAXOVL;
        const int no = min(*novl, MAXOVL);
        for (int q = 0; q < no; ++q) {
            const bool* row = UtU + ovl_f[q];
            for (int i = tid; i < Nfilt; i += TPBF)
                if (row[(long long)Nfilt * i]) mark[i] = 1;
        }
        __syncthreads();
        // ordered compaction of the marked templates
        for (int c0 = 0; c0 < Nfilt; c0 += TPBF) {
            int i = c0 + tid;
            bool m = (i < Nfilt) && mark[i];
            unsigned ballot = __ballot_sync(0xffffffffu, m);
            int within = __popc(ballot & ((1u << lane) - 1));
            if (lane == 0) warp_sum[warp] = __popc(ballot);
            __syncthreads();
            int off = *nlist;
            for (int w = 0; w < warp; ++w) off += warp_sum[w];
            if (m) list[off + within] = i;
            __syncthreads();
            if (tid == 0) for (int w = 0; w < TPBF / 32; ++w) *nlist += warp_sum[w];
            __syncthreads();
        }
        const int nl = *nlist;
        if (tid < NLAG) {
            int t = s - NT0 + tid;
            if (t >= NT0 && t < NT - NT0) {
                float Cbest = 0.0f, Cnextbest = 0.0f;
                int ibest = 0;
                const int istar = ftype[t];
                if (full || mark[istar]) {
                    best_at(dout, mu, t, NT, Nfilt, lam, Cbest, Cnextbest, ibest);
                } else {
                    const float a = 1 + lam;
                    bool star_done = false;
                    for (int j = 0; j <= nl; ++j) {
                        int i;
                        if (j < nl) {
                            i = list[j];
                            if (!star_done && istar < i) { i = istar; star_done = true; --j; }
                        } else {
                            if (star_done) break;
                            i = istar;
                            star_done = true;
                        }
                        float bb = fmaxf(0.0f, dout[t + (long long)NT * i]) + lam * mu[i];
                        float Cf = bb * bb / a - lam * mu[i] * mu[i];
                        if (Cf > Cbest + 1e-6f) { Cnextbest = Cbest; Cbest = Cf; ibest = i; }
                        else if (Cf > Cnextbest + 1e-6f) Cnextbest = Cf;
                    }
                }
                err[t] = Cbest;
                ftype[t] = ibest;
            }
        }
    }
}

// cleanup_spikes: local maxima of err above Th^2 become new spikes
extern "C" __global__
void mp_cleanup_spikes(ptr_t pdout, ptr_t perr, ptr_t peloss, ptr_t pftype, ptr_t pst, ptr_t pid,
                       ptr_t px, ptr_t py, ptr_t pz, ptr_t pcounter, int NT, int Tlo, int Thi, float Th, int maxFR)
{
    const float* dout = (const float*)pdout;
    const float* err = (const float*)perr;
    const float* eloss = (const float*)peloss;
    const int* ftype = (const int*)pftype;
    int* st = (int*)pst;
    int* id = (int*)pid;
    float* x = (float*)px;
    float* y = (float*)py;
    float* z = (float*)pz;
    int* counter = (int*)pcounter;
    const int lockout = NT0 - 1;
    for (int t0 = Tlo + threadIdx.x + blockIdx.x * blockDim.x; t0 < Thi; t0 += blockDim.x * gridDim.x) {
        float err0 = err[t0];
        if (err0 > Th * Th) {
            bool flag = false;
            for (int j = -lockout; j <= lockout; ++j)
                if (err[t0 + j] > err0) { flag = true; break; }
            if (!flag) {
                int indx = atomicAdd(&counter[0], 1);
                if (indx < maxFR) {
                    int id0 = ftype[t0];
                    st[indx] = t0;
                    id[indx] = id0;
                    y[indx] = dout[t0 + (long long)NT * id0];
                    x[indx] = sqrtf(err0);
                    z[indx] = eloss[t0];
                }
            }
        }
    }
}

// counter[1] = counter[0] = min(counter[0], maxFR)
extern "C" __global__
void mp_advance_counter(ptr_t pcounter, int maxFR)
{
    int* counter = (int*)pcounter;
    int c = min(counter[0], maxFR);
    counter[0] = c;
    counter[1] = c;
}

// extractFEAT: template features of the new spikes (before subtraction)
extern "C" __global__
void mp_extract_feat(ptr_t pst, ptr_t pid, ptr_t pcounter, ptr_t pdout, ptr_t piList, ptr_t pmu, ptr_t pfeat,
                     int NT, int Nnearest, float lam, int maxFR)
{
    const int* st = (const int*)pst;
    const int* id = (const int*)pid;
    const int* counter = (const int*)pcounter;
    const float* dout = (const float*)pdout;
    const int* iList = (const int*)piList;
    const float* mu = (const float*)pmu;
    float* feat = (float*)pfeat;
    const int tidx = threadIdx.x, tidy = threadIdx.y;
    const int n0 = min(counter[0], maxFR);
    for (int ind = counter[1] + tidx + blockDim.x * blockIdx.x; ind < n0; ind += blockDim.x * gridDim.x) {
        int tcurr = st[ind];
        int idF = iList[tidy + Nnearest * id[ind]];
        float rMax = 0.0f;
        for (int t = -3; t < 3; ++t) {
            if (tcurr + t < 0 || tcurr + t >= NT) continue;  // (never true: tcurr >= NT0 - 1)
            float Ci = dout[tcurr + t + (long long)idF * NT] + lam / mu[idF];
            float Cf = Ci / sqrtf(lam / (mu[idF] * mu[idF]) + 1.0f);
            rMax = fmaxf(rMax, Cf);
        }
        feat[tidy + (long long)ind * Nnearest] = rMax;
    }
}

// subtract_spikes (float atomics) / subtract_spikes_v4 (double atomics).
// The original loops over all channels; U is zero outside the NCHANU
// channels of the template (mexSVDsmall2 blanks them), so only those are
// touched here (adding -0 or +0 would leave the value unchanged).
#define SUBTRACT_SPIKES(NAME, T)                                                                \
extern "C" __global__                                                                           \
void NAME(ptr_t pst, ptr_t pid, ptr_t py, ptr_t pcounter, ptr_t pdata, ptr_t pW, ptr_t pU,      \
          ptr_t piC, ptr_t piW, int NT, int Nchan, int Nfilt, int maxFR)                        \
{                                                                                               \
    const int* st = (const int*)pst;                                                            \
    const int* id = (const int*)pid;                                                            \
    const float* y = (const float*)py;                                                          \
    const int* counter = (const int*)pcounter;                                                  \
    T* data = (T*)pdata;                                                                        \
    const float* W = (const float*)pW;                                                          \
    const float* U = (const float*)pU;                                                          \
    const int* iC = (const int*)piC;                                                            \
    const int* iW = (const int*)piW;                                                            \
    const int tidx = threadIdx.x;                                                               \
    const int n0 = min(counter[0], maxFR);                                                      \
    for (int ind = counter[1] + blockIdx.x; ind < n0; ind += gridDim.x) {                       \
        const int f = id[ind];                                                                  \
        const int* row = iC + NCHANU * iW[f];                                                   \
        for (int j = threadIdx.y; j < NCHANU; j += blockDim.y) {                                \
            int tidy = row[j];                                                                  \
            T X = 0;                                                                            \
            for (int k = 0; k < NRANK; ++k)                                                     \
                X += (T)(W[tidx + f * NT0 + (long long)NT0 * Nfilt * k]) *                      \
                     (T)(U[tidy + f * Nchan + (long long)Nchan * Nfilt * k]);                   \
            X = -(T)(y[ind]) * X;                                                               \
            atomicAdd(&data[tidx + st[ind] + (long long)NT * tidy], X);                         \
        }                                                                                       \
    }                                                                                           \
}
SUBTRACT_SPIKES(mp_subtract_f32, float)
SUBTRACT_SPIKES(mp_subtract_f64, double)

// convToSingle (round toward zero, as the original)
extern "C" __global__
void mp_to_single_rz(ptr_t pdouble, ptr_t psingle, long long n)
{
    const double* d = (const double*)pdouble;
    float* s = (float*)psingle;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x; i < n; i += (long long)blockDim.x * gridDim.x)
        s[i] = __double2float_rz(d[i]);
}

// computePCfeatures: PC projections of the reconstructed spikes (residual +
// subtracted template) on the template's channels
extern "C" __global__
void mp_pc_features(ptr_t pdata, ptr_t pst, ptr_t pid, ptr_t py, ptr_t pW, ptr_t pU, ptr_t piW, ptr_t piC,
                    ptr_t pwPCA, ptr_t pfeatPC, int NT, int Nchan, int Nfilt, int n)
{
    const float* data = (const float*)pdata;
    const int* st = (const int*)pst;
    const int* id = (const int*)pid;
    const float* y = (const float*)py;
    const float* W = (const float*)pW;
    const float* U = (const float*)pU;
    const int* iW = (const int*)piW;
    const int* iC = (const int*)piC;
    const float* wPCA = (const float*)pwPCA;
    float* featPC = (float*)pfeatPC;
    __shared__ float sPCA[NT0 * NRANK], sW[NT0 * NRANK], sU[NCHANU * NRANK];
    __shared__ int iU[NCHANU];
    const int tidx = threadIdx.x, tidy = threadIdx.y, bid = blockIdx.x;
    if (tidy == 0) iU[tidx] = iC[tidx + NCHANU * iW[bid]];
    __syncthreads();
    sU[tidx + tidy * NCHANU] = U[iU[tidx] + Nchan * bid + (long long)Nchan * Nfilt * tidy];
    for (int t = tidx; t < NT0; t += NCHANU) {
        sW[t + tidy * NT0] = W[t + bid * NT0 + (long long)Nfilt * NT0 * tidy];
        sPCA[t + tidy * NT0] = wPCA[t + NT0 * tidy];
    }
    __syncthreads();
    float Y = 0.0f;
    for (int k = 0; k < NRANK; ++k) {
        float X = 0.0f;
        for (int t = 0; t < NT0; ++t) X += sW[t + k * NT0] * sPCA[t + tidy * NT0];
        Y += X * sU[tidx + k * NCHANU];
    }
    const float* src = data + (long long)NT * iU[tidx];
    for (int ind = 0; ind < n; ++ind) {
        if (id[ind] == bid) {
            float X = Y * y[ind];
            for (int t = 0; t < NT0; ++t) X += src[st[ind] + t] * sPCA[t + NT0 * tidy];
            featPC[tidx + tidy * NCHANU + (long long)ind * NCHANU * NRANK] = X;
        }
    }
}

// average_snips: sum of the reconstructed spikes with vexp > Th per template
// and their count. The spikes of template bid are order[offsets[bid] ..
// offsets[bid+1]) (grouped by template on the host, in the order the
// original would visit them), so each block only visits its own spikes.
// Each thread owns the (t, channel) elements it accumulates (in registers,
// written once), and the template value W*U of an element is computed once.
// The channels are split over CSPLIT blocks (grid.y) to bound the registers.
#define CSPLIT 4
#define CHPT ((NCHAN + 16 * CSPLIT - 1) / (16 * CSPLIT))
extern "C" __global__ void __launch_bounds__(TPB)
mp_average_snips(ptr_t pst, ptr_t porder, ptr_t poffsets, ptr_t px, ptr_t py, ptr_t pdata, ptr_t pW, ptr_t pU,
                      ptr_t pWU, ptr_t pnsp, int NT, int Nchan, int Nfilt, float Th)
{
    const int* st = (const int*)pst;
    const int* order = (const int*)porder;
    const int* offsets = (const int*)poffsets;
    const float* x = (const float*)px;
    const float* y = (const float*)py;
    const float* data = (const float*)pdata;
    const float* W = (const float*)pW;
    const float* U = (const float*)pU;
    double* WU = (double*)pWU;
    int* nsp = (int*)pnsp;
    const int tidx = threadIdx.x, bid = blockIdx.x;
    const int ty0 = blockIdx.y * 16 * CHPT + threadIdx.y;
    const int i0 = offsets[bid], i1 = offsets[bid + 1];
    if (i0 == i1) return;
    double X[CHPT], acc[CHPT];
    #pragma unroll
    for (int j = 0; j < CHPT; ++j) {
        int tidy = ty0 + 16 * j;
        X[j] = 0.0;
        acc[j] = 0.0;
        if (tidy < Nchan)
            for (int k = 0; k < NRANK; ++k)
                X[j] += W[tidx + bid * NT0 + (long long)NT0 * Nfilt * k] * U[tidy + bid * Nchan + (long long)Nchan * Nfilt * k];
    }
    int count = 0;
    for (int ind = i0; ind < i1; ++ind) {
        int cur = order[ind];
        if (x[cur] > Th) {
            ++count;
            const float* d = data + st[cur] + tidx;
            const float yc = y[cur];
            #pragma unroll
            for (int j = 0; j < CHPT; ++j) {
                int tidy = ty0 + 16 * j;
                if (tidy < Nchan) acc[j] += d[(long long)NT * tidy] + yc * X[j];
            }
        }
    }
    if (tidx == 0 && threadIdx.y == 0 && blockIdx.y == 0) nsp[bid] += count;
    #pragma unroll
    for (int j = 0; j < CHPT; ++j) {
        int tidy = ty0 + 16 * j;
        if (tidy < Nchan) WU[tidx + tidy * NT0 + (long long)NT0 * Nchan * bid] += acc[j];
    }
}
"""


_workspace = {}


def free_workspace():
    """Release the work buffers that `mpnu8` keeps between calls."""
    _workspace.clear()


def _mpnu_workspace(dev, NT, Nfilt, Nrank, Nchan, Nnearest, maxFR, flag, stable):
    """Work buffers of mpnu8, reused while the problem size is unchanged
    (about 1.7 GB at the Neuropixels batch size)."""
    key = (str(dev), NT, Nfilt, Nrank, Nchan, Nnearest, maxFR, flag > 1, stable)
    ws = _workspace.get(key)
    if ws is None:
        _workspace.clear()
        ws = dict(
            dprod=torch.empty((Nrank, Nfilt, NT), device=dev),
            dout=torch.empty((Nfilt, NT), device=dev),
            err=torch.empty(NT, device=dev),
            eloss=torch.empty(NT, device=dev),
            ftype=torch.empty(NT, dtype=torch.int32, device=dev),
            st=torch.empty(maxFR, dtype=torch.int32, device=dev),
            sid=torch.empty(maxFR, dtype=torch.int32, device=dev),
            x=torch.empty(maxFR, device=dev),
            y=torch.empty(maxFR, device=dev),
            z=torch.empty(maxFR, device=dev),
            counter=torch.empty(2, dtype=torch.int32, device=dev),
            feat=torch.empty((maxFR if flag > 1 else 1, Nnearest), device=dev),
            draw64=torch.empty((Nchan, NT), dtype=torch.float64, device=dev) if stable else None,
            counts=torch.empty((512, 2), dtype=torch.int32, pin_memory=True),
        )
        _workspace[key] = ws
    return ws


def mpnu8(Params, X, U, W, mu, iC, iW, UtU, iList, wPCA, maxFR=100000,
          early_stop=True, sync_every=8, matlab_tail=False):
    """mexMPnu8: matching pursuit of the templates on one batch.

    X (Nchan, NT) f32 (not modified), U (Nrank, Nfilt, Nchan) f32,
    W (Nrank, Nfilt, nt0) f32, mu (Nfilt,) f32, iC (Nchan, NchanNear) i32,
    iW (Nfilt,) i32, UtU (Nfilt, Nfilt) bool, iList (Nfilt, Nnearest) i32,
    wPCA (>=Nrank, nt0) f32 (the first Nrank rows are used).

    Returns a dict: st0 (nsp,) i32 template start times, id0 (nsp,) i32,
    x0 (nsp,) f32 amplitudes (MATLAB x0 = d_y), vexp (nsp,) f32 (sqrt of the
    explained variance, MATLAB vexp = d_x), featW (nsp, Nnearest) f32,
    featPC (nsp, Nrank, NchanNear) f32, dWU0 (Nfilt, Nchan, nt0) f64,
    nsp0 (Nfilt,) i32, drez (Nchan, NT) f32 (the residual).

    `early_stop` ends the inner loop once an iteration finds no new spike
    (identical outputs); the spike counter is checked on the host every
    `sync_every` iterations (the only host synchronizations of the loop).
    `matlab_tail` reproduces the loop bounds of the mex file, which never
    scan the last ~Nthreads+nt0 samples. Work buffers are kept between calls
    (see `free_workspace`).
    """
    p = _params(Params)
    NT, Nfilt, Th, nInnerIter, nt0 = int(p[0]), int(p[1]), float(p[2]), int(p[3]), int(p[4])
    Nnearest, Nrank, lam, Nchan, NchanU = int(p[5]), int(p[6]), float(p[7]), int(p[9]), int(p[10])
    flag, Th2, stable = int(p[12]), float(p[15]), bool(int(p[16]))
    X = _tensor(X, torch.float32, (Nchan, NT), 'X')
    dev = X.device
    U = _tensor(U, torch.float32, (Nrank, Nfilt, Nchan), 'U', dev)
    W = _tensor(W, torch.float32, (Nrank, Nfilt, nt0), 'W', dev)
    mu = _tensor(mu, torch.float32, (Nfilt,), 'mu', dev)
    iC = _tensor(iC, torch.int32, (Nchan, NchanU), 'iC', dev)
    iW = _tensor(iW, torch.int32, (Nfilt,), 'iW', dev)
    UtU = _tensor(UtU, torch.bool, (Nfilt, Nfilt), 'UtU', dev)
    iList = _tensor(iList, torch.int32, (Nfilt, Nnearest), 'iList', dev)
    wPCA = _tensor(wPCA, torch.float32, None, 'wPCA', dev)
    if wPCA.shape[0] < Nrank or wPCA.shape[1] != nt0:
        raise ValueError('wPCA must be (>=Nrank, nt0)')
    if nt0 * Nrank > TPB or NchanU * Nrank > TPB or NchanU > TPB or 16 * Nnearest > TPB or 16 * nt0 > TPB:
        raise ValueError('unsupported nt0 / Nrank / NchanNear / Nnearest')
    if 2 * nt0 - 1 > 128:
        raise ValueError('nt0 too large (2 * nt0 - 1 must fit in 128 threads)')

    drez = X.clone()
    out = dict(st0=torch.empty(0, dtype=torch.int32, device=dev),
               id0=torch.empty(0, dtype=torch.int32, device=dev),
               x0=torch.empty(0, device=dev), vexp=torch.empty(0, device=dev),
               featW=torch.zeros((0, Nnearest), device=dev),
               featPC=torch.zeros((0, Nrank, NchanU), device=dev),
               dWU0=torch.zeros((Nfilt, Nchan, nt0), dtype=torch.float64, device=dev),
               nsp0=torch.zeros(Nfilt, dtype=torch.int32, device=dev), drez=drez)
    if Nfilt == 0 or NT < 2 * nt0 or nInnerIter <= 0:
        return out
    defs = (('NT0', nt0), ('NRANK', Nrank), ('NCHANU', NchanU), ('TPB', TPB),
            ('MAXSLOTS', MAXSLOTS), ('NCHAN', Nchan))
    kern = lambda name: _function('mpnu', name, defs)

    with _device(X), _stream():
        ws = _mpnu_workspace(dev, NT, Nfilt, Nrank, Nchan, Nnearest, maxFR, flag, stable)
        dprod, dout, err, eloss, ftype = ws['dprod'], ws['dout'], ws['err'], ws['eloss'], ws['ftype']
        st, sid, x, y, z, counter, feat = (ws[n] for n in ('st', 'sid', 'x', 'y', 'z', 'counter', 'feat'))
        draw64 = ws['draw64']
        if stable:
            draw64.copy_(X)
        # err / eloss / ftype are only written in [nt0, NT - nt0) (the mex
        # files memset them); dout only below Tend
        err.zero_()
        eloss.zero_()
        ftype.zero_()
        counter.zero_()
        Tend = _tail(NT, nt0, matlab_tail)
        if Tend < NT:
            dout[:, Tend:].zero_()

        nblk = (NT + TPB - 1) // TPB
        kern('mp_space_filter')((Nfilt,), (TPB,),
                                (_p(drez), _p(U), _p(iC), _p(iW), _p(dprod), *_i32(NT, Nfilt, Nchan)))
        if Tend > 0:
            kern('mp_time_filter')(((Tend + 1023) // 1024, Nfilt), (256,),
                                   (_p(dprod), _p(W), _p(dout), *_i32(NT, Nfilt, Tend)))
        kern('mp_best_filter')((nblk,), (TPB,),
                               (_p(dout), _p(mu), _p(err), _p(eloss), _p(ftype), *_i32(NT, Nfilt), *_f32(lam)))

        Tlo, Thi = _cleanup_range(NT, nt0 - 1, matlab_tail, strict=False)
        cleanup = kern('mp_cleanup_spikes')
        extract_feat = kern('mp_extract_feat')
        if stable:
            subtract, sfu, data_sub = kern('mp_subtract_f64'), kern('mp_space_filter_update_f64'), draw64
        else:
            subtract, sfu, data_sub = kern('mp_subtract_f32'), kern('mp_space_filter_update_f32'), drez
        tfu = kern('mp_time_filter_update')
        bfu = kern('mp_best_filter_update')
        advance = kern('mp_advance_counter')
        ty = max(1, min(16, TPB // nt0))
        # the new spikes of an iteration are local maxima over +-(nt0-1), so
        # there are at most about NT / nt0 of them: one block each
        gspk = max(1, min(4096, NT // nt0 + 1))
        smem_sfu = 4 * (MAXSLOTS * nt0 + Nchan + MAXSLOTS + Nfilt + 2)
        smem_tfu = 4 * (Nfilt + 1)
        smem_bfu = 4 * (Nfilt + 32 + 1 + 4 + 1) + Nfilt + 16
        sfu.max_dynamic_shared_size_bytes = smem_sfu
        tfu.max_dynamic_shared_size_bytes = smem_tfu
        counts = ws['counts']
        if early_stop and nInnerIter > counts.shape[0]:
            counts = torch.empty((nInnerIter, 2), dtype=torch.int32, pin_memory=True)
        sync_every = max(1, int(sync_every))
        for k in range(nInnerIter):
            if Thi > Tlo:
                cleanup((nblk,), (TPB,),
                        (_p(dout), _p(err), _p(eloss), _p(ftype), _p(st), _p(sid), _p(x), _p(y), _p(z),
                         _p(counter), *_i32(NT, Tlo, Thi), *_f32(Th), *_i32(maxFR)))
            if early_stop:
                counts[k].copy_(counter, non_blocking=True)
            if flag > 1:
                extract_feat((64,), (16, Nnearest),
                             (_p(st), _p(sid), _p(counter), _p(dout), _p(iList), _p(mu), _p(feat),
                              *_i32(NT, Nnearest), *_f32(lam), *_i32(maxFR)))
            subtract((Nfilt,), (nt0, ty),
                     (_p(st), _p(sid), _p(y), _p(counter), _p(data_sub), _p(W), _p(U), _p(iC), _p(iW),
                      *_i32(NT, Nchan, Nfilt, maxFR)))
            sfu((gspk,), (TPB,),
                (_p(data_sub), _p(U), _p(UtU), _p(iC), _p(iW), _p(dprod), _p(st), _p(sid), _p(counter),
                 *_i32(NT, Nfilt, Nchan, maxFR)), shared_mem=smem_sfu)
            tfu((gspk,), (TPB,),
                (_p(dprod), _p(W), _p(UtU), _p(dout), _p(st), _p(sid), _p(counter), *_i32(NT, Nfilt, maxFR)),
                shared_mem=smem_tfu)
            bfu((gspk,), (128,),
                (_p(dout), _p(mu), _p(err), _p(ftype), _p(st), _p(sid), _p(counter), _p(UtU), *_i32(NT, Nfilt),
                 *_f32(lam), *_i32(maxFR)), shared_mem=smem_bfu)
            advance((1,), (1,), (_p(counter), *_i32(maxFR)))
            if early_stop and (k + 1) % sync_every == 0 and k + 1 < nInnerIter:
                # One host synchronization per `sync_every` iterations: an
                # iteration that found no new spike is a fixed point, so the
                # remaining iterations would be no-ops.
                torch.cuda.current_stream().synchronize()
                c = counts[k + 1 - sync_every:k + 1]
                if bool((c[:, 0] == c[:, 1]).any()):
                    break

        n = min(int(counter[0].item()), maxFR)
        if stable:
            kern('mp_to_single_rz')((1024,), (TPB,), (_p(draw64), _p(drez), np.int64(NT * Nchan)))
        featPC = torch.zeros((n, Nrank, NchanU), device=dev)
        if flag > 0 and n > 0:
            kern('mp_pc_features')((Nfilt,), (NchanU, Nrank),
                                   (_p(drez), _p(st), _p(sid), _p(y), _p(W), _p(U), _p(iW), _p(iC), _p(wPCA),
                                    _p(featPC), *_i32(NT, Nchan, Nfilt, n)))
        dWU0, nsp0 = out['dWU0'], out['nsp0']
        if n > 0:
            # visiting order of the original: time-sorted in stable mode,
            # detection order otherwise; then grouped by template (stable)
            ids = sid[:n].long()
            if stable:
                idx = torch.argsort(st[:n], stable=True)
                ids = ids[idx]
            else:
                idx = torch.arange(n, device=dev)
            perm = torch.argsort(ids, stable=True)
            order = idx[perm].to(torch.int32).contiguous()
            offsets = torch.zeros(Nfilt + 1, dtype=torch.int64, device=dev)
            offsets[1:] = torch.cumsum(torch.bincount(ids, minlength=Nfilt), 0)
            offsets = offsets.to(torch.int32).contiguous()
            kern('mp_average_snips')((Nfilt, 4), (nt0, 16),
                                     (_p(st), _p(order), _p(offsets), _p(x), _p(y), _p(drez), _p(W), _p(U),
                                      _p(dWU0), _p(nsp0), *_i32(NT, Nchan, Nfilt), *_f32(Th2)))
        out.update(st0=st[:n].clone(), id0=sid[:n].clone(), x0=y[:n].clone(), vexp=x[:n].clone(),
                   featW=feat[:n].clone() if flag > 1 else torch.zeros((n, Nnearest), device=dev),
                   featPC=featPC)
    return out
