"""Optional fused CUDA kernels for the hot loops of Kilosort4.

The kernels are compiled at runtime with NVRTC through CuPy, so no CUDA
toolkit is needed, only a CuPy wheel matching the CUDA version of PyTorch
(e.g. ``pip install cupy-cuda12x`` or ``cupy-cuda13x``). Without CuPy, or if a
kernel fails to compile, Kilosort uses the PyTorch implementations.

The kernels perform the same floating point operations in the same order as
the PyTorch code they replace (compiled with ``--fmad=false``, IEEE division
and square root), and the matrix products that go through cuBLAS / cuDNN
are left to PyTorch. As a safeguard, `verified_call` runs both versions on
the first call and only switches to the fused kernel if the outputs are
bit-identical.

Set the environment variable ``KILOSORT_FUSED_KERNELS=0`` to disable them.

``KILOSORT_FUSED_VERIFY`` controls how many calls are checked against the
PyTorch implementation: ``1`` (default) checks the first call of each kernel
and silently falls back on a mismatch, an integer ``N`` checks the first N
calls, and ``all`` checks every call. With an explicit setting a mismatch
raises an error, so a full run with ``KILOSORT_FUSED_VERIFY=all`` proves that
the kernels are bit-identical on that data.
"""
import logging
import os
import warnings

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

logger = logging.getLogger(__name__)

_state = {}

_INSTALL_HINT = (
    "Install CuPy with the CUDA headers matching PyTorch's CUDA version, e.g. "
    "`pip install \"cupy-cuda12x[ctk]\"` or `pip install \"cupy-cuda13x[ctk]\"` "
    "(torch.version.cuda = {cuda}). Without them Kilosort uses the slower "
    "PyTorch code paths; set KILOSORT_FUSED_KERNELS=0 to silence this warning."
    )


def enabled():
    return os.environ.get('KILOSORT_FUSED_KERNELS', '1') != '0'


def _verify_setting():
    """(number of calls to verify, whether a mismatch is an error)."""
    v = os.environ.get('KILOSORT_FUSED_VERIFY')
    if v is None:
        return 1, False
    v = v.strip().lower()
    if v == 'all':
        return float('inf'), True
    try:
        return int(v), True
    except ValueError:
        logger.warning(f'Ignoring KILOSORT_FUSED_VERIFY={v!r} (expected an '
                       'integer or "all"); verifying the first call only.')
        return 1, False


def _describe_mismatch(ref, new):
    if len(ref) != len(new):
        return f'{len(ref)} vs {len(new)} outputs'
    parts = []
    for i, (a, b) in enumerate(zip(ref, new)):
        if a.shape != b.shape or a.dtype != b.dtype:
            parts.append(f'output {i}: {tuple(a.shape)} {a.dtype} vs '
                         f'{tuple(b.shape)} {b.dtype}')
        elif not torch.equal(a, b):
            n = int((a != b).sum())
            parts.append(f'output {i}: {n} of {a.numel()} elements differ')
    return '; '.join(parts)


def verified_call(key, reference, fused, *args, **kwargs):
    """Call `fused` instead of `reference` once it has been checked.

    On the first call(s) both are run and the reference output is returned.
    Later calls use `fused` only if the checked outputs were bit-identical;
    any exception from the fused path (e.g. CuPy not installed) also falls
    back to `reference`. See the module docstring for KILOSORT_FUSED_VERIFY.
    """
    state = _state.setdefault(key, {'n_checked': 0, 'use_fused': None})
    n_verify, strict = _verify_setting()
    if state['use_fused'] is None and state['n_checked'] >= n_verify:
        # n_verify == 0: use the fused kernel without checking it.
        state['use_fused'] = True
    if state['use_fused'] is None:
        ref = reference(*args, **kwargs)
        try:
            new = fused(*args, **kwargs)
        except Exception as e:
            state['use_fused'] = False
            cuda = getattr(torch.version, 'cuda', None)
            logger.warning(f'Fused CUDA kernel for {key} unavailable ({e!r}); '
                           'using the PyTorch implementation. '
                           + _INSTALL_HINT.format(cuda=cuda))
            return ref
        state['n_checked'] += 1
        same = (len(ref) == len(new)
                and all(torch.equal(a, b) for a, b in zip(ref, new)))
        if not same:
            msg = (f'Fused kernel for {key} does not match the PyTorch '
                   f'implementation on call {state["n_checked"]}: '
                   f'{_describe_mismatch(ref, new)}.')
            if strict:
                raise RuntimeError(msg)
            state['use_fused'] = False
            logger.warning(msg + ' Using the PyTorch implementation.')
        elif state['n_checked'] >= n_verify:
            state['use_fused'] = True
            logger.info(f'Fused kernel for {key} verified on '
                        f'{state["n_checked"]} call(s).')
        return ref
    if state['use_fused']:
        return fused(*args, **kwargs)
    return reference(*args, **kwargs)


def reset_verification():
    """Forget the verification results (e.g. after changing the kernels)."""
    _state.clear()


_modules = {}


def _function(src_name, name):
    import cupy as cp
    if src_name not in _modules:
        _modules[src_name] = cp.RawModule(code=_SOURCES[src_name],
                                          options=('--fmad=false',))
    return _modules[src_name].get_function(name)


def _stream():
    """CuPy view of PyTorch's current CUDA stream."""
    import cupy as cp
    stream = torch.cuda.current_stream()
    if hasattr(stream, '__cuda_stream__'):
        return cp.cuda.Stream.from_external(stream)
    # Older PyTorch (< 2.6 or so) doesn't implement the CUDA stream protocol.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return cp.cuda.ExternalStream(stream.cuda_stream)


def _p(t):
    # Pass raw device pointers (declared as unsigned long long in the kernels)
    # to avoid a DLPack conversion per argument per launch.
    return np.uint64(t.data_ptr())


_SOURCES = {}


# --------------------------------------------------------------------------
# spikedetect.template_match
# --------------------------------------------------------------------------
_SOURCES['template_match'] = r"""
typedef unsigned long long ptr_t;

// For each (template f, time t): A[s, l] = sum_j weigh[s, j, f] * B[iC[j, f], l, t]
// and the max of |A| over (s, l), its flat index q = s*NK + l and the sign.
extern "C" __global__
void as_kernel(ptr_t pB, ptr_t pw, ptr_t piC, ptr_t pAs, ptr_t pim,
               int NT, int F, int NK, int NC, int S)
{
    const float* B = (const float*)pB;           // (C, NK, NT)
    const float* weigh = (const float*)pw;       // (S, NC, F)
    const long long* iC = (const long long*)piC; // (NC, F)
    float* As = (float*)pAs;                     // (F, NT)
    int* imax = (int*)pim;                       // (F, NT), (1 + q) * sign
    const int f = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ float sh[];
    float* w = sh;                               // S*NC
    int* ch = (int*)(sh + S * NC);               // NC
    for (int k = threadIdx.x; k < S * NC; k += blockDim.x) {
        int s = k / NC, j = k % NC;
        w[k] = weigh[((long long)s * NC + j) * F + f];
    }
    for (int j = threadIdx.x; j < NC; j += blockDim.x) ch[j] = (int)iC[(long long)j * F + f];
    __syncthreads();
    if (t >= NT) return;

    float best = -1.0f, bval = 0.0f;
    int bq = 0;
    float b[16];
    for (int l = 0; l < NK; ++l) {
        for (int j = 0; j < NC; ++j)
            b[j] = B[((long long)ch[j] * NK + l) * NT + t];
        for (int s = 0; s < S; ++s) {
            // Same rounding as the cuBLAS batched matmul used by torch.einsum:
            // the two halves of the sum are accumulated separately with FMA.
            float a0 = 0.0f, a1 = 0.0f;
            const int h = NC / 2;
            for (int j = 0; j < h; ++j) a0 = fmaf(w[s * NC + j], b[j], a0);
            for (int j = h; j < NC; ++j) a1 = fmaf(w[s * NC + j], b[j], a1);
            float acc = a0 + a1;
            float v = fabsf(acc);
            int q = s * NK + l;
            // torch.max returns the lowest index among ties
            if (v > best || (v == best && q < bq)) { best = v; bq = q; bval = acc; }
        }
    }
    int sgn = (bval > 0.0f) - (bval < 0.0f);
    As[(long long)f * NT + t] = best;
    imax[(long long)f * NT + t] = (1 + bq) * sgn;
}

// Amax[f, t] = max_n As[iC2[n, f], t]
extern "C" __global__
void nbr_kernel(ptr_t pAs, ptr_t piC2, ptr_t pAm, int NT, int F, int N2)
{
    const float* As = (const float*)pAs;
    const long long* iC2 = (const long long*)piC2;
    float* Amax = (float*)pAm;
    const int f = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ int nb[];
    for (int n = threadIdx.x; n < N2; n += blockDim.x) nb[n] = (int)iC2[(long long)n * F + f];
    __syncthreads();
    if (t >= NT) return;
    float m = As[(long long)nb[0] * NT + t];
    for (int n = 1; n < N2; ++n) m = fmaxf(m, As[(long long)nb[n] * NT + t]);
    Amax[(long long)f * NT + t] = m;
}
"""


def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    """Fused version of `spikedetect.template_match`.

    The PyTorch version materializes B[iC] and the (S*NK, Nfilt, t) products
    in 40 chunks (~1.5 GB of memory traffic per batch). Here each
    (template, time) pair is computed in place, and the neighbor max is a
    second small kernel.
    """
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    S, NC, _ = weigh.shape
    if NC > 16 or weigh.dtype != torch.float32:
        raise ValueError('unsupported shape or dtype for fused template_match')
    tpb = 256

    W = ops['wTEMP'].unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=nt//2).contiguous()

    As = torch.empty((Nfilt, NT), device=device)
    imaxs = torch.empty((Nfilt, NT), dtype=torch.int32, device=device)
    Amaxs = torch.empty((Nfilt, NT), device=device)
    weigh = weigh.contiguous()
    iC_l = iC.long().contiguous()
    iC2_l = iC2.long().contiguous()

    as_kernel = _function('template_match', 'as_kernel')
    nbr_kernel = _function('template_match', 'nbr_kernel')
    grid = (Nfilt, (NT + tpb - 1) // tpb)
    with _stream():
        as_kernel(grid, (tpb,),
                  (_p(B), _p(weigh), _p(iC_l), _p(As), _p(imaxs),
                   np.int32(NT), np.int32(Nfilt), np.int32(nk), np.int32(NC),
                   np.int32(S)),
                  shared_mem=(S * NC + NC) * 4)
        nbr_kernel(grid, (tpb,),
                   (_p(As), _p(iC2_l), _p(Amaxs), np.int32(NT), np.int32(Nfilt),
                    np.int32(iC2_l.shape[0])),
                   shared_mem=iC2_l.shape[0] * 4)

    # Same as the PyTorch implementation from here on.
    Amaxs[:,:nt] = 0
    Amaxs[:,-nt:] = 0
    Amaxs  = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride = 1, padding = nt0).squeeze(0)
    xy = torch.logical_and(Amaxs==As, As > ops['Th_universal']).nonzero()
    imax = imaxs[xy[:,0], xy[:,1]].long()
    amp = As[xy[:,0], xy[:,1]]

    ssign = imax.sign()
    imax = imax.abs()-1
    adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] * ssign

    return xy, imax, amp, adist


# --------------------------------------------------------------------------
# template_matching.run_matching
# --------------------------------------------------------------------------
_SOURCES['run_matching'] = r"""
typedef unsigned long long ptr_t;

// Cf = relu(B)**2 / nm (zero at the edges), then max over templates with the
// lowest index among ties, without materializing Cf.
extern "C" __global__
void cfmax_kernel(ptr_t pB, ptr_t pnm, ptr_t pCf, ptr_t pim, int NU, int NT, int nt)
{
    const float* B = (const float*)pB;
    const float* nm = (const float*)pnm;
    float* Cfmax = (float*)pCf;
    long long* imax = (long long*)pim;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NT) return;
    float best = 0.0f;
    long long bi = 0;
    if (t >= nt && t < NT - nt) {
        float r = B[t];
        r = r < 0.0f ? 0.0f : r;
        best = (r * r) / nm[0];
        for (int i = 1; i < NU; ++i) {
            float v = B[(long long)i * NT + t];
            v = v < 0.0f ? 0.0f : v;
            float c = (v * v) / nm[i];
            if (c > best) { best = c; bi = i; }
        }
    }
    Cfmax[t] = best;
    imax[t] = bi;
}

// Cmax = max_pool1d(Cfmax, 2*nt+1); mask = (Cmax > Th**2) & (|Cmax - Cfmax| < 1e-9)
extern "C" __global__
void pool_kernel(ptr_t pCf, ptr_t pCm, ptr_t pmask, int NT, int nt, float th2, float eps)
{
    const float* Cfmax = (const float*)pCf;
    float* Cmax = (float*)pCm;
    bool* mask = (bool*)pmask;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NT) return;
    int a = t - nt < 0 ? 0 : t - nt;
    int b = t + nt > NT - 1 ? NT - 1 : t + nt;
    float m = Cfmax[a];
    for (int i = a + 1; i <= b; ++i) { float v = Cfmax[i]; if (v > m) m = v; }
    Cmax[t] = m;
    float d = m - Cfmax[t];
    mask[t] = (m > th2) && (fabsf(d) < eps);
}

// st[k+s] = (x, imax[x]); amps[k+s] = B[y, x] / nm[y]; th_amps[k+s] = Cmax[x]**0.5
extern "C" __global__
void record_kernel(ptr_t pxs, ptr_t pim, ptr_t pB, ptr_t pnm, ptr_t pCm,
                   ptr_t pst, ptr_t pamp, ptr_t pth, int n, int k, int NT)
{
    const long long* xs = (const long long*)pxs;
    const long long* imax = (const long long*)pim;
    const float* B = (const float*)pB;
    const float* nm = (const float*)pnm;
    const float* Cmax = (const float*)pCm;
    long long* st = (long long*)pst;
    float* amps = (float*)pamp;
    float* th = (float*)pth;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    long long x = xs[s];
    long long y = imax[x];
    st[(long long)(k + s) * 2] = x;
    st[(long long)(k + s) * 2 + 1] = y;
    amps[k + s] = B[y * NT + x] / nm[y];
    th[k + s] = sqrtf(Cmax[x]);
}

// Peel off half j of the spikes (s = j, j+2, ...), step 1: read and subtract
// into temporaries (like the gather in `B[:, idx] -= ...`):
//   tB[q,u,tau] = B[u, x+tau-nt]    - amp * ctc[u, y, tau]
//   tX[q,c,l]   = Xres[c, x+l-nt/2] - amp * E[c, q, l]
extern "C" __global__
void upd1_kernel(ptr_t pst, ptr_t pamp, ptr_t pB, ptr_t pctc, ptr_t pX, ptr_t pE,
                 ptr_t ptB, ptr_t ptX, int k, int j, int m, int NU, int NC, int NT, int nt,
                 long long e0, long long e1, long long e2)
{
    const long long* st = (const long long*)pst;
    const float* amps = (const float*)pamp;
    const float* B = (const float*)pB;
    const float* ctc = (const float*)pctc;
    const float* X = (const float*)pX;
    const float* E = (const float*)pE;
    float* tB = (float*)ptB;
    float* tX = (float*)ptX;
    int row = blockIdx.x;                  // 0 .. m*(NU+NC)-1
    int q = row / (NU + NC), r = row % (NU + NC);
    int s = k + j + 2 * q;
    long long x = st[(long long)s * 2];
    float amp = amps[s];
    int tau = threadIdx.x;
    if (r < NU) {
        int nw = 2 * nt + 1;
        if (tau >= nw) return;
        long long y = st[(long long)s * 2 + 1];
        float p = amp * ctc[((long long)r * NU + y) * nw + tau];
        tB[((long long)q * NU + r) * nw + tau] = B[(long long)r * NT + x + tau - nt] - p;
    } else {
        int c = r - NU, nw = 2 * (nt / 2) + 1;
        if (tau >= nw) return;
        float p = amp * E[c * e0 + q * e1 + tau * e2];
        tX[((long long)q * NC + c) * nw + tau] = X[(long long)c * NT + x + tau - nt / 2] - p;
    }
}

// Step 2: write the temporaries back (like the scatter in `B[:, idx] -= ...`).
extern "C" __global__
void upd2_kernel(ptr_t pst, ptr_t pB, ptr_t pX, ptr_t ptB, ptr_t ptX,
                 int k, int j, int m, int NU, int NC, int NT, int nt)
{
    const long long* st = (const long long*)pst;
    float* B = (float*)pB;
    float* X = (float*)pX;
    const float* tB = (const float*)ptB;
    const float* tX = (const float*)ptX;
    int row = blockIdx.x;
    int q = row / (NU + NC), r = row % (NU + NC);
    int s = k + j + 2 * q;
    long long x = st[(long long)s * 2];
    int tau = threadIdx.x;
    if (r < NU) {
        int nw = 2 * nt + 1;
        if (tau >= nw) return;
        B[(long long)r * NT + x + tau - nt] = tB[((long long)q * NU + r) * nw + tau];
    } else {
        int c = r - NU, nw = 2 * (nt / 2) + 1;
        if (tau >= nw) return;
        X[(long long)c * NT + x + tau - nt / 2] = tX[((long long)q * NC + c) * nw + tau];
    }
}
"""


def run_matching(ops, X, U, ctc, device=torch.device('cuda')):
    """Fused version of `template_matching.run_matching`.

    Each peel of the PyTorch version launches ~70 small kernels and writes
    Cf = relu(B)**2 / nm (n_templates x NT) three times. Here the max over
    templates, the pooling / threshold test, the bookkeeping and the B / Xres
    updates are a handful of kernels. The matrix products (conv1d, the B
    einsum and the per-spike waveform einsum) are the same PyTorch calls.
    """
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()

    nm = (U**2).sum(-1).sum(-1)

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', U, B)

    st = torch.zeros((100000,2), dtype = torch.int64, device = device)
    amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    th_amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    k = 0

    Xres = X.clone(memory_format=torch.contiguous_format)

    B = B.contiguous()
    nm = nm.contiguous()
    ctc = ctc.contiguous()
    NU, NT = B.shape
    NC = Xres.shape[0]
    if (ctc.shape != (NU, NU, 2*nt + 1) or Xres.shape[1] != NT
            or not (B.dtype == nm.dtype == ctc.dtype == Xres.dtype == torch.float32)):
        raise ValueError('unsupported shape or dtype for fused run_matching')
    Cfmax = torch.empty(NT, device=device)
    imax = torch.empty(NT, dtype=torch.int64, device=device)
    Cmax = torch.empty(NT, device=device)
    mask = torch.empty(NT, dtype=torch.bool, device=device)
    th2 = np.float32(Th**2)
    eps = np.float32(1e-9)
    tpb = 128
    gridT = ((NT + tpb - 1) // tpb,)
    nwB, nwX = 2*nt + 1, 2*(nt//2) + 1
    tpb_upd = (max(nwB, nwX) + 31) // 32 * 32

    cfmax = _function('run_matching', 'cfmax_kernel')
    pool = _function('run_matching', 'pool_kernel')
    record = _function('run_matching', 'record_kernel')
    upd1 = _function('run_matching', 'upd1_kernel')
    upd2 = _function('run_matching', 'upd2_kernel')

    with _stream():
        for t in range(max_peels):
            cfmax(gridT, (tpb,), (_p(B), _p(nm), _p(Cfmax), _p(imax),
                                  np.int32(NU), np.int32(NT), np.int32(nt)))
            pool(gridT, (tpb,), (_p(Cfmax), _p(Cmax), _p(mask),
                                 np.int32(NT), np.int32(nt), th2, eps))
            xs = torch.nonzero(mask)
            nsp = len(xs)
            if nsp == 0:
                break
            if k + nsp > st.shape[0]:
                raise RuntimeError('Too many spikes in one batch for run_matching.')
            record(((nsp + tpb - 1) // tpb,), (tpb,),
                   (_p(xs), _p(imax), _p(B), _p(nm), _p(Cmax),
                    _p(st), _p(amps), _p(th_amps),
                    np.int32(nsp), np.int32(k), np.int32(NT)))
            iY = st[k:k+nsp, 1:2]
            # As in the PyTorch version, every other spike is subtracted at
            # once so that the subtracted windows don't overlap.
            for j in range(2):
                m = (nsp - j + 1) // 2
                if m == 0:
                    continue
                # Same call as the PyTorch version, so the waveforms have the
                # same cuBLAS rounding.
                E = torch.einsum('ijk, jl -> kil', U[iY[j::2,0]], W)
                tB = torch.empty((m, NU, nwB), device=device)
                tX = torch.empty((m, NC, nwX), device=device)
                e0, e1, e2 = E.stride()
                upd1((m * (NU + NC),), (tpb_upd,),
                     (_p(st), _p(amps), _p(B), _p(ctc), _p(Xres), _p(E),
                      _p(tB), _p(tX), np.int32(k), np.int32(j), np.int32(m),
                      np.int32(NU), np.int32(NC), np.int32(NT), np.int32(nt),
                      np.int64(e0), np.int64(e1), np.int64(e2)))
                upd2((m * (NU + NC),), (tpb_upd,),
                     (_p(st), _p(B), _p(Xres), _p(tB), _p(tX),
                      np.int32(k), np.int32(j), np.int32(m),
                      np.int32(NU), np.int32(NC), np.int32(NT), np.int32(nt)))
            k += nsp

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    return st, amps, th_amps, Xres
