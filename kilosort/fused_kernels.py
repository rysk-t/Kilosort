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
(for each distinct problem size; calls whose outputs are empty, e.g. a
silent first batch, do not count) and silently falls back on a mismatch, an
integer ``N`` checks the first N calls, and ``all`` checks every call. With an explicit setting a mismatch
raises an error, so a full run with ``KILOSORT_FUSED_VERIFY=all`` proves that
the kernels are bit-identical on that data.
"""
import itertools
import logging
import os
import warnings
import weakref

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


def _signature(args, kwargs):
    """Shapes / dtypes of the tensor arguments and the plain scalars, so that
    the verification is redone for every distinct problem size (a second
    run_kilosort in the same process, another probe, other settings)."""
    def sig(v):
        if torch.is_tensor(v):
            return (tuple(v.shape), str(v.dtype), str(v.device))
        if isinstance(v, (bool, int, float, str, np.integer, np.floating)):
            return v
        if isinstance(v, dict):
            return tuple((k, sig(x)) for k, x in v.items()
                         if torch.is_tensor(x) or isinstance(
                             x, (bool, int, float, str, dict, np.integer, np.floating)))
        return None
    return (tuple(sig(a) for a in args), tuple((k, sig(v)) for k, v in kwargs.items()))


def _on_device(fused, args, kwargs):
    """Call `fused` with PyTorch's and CuPy's current device set to the
    device of its first CUDA tensor argument, so that the kernels are loaded
    and launched on that GPU (on PyTorch's current stream there) when
    Kilosort runs on a device other than the current one, e.g. cuda:1."""
    for v in itertools.chain(args, kwargs.values()):
        if torch.is_tensor(v) and v.is_cuda:
            import cupy as cp
            index = v.device.index
            with torch.cuda.device(index), cp.cuda.Device(index):
                return fused(*args, **kwargs)
    return fused(*args, **kwargs)


def verified_call(key, reference, fused, *args, **kwargs):
    """Call `fused` instead of `reference` once it has been checked.

    On the first call(s) with a given problem size both are run and the
    reference output is returned. Later calls use `fused` only if the
    checked outputs were bit-identical; any exception from the fused path
    (e.g. CuPy not installed) also falls back to `reference`, as does a
    ValueError (unsupported shape) raised later. See the module docstring
    for KILOSORT_FUSED_VERIFY.
    """
    state = _state.setdefault((key, _signature(args, kwargs)),
                              {'n_checked': 0, 'use_fused': None})
    n_verify, strict = _verify_setting()
    if state['use_fused'] is None and state['n_checked'] >= n_verify:
        # n_verify == 0: use the fused kernel without checking it.
        state['use_fused'] = True
    if state['use_fused'] is None:
        ref = reference(*args, **kwargs)
        try:
            new = _on_device(fused, args, kwargs)
        except ValueError as e:
            # unsupported shape / dtype for this fused kernel
            state['use_fused'] = False
            logger.info(f'Fused kernel for {key}: {e}; using the PyTorch '
                        'implementation.')
            return ref
        except Exception as e:
            state['use_fused'] = False
            cuda = getattr(torch.version, 'cuda', None)
            logger.warning(f'Fused CUDA kernel for {key} unavailable ({e!r}); '
                           'using the PyTorch implementation. '
                           + _INSTALL_HINT.format(cuda=cuda))
            return ref
        # A call without any output (e.g. no spikes in the batch) exercises
        # almost none of the kernel, so it does not count as a check.
        if any(torch.is_tensor(a) and a.numel() > 0 for a in ref):
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
        try:
            return _on_device(fused, args, kwargs)
        except ValueError as e:
            # unsupported shape for this fused kernel: use the reference
            state['use_fused'] = False
            logger.warning(f'Fused kernel for {key}: {e}; using the PyTorch '
                           'implementation.')
    return reference(*args, **kwargs)


def reset_verification():
    """Forget the verification results (e.g. after changing the kernels)."""
    _state.clear()




_modules = {}


def _function(src_name, name, defines=()):
    """Compiled kernel `name` from `_SOURCES[src_name]`.

    `defines` are (NAME, value) pairs passed as -D options, so loop bounds
    can be compile-time constants (one module per distinct set of values).
    """
    import cupy as cp
    # Modules are loaded on the current device, so keep one per device.
    key = (src_name, tuple(defines), cp.cuda.Device().id)
    if key not in _modules:
        options = ('--fmad=false',) + tuple(f'-D{k}={v}' for k, v in defines)
        _modules[key] = cp.RawModule(code=_SOURCES[src_name], options=options)
    return _modules[key].get_function(name)


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


def _i32(*xs):
    return tuple(np.int32(x) for x in xs)


_SOURCES = {}


# --------------------------------------------------------------------------
# conv1d with a few short filters (the universal / PC templates)
# --------------------------------------------------------------------------
_SOURCES['conv'] = r"""
typedef unsigned long long ptr_t;
// out[c, k, t] = sum_i w[k, i] * x[c, t + i - PAD], zero padded, for K filters
// of NTAP taps. The taps are accumulated in ascending order with FMA, which
// is the summation order of cuDNN / PyTorch's conv1d for this shape.
extern "C" __global__
void conv_kernel(ptr_t px, ptr_t pw, ptr_t po, int C, int NT)
{
    const float* x = (const float*)px;
    const float* w = (const float*)pw;
    float* o = (float*)po;
    __shared__ float xs[TPB + NTAP - 1];
    __shared__ float ws[K * NTAP];
    const int c = blockIdx.y;
    const int t0 = blockIdx.x * TPB;
    for (int i = threadIdx.x; i < K * NTAP; i += TPB) ws[i] = w[i];
    for (int i = threadIdx.x; i < TPB + NTAP - 1; i += TPB) {
        int s = t0 + i - PAD;
        xs[i] = (s >= 0 && s < NT) ? x[(long long)c * NT + s] : 0.0f;
    }
    __syncthreads();
    const int t = t0 + threadIdx.x;
    if (t >= NT) return;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < NTAP; ++i) acc = fmaf(ws[k * NTAP + i], xs[threadIdx.x + i], acc);
#if KMAJOR
        o[((long long)k * C + c) * NT + t] = acc;
#else
        o[((long long)c * K + k) * NT + t] = acc;
#endif
    }
}
"""


def _conv1d_torch(X, W, padding, kmajor=False):
    out = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=padding)
    return out.permute(1, 0, 2) if kmajor else out


def _conv1d_fused(X, W, padding, kmajor=False):
    X = X.contiguous()
    W = W.contiguous()
    C, NT = X.shape
    K, NTAP = W.shape
    if X.dtype != torch.float32 or W.dtype != torch.float32 or K > 16 or NTAP > 255:
        raise ValueError('unsupported shape or dtype for fused conv1d')
    out = torch.empty((K, C, NT) if kmajor else (C, K, NT), device=X.device)
    tpb = 256
    kern = _function('conv', 'conv_kernel',
                     (('K', K), ('NTAP', NTAP), ('PAD', padding), ('TPB', tpb),
                      ('KMAJOR', int(bool(kmajor)))))
    with _stream():
        kern(((NT + tpb - 1) // tpb, C), (tpb,), (_p(X), _p(W), _p(out), *_i32(C, NT)))
    return out


# verified_call expects tuples of tensors
def _conv1d_torch_t(X, W, padding, kmajor=False):
    return (_conv1d_torch(X, W, padding, kmajor),)


def _conv1d_fused_t(X, W, padding, kmajor=False):
    return (_conv1d_fused(X, W, padding, kmajor),)


def conv1d_templates(X, W, padding, kmajor=False):
    """conv1d(X.unsqueeze(1), W.unsqueeze(1), padding) as (C, K, NT), or as
    (K, C, NT) with `kmajor` (contiguous when the fused kernel is used), with
    the fused kernel once it has been verified bit-identical to cuDNN."""
    return verified_call('conv1d', _conv1d_torch_t, _conv1d_fused_t,
                         X, W, padding, kmajor)[0]


# --------------------------------------------------------------------------
# spikedetect.template_match
# --------------------------------------------------------------------------
_SOURCES['template_match'] = r"""
typedef unsigned long long ptr_t;

// A[s, l] = sum_j weigh[s, j, f] * B[iC[j, f], l, t] for one (template f, time t),
// with the same rounding as the cuBLAS batched matmul behind torch.einsum (the
// two halves of the sum are accumulated separately with FMA). Returns the max
// of |A| and, in *bq, its flat index s*NK + l (lowest index among ties) and
// in *bval the signed value.
__device__ __forceinline__
float a_max(const float* B, const float* w, const int* ch, long long NT, int t,
            int* bq, float* bval)
{
    float best = -1.0f, bv = 0.0f;
    int q0 = 0;
    float b[NC];
    #pragma unroll
    for (int l = 0; l < NK; ++l) {
        #pragma unroll
        for (int j = 0; j < NC; ++j) b[j] = B[((long long)ch[j] * NK + l) * NT + t];
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < NC / 2; ++j) a0 = fmaf(w[s * NC + j], b[j], a0);
            #pragma unroll
            for (int j = NC / 2; j < NC; ++j) a1 = fmaf(w[s * NC + j], b[j], a1);
            float acc = a0 + a1;
            float v = fabsf(acc);
            int q = s * NK + l;
            if (v > best || (v == best && q < q0)) { best = v; q0 = q; bv = acc; }
        }
    }
    *bq = q0;
    *bval = bv;
    return best;
}

// As[f, t] = max_{s,l} |A[s, l]|
extern "C" __global__
void as_kernel(ptr_t pB, ptr_t pw, ptr_t piC, ptr_t pAs, int NT, int F)
{
    const float* B = (const float*)pB;           // (C, NK, NT)
    const float* weigh = (const float*)pw;       // (S, NC, F)
    const long long* iC = (const long long*)piC; // (NC, F)
    float* As = (float*)pAs;                     // (F, NT)
    const int f = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    __shared__ float w[S * NC];
    __shared__ int ch[NC];
    for (int k = threadIdx.x; k < S * NC; k += blockDim.x) {
        int s = k / NC, j = k % NC;
        w[k] = weigh[((long long)s * NC + j) * F + f];
    }
    for (int j = threadIdx.x; j < NC; j += blockDim.x) ch[j] = (int)iC[(long long)j * F + f];
    __syncthreads();
    if (t >= NT) return;
    int bq; float bval;
    As[(long long)f * NT + t] = a_max(B, w, ch, NT, t, &bq, &bval);
}

// Same as as_kernel for a block of G_ adjacent templates x TL_ time samples:
// the B values of the union of their nearest channels (UMAX at most) are
// staged in shared memory once, each thread then handles one template and
// TT_ = 4 consecutive samples (float4 reads) with the weights in registers.
#ifdef G_
extern "C" __global__ void __launch_bounds__(G_ * TL_ / TT_)
as_tile_kernel(ptr_t pB, ptr_t pw, ptr_t pumap, ptr_t puch, ptr_t pAs, int NT, int F, int UMAX)
{
    const float* B = (const float*)pB;           // (C, NK, NT)
    const float* weigh = (const float*)pw;       // (S, NC, F)
    const int* umap = (const int*)pumap;         // (ngroups, G_, NC): index into the union
    const int* uch = (const int*)puch;           // (ngroups, UMAX): channel ids, -1 = none
    float* As = (float*)pAs;                     // (F, NT)
    extern __shared__ float4 sh4[];
    float* sB = (float*)sh4;                     // UMAX * NK * TL_
    const int g = blockIdx.x;
    const int f0 = g * G_;
    const int t0 = blockIdx.y * TL_;
    const int tid = threadIdx.x;
    const int NTHR = G_ * TL_ / TT_;
    for (int k = tid; k < UMAX * NK * TL_; k += NTHR) {
        int u = k / (NK * TL_), r = k % (NK * TL_);
        int l = r / TL_, tt = r % TL_;
        int c = uch[(long long)g * UMAX + u];
        int t = t0 + tt;
        sB[k] = (c >= 0 && t < NT) ? B[((long long)c * NK + l) * NT + t] : 0.0f;
    }
    __syncthreads();
    const int lane = tid / (TL_ / TT_);          // template within the group
    const int tt = (tid % (TL_ / TT_)) * TT_;
    const int f = f0 + lane;
    const int t = t0 + tt;
    if (f >= F || t >= NT) return;
    float w[S * NC];
    #pragma unroll
    for (int k = 0; k < S * NC; ++k) w[k] = weigh[(long long)k * F + f];
    int mp[NC];
    #pragma unroll
    for (int j = 0; j < NC; ++j) mp[j] = umap[((long long)g * G_ + lane) * NC + j];
    float best[TT_];
    #pragma unroll
    for (int u = 0; u < TT_; ++u) best[u] = -1.0f;
    #pragma unroll
    for (int l = 0; l < NK; ++l) {
        float b[NC][TT_];
        #pragma unroll
        for (int j = 0; j < NC; ++j) {
            float4 v4 = *(const float4*)(sB + (mp[j] * NK + l) * TL_ + tt);
            b[j][0] = v4.x; b[j][1] = v4.y; b[j][2] = v4.z; b[j][3] = v4.w;
        }
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            #pragma unroll
            for (int u = 0; u < TT_; ++u) {
                // same summation order as a_max
                float a0 = 0.0f, a1 = 0.0f;
                #pragma unroll
                for (int j = 0; j < NC / 2; ++j) a0 = fmaf(w[s * NC + j], b[j][u], a0);
                #pragma unroll
                for (int j = NC / 2; j < NC; ++j) a1 = fmaf(w[s * NC + j], b[j][u], a1);
                float v = fabsf(a0 + a1);
                best[u] = fmaxf(best[u], v);
            }
        }
    }
    #pragma unroll
    for (int u = 0; u < TT_; ++u) if (t + u < NT) As[(long long)f * NT + t + u] = best[u];
}
#endif

// imax[s] = (1 + q) * sign(A[q]) at the detected (f, t) = xy[s]
extern "C" __global__
void imax_kernel(ptr_t pB, ptr_t pw, ptr_t piC, ptr_t pxy, ptr_t pim, int NT, int F, int n)
{
    const float* B = (const float*)pB;
    const float* weigh = (const float*)pw;
    const long long* iC = (const long long*)piC;
    const long long* xy = (const long long*)pxy; // (n, 2)
    long long* imax = (long long*)pim;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int f = (int)xy[2 * s], t = (int)xy[2 * s + 1];
    float w[S * NC];
    int ch[NC];
    #pragma unroll
    for (int k = 0; k < S * NC; ++k) w[k] = weigh[((long long)(k / NC) * NC + (k % NC)) * F + f];
    #pragma unroll
    for (int j = 0; j < NC; ++j) ch[j] = (int)iC[(long long)j * F + f];
    int bq; float bval;
    a_max(B, w, ch, NT, t, &bq, &bval);
    int sgn = (bval > 0.0f) - (bval < 0.0f);
    imax[s] = (long long)((1 + bq) * sgn);
}

// Q[f, t] = max_{|tau - t| <= nt0, nt <= tau < NT - nt} As[f, tau]  (0 if empty).
// This is max_pool1d(2*nt0+1) of As with the first and last nt columns zeroed,
// since As >= 0.
extern "C" __global__
void qpool_kernel(ptr_t pAs, ptr_t pQ, int NT, int nt, int nt0)
{
    const float* As = (const float*)pAs;
    float* Q = (float*)pQ;
    extern __shared__ float sh[];                // TPB + 2*nt0
    const int f = blockIdx.y;
    const int t0 = blockIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < blockDim.x + 2 * nt0; i += blockDim.x) {
        int tau = t0 + i - nt0;
        sh[i] = (tau >= nt && tau < NT - nt) ? As[(long long)f * NT + tau] : 0.0f;
    }
    __syncthreads();
    const int t = t0 + threadIdx.x;
    if (t >= NT) return;
    float m = 0.0f;
    for (int i = 0; i <= 2 * nt0; ++i) m = fmaxf(m, sh[threadIdx.x + i]);
    Q[(long long)f * NT + t] = m;
}

// Appends (f, t) to list (in no particular order; sorted on the host) where
// As[f, t] > th && As[f, t] == max_n Q[iC2[n, f], t]
extern "C" __global__
void cand_kernel(ptr_t pAs, ptr_t pQ, ptr_t piC2, ptr_t plist, ptr_t pcount,
                 int NT, int F, int N2, float th, int cap)
{
    const float* As = (const float*)pAs;
    const float* Q = (const float*)pQ;
    const long long* iC2 = (const long long*)piC2;
    int* list = (int*)plist;
    int* count = (int*)pcount;
    const int f = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ int nb[];
    for (int n = threadIdx.x; n < N2; n += blockDim.x) nb[n] = (int)iC2[(long long)n * F + f];
    __syncthreads();
    if (t >= NT) return;
    const float v = As[(long long)f * NT + t];
    bool ok = false;
    if (v > th) {
        float m = 0.0f;
        for (int n = 0; n < N2; ++n) {
            float q = Q[(long long)nb[n] * NT + t];
            if (q > v) { m = q; break; }
            m = fmaxf(m, q);
        }
        ok = (m == v);
    }
    if (ok) {
        int i = atomicAdd(count, 1);
        if (i < cap) { list[2 * i] = f; list[2 * i + 1] = t; }
    }
}
"""


_TILE_G, _TILE_TL, _TILE_TT = 16, 32, 4
_groups_cache = {}
_smem_limit = {}


def _max_shared_memory():
    """Largest dynamic shared memory per block (opt-in) of the current GPU."""
    import cupy as cp
    dev = cp.cuda.Device()
    if dev.id not in _smem_limit:
        attrs = dev.attributes
        _smem_limit[dev.id] = attrs.get('MaxSharedMemoryPerBlockOptin',
                                        attrs.get('MaxSharedMemoryPerBlock', 48 * 1024))
    return _smem_limit[dev.id]


def _template_groups(iC, iC_l, G):
    """For each group of G consecutive templates: the union of their nearest
    channels (-1 padded, (ngroups, UMAX)) and, per template and nearest
    channel, the index into that union ((ngroups, G, NC)). Cached for the
    iC tensor of the current run."""
    hit = _groups_cache.get('hit')
    if hit is not None and hit[0]() is iC and hit[1] == G:
        return hit[2]
    iCt = iC_l.t().cpu().numpy()                      # (F, NC)
    F, NC = iCt.shape
    ng = (F + G - 1) // G
    unions, maps = [], np.zeros((ng, G, NC), np.int32)
    for g in range(ng):
        block = iCt[g * G:(g + 1) * G]
        u, inv = np.unique(block, return_inverse=True)
        unions.append(u)
        maps[g, :len(block)] = inv.reshape(block.shape)
    umax = max(len(u) for u in unions)
    uch = -np.ones((ng, umax), np.int32)
    for g, u in enumerate(unions):
        uch[g, :len(u)] = u
    payload = (torch.from_numpy(maps).to(iC_l.device),
               torch.from_numpy(uch).to(iC_l.device), umax, ng)
    _groups_cache['hit'] = (weakref.ref(iC), G, payload)
    return payload


def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    """Fused version of `spikedetect.template_match`.

    The PyTorch version materializes B[iC] and the (S*NK, Nfilt, t) products
    in 40 chunks, takes the max over the 100 neighboring templates at every
    (template, time) and max-pools it over time. Here As is computed in
    place, the time pooling is done per template (Q), and the neighbor max
    is only evaluated at the (template, time) pairs above threshold, which
    is the only place where it can change the output: a value <= Th can
    neither be detected nor be the maximum that a detected value has to
    equal. The index of the best (size, template) is recomputed at the
    detected points instead of being stored for all of them.
    """
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    S, NC, _ = weigh.shape
    N2 = iC2.shape[0]
    if NC > 16 or weigh.dtype != torch.float32 or nk > 16:
        raise ValueError('unsupported shape or dtype for fused template_match')
    tpb = 256
    consts = (('NC', NC), ('S', S), ('NK', nk))

    B = conv1d_templates(X, ops['wTEMP'], nt//2).contiguous()

    As = torch.empty((Nfilt, NT), device=device)
    Q = torch.empty((Nfilt, NT), device=device)
    cap = 1 << 18
    clist = torch.empty((cap, 2), dtype=torch.int32, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    weigh = weigh.contiguous()
    iC_l = iC.long().contiguous()
    iC2_l = iC2.long().contiguous()
    th = np.float32(ops['Th_universal'])

    qpool_kernel = _function('template_match', 'qpool_kernel', consts)
    cand_kernel = _function('template_match', 'cand_kernel', consts)
    imax_kernel = _function('template_match', 'imax_kernel', consts)
    ntile = (NT + tpb - 1) // tpb
    # The tiled kernel needs the B values of the union of the nearest channels
    # of G adjacent templates in shared memory; fall back to the one-thread-
    # per-(template, time) kernel if that does not fit.
    G, TL, TT = _TILE_G, _TILE_TL, _TILE_TT
    umap, uch, umax, ngroups = _template_groups(iC, iC_l, G)
    smem = umax * nk * TL * 4
    tiled = smem <= _max_shared_memory()
    with _stream():
        if tiled:
            as_tile = _function('template_match', 'as_tile_kernel',
                                consts + (('G_', G), ('TL_', TL), ('TT_', TT)))
            if smem > 48 * 1024:
                as_tile.max_dynamic_shared_size_bytes = smem
            as_tile((ngroups, (NT + TL - 1) // TL), (G * TL // TT,),
                    (_p(B), _p(weigh), _p(umap), _p(uch), _p(As), *_i32(NT, Nfilt, umax)),
                    shared_mem=smem)
        else:
            as_kernel = _function('template_match', 'as_kernel', consts)
            as_kernel((Nfilt, ntile), (tpb,),
                      (_p(B), _p(weigh), _p(iC_l), _p(As), *_i32(NT, Nfilt)))
        qpool_kernel((ntile, Nfilt), (tpb,), (_p(As), _p(Q), *_i32(NT, nt, nt0)),
                     shared_mem=(tpb + 2 * nt0) * 4)
        cand_kernel((Nfilt, ntile), (tpb,),
                    (_p(As), _p(Q), _p(iC2_l), _p(clist), _p(count),
                     *_i32(NT, Nfilt, N2), th, np.int32(cap)),
                    shared_mem=N2 * 4)
    nsp = int(count.item())
    while nsp > cap:
        # more detections than the list holds: rerun with a large enough one
        cap = nsp
        clist = torch.empty((cap, 2), dtype=torch.int32, device=device)
        count.zero_()
        with _stream():
            cand_kernel((Nfilt, ntile), (tpb,),
                        (_p(As), _p(Q), _p(iC2_l), _p(clist), _p(count),
                         *_i32(NT, Nfilt, N2), th, np.int32(cap)),
                        shared_mem=N2 * 4)
        nsp = int(count.item())
    # same order as logical_and(...).nonzero(): row-major in (f, t)
    xy = clist[:nsp].long()
    xy = xy[(xy[:, 0] * NT + xy[:, 1]).argsort()].contiguous()
    imax = torch.empty(nsp, dtype=torch.int64, device=device)
    if nsp > 0:
        with _stream():
            imax_kernel(((nsp + 127) // 128,), (128,),
                        (_p(B), _p(weigh), _p(iC_l), _p(xy), _p(imax), *_i32(NT, Nfilt, nsp)))
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
__device__ __forceinline__
void cf_column(const float* B, const float* nm, float* Cfmax, long long* imax,
               int NU, int NT, int nt, int t)
{
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

extern "C" __global__
void cfmax_kernel(ptr_t pB, ptr_t pnm, ptr_t pCf, ptr_t pim, int NU, int NT, int nt)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NT) return;
    cf_column((const float*)pB, (const float*)pnm, (float*)pCf, (long long*)pim, NU, NT, nt, t);
}

// Same, only for the columns x - nt .. x + nt around the spikes st[k0 .. k0+n)
// (the only columns of B that the last peel changed).
extern "C" __global__
void cfmax_dirty_kernel(ptr_t pB, ptr_t pnm, ptr_t pCf, ptr_t pim, ptr_t pst,
                        int k0, int NU, int NT, int nt)
{
    const long long* st = (const long long*)pst;
    int t = (int)st[(long long)(k0 + blockIdx.x) * 2] + (int)threadIdx.x - nt;
    if (threadIdx.x > 2 * nt || t < 0 || t >= NT) return;
    cf_column((const float*)pB, (const float*)pnm, (float*)pCf, (long long*)pim, NU, NT, nt, t);
}

// Cmax = max_pool1d(Cfmax, 2*nt+1); mask = (Cmax > Th**2) & (|Cmax - Cfmax| < 1e-9)
__device__ __forceinline__
void pool_column(const float* Cfmax, float* Cmax, bool* mask, int NT, int nt, float th2,
                 float eps, int t)
{
    int a = t - nt < 0 ? 0 : t - nt;
    int b = t + nt > NT - 1 ? NT - 1 : t + nt;
    float m = Cfmax[a];
    for (int i = a + 1; i <= b; ++i) { float v = Cfmax[i]; if (v > m) m = v; }
    Cmax[t] = m;
    float d = m - Cfmax[t];
    mask[t] = (m > th2) && (fabsf(d) < eps);
}

extern "C" __global__
void pool_kernel(ptr_t pCf, ptr_t pCm, ptr_t pmask, int NT, int nt, float th2, float eps)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NT) return;
    pool_column((const float*)pCf, (float*)pCm, (bool*)pmask, NT, nt, th2, eps, t);
}

// Same for the columns x - 2nt .. x + 2nt around the spikes st[k0 .. k0+n).
extern "C" __global__
void pool_dirty_kernel(ptr_t pCf, ptr_t pCm, ptr_t pmask, ptr_t pst, int k0,
                       int NT, int nt, float th2, float eps)
{
    const long long* st = (const long long*)pst;
    int t = (int)st[(long long)(k0 + blockIdx.x) * 2] + (int)threadIdx.x - 2 * nt;
    if (threadIdx.x > 4 * nt || t < 0 || t >= NT) return;
    pool_column((const float*)pCf, (float*)pCm, (bool*)pmask, NT, nt, th2, eps, t);
}

// One block: xs = nonzero(mask) in ascending order; for the n new spikes
//   st[k+s] = (x, imax[x]); amps[k+s] = B[y, x] / nm[y]; th_amps[k+s] = Cmax[x]**0.5
// info[0] = n, info[1] = 1 if two spikes of the same parity are closer than
// 2*nt + 1 (their update windows overlap), info[2] = 1 if k + n > capacity.
extern "C" __global__
void compact_kernel(ptr_t pmask, ptr_t pim, ptr_t pB, ptr_t pnm, ptr_t pCm,
                    ptr_t pst, ptr_t pamp, ptr_t pth, ptr_t pxs, ptr_t pinfo,
                    int k, int capacity, int NT, int nt)
{
    const bool* mask = (const bool*)pmask;
    const long long* imax = (const long long*)pim;
    const float* B = (const float*)pB;
    const float* nm = (const float*)pnm;
    const float* Cmax = (const float*)pCm;
    long long* st = (long long*)pst;
    float* amps = (float*)pamp;
    float* th = (float*)pth;
    long long* xs = (long long*)pxs;
    int* info = (int*)pinfo;
    __shared__ int cnt[CTPB];
    __shared__ int overlap;
    const int chunk = (NT + CTPB - 1) / CTPB;
    const int a = threadIdx.x * chunk;
    const int b = min(a + chunk, NT);
    int c = 0;
    for (int t = a; t < b; ++t) c += mask[t] ? 1 : 0;
    cnt[threadIdx.x] = c;
    if (threadIdx.x == 0) overlap = 0;
    __syncthreads();
    // exclusive scan (Hillis-Steele)
    for (int off = 1; off < CTPB; off *= 2) {
        int v = threadIdx.x >= off ? cnt[threadIdx.x - off] : 0;
        __syncthreads();
        cnt[threadIdx.x] += v;
        __syncthreads();
    }
    const int n = cnt[CTPB - 1];
    int pos = cnt[threadIdx.x] - c;
    for (int t = a; t < b; ++t) {
        if (mask[t]) {
            xs[pos] = t;
            if (k + pos < capacity) {
                long long y = imax[t];
                st[(long long)(k + pos) * 2] = t;
                st[(long long)(k + pos) * 2 + 1] = y;
                amps[k + pos] = B[y * NT + t] / nm[y];
                th[k + pos] = sqrtf(Cmax[t]);
            }
            ++pos;
        }
    }
    __syncthreads();
    // windows of spikes s and s+2 overlap if xs[s+2] - xs[s] <= 2*nt
    for (int s = threadIdx.x; s + 2 < n; s += CTPB)
        if (xs[s + 2] - xs[s] <= 2 * nt) overlap = 1;
    __syncthreads();
    if (threadIdx.x == 0) { info[0] = n; info[1] = overlap; info[2] = (k + n > capacity); }
}

// In-place update for the spikes s = k + j, k + j + 2, ... (half j), whose
// windows don't overlap:
//   B[r, x+tau-nt]    -= amp * ctc[r, y, tau]   for the rows r with ctc[r, y, :] != 0
//   Xres[c, x+l-nt/2] -= amp * E[c, q, l]       for the channels c with U[y, :, c] != 0
// Rows / channels with all-zero ctc / E are skipped: subtracting amp * 0
// leaves B and Xres bit-identical.
extern "C" __global__
void upd_kernel(ptr_t pst, ptr_t pamp, ptr_t pB, ptr_t pctc, ptr_t pX, ptr_t pE,
                ptr_t prows, ptr_t prow0, ptr_t pchans, ptr_t pch0,
                int k, int j, int LR, int NU, int NT, int nt,
                long long e0, long long e1, long long e2)
{
    const long long* st = (const long long*)pst;
    const float* amps = (const float*)pamp;
    float* B = (float*)pB;
    const float* ctc = (const float*)pctc;
    float* X = (float*)pX;
    const float* E = (const float*)pE;
    const int* rows = (const int*)prows;   // CSR of nonzero rows of ctc[:, y, :]
    const int* row0 = (const int*)prow0;   // (NU + 1)
    const int* chans = (const int*)pchans; // CSR of nonzero channels of U[y]
    const int* ch0 = (const int*)pch0;     // (NU + 1)
    const int q = blockIdx.x;
    const int s = k + j + 2 * q;
    const long long x = st[(long long)s * 2];
    const int y = (int)st[(long long)s * 2 + 1];
    const float amp = amps[s];
    const int tau = threadIdx.x;
    if (blockIdx.y < LR) {
        int idx = row0[y] + blockIdx.y;
        if (idx >= row0[y + 1]) return;
        int nw = 2 * nt + 1;
        if (tau >= nw) return;
        int r = rows[idx];
        float p = amp * ctc[((long long)r * NU + y) * nw + tau];
        long long i = (long long)r * NT + x + tau - nt;
        B[i] = B[i] - p;
    } else {
        int idx = ch0[y] + (blockIdx.y - LR);
        if (idx >= ch0[y + 1]) return;
        int nw = 2 * (nt / 2) + 1;
        if (tau >= nw) return;
        int c = chans[idx];
        float p = amp * E[c * e0 + q * e1 + tau * e2];
        long long i = (long long)c * NT + x + tau - nt / 2;
        X[i] = X[i] - p;
    }
}

// Fallback when windows of the same half overlap (ties): read and subtract
// into temporaries, then write back, like the gather / scatter of
// `B[:, idx] -= ...` in PyTorch.
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

_CTPB = 1024
_sparsity_cache = {}


def _csr(nz):
    """CSR (indices, offsets) of the True entries of each column of a bool
    (rows, cols) tensor, as int32 CUDA tensors, plus the longest column."""
    nzt = nz.t().contiguous()
    counts = nzt.sum(1, dtype=torch.int32)
    offsets = torch.zeros(nz.shape[1] + 1, dtype=torch.int32, device=nz.device)
    offsets[1:] = torch.cumsum(counts, 0)
    idx = nzt.nonzero()[:, 1].to(torch.int32).contiguous()
    return idx, offsets.contiguous(), int(counts.max()) if len(counts) else 0


def _template_sparsity(U, ctc):
    """Nonzero rows of ctc[:, y, :] and nonzero channels of U[y], cached for
    the (U, ctc) tensors of the current `extract` since they are the same
    for every batch. The cache holds weak references, so it is dropped as
    soon as the tensors are, and a new tensor at the same address is never
    mistaken for the old one."""
    hit = _sparsity_cache.get('hit')
    if hit is not None and hit[0]() is U and hit[1]() is ctc:
        return hit[2]
    rows, row0, LR = _csr((ctc != 0).any(-1))         # (NU rows, NU templates)
    chans, ch0, LC = _csr((U != 0).any(1).t())        # (C channels, NU templates)
    payload = (rows, row0, LR, chans, ch0, LC)
    _sparsity_cache['hit'] = (weakref.ref(U), weakref.ref(ctc), payload)
    return payload


def run_matching(ops, X, U, ctc, device=torch.device('cuda')):
    """Fused version of `template_matching.run_matching`.

    Per peel the PyTorch version launches ~70 kernels, writes
    Cf = relu(B)**2 / nm (n_templates x NT) three times and gathers /
    scatters a window of all n_templates rows of B for every spike. Here:
    - Cf is never materialized, and after the first peel only the columns
      that the last peel changed are recomputed;
    - the spikes are compacted, recorded and checked for overlapping windows
      by one block, so one small copy per peel brings back the count;
    - B and Xres are updated in place, only in the rows of ctc / channels of
      the template that are not identically zero.
    The matrix products (conv1d, the B einsum and the per-spike waveform
    einsum) are the same PyTorch calls, so they round the same way.
    """
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()
    if 4 * nt + 1 > 1024:
        # the kernels use one thread per sample of a 2nt+1 / 4nt+1 window
        raise ValueError('nt > 255 is not supported by fused run_matching')

    nm = (U**2).sum(-1).sum(-1)

    # einsum('ijk, kjl -> il', U, conv1d(X, W)) permutes the conv output to
    # (j, k, l) and runs this matmul; producing it in that layout directly
    # saves the copy and rounds the same way.
    B = conv1d_templates(X, W, nt//2, kmajor=True)
    NU = U.shape[0]
    B = U.reshape(NU, -1) @ B.reshape(-1, B.shape[-1])

    capacity = 100000
    st = torch.zeros((capacity,2), dtype = torch.int64, device = device)
    amps = torch.zeros((capacity,1), dtype = torch.float, device = device)
    th_amps = torch.zeros((capacity,1), dtype = torch.float, device = device)
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
    rows, row0, LR, chans, ch0, LC = _template_sparsity(U, ctc)
    Cfmax = torch.empty(NT, device=device)
    imax = torch.empty(NT, dtype=torch.int64, device=device)
    Cmax = torch.empty(NT, device=device)
    mask = torch.empty(NT, dtype=torch.bool, device=device)
    xs = torch.empty(NT, dtype=torch.int64, device=device)
    info = torch.zeros(3, dtype=torch.int32, device=device)
    th2 = np.float32(Th**2)
    eps = np.float32(1e-9)
    tpb = 128
    gridT = ((NT + tpb - 1) // tpb,)
    nwB, nwX = 2*nt + 1, 2*(nt//2) + 1
    tpb_upd = (max(nwB, nwX) + 31) // 32 * 32

    consts = (('CTPB', _CTPB),)
    cfmax = _function('run_matching', 'cfmax_kernel', consts)
    cfmax_dirty = _function('run_matching', 'cfmax_dirty_kernel', consts)
    pool = _function('run_matching', 'pool_kernel', consts)
    pool_dirty = _function('run_matching', 'pool_dirty_kernel', consts)
    compact = _function('run_matching', 'compact_kernel', consts)
    upd = _function('run_matching', 'upd_kernel', consts)
    upd1 = _function('run_matching', 'upd1_kernel', consts)
    upd2 = _function('run_matching', 'upd2_kernel', consts)

    with _stream():
        for t in range(max_peels):
            # Only the columns around the spikes of the last peel changed, but
            # when there are many spikes their (overlapping) windows cover
            # more columns than the full pass, which is then cheaper. Both
            # paths use the same per-column device functions.
            if t == 0 or nsp * nwB >= NT:
                cfmax(gridT, (tpb,), (_p(B), _p(nm), _p(Cfmax), _p(imax), *_i32(NU, NT, nt)))
            else:
                cfmax_dirty((nsp,), (nwB,), (_p(B), _p(nm), _p(Cfmax), _p(imax), _p(st),
                                             *_i32(k - nsp, NU, NT, nt)))
            if t == 0 or nsp * (2 * nwB - 1) >= NT:
                pool(gridT, (tpb,), (_p(Cfmax), _p(Cmax), _p(mask), *_i32(NT, nt), th2, eps))
            else:
                pool_dirty((nsp,), (2 * nwB - 1,), (_p(Cfmax), _p(Cmax), _p(mask), _p(st),
                                                    *_i32(k - nsp, NT, nt), th2, eps))
            compact((1,), (_CTPB,), (_p(mask), _p(imax), _p(B), _p(nm), _p(Cmax),
                                      _p(st), _p(amps), _p(th_amps), _p(xs), _p(info),
                                      *_i32(k, capacity, NT, nt)))
            nsp, overlap, overflow = info.tolist()
            if nsp == 0:
                break
            if overflow:
                raise RuntimeError('Too many spikes in one batch for run_matching.')
            iY = st[k:k+nsp, 1:2]
            # As in the PyTorch version, every other spike is subtracted at
            # once so that the subtracted windows don't overlap.
            for j in range(2):
                m = (nsp - j + 1) // 2
                if m == 0:
                    continue
                # einsum('ijk, jl -> kil', U[iY], W) permutes U[iY] to (i, k, j)
                # and runs this batched matmul (same cuBLAS call, same
                # rounding) with ~25 us less host overhead per call.
                E = (U[iY[j::2,0]].permute(0, 2, 1) @ W).permute(1, 0, 2)
                e0, e1, e2 = E.stride()
                if not overlap:
                    upd((m, LR + LC), (tpb_upd,),
                        (_p(st), _p(amps), _p(B), _p(ctc), _p(Xres), _p(E),
                         _p(rows), _p(row0), _p(chans), _p(ch0),
                         *_i32(k, j, LR, NU, NT, nt),
                         np.int64(e0), np.int64(e1), np.int64(e2)))
                else:
                    tB = torch.empty((m, NU, nwB), device=device)
                    tX = torch.empty((m, NC, nwX), device=device)
                    upd1((m * (NU + NC),), (tpb_upd,),
                         (_p(st), _p(amps), _p(B), _p(ctc), _p(Xres), _p(E),
                          _p(tB), _p(tX), *_i32(k, j, m, NU, NC, NT, nt),
                          np.int64(e0), np.int64(e1), np.int64(e2)))
                    upd2((m * (NU + NC),), (tpb_upd,),
                         (_p(st), _p(B), _p(Xres), _p(tB), _p(tX),
                          *_i32(k, j, m, NU, NC, NT, nt)))
            k += nsp

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    return st, amps, th_amps, Xres
