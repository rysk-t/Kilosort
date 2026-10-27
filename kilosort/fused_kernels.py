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
"""
import logging
import os

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

logger = logging.getLogger(__name__)

_state = {}


def enabled():
    return os.environ.get('KILOSORT_FUSED_KERNELS', '1') != '0'


def verified_call(key, reference, fused, *args, **kwargs):
    """Call `fused` instead of `reference` once it has been checked.

    On the first call both are run and the reference output is returned.
    Later calls use `fused` only if the first outputs were bit-identical;
    any exception from the fused path (e.g. CuPy not installed) also falls
    back to `reference`.
    """
    state = _state.setdefault(key, {'checked': False, 'use_fused': False})
    if not state['checked']:
        state['checked'] = True
        ref = reference(*args, **kwargs)
        try:
            new = fused(*args, **kwargs)
            same = (len(ref) == len(new)
                    and all(torch.equal(a, b) for a, b in zip(ref, new)))
        except Exception as e:
            logger.info(f'Fused kernel for {key} unavailable ({e!r}), '
                        'using the PyTorch implementation.')
            return ref
        state['use_fused'] = same
        if same:
            logger.info(f'Fused kernel for {key} verified on first call.')
        else:
            logger.warning(f'Fused kernel for {key} does not match the PyTorch '
                           'implementation, using the PyTorch implementation.')
        return ref
    if state['use_fused']:
        return fused(*args, **kwargs)
    return reference(*args, **kwargs)


_modules = {}


def _function(src_name, name):
    import cupy as cp
    if src_name not in _modules:
        _modules[src_name] = cp.RawModule(code=_SOURCES[src_name],
                                          options=('--fmad=false',))
    return _modules[src_name].get_function(name)


def _stream():
    import cupy as cp
    return cp.cuda.Stream.from_external(torch.cuda.current_stream())


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
