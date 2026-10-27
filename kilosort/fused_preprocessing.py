"""Optional fused CUDA kernels for `io.BinaryFiltered.filter`.

Uses the CuPy / NVRTC setup and the first-call verification of
`kilosort.fused_kernels`; see that module for requirements and for the
``KILOSORT_FUSED_KERNELS=0`` switch.
"""
import numpy as np
import torch

from kilosort import fused_kernels
from kilosort.fused_kernels import _p, _stream


_SOURCE = r"""
typedef unsigned long long ptr_t;

// Order-preserving map from float to uint32, the same one PyTorch's radix
// select uses (TopKTypeConfig<float>): -0.0 sorts below +0.0, NaN above all.
__device__ __forceinline__ unsigned int to_key(float v)
{
    unsigned int x = __float_as_uint(v);
    unsigned int mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return (v == v) ? (x ^ mask) : 0xffffffffu;
}

__device__ __forceinline__ float from_key(unsigned int k)
{
    unsigned int x = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
    return __uint_as_float(x);
}

// Median over rows (dim 0) of a row-major (n, T) float matrix, matching
// torch.median(X, 0)[0]: the element of rank (n - 1) / 2 in key order, or
// NaN if the column has a NaN. One warp per column, COLS columns per block;
// the block's tile is staged in shared memory as keys. PER = ceil(n / 32).
extern "C" __global__
void column_median(ptr_t pX, ptr_t pOut, int n, int T)
{
    const float* X = (const float*)pX;
    float* out = (float*)pOut;
    extern __shared__ unsigned int keys[];       // (COLS, n)
    __shared__ int has_nan[COLS];
    const int c0 = blockIdx.x * COLS;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x < COLS) has_nan[threadIdx.x] = 0;
    __syncthreads();

    for (int idx = threadIdx.x; idx < n * COLS; idx += blockDim.x) {
        int row = idx / COLS, col = idx % COLS;
        int t = c0 + col;
        if (t < T) {
            float v = X[(long long)row * T + t];
            if (v != v) has_nan[col] = 1;
            keys[col * n + row] = to_key(v);
        }
    }
    __syncthreads();

    const int t = c0 + warp;
    if (t >= T) return;
    if (has_nan[warp]) {
        if (lane == 0) out[t] = __int_as_float(0x7fffffff);
        return;
    }

    unsigned int k[PER];
    #pragma unroll
    for (int j = 0; j < PER; ++j) {
        int row = lane + 32 * j;
        k[j] = row < n ? keys[warp * n + row] : 0u;
    }

    // Select the key of rank r one bit at a time, from the most significant.
    int r = (n - 1) / 2;
    unsigned int ans = 0u;
    for (int b = 31; b >= 0; --b) {
        unsigned int hi = (b == 31) ? 0u : (0xffffffffu << (b + 1));
        int cnt0 = 0;
        #pragma unroll
        for (int j = 0; j < PER; ++j) {
            int row = lane + 32 * j;
            bool p = row < n && ((k[j] ^ ans) & hi) == 0u && ((k[j] >> b) & 1u) == 0u;
            cnt0 += __popc(__ballot_sync(0xffffffffu, p));
        }
        if (r >= cnt0) { ans |= (1u << b); r -= cnt0; }
    }
    if (lane == 0) out[t] = from_key(ans);
}
"""

_modules = {}


def _median_kernel(n):
    """Compile (once per channel count) and return the kernel and block shape."""
    import cupy as cp
    per = (n + 31) // 32
    cols = 16
    while cols > 1 and n * cols * 4 > 40 * 1024:
        cols //= 2
    if n * cols * 4 > 40 * 1024 or per > 64:
        raise ValueError(f'column_median: unsupported number of rows {n}')
    key = (per, cols)
    if key not in _modules:
        _modules[key] = cp.RawModule(
            code=_SOURCE, options=('--fmad=false', f'-DPER={per}', f'-DCOLS={cols}')
            )
    return _modules[key].get_function('column_median'), cols


def _median_torch(X):
    return (torch.median(X, 0)[0],)


def _median_fused(X):
    if X.dtype != torch.float32 or X.dim() != 2:
        raise ValueError('column_median expects a 2D float32 tensor')
    X = X.contiguous()
    n, T = X.shape
    kernel, cols = _median_kernel(n)
    out = torch.empty(T, dtype=X.dtype, device=X.device)
    with _stream():
        kernel(((T + cols - 1) // cols,), (32 * cols,),
               (_p(X), _p(out), np.int32(n), np.int32(T)),
               shared_mem=n * cols * 4)
    return (out,)


def channel_median(X):
    """`torch.median(X, 0)[0]`, using a fused kernel on CUDA when possible."""
    if X.device.type == 'cuda' and fused_kernels.enabled():
        return fused_kernels.verified_call(
            'channel_median', _median_torch, _median_fused, X
            )[0]
    return _median_torch(X)[0]
