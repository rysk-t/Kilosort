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

// Common average reference of a row-major (n, T) batch, matching
//     Y = X - mean[:, None];  out = Y - torch.median(Y, 0)[0]
// with the row means computed by PyTorch. Same selection as column_median;
// the keys are a lossless encoding of Y, so the tile is written back from
// shared memory without reading X again.
extern "C" __global__
void center_car(ptr_t pX, ptr_t pMean, ptr_t pOut, int n, int T)
{
    const float* X = (const float*)pX;
    const float* mean = (const float*)pMean;
    float* out = (float*)pOut;
    extern __shared__ unsigned int keys[];       // (COLS, n)
    __shared__ int has_nan[COLS];
    __shared__ float med[COLS];
    const int c0 = blockIdx.x * COLS;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x < COLS) has_nan[threadIdx.x] = 0;
    __syncthreads();

    for (int idx = threadIdx.x; idx < n * COLS; idx += blockDim.x) {
        int row = idx / COLS, col = idx % COLS;
        int t = c0 + col;
        if (t < T) {
            float v = X[(long long)row * T + t] - mean[row];
            if (v != v) has_nan[col] = 1;
            keys[col * n + row] = to_key(v);
        }
    }
    __syncthreads();

    const int t = c0 + warp;
    if (t < T) {
        if (has_nan[warp]) {
            if (lane == 0) med[warp] = __int_as_float(0x7fffffff);
        } else {
            unsigned int k[PER];
            #pragma unroll
            for (int j = 0; j < PER; ++j) {
                int row = lane + 32 * j;
                k[j] = row < n ? keys[warp * n + row] : 0u;
            }
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
            if (lane == 0) med[warp] = from_key(ans);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n * COLS; idx += blockDim.x) {
        int row = idx / COLS, col = idx % COLS;
        int tt = c0 + col;
        if (tt < T) {
            unsigned int key = keys[col * n + row];
            // NaN keys do not decode to the original value, but a column
            // with a NaN gives NaN everywhere (Y - NaN) anyway.
            float v = has_nan[col] ? med[col] : from_key(key);
            out[(long long)row * T + tt] = v - med[col];
        }
    }
}

// Raw int16 batch (nsamp, Cbin), time-major as in the file, to the float
// (Nchan, NTp) batch that BinaryRWFile.padded_batch_to_torch + chan_map +
// invert_sign produce: X[c, t] = R[s(t), chan_map[c]] with s(t) = t - off,
// and for s < 0 the first sample (left_fill) or 0, for s >= nsamp the last
// sample (tail_fill) or 0. Transposed through a shared-memory tile.
extern "C" __global__
void batch_load(ptr_t pR, ptr_t pCm, ptr_t pX, int nsamp, int Cbin, int Nchan,
                int NTp, int off, int left_fill, int tail_fill, int invert)
{
    const short* R = (const short*)pR;
    const long long* cm = (const long long*)pCm;
    float* X = (float*)pX;
    __shared__ float tile[32][33];
    const int t0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        int tt = t0 + ty + k, c = c0 + tx;
        float v = 0.0f;
        if (tt < NTp && c < Nchan) {
            long long s = (long long)tt - off;
            bool valid = true;
            if (s < 0) { if (left_fill) s = 0; else valid = false; }
            if (s >= nsamp) { if (tail_fill) s = nsamp - 1; else valid = false; }
            if (valid) v = (float)R[s * Cbin + cm[c]];
            if (invert) v = v * -1.0f;
        }
        tile[ty + k][tx] = v;
    }
    __syncthreads();
    for (int k = 0; k < 32; k += 8) {
        int c = c0 + ty + k, tt = t0 + tx;
        if (c < Nchan && tt < NTp) X[(long long)c * NTp + tt] = tile[tx][ty + k];
    }
}
"""

_modules = {}


def _median_kernel(n, name='column_median'):
    """Compile (once per channel count) and return the kernel and block shape."""
    import cupy as cp
    per = (n + 31) // 32
    cols = 16
    while cols > 1 and n * cols * 4 > 40 * 1024:
        cols //= 2
    if n * cols * 4 > 40 * 1024 or per > 64:
        raise ValueError(f'{name}: unsupported number of rows {n}')
    key = (per, cols)
    if key not in _modules:
        _modules[key] = cp.RawModule(
            code=_SOURCE, options=('--fmad=false', f'-DPER={per}', f'-DCOLS={cols}')
            )
    return _modules[key].get_function(name), cols


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


def _car_torch(X, mean):
    X = X - mean.unsqueeze(1)
    return (X - torch.median(X, 0)[0],)


def _car_fused(X, mean):
    if X.dtype != torch.float32 or X.dim() != 2 or mean.dtype != torch.float32:
        raise ValueError('center_car expects 2D float32 tensors')
    X = X.contiguous()
    mean = mean.contiguous()
    n, T = X.shape
    kernel, cols = _median_kernel(n, 'center_car')
    out = torch.empty_like(X)
    with _stream():
        kernel(((T + cols - 1) // cols,), (32 * cols,),
               (_p(X), _p(mean), _p(out), np.int32(n), np.int32(T)),
               shared_mem=n * cols * 4)
    return (out,)


def subtract_mean_and_median(X, mean):
    """`Y = X - mean[:, None]; Y - torch.median(Y, 0)[0]` in one pass on CUDA.

    Same as subtracting the channel means and then `channel_median`, but the
    batch is read once and written once instead of three reads and two
    writes.
    """
    if X.device.type == 'cuda' and fused_kernels.enabled():
        return fused_kernels.verified_call(
            'center_car', _car_torch, _car_fused, X, mean
            )[0]
    return _car_torch(X, mean)[0]


def _load_torch(raw, chan_map, NTp, off, left_fill, tail_fill, invert):
    # Same steps as BinaryRWFile.padded_batch_to_torch followed by the
    # chan_map selection and sign inversion of BinaryFiltered.filter.
    nsamp = raw.shape[0]
    X = torch.zeros((raw.shape[1], NTp), device=raw.device)
    data = raw.T.float()
    if left_fill:
        X[:, off : off+nsamp] = data
        X[:, :off] = X[:, off : off+1]
    elif tail_fill:
        X[:, :nsamp] = data
        X[:, nsamp:] = X[:, nsamp-1:nsamp]
    else:
        X[:] = data
    X = X[chan_map]
    if invert:
        X = X * -1
    return (X,)


def _load_fused(raw, chan_map, NTp, off, left_fill, tail_fill, invert):
    import cupy as cp
    if raw.dtype != torch.int16 or raw.dim() != 2 or chan_map.dtype != torch.int64:
        raise ValueError('batch_load expects an int16 batch and int64 chan_map')
    if not (left_fill or tail_fill) and raw.shape[0] != NTp:
        raise ValueError('batch_load: batch size does not match')
    raw = raw.contiguous()
    nsamp, Cbin = raw.shape
    Nchan = chan_map.shape[0]
    _median_kernel(32)   # make sure the module is compiled
    kernel = _modules[(1, 16)].get_function('batch_load')
    X = torch.empty((Nchan, NTp), device=raw.device)
    with _stream():
        kernel(((NTp + 31) // 32, (Nchan + 31) // 32), (32, 8),
               (_p(raw), _p(chan_map), _p(X), np.int32(nsamp), np.int32(Cbin),
                np.int32(Nchan), np.int32(NTp), np.int32(off),
                np.int32(left_fill), np.int32(tail_fill), np.int32(invert)))
    return (X,)


def load_batch(raw, chan_map, NTp, off, left_fill, tail_fill, invert):
    """Raw int16 batch on the GPU to the padded, channel-selected float batch.

    `raw` is (n_samples, n_chan_bin) as stored in the file. The batch is
    placed at column `off` of an (len(chan_map), NTp) tensor; columns before
    it repeat the first sample if `left_fill`, columns after it repeat the
    last sample if `tail_fill`, otherwise they are 0. Signs are flipped if
    `invert`. Replaces several full-size copies with one transposing kernel.
    """
    args = (raw, chan_map, NTp, off, int(left_fill), int(tail_fill), int(invert))
    if raw.device.type == 'cuda' and fused_kernels.enabled():
        return fused_kernels.verified_call(
            'batch_load', _load_torch, _load_fused, *args
            )[0]
    return _load_torch(*args)[0]
