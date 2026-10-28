"""KS2.5 refractory merging, bimodal pursuit, cutoff and Phy export.

Array layouts and zero-based template IDs follow ``KS25_BRIEF.md``.  The
equations/thresholds come from the MATLAB postProcess directory, not KS4's
CCG implementation (which has different histogram endpoints and shoulders).
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from numba import njit

from kilosort.utils import fastest

logger = logging.getLogger(__name__)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _opt(ops, key, default=None):
    return ops.get('ks25', {}).get(key, ops.get(key, ops.get('settings', {}).get(key, default)))


def _rows(array, indices):
    if torch.is_tensor(array):
        return array.index_select(0, torch.as_tensor(indices, device=array.device, dtype=torch.long))
    return array[indices]


def _groups(labels, count):
    order = np.argsort(labels, kind='stable')
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(count)]


@njit(cache=True, nogil=True)
def _ccg_counts(st1, st2, nbins, tbin):
    counts = np.zeros(2 * nbins + 1, dtype=np.float64)
    low = high = 0
    dt = nbins * tbin
    for t in st2:
        while high < len(st1) and st1[high] < t + dt:
            high += 1
        while low < len(st1) and st1[low] <= t - dt:
            low += 1
        if low == len(st1):
            break
        for k in range(low, high):
            lag = (t - st1[k]) / tbin
            # MATLAB round uses ties away from zero, unlike numpy.round.
            index = int(math.floor(lag + .5) if lag >= 0 else math.ceil(lag - .5))
            counts[index + nbins] += 1
    return counts


def ccg(st1, st2, nbins=500, tbin=1 / 1000):
    """Return MATLAB ``ccg.m``'s (K, Qi, Q00, Q01, Ri); times are seconds.

    K retains the zero-lag bin; only the refractory metrics exclude it.
    Undefined metrics (empty trains or no shoulder counts) retain NaNs.
    The fixed shoulder windows require an even ``nbins >= 50``.
    """
    if int(nbins) != nbins or nbins < 50 or nbins % 2 or not np.isfinite(tbin) or tbin <= 0:
        raise ValueError('ccg requires even nbins >= 50 and positive finite tbin')
    nbins = int(nbins)
    a = np.sort(np.asarray(st1, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(st2, dtype=np.float64).reshape(-1))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('spike times must be finite')
    K = _ccg_counts(a, b, nbins, tbin)
    if not len(a) or not len(b):
        return K, np.full(10, np.nan), np.nan, np.nan, np.full(10, np.nan)
    duration = max(a[-1], b[-1]) - min(a[0], b[0])
    shoulders = np.r_[K[1:nbins // 2], K[3 * nbins // 2:2 * nbins]]
    left, right = K[nbins - 50:nbins - 10], K[nbins + 11:nbins + 50]
    r00 = max(shoulders.mean(), left.mean(), right.mean())
    Qi, Ri = np.empty(10), np.empty(10)
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = np.divide(tbin * len(a) * len(b), duration)
        Q00 = shoulders.mean() / expected
        Q01 = max(left.mean(), right.mean()) / expected
        for i in range(1, 11):
            total = K[nbins - i:nbins + i + 1].sum() - K[nbins]
            Qi[i - 1] = total / (2 * i * expected)
            lam = r00 * i
            Ri[i - 1] = .5 * (1 + math.erf(np.divide(total / 2 - lam, np.sqrt(2 * lam))))
    return K, Qi, Q00, Q01, Ri


def _refractory(st1, st2):
    K, Qi, Q00, Q01, Ri = ccg(st1, st2)
    with np.errstate(invalid='ignore', divide='ignore'):
        Q = np.min(Qi / np.maximum(Q00, Q01))
    return K, Q, np.min(Ri)


def _remap_merge_pc(rez, indices, source, target):
    """Reorder overlapping channels; unrecorded target channels are zero."""
    features = rez['cProjPC']
    near = _numpy(rez['iNeighPC'])
    src = {int(channel): i for i, channel in enumerate(near[source])}
    pairs = [(j, src[int(c)]) for j, c in enumerate(near[target]) if int(c) in src]
    for start in range(0, len(indices), 65536):
        ii = indices[start:start + 65536]
        old = _rows(features, ii)
        new = torch.zeros_like(old) if torch.is_tensor(old) else np.zeros_like(old)
        for dest, origin in pairs:
            new[:, :, dest] = old[:, :, origin]
        if torch.is_tensor(features):
            features[torch.as_tensor(ii, device=features.device)] = new
        else:
            features[ii] = new


def find_merges(rez, ops, flag=True):
    """Merge correlated, cross-refractory clusters in spike-count order.

    ``flag=False`` records symmetric diagnostic CCGs without changing IDs.
    ``ks25.fix_merge_pc=False`` retains MATLAB's unreordered PC features.
    """
    st = _numpy(rez['st3'])
    rez['st3'] = st
    similarity = _numpy(rez['simScore']).copy()
    np.fill_diagonal(similarity, 0)
    count = len(similarity)
    groups = _groups(st[:, 1], count)
    nsp = np.array([len(g) for g in groups])
    order = np.argsort(nsp, kind='stable')
    fs = float(_opt(ops, 'fs', 30000))
    if not flag:
        rez['R_CCG'] = np.full((count, count), np.inf)
        rez['Q_CCG'] = np.full((count, count), np.inf)
        rez['K_CCG'] = {}  # Sparse MATLAB cell array: only evaluated pairs.
    for source in order:
        indices = groups[source]
        if not len(indices):
            continue
        ss = st[indices, 0] / fs
        scores = similarity[source] * (nsp > len(indices))
        for target in np.argsort(-scores, kind='stable'):
            if scores[target] < .5:
                break
            K, Q, R = _refractory(ss, st[groups[target], 0] / fs)
            if flag:
                if Q < .2 and R < .05:
                    if _opt(ops, 'fix_merge_pc', True) and 'cProjPC' in rez:
                        _remap_merge_pc(rez, indices, source, target)
                    st[indices, 1] = target
                    groups[target] = np.concatenate((groups[target], indices))
                    groups[source] = np.empty(0, dtype=np.int64)
                    nsp[target] += nsp[source]
                    logger.debug('Merged %d into %d', source, target)
                    break
            else:
                rez['Q_CCG'][source, target] = rez['Q_CCG'][target, source] = Q
                rez['R_CCG'][source, target] = rez['R_CCG'][target, source] = R
                rez['K_CCG'][source, target] = K
                rez['K_CCG'][target, source] = K[::-1].copy()
    return rez


def _bimodal_pursuit(features, flag, device):
    """The 50 EM/pursuit iterations; return responsibilities or None."""
    raw = torch.as_tensor(features, dtype=torch.float64, device=device)
    flat = raw.reshape(len(raw), -1)
    clp = flat - flat.mean(0)
    if not bool(torch.isfinite(clp).all()) or not bool(clp.square().sum() > 0):
        return None
    if flag:
        # svdecon(clp')'s leading left vector, without storing nsp x nsp.
        _, vectors = torch.linalg.eigh(clp.T @ clp)
        w = vectors[:, -1]
    else:
        w = flat.mean(0)
        norm = torch.linalg.vector_norm(w)
        if not bool(norm > 0):
            return None
        w = w / norm
    x = clp @ w
    positive = x > x.mean()
    negative = x < x.mean()
    if int(positive.sum()) < 2 or int(negative.sum()) < 2:
        return None
    means = torch.stack((x[positive].mean(), x[negative].mean()))
    variances = torch.stack((x[positive].var(), x[negative].var()))
    p = positive.double().mean()
    for iteration in range(1, 51):
        if not bool(torch.isfinite(variances).all() & (variances > 0).all() & (p > 0) & (p < 1)):
            return None
        logp = -.5 * torch.log(variances) - (x[:, None] - means).square() / (2 * variances)
        logp += torch.log(torch.stack((p, 1 - p)))
        rs = torch.softmax(logp, dim=1)
        mass = rs.sum(0)
        p = rs[:, 0].mean()
        means = (rs * x[:, None]).sum(0) / mass
        variances = (rs * (x[:, None] - means).square()).sum(0) / mass
        if iteration > 10 and iteration % 2:
            if not bool((variances > 0).all()):
                return None
            weights = (rs / variances).sum(1)
            StS = clp.T @ (clp * weights[:, None]) / len(clp)
            StMu = clp.T @ (rs @ (means / variances)) / len(clp)
            w, info = torch.linalg.solve_ex(StS, StMu)
            if int(info) or not bool(torch.isfinite(w).all()):
                # Rank-deficient features: MATLAB mrdivide also uses a
                # least-squares solution. Never add a ridge to regular data.
                w = torch.linalg.pinv(StS, hermitian=True) @ StMu
            norm = torch.linalg.vector_norm(w)
            if not bool(torch.isfinite(norm) & (norm > 0)):
                return None
            x = clp @ (w / norm)
    return _numpy(rs)


def _bimodal_pursuit_batch(features, flag, device):
    """Fifty masked EM iterations, without per-iteration host checks.

    Ragged clusters share batched GEMMs/solves. Singular systems fall back
    to the scalar pseudoinverse path after the batch, rather than changing
    the equations with regularization. Initial eigenvectors retain their
    scalar signs so the two child IDs do not silently exchange places.
    """
    result = [None] * len(features)
    if not features:
        return result
    lengths = [len(f) for f in features]
    count, maximum, dim = len(features), max(lengths), int(np.prod(features[0].shape[1:]))
    clp = torch.zeros((count, maximum, dim), dtype=torch.float64, device=device)
    covariance, uncentered_means, valid = [], [], []
    for j, f in enumerate(features):
        raw = torch.as_tensor(f, dtype=torch.float64, device=device).reshape(len(f), -1)
        mean = raw.mean(0)
        centered = raw - mean
        finite = torch.isfinite(centered).all()
        valid.append(finite & (centered.square().sum() > 0))
        centered = torch.where(finite, centered, 0.)
        clp[j, :len(f)] = centered
        # Unpadded GEMMs keep covariance entries identical to scalar init.
        covariance.append(centered.T @ centered)
        uncentered_means.append(mean)
    n = torch.tensor(lengths, dtype=torch.float64, device=device)
    mask = torch.arange(maximum, device=device)[None, :] < n[:, None]
    alive = torch.stack(valid)
    if flag:
        _, vectors = torch.linalg.eigh(torch.stack(covariance))
        w = vectors[:, :, -1]
    else:
        w = torch.stack(uncentered_means)
        norm = torch.linalg.vector_norm(w, dim=1)
        alive &= norm > 0
        w = w / norm[:, None]
    w = torch.where(alive[:, None], w, 0.)
    x = torch.bmm(clp, w[:, :, None]).squeeze(2)
    midpoint = x.sum(1) / n
    positive = (x > midpoint[:, None]) & mask
    negative = (x < midpoint[:, None]) & mask
    sides = torch.stack((positive, negative), dim=2)
    sizes = sides.sum(1)
    alive &= (sizes >= 2).all(1)
    means = (x[:, :, None] * sides).sum(1) / sizes
    variances = ((x[:, :, None] - means[:, None, :]).square() * sides).sum(1) / (sizes - 1)
    p = sizes[:, 0] / n
    iterations = (_pursuit_cuda_iterations if clp.is_cuda and fastest('ks25_split_em_fused')
                  else _pursuit_torch_iterations)
    rs, alive, fallback = iterations(clp, x, means, variances, p, n, alive)
    status = _numpy(torch.stack((alive, fallback), dim=1))
    responsibilities = _numpy(rs)
    for j in range(count):
        if status[j, 1]:
            result[j] = _bimodal_pursuit(features[j], flag, device)
        elif status[j, 0]:
            result[j] = responsibilities[j, :lengths[j]].copy()
    return result


def _pursuit_torch_iterations(clp, x, means, variances, p, n, alive):
    count, maximum, dim = clp.shape
    device = clp.device
    mask = torch.arange(maximum, device=device)[None, :] < n[:, None]
    fallback = torch.zeros_like(alive)
    transpose = clp.transpose(1, 2)
    identity = torch.eye(dim, dtype=clp.dtype, device=device)[None]
    for iteration in range(1, 51):
        alive &= torch.isfinite(variances).all(1) & (variances > 0).all(1) & (p > 0) & (p < 1)
        safe_var = torch.where(alive[:, None], variances, 1.)
        safe_mu = torch.where(alive[:, None], means, 0.)
        safe_p = torch.where(alive, p, .5)
        logp = (-.5 * safe_var.log()[:, None, :]
                - (x[:, :, None] - safe_mu[:, None, :]).square() / (2 * safe_var[:, None, :])
                + torch.stack((safe_p, 1 - safe_p), dim=1).log()[:, None, :])
        rs = torch.softmax(logp, dim=2) * mask[:, :, None]
        mass = rs.sum(1)
        p = rs[:, :, 0].sum(1) / n
        means = (rs * x[:, :, None]).sum(1) / mass
        variances = (rs * (x[:, :, None] - means[:, None, :]).square()).sum(1) / mass
        if iteration > 10 and iteration % 2:
            alive &= (variances > 0).all(1)
            safe_var = torch.where(alive[:, None], variances, 1.)
            weights = (rs / safe_var[:, None, :]).sum(2)
            StS = torch.bmm(transpose, clp * weights[:, :, None]) / n[:, None, None]
            target = torch.bmm(rs, (means / safe_var)[:, :, None])
            StMu = torch.bmm(transpose, target).squeeze(2) / n[:, None]
            StS = torch.where(alive[:, None, None], StS, identity)
            StMu = torch.where(alive[:, None], StMu, 0.)
            lower, info = torch.linalg.cholesky_ex(StS, check_errors=False)
            y = torch.linalg.solve_triangular(lower, StMu[:, :, None], upper=False)
            w = torch.linalg.solve_triangular(lower.transpose(1, 2), y, upper=True).squeeze(2)
            failed = (info != 0) | ~torch.isfinite(w).all(1)
            fallback |= alive & failed
            norm = torch.linalg.vector_norm(w, dim=1)
            alive &= ~failed & torch.isfinite(norm) & (norm > 0)
            w = torch.where(alive[:, None], w / norm[:, None], 0.)
            x = torch.bmm(clp, w[:, :, None]).squeeze(2)
    return rs, alive, fallback


_PURSUIT_CUDA = r'''
__device__ double sum_block(double value, double* buffer) {
    int t = threadIdx.x;
    buffer[t] = value;
    __syncthreads();
    for (int stride=128; stride; stride/=2) {
        if (t < stride) buffer[t] += buffer[t+stride];
        __syncthreads();
    }
    double result = buffer[0];
    __syncthreads();
    return result;
}
extern "C" __global__ void pursuit_em(
        const double* x, const long long* lengths, double* model, bool* alive,
        double* rs, double* weights, double* target, int maximum, int update) {
    int b=blockIdx.x, t=threadIdx.x, n=lengths[b];
    const double *xb=x+(long long)b*maximum;
    double *rb=rs+(long long)b*maximum*2;
    double *wb=weights+(long long)b*maximum, *tb=target+(long long)b*maximum;
    double *m=model+b*5;
    double mu1=m[0], mu2=m[1], s1=m[2], s2=m[3], p=m[4];
    __shared__ double scratch[256];
    bool valid=alive[b] && isfinite(s1) && isfinite(s2) && s1>0 && s2>0 && p>0 && p<1;
    if (!valid) {
        if (!t) alive[b]=false;
        for (int i=t; i<n; i+=256) rb[2*i]=rb[2*i+1]=wb[i]=tb[i]=0;
        return;
    }
    double log1=-.5*log(s1)+log(p), log2=-.5*log(s2)+log(1-p);
    double r1sum=0, r2sum=0, x1sum=0, x2sum=0;
    for (int i=t; i<n; i+=256) {
        double v=xb[i];
        double a=log1-(v-mu1)*(v-mu1)/(2*s1), c=log2-(v-mu2)*(v-mu2)/(2*s2);
        double top=fmax(a,c), e1=exp(a-top), e2=exp(c-top);
        double r1=e1/(e1+e2), r2=e2/(e1+e2);
        rb[2*i]=r1; rb[2*i+1]=r2;
        r1sum+=r1; r2sum+=r2; x1sum+=r1*v; x2sum+=r2*v;
    }
    r1sum=sum_block(r1sum,scratch); r2sum=sum_block(r2sum,scratch);
    x1sum=sum_block(x1sum,scratch); x2sum=sum_block(x2sum,scratch);
    mu1=x1sum/r1sum; mu2=x2sum/r2sum;
    double v1=0, v2=0;
    for (int i=t; i<n; i+=256) {
        double a=xb[i]-mu1, c=xb[i]-mu2;
        v1+=rb[2*i]*a*a; v2+=rb[2*i+1]*c*c;
    }
    s1=sum_block(v1,scratch)/r1sum; s2=sum_block(v2,scratch)/r2sum;
    if (!t) {m[0]=mu1; m[1]=mu2; m[2]=s1; m[3]=s2; m[4]=r1sum/n;}
    if (update) {
        bool ok=s1>0 && s2>0;
        if (!ok && !t) alive[b]=false;
        for (int i=t; i<n; i+=256) {
            wb[i]=ok ? rb[2*i]/s1+rb[2*i+1]/s2 : 0;
            tb[i]=ok ? rb[2*i]*mu1/s1+rb[2*i+1]*mu2/s2 : 0;
        }
    }
}
extern "C" __global__ void pursuit_prepare(
        double* matrix, double* rhs, const long long* lengths, const bool* alive, int dim) {
    int b=blockIdx.x;
    for (int i=threadIdx.x; i<dim*dim; i+=256)
        matrix[(long long)b*dim*dim+i]=alive[b] ? matrix[(long long)b*dim*dim+i]/lengths[b] : (i/dim==i%dim);
    for (int i=threadIdx.x; i<dim; i+=256)
        rhs[b*dim+i]=alive[b] ? rhs[b*dim+i]/lengths[b] : 0;
}
extern "C" __global__ void pursuit_normalize(
        double* direction, const int* info, bool* alive, bool* fallback, int dim) {
    int b=blockIdx.x, t=threadIdx.x;
    __shared__ double scratch[256];
    __shared__ bool ok;
    double norm2=0;
    for (int i=t; i<dim; i+=256) norm2+=direction[b*dim+i]*direction[b*dim+i];
    double norm=sqrt(sum_block(norm2,scratch));
    if (!t) {
        bool failed=info[b]!=0 || !isfinite(norm);
        fallback[b]=fallback[b] || (alive[b] && failed);
        ok=alive[b] && !failed && norm>0;
        alive[b]=ok;
    }
    __syncthreads();
    for (int i=t; i<dim; i+=256) direction[b*dim+i]=ok ? direction[b*dim+i]/norm : 0;
}
'''


@lru_cache(maxsize=None)
def _pursuit_cuda_module(device_index):
    import cupy as cp
    with cp.cuda.Device(device_index):
        module = cp.RawModule(code=_PURSUIT_CUDA, options=('--fmad=false',))
        return tuple(module.get_function(name) for name in
                     ('pursuit_em', 'pursuit_prepare', 'pursuit_normalize'))


def _pursuit_cuda_iterations(clp, x, means, variances, p, n, alive):
    from kilosort.fused_kernels import _device, _stream, _p

    count, maximum, dim = clp.shape
    model = torch.cat((means, variances, p[:, None]), dim=1).contiguous()
    lengths = n.long()
    rs = clp.new_zeros((count, maximum, 2))
    weights, target = clp.new_zeros((count, maximum)), clp.new_zeros((count, maximum))
    fallback = torch.zeros_like(alive)
    transpose = clp.transpose(1, 2)
    with _device(clp), _stream():
        em, prepare, normalize = _pursuit_cuda_module(clp.device.index)
        for iteration in range(1, 51):
            update = iteration > 10 and iteration % 2
            em((count,), (256,), (_p(x), _p(lengths), _p(model), _p(alive), _p(rs),
                                  _p(weights), _p(target), np.int32(maximum), np.int32(update)))
            if update:
                matrix = torch.bmm(transpose, clp * weights[:, :, None])
                rhs = torch.bmm(transpose, target[:, :, None])
                prepare((count,), (256,), (_p(matrix), _p(rhs), _p(lengths), _p(alive), np.int32(dim)))
                lower, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
                y = torch.linalg.solve_triangular(lower, rhs, upper=False)
                w = torch.linalg.solve_triangular(lower.transpose(1, 2), y, upper=True)
                normalize((count,), (256,), (_p(w), _p(info), _p(alive), _p(fallback), np.int32(dim)))
                x = torch.bmm(clp, w).squeeze(2)
    return rs, alive, fallback


def _refit_templates(rez, ops, iC, iW):
    """Rebuild SVD, lagged similarities and neighbor tables after pursuit."""
    from . import learn

    W, dWU = rez['W'], rez['dWU']
    rank, nfilt, nt = W.shape
    nchan, nnear = dWU.shape[1], iC.shape[1]
    nearest = min(nchan, int(_opt(ops, 'Nnearest', 32)))
    if nfilt:
        p = np.zeros(17, dtype=np.float64)
        p[[1, 4, 5, 6, 9, 10, 11]] = [nfilt, nt, nearest, rank, nchan, nnear, _opt(ops, 'nt0min', 20)]
        local_ops = dict(ops, device=W.device, nt=nt, nt0=nt)
        Ka, Kb = learn.get_kernels(local_ops, 10, 1)
        if W.is_cuda:
            from . import kernels
            svd = kernels.svd_small2
        else:
            # Same small power-iteration reference used by CPU learning.
            svd = learn._svd_reference
        W, U, mu = svd(p, dWU.double().contiguous(), W.double().contiguous(),
                       iC.int().contiguous(), iW.int().contiguous(), Ka, Kb)
        similarity, iList = learn.get_me_wtw(W.float(), U.float(), nearest)
        same_parent = _numpy(rez['simScore']) == 1
        similarity = _numpy(similarity).copy()
        similarity[same_parent] = 1
        rez.update(W=W, U=U, mu=mu, simScore=similarity, isplit=same_parent,
                   iList=iList, iNeigh=_numpy(iList), iNeighPC=_numpy(iC[iW.long()]))
    else:
        rez['isplit'] = np.zeros((0, 0), dtype=bool)
    rez['iW'] = iW
    rez['iC'] = iC
    rez['Wphy'] = torch.nn.functional.pad(rez['W'], (int(_opt(ops, 'nt0min', 20)) + 1, 0))


def split_all_clusters(rez, ops, flag=True):
    """MATLAB bimodal pursuit with optional correction of its c1/c2 swap.

    ``flag`` selects initialization (top PC vs. mean), not whether to split.
    Set ``ks25.fix_split_swap=False`` for MATLAB's swapped child waveform.
    Batched EM evaluates independent clusters in rounds. Replaying the
    resulting split trees preserves MATLAB's parent-revisit/child-ID order.
    Disable ``ks25_split_batch`` in KS_FASTEST_OFF for the scalar reference.
    """
    batched = fastest('ks25_split_batch')
    pool = (ThreadPoolExecutor(max_workers=6 if batched else 1, thread_name_prefix='ks25-split-ccg')
            if fastest('ks25_split_ccg_threads') else nullcontext())
    with pool as executor:
        split = _split_rounds if batched else _split_all_clusters
        return split(rez, ops, flag, executor)


def _split_waveforms(features, low, pc, device):
    f = torch.as_tensor(features, device=device, dtype=torch.float64)
    mask = torch.as_tensor(low, device=device)
    c1, c2 = f[mask].mean(0).T @ pc, f[~mask].mean(0).T @ pc
    a, b = c1.flatten(), c2.flatten()
    ac, bc = a - a.mean(), b - b.mean()
    correlation = (ac @ bc) / (torch.linalg.vector_norm(ac) * torch.linalg.vector_norm(bc))
    n1, n2 = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    amplitude_difference = 2 * (n1 - n2).abs() / (n1 + n2)
    return c1, c2, correlation, amplitude_difference


def _split_chunks(active, nodes, dim, device):
    # clp, its weighted copy, initial states, and transient linear algebra.
    budget = 512 * 1024 ** 2
    if device.type == 'cuda':
        available = (torch.cuda.mem_get_info(device)[0] + torch.cuda.memory_reserved(device)
                     - torch.cuda.memory_allocated(device))
        budget = min(2 * 1024 ** 3, int(available * .25))
    ordered = sorted(active, key=lambda i: len(nodes[i]['indices']))
    batch = []
    for node in ordered:
        size = len(nodes[node]['indices'])
        if batch and (len(batch) >= 128 or (len(batch) + 1) * size * dim * 8 * 4 > budget):
            yield batch
            batch = []
        batch.append(node)
    if batch:
        yield batch


def _split_rounds(rez, ops, flag, ccg_executor):
    from . import learn

    original = rez['dWU']
    device = original.device if torch.is_tensor(original) else torch.device('cpu')
    W = torch.as_tensor(rez['W'], device=device, dtype=torch.float64)
    dWU = torch.as_tensor(original, device=device, dtype=torch.float64)
    rank, nroots, nt = W.shape
    nchan = dWU.shape[1]
    nnear = min(nchan, int(_opt(ops, 'NchanNear', 32)))
    local_ops = dict(ops, device=device)
    for name in ('xc', 'yc'):
        if name not in local_ops and name in rez:
            local_ops[name] = rez[name]
    iC, _, _ = learn.get_closest_channels(local_ops, _opt(ops, 'sigmaMask', 30), nnear)
    peak = int(_opt(ops, 'nt0min', 20)) - 1
    if not 0 <= peak < nt:
        raise ValueError('nt0min must identify a sample in dWU')
    iW = dWU[:, :, peak].abs().argmax(1).int()
    pc = torch.as_tensor(rez['wPCA'], device=device, dtype=torch.float64)[:rank]
    if tuple(pc.shape) != (rank, nt) or rez['cProjPC'].shape[1:] != (rank, nnear):
        raise ValueError('wPCA/cProjPC must match template rank, nt and channel neighborhood')
    st = _numpy(rez['st3'])
    groups = _groups(st[:, 1], nroots)
    nodes = [dict(indices=indices, root=i) for i, indices in enumerate(groups)]
    active = [i for i, node in enumerate(nodes) if len(node['indices']) >= 300]
    fs = float(_opt(ops, 'fs', 30000))
    threshold = float(_opt(ops, 'AUCsplit', .9))
    fix_swap = bool(_opt(ops, 'fix_split_swap', True))
    nsplits = vetoes = round_number = 0
    while active:
        round_number += 1
        evaluated = {}
        for chunk in _split_chunks(active, nodes, rank * nnear, device):
            features = [_rows(rez['cProjPC'], nodes[i]['indices']) for i in chunk]
            responsibilities = _bimodal_pursuit_batch(features, flag, device)
            for i, f, rs in zip(chunk, features, responsibilities):
                if rs is None or not np.isfinite(rs).all():
                    continue
                low = rs[:, 0] > rs[:, 1]
                if not low.any() or low.all():
                    continue
                ss = st[nodes[i]['indices'], 0] / fs
                ccg = (ccg_executor.submit(_refractory, ss[low], ss[~low])
                       if ccg_executor else _refractory(ss[low], ss[~low]))
                waves = _split_waveforms(f, low, pc, device)
                evaluated[i] = (low, rs, ccg, waves)
        following = []
        for i in active:
            if i not in evaluated:
                continue
            low, rs, ccg, (c1, c2, correlation, amplitude_difference) = evaluated[i]
            _, Q, R = ccg.result() if ccg_executor else ccg
            if Q < .25 and R < .05:
                vetoes += 1
                continue
            if bool((correlation > .9) & (amplitude_difference < .2)):
                continue
            if (min(low.mean(), (~low).mean()) <= .05 or min(low.sum(), (~low).sum()) <= 300
                    or min(rs[low, 0].mean(), rs[~low, 1].mean()) <= threshold):
                continue
            node = nodes[i]
            remaining, child = len(nodes), len(nodes) + 1
            nodes.extend([dict(indices=node['indices'][~low], root=node['root']),
                          dict(indices=node['indices'][low], root=node['root'])])
            node['children'] = (remaining, child)
            node['waves'] = (c2, c1) if fix_swap else (c1, c2)
            following.extend((remaining, child))
            nsplits += 1
        logger.info('Split round %d: checked %d clusters, %d splits, %d CCG vetoes',
                    round_number, len(active), nsplits, vetoes)
        active = following

    # Replay exactly the scalar traversal: revisit the remaining parent
    # immediately, append each new child at the end of the template queue.
    # This also preserves the original warm starts and simScore==1 ancestry.
    total = nroots + nsplits
    waveforms = dWU.new_zeros((total, nchan, nt))
    waveforms[:nroots] = dWU
    warm = torch.cat((W, pc[:, None, :].expand(-1, nsplits, -1)), dim=1)
    sim = np.empty((total, total), dtype=_numpy(rez['simScore']).dtype)
    sim[:nroots, :nroots] = _numpy(rez['simScore'])
    queue = list(range(nroots))
    roots = list(range(nroots))
    for cluster, index in enumerate(queue):
        node = nodes[index]
        while 'children' in node:
            remaining, child = node['children']
            new = len(queue)
            channels = iC[iW[node['root']].long()].long()
            waveforms[cluster, channels], waveforms[new, channels] = node['waves']
            sim[:new, new] = sim[:new, cluster]
            sim[new, :new + 1] = sim[cluster, :new + 1]
            sim[cluster, new] = sim[new, cluster] = 1
            queue.append(child)
            roots.append(node['root'])
            node = nodes[remaining]
        st[node['indices'], 1] = cluster
    iW = iW[torch.as_tensor(roots, device=device, dtype=torch.long)]
    rez.update(st3=st, dWU=waveforms, W=warm, simScore=sim,
               iNeigh=_numpy(rez['iNeigh'])[roots], iNeighPC=_numpy(rez['iNeighPC'])[roots])
    _refit_templates(rez, ops, iC, iW)
    logger.info('Finished batched pursuit: %d splits, %d CCG vetoes', nsplits, vetoes)
    return rez


def _split_all_clusters(rez, ops, flag, ccg_executor):
    from . import learn

    original = rez['dWU']
    device = original.device if torch.is_tensor(original) else torch.device('cpu')
    dWU = torch.as_tensor(original, device=device, dtype=torch.float64).clone()
    W = torch.as_tensor(rez['W'], device=device, dtype=torch.float64).clone()
    rank, nfilt, nt = W.shape
    nchan = dWU.shape[1]
    nnear = min(nchan, int(_opt(ops, 'NchanNear', 32)))
    local_ops = dict(ops, device=device)
    for name in ('xc', 'yc'):
        if name not in local_ops and name in rez:
            local_ops[name] = rez[name]
    iC, _, _ = learn.get_closest_channels(local_ops, _opt(ops, 'sigmaMask', 30), nnear)
    peak = int(_opt(ops, 'nt0min', 20)) - 1  # MATLAB dWU(nt0min,...)
    if not 0 <= peak < nt:
        raise ValueError('nt0min must identify a sample in dWU')
    iW = dWU[:, :, peak].abs().argmax(1).int()
    pc = torch.as_tensor(rez['wPCA'], device=device, dtype=torch.float64)[:rank]
    if tuple(pc.shape) != (rank, nt) or rez['cProjPC'].shape[1:] != (rank, nnear):
        raise ValueError('wPCA/cProjPC must match template rank, nt and channel neighborhood')
    st = _numpy(rez['st3'])
    rez['st3'] = st
    groups = _groups(st[:, 1], nfilt)
    sim = _numpy(rez['simScore']).copy()
    ineigh, ineighpc = _numpy(rez['iNeigh']).copy(), _numpy(rez['iNeighPC']).copy()
    fs = float(_opt(ops, 'fs', 30000))
    threshold = float(_opt(ops, 'AUCsplit', .9))
    fix_swap = bool(_opt(ops, 'fix_split_swap', True))
    cluster = nsplits = vetoes = 0
    while cluster < nfilt:
        if cluster % 100 == 0:
            logger.info('Split pursuit: checked %d/%d clusters, %d splits, %d CCG vetoes', cluster, nfilt, nsplits, vetoes)
        indices = groups[cluster]
        if len(indices) < 300:
            cluster += 1
            continue
        features = _rows(rez['cProjPC'], indices)
        rs = _bimodal_pursuit(features, flag, device)
        if rs is None or not np.isfinite(rs).all():
            cluster += 1
            continue
        low = rs[:, 0] > rs[:, 1]
        if not low.any() or low.all():
            cluster += 1
            continue
        ss = st[indices, 0] / fs
        if ccg_executor is not None:
            ccg_future = ccg_executor.submit(_refractory, ss[low], ss[~low])
            # Torch/GPU work remains on the caller's stream. Only NumPy and
            # nogil numba CCG code runs in the worker, with immutable inputs.
            c1, c2, correlation, amplitude_difference = _split_waveforms(features, low, pc, device)
            _, Q, R = ccg_future.result()
        else:
            _, Q, R = _refractory(ss[low], ss[~low])
        if Q < .25 and R < .05:
            vetoes += 1
            cluster += 1
            continue
        if ccg_executor is None:
            c1, c2, correlation, amplitude_difference = _split_waveforms(features, low, pc, device)
        if bool((correlation > .9) & (amplitude_difference < .2)):
            cluster += 1
            continue
        if (min(low.mean(), (~low).mean()) <= .05 or min(low.sum(), (~low).sum()) <= 300
                or min(rs[low, 0].mean(), rs[~low, 1].mean()) <= threshold):
            cluster += 1
            continue
        channels = iC[iW[cluster].long()].long()
        child = dWU.new_zeros((1, nchan, nt))
        # ilow gets the new ID in MATLAB, but c2 was assigned to that ID.
        child[0, channels] = c1 if fix_swap else c2
        dWU[cluster, channels] = c2 if fix_swap else c1
        dWU = torch.cat((dWU, child))
        W = torch.cat((W, pc[:, None, :]), dim=1)
        iW = torch.cat((iW, iW[cluster:cluster + 1]))
        st[indices[low], 1] = nfilt
        groups[cluster] = indices[~low]
        groups.append(indices[low])
        sim = np.pad(sim, ((0, 1), (0, 1)))
        sim[:-1, -1] = sim[:-1, cluster]
        sim[-1] = sim[cluster]
        sim[cluster, -1] = sim[-1, cluster] = 1
        ineigh = np.concatenate((ineigh, ineigh[cluster:cluster + 1]))
        ineighpc = np.concatenate((ineighpc, ineighpc[cluster:cluster + 1]))
        nfilt += 1
        nsplits += 1
        # Revisit the remaining parent; the appended child is visited later.
    rez.update(dWU=dWU, W=W, simScore=sim, iNeigh=ineigh, iNeighPC=ineighpc)
    _refit_templates(rez, ops, iC, iW)
    logger.info('Finished pursuit: %d splits, %d CCG vetoes', nsplits, vetoes)
    return rez


def set_cutoff(rez, ops):
    """Lower vexp thresholds by .5 while the ACG remains refractory.

    Independent clusters run on up to six CPU workers. Results and spike
    masks are applied in cluster-index order, never by completion order.
    """
    st = _numpy(rez['st3'])
    count = len(rez['mu'])
    good, contamination = np.zeros(count, dtype=bool), np.ones(count)
    thresholds = np.zeros(count)
    keep = np.ones(len(st), dtype=bool)
    high, low = np.asarray(_opt(ops, 'Th', (10., 4.)), dtype=float)
    if not np.isfinite([high, low]).all() or high < low:
        raise ValueError('Th must contain finite descending learning/extraction thresholds')
    fs = float(_opt(ops, 'fs', 30000))
    groups = _groups(st[:, 1], count)

    def one_cluster(indices):
        if not len(indices):
            return False, 1., 0., indices
        ss, vexp = st[indices, 0] / fs, st[indices, 3]
        threshold, allowed = high, .1
        is_good, contam = False, 1.
        while threshold >= low:
            selected = ss[vexp > threshold]
            if not len(selected):
                threshold -= .5
                continue
            _, Q, R = _refractory(selected, selected)
            if Q > allowed or R > .05:
                break
            if threshold == high and Q < .05:
                allowed = min(.05, max(.01, Q * 2))
            is_good = True
            threshold -= .5
        threshold += .5
        selected = ss[vexp > threshold]
        if len(selected):
            _, contam, _ = _refractory(selected, selected)
        return is_good, contam, threshold, indices[vexp <= threshold]

    pool = (ThreadPoolExecutor(max_workers=min(6, count), thread_name_prefix='ks25-cutoff')
            if count > 1 and fastest('ks25_cutoff_threads') else nullcontext())
    with pool as executor:
        results = executor.map(one_cluster, groups) if executor else map(one_cluster, groups)
        for cluster, (is_good, contam, threshold, removed) in enumerate(results):
            good[cluster], contamination[cluster], thresholds[cluster] = is_good, contam, threshold
            keep[removed] = False
    contamination[np.isnan(contamination)] = 1
    rez.update(good=good, est_contam_rate=contamination, Ths=thresholds)
    # Zero is a valid template ID. A mask replaces MATLAB's 0 sentinel.
    remaining = np.flatnonzero(keep)
    rez['st3'] = st[remaining]
    for name in ('cProj', 'cProjPC', 'st2'):
        if name in rez and len(rez[name]) == len(st):
            rez[name] = _rows(rez[name], remaining)
    return rez


def get_good_units(rez, ops):
    """Reject spatially diffuse templates (weighted radius >= 100 um)."""
    dWU = _numpy(rez['dWU'])
    energy = np.square(dWU).sum(-1)
    energy[energy < energy.max(1, keepdims=True) / 10] = 0
    coords = []
    for name in ('xc', 'yc'):
        value = rez.get(name, ops.get(name, ops.get('probe', {}).get(name)))
        coords.append(_numpy(value).reshape(-1))
    xy = np.column_stack(coords)
    mass = energy.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        center = energy @ xy / mass[:, None]
        distance = np.linalg.norm(xy[None] - center[:, None], axis=-1)
        radius = (distance * energy).sum(1) / mass
    return _numpy(rez['good']).astype(bool) & (radius < 100)


def _save_feature(path, values, order):
    """Stable row permutation into mmap; avoid a second full PC feature copy."""
    shape = (len(order), *values.shape[1:])
    dtype = _numpy(values[:0]).dtype
    if not len(order) or np.prod(shape) == 0:
        np.save(path, np.empty(shape, dtype=dtype))
        return
    mapped = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
    chunk = max(1, (32 * 1024 ** 2) // max(1, int(np.prod(shape[1:])) * dtype.itemsize))
    for start in range(0, len(order), chunk):
        mapped[start:start + chunk] = _numpy(_rows(values, order[start:start + chunk]))
    mapped.flush()
    del mapped


def save_to_phy25(rez, ops, results_dir):
    """Write raw-binary Phy results and preserve the six-PC KS4 bench ops.

    MATLAB exported a prewhitened int16 file. Here Phy reads the original
    recording, so channel_map/Wrot are real mappings and hp_filtered=False.
    """
    from kilosort import io

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    st = _numpy(rez['st3'])
    order = np.argsort(st[:, 0], kind='stable')
    sorted_st = st[order]
    ids = sorted_st[:, 1].astype(np.int32)
    W = _numpy(rez['Wphy']) if 'Wphy' in rez else np.pad(
        _numpy(rez['W']), ((0, 0), (0, 0), (int(_opt(ops, 'nt0min', 20)) + 1, 0)))
    U = _numpy(rez['U'])
    templates = np.einsum('rfc,rft->ftc', U.astype(np.float32), W.astype(np.float32))
    nfilt, _, nchan = templates.shape
    whitening = _numpy(rez.get('Wrot', ops.get('Wrot')))
    try:
        inv_whitening = np.linalg.inv(whitening)
    except np.linalg.LinAlgError:
        # Preserve the ordinary inverse for full-rank recordings, while
        # allowing dead/redundant channels to be exported to Phy as well.
        logger.warning('Singular whitening matrix; exporting its pseudoinverse.')
        inv_whitening = np.linalg.pinv(whitening)
    positions = np.column_stack([_numpy(rez.get(k, ops.get(k, ops.get('probe', {}).get(k)))).reshape(-1)
                                 for k in ('xc', 'yc')])
    arrays = {
        'spike_times': sorted_st[:, 0].astype(np.int64),
        'spike_templates': ids, 'spike_clusters': ids,
        'amplitudes': sorted_st[:, 2], 'templates': templates,
        'templates_ind': np.tile(np.arange(nchan, dtype=np.int32), (nfilt, 1)),
        'channel_map': _numpy(ops['chanMap']).astype(np.int32).reshape(-1),
        'channel_positions': positions,
        'template_feature_ind': _numpy(rez['iNeigh']).astype(np.int32),
        'pc_feature_ind': _numpy(rez['iNeighPC']).astype(np.int32),
        'whitening_mat': whitening, 'whitening_mat_inv': inv_whitening,
        'similar_templates': _numpy(rez['simScore']),
    }
    for name, array in arrays.items():
        io._save_mapped(output / f'{name}.npy', array) if array.size else np.save(output / f'{name}.npy', array)
    _save_feature(output / 'template_features.npy', rez['cProj'], order)
    _save_feature(output / 'pc_features.npy', rez['cProjPC'], order)
    unwhitened = templates @ inv_whitening
    peak_to_peak = np.ptp(unwhitened, axis=1).max(1)
    counts = np.bincount(ids, minlength=nfilt)
    sums = np.bincount(ids, weights=peak_to_peak[ids] * sorted_st[:, 2], minlength=nfilt)
    amplitudes = np.divide(sums, counts, out=np.zeros(nfilt), where=counts > 0) * float(_opt(ops, 'gain', 1))
    good = _numpy(rez['good']).astype(bool)
    contamination = _numpy(rez['est_contam_rate']).copy()
    contamination[np.isnan(contamination)] = 1
    columns = {'KSLabel': ['good' if g else 'mua' for g in good],
               'group': ['good' if g else 'mua' for g in good],
               'ContamPct': [f'{q * 100:.1f}' for q in contamination],
               'Amplitude': [f'{a:.1f}' for a in amplitudes]}
    for name, values in columns.items():
        with (output / f'cluster_{name}.tsv').open('w', encoding='utf8', newline='') as stream:
            stream.write(f'cluster_id\t{name}\n')
            stream.writelines(f'{i}\t{v}\n' for i, v in enumerate(values))
    filename = ops['filename']
    dat_path = [str(Path(f).resolve()).replace('\\', '/') for f in filename] if isinstance(filename, list) else str(Path(filename).resolve()).replace('\\', '/')
    params = (f'dat_path = {dat_path!r}\nn_channels_dat = {int(ops["n_chan_bin"])}\n'
              f'dtype = {np.dtype(ops.get("data_dtype", "int16")).name!r}\noffset = 0\n'
              f'sample_rate = {float(ops["fs"])!r}\nhp_filtered = False\n')
    (output / 'params.py').write_text(params, encoding='utf8')
    saved_ops = dict(ops)
    # save_ops normalizes paths inside settings; do not mutate the caller.
    saved_ops['settings'] = dict(ops['settings'])
    for key in ('W', 'U', 'mu', 'simScore', 'iNeigh', 'iNeighPC', 'good', 'est_contam_rate', 'Ths', 'isplit'):
        if key in rez:
            saved_ops[key] = _numpy(rez[key])
    saved_ops['wPCA25'] = _numpy(rez['wPCA'])
    # Deliberately preserve ops['wPCA'] and ops['n_pcs'] for bench.avg_wav.
    io.save_ops(saved_ops, output)
