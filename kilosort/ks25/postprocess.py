"""KS2.5 refractory merging, bimodal pursuit, cutoff and Phy export.

Array layouts and zero-based template IDs follow ``KS25_BRIEF.md``.  The
equations/thresholds come from the MATLAB postProcess directory, not KS4's
CCG implementation (which has different histogram endpoints and shoulders).
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
    CCG runs on a CPU worker alongside waveform reconstruction; all parent
    revisits and template mutations stay ordered on the calling thread.
    """
    pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='ks25-split-ccg')
            if fastest('ks25_split_ccg_threads') else nullcontext())
    with pool as executor:
        return _split_all_clusters(rez, ops, flag, executor)


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
