"""Kilosort 2.5 template learning and final extraction.

Tensor layouts follow the reversed MATLAB dimensions documented in KS25_BRIEF.
"""
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _opt(ops, name, default=None):
    return ops.get(name, ops.get('ks25', {}).get(name, ops.get('settings', {}).get(name, default)))


def _device(ops):
    device = ops.get('device', 'cpu')
    return torch.device(device)


def _coords(ops, name):
    return ops.get(name, ops.get('probe', {}).get(name))


def get_kernels(ops, nup=10, sig=1):
    nt = int(_opt(ops, 'nt0', _opt(ops, 'nt', 61)))
    peak = int(_opt(ops, 'nt0min', 20)) - 1
    if not 0 <= peak < nt or nup <= 0 or sig <= 0:
        raise ValueError('invalid interpolation parameters')
    device = _device(ops)
    xs = torch.arange(1, nt + 1, device=device, dtype=torch.float64)
    ys = .5 + torch.arange(nt * nup, device=device, dtype=torch.float64) / nup
    def kernel(a, b):
        d = torch.remainder(a[:, None] - b[None, :], nt)
        d = torch.minimum(d, nt - d)
        return torch.exp(-d.square() / sig**2)
    kxx, kyx = kernel(xs, xs), kernel(ys, xs)
    b = torch.linalg.solve(kxx + .01 * torch.eye(nt, dtype=xs.dtype, device=device), kyx.T).T
    b = b.reshape(nt, nup, nt).permute(1, 0, 2).contiguous()
    return b[:, peak, :].T.contiguous(), b


def get_closest_channels(ops, sigma, NchanClosest):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    device = _device(ops)
    x = torch.as_tensor(_coords(ops, 'xc'), dtype=torch.float32, device=device).flatten()
    y = torch.as_tensor(_coords(ops, 'yc'), dtype=torch.float32, device=device).flatten()
    if x.numel() != y.numel() or not 0 < NchanClosest <= x.numel():
        raise ValueError('invalid channel coordinates or neighbor count')
    distances = torch.sqrt((x[:, None] - x).square() + (y[:, None] - y).square())
    order = torch.argsort(distances, dim=1, stable=True)[:, :NchanClosest]
    weights = torch.exp(-distances.gather(1, order).square() / (2 * sigma**2))
    weights /= torch.sqrt(1e-3 + weights.square().sum(dim=1, keepdim=True))
    return order.to(torch.int32).contiguous(), weights.contiguous(), distances


def get_me_utu(iU, iC, mask, Nnearest, Nchan):
    del Nnearest
    iU = torch.as_tensor(iU, device=iC.device, dtype=torch.long).flatten()
    channels = iC.long()[iU]
    membership = torch.zeros((iU.numel(), Nchan), device=iC.device, dtype=torch.float32)
    membership.scatter_(1, channels, 1)
    return (membership @ membership.T > 0), mask[iU].contiguous()


def _neighbors(cc, count):
    n = cc.shape[0]
    if count is None:
        return None
    if n == 0:
        return torch.empty((0, count), dtype=torch.int32, device=cc.device)
    order = torch.argsort(cc, dim=1, descending=True, stable=True)
    return order[:, torch.arange(count, device=cc.device) % n].to(torch.int32).contiguous()


def _wtw_reference(w1, w2, utu):
    n, nt = w1.shape
    result = w1.new_zeros((2 * nt - 1, n, n))
    # Output rows correspond to MATLAB's second index, columns to its first.
    for lag in range(2 * nt - 1):
        shift = nt - lag - 1
        a = w1[:, max(0, -shift):min(nt, nt-shift)]
        b = w2[:, max(0, shift):min(nt, nt+shift)]
        result[lag] = (b @ a.T) * utu
    return result


def get_me_wtw(W, U, Nnearest):
    """Maximum full-template correlation over all lags and nearest templates."""
    rank, n, nt = W.shape
    try:
        from . import kernels
        wtw_max = getattr(kernels, 'wtw_max', None) if W.is_cuda else None
    except ImportError:
        wtw_max = None
    if wtw_max is not None:
        cc = wtw_max(W.float().contiguous(), U.float().contiguous())
    else:
        result = W.new_zeros((2 * nt - 1, n, n), dtype=torch.float32)
        for i in range(rank):
            for j in range(rank):
                spatial = (U[j].float() @ U[i].float().T).contiguous()
                result += _wtw_reference(W[i].float(), W[j].float(), spatial)
        cc = result.max(dim=0).values
    return cc, _neighbors(cc, Nnearest)


def get_me_wtw2(W, U, Nnearest=None):
    # Matrix row = reference template, column = neighboring template.
    W, U = W.float(), U.float()
    cc = W.new_zeros((W.shape[1], W.shape[1]))
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            cc += (W[i] @ W[j].T) * (U[i] @ U[j].T)
    return cc, _neighbors(cc, Nnearest)


def triage_templates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop):
    del iW, C2C
    threshold = float(_opt(ops, 'minFR', 1/50)) * int(_opt(ops, 'batch_size', 60000)) / float(_opt(ops, 'fs', 30000))
    keep = nsp >= threshold
    ndrop[0] = .9 * ndrop[0] + .1 * int((~keep).sum())
    W, U, dWU, mu, nsp = W[:, keep], U[:, keep], dWU[keep], mu[keep], nsp[keep]
    if nsp.numel():
        cc, _ = get_me_wtw2(W, U)
        cc.fill_diagonal_(0)
        delta = (mu[:, None] - mu[None, :]).abs()
        drop = ((cc > .9) & (4 * math.sqrt(10) > delta) & (nsp[:, None] < nsp[None, :])).any(dim=1)
        ndrop[1] = .9 * ndrop[1] + .1 * int(drop.sum())
        keep = ~drop
        W, U, dWU, mu, nsp = W[:, keep], U[:, keep], dWU[keep], mu[keep], nsp[keep]
    else:
        ndrop[1] *= .9
    return W.contiguous(), U.contiguous(), dWU.contiguous(), mu.contiguous(), nsp.contiguous(), ndrop


def extract_templates_from_snippets(ops, bfile, nPCs=6):
    nt0 = int(_opt(ops, 'nt0', _opt(ops, 'nt', 61)))
    nt0min = int(_opt(ops, 'nt0min', 20))
    nskip = int(_opt(ops, 'nskip', 25))
    threshold = float(_opt(ops, 'spkTh', -6))
    loc_t, loc_c = _opt(ops, 'loc_range', (5, 4))
    long_t, long_c = _opt(ops, 'long_range', (30, 6))
    clips = []
    count = 0
    for batch in range(0, bfile.n_batches, nskip):
        X = bfile.padded_batch_to_torch(batch, ops)
        # MATLAB uses the unpadded batch for this stage.
        pad = int(_opt(ops, 'nt', nt0))
        X = X[:, pad:pad + int(_opt(ops, 'batch_size', X.shape[1] - 2 * pad))]
        xx = X[None, None]
        local_min = -F.max_pool2d(-xx, (2*loc_c+1, 2*loc_t+1), stride=1, padding=(loc_c, loc_t))[0, 0]
        peaks = (X < local_min + 1e-3) & (X < threshold)
        # Count, not average, the peaks in the isolation rectangle.
        sums = F.avg_pool2d(peaks[None, None].float(), (2*long_c+1, 2*long_t+1), stride=1, padding=(long_c, long_t), divisor_override=1)[0, 0]
        peaks &= sums < 1.2
        peaks[:, :nt0] = False
        peaks[:, -nt0-1:] = False
        ch, t = peaks.nonzero(as_tuple=True)
        if t.numel():
            offsets = torch.arange(nt0, device=X.device) + 1 - nt0min
            c = X[ch[:, None], t[:, None] + offsets].float()
            clips.append(c)
            count += len(c)
        if count > 100000:
            break
    if count < nPCs:
        raise ValueError(f'Not enough isolated spikes to learn {nPCs} waveforms: {count}')
    dd = torch.cat(clips, dim=0)
    # MATLAB round is half away from zero; linspace is one-based here.
    ind = torch.floor(torch.linspace(0, count-1, nPCs, device=dd.device) + .5).long()
    temp = F.normalize(dd[ind], dim=1)
    for _ in range(10):
        corr = temp @ dd.T
        amplitude, labels = corr.max(dim=0)
        temp = torch.stack([(dd[labels == j] * amplitude[labels == j, None]).sum(dim=0) for j in range(nPCs)])
        temp = F.normalize(temp, dim=1)
    _, _, vh = torch.linalg.svd(dd.double(), full_matrices=False)
    pca = vh[:nPCs].float()
    pca[0] *= -torch.sign(pca[0, nt0min-1])
    return temp.contiguous(), pca.contiguous()


def _get_spikes_reference(p, X, wTEMP, iC):
    nt0, nsum = int(p[4]), int(p[13])
    nchan, nt = X.shape
    filtered = F.conv1d(X[:, None, :], wTEMP[:, None, :]).permute(1, 0, 2)
    cs = filtered[:, iC.long()[:, :nsum], :].cumsum(dim=2)
    scores = cs.square() / torch.arange(1, nsum+1, device=X.device)[None, None, :, None]
    best, choice = scores.permute(1, 0, 2, 3).reshape(nchan, -1, filtered.shape[-1]).max(dim=1)
    err, chan = best.max(dim=0)
    lock = nt0-1
    local = F.max_pool1d(err[None, None], 2*lock+1, stride=1, padding=lock)[0, 0]
    valid = (err > float(p[2])**2) & (err >= local)
    valid[:lock] = False
    valid[max(0, nt-lock-1):] = False
    times = valid.nonzero().flatten()[:5000]
    ch = chan[times]
    # cleanup_heights retains only the strongest candidate within four channel indices.
    if times.numel():
        near = (ch[:, None] - ch[None, :]).abs() < 5
        keep = ~((err[times][None, :] > err[times][:, None]) & near).any(dim=1)
        times, ch = times[keep], ch[keep]
    clips = X.new_zeros((times.numel(), nchan, nt0))
    for idx in range(times.numel()):
        k = int(choice[ch[idx], times[idx]])
        nch, rank = k % nsum + 1, k // nsum
        channels = iC[ch[idx], :nch].long()
        clips[idx, channels] = torch.sqrt(err[times[idx]] / nch) * wTEMP[rank]
    dout = X.new_zeros(X.shape)
    dout[:, :best.shape[-1]] = best
    return clips, dout


def _svd_reference(p, dWU, W, iC, iW, Ka, Kb):
    rank, n, nt = W.shape
    nchan = dWU.shape[1]
    outW, outU = W.clone(), dWU.new_zeros((rank, n, nchan))
    mu = dWU.new_zeros(n)
    target = int(p[11])
    for f in range(n):
        a = dWU.new_zeros((nchan, nt))
        channels = iC[iW[f].long()].long()
        a[channels] = dWU[f, channels]
        cov = a.T @ a
        for k in range(rank):
            v = outW[k, f].clone()
            for _ in range(99):
                v = cov @ v
                v /= torch.sqrt(1e-5 + v.square().sum())
            x = cov @ v
            cov -= x[:, None] * v[None, :]
            outW[k, f] = v
        outW[0, f] *= -torch.where(outW[0, f, target] < 0, -1., 1.)
        uu = outW[:, f] @ a.T
        mu[f] = torch.sqrt(1e-7 + uu.square().sum())
        outU[:, f] = uu / mu[f]
        shift = int(outW[0, f].abs().argmax()) - target
        shifted = torch.roll(outW[:, f], -shift, dims=-1)
        ups = Ka.T @ shifted[0]
        best = int(ups.abs().argmax())
        outW[:, f] = shifted @ Kb[best].T
        outW[0, f] *= -torch.where(ups[best] < 0, -1., 1.)
    return outW.contiguous(), outU.contiguous(), mu


def _mpnu_reference(p, X, U, W, mu, iC, iW, UtU, iList, wPCA):
    del UtU
    nchan, nt = X.shape
    rank, nfilt, nt0 = W.shape
    nnearest, nnear = int(p[5]), int(p[10])
    lam = float(p[7])
    templates = torch.einsum('rfc,rft->fct', U, W)
    residual = X.double().clone() if int(p[16]) else X.clone()
    ids, times, amps, values, features = [], [], [], [], []
    for _ in range(int(p[3])):
        corr = F.conv1d(residual.float()[None], templates)[0]
        score = (corr.clamp_min(0) + lam * mu[:, None]).square() / (1+lam) - lam * mu[:, None].square()
        best, which = score.max(dim=0)
        best = best.clamp_min(0)
        best[:nt0] = 0
        lock = nt0 - 1
        local = F.max_pool1d(best[None, None], 2*lock+1, stride=1, padding=lock)[0, 0]
        valid = (best > float(p[2])**2) & (best >= local)
        valid[:lock] = False
        valid[max(0, nt-lock-1):] = False
        tt = valid.nonzero().flatten()
        if not tt.numel():
            break
        ii = which[tt]
        aa = corr[ii, tt]
        vv = best[tt].sqrt()
        for t, i, a in zip(tt.tolist(), ii.tolist(), aa):
            if int(p[12]) > 1:
                near = iList[i].long()
                lo, hi = max(0, t-3), min(corr.shape[1], t+3)
                mm = mu[near]
                feat = ((corr[near, lo:hi] + lam/mm[:, None]) / torch.sqrt(1+lam/mm[:, None].square())).clamp_min(0).max(dim=1).values
                features.append(feat)
            residual[:, t:t+nt0] -= a.to(residual.dtype) * templates[i].to(residual.dtype)
        times.append(tt); ids.append(ii); amps.append(aa); values.append(vv)
    st = torch.cat(times).int() if times else torch.empty(0, dtype=torch.int32, device=X.device)
    ident = torch.cat(ids).int() if ids else st.clone()
    amp = torch.cat(amps) if amps else X.new_empty(0)
    vexp = torch.cat(values) if values else X.new_empty(0)
    feat = torch.stack(features) if features else X.new_zeros((len(st), nnearest))
    accum = torch.zeros((nfilt, nchan, nt0), dtype=torch.float64, device=X.device)
    counts = torch.zeros(nfilt, dtype=torch.int32, device=X.device)
    pc = X.new_zeros((len(st), rank, nnear))
    residual = residual.float()
    for idx in torch.argsort(st, stable=True).tolist():
        t, i = int(st[idx]), int(ident[idx])
        snippet = residual[:, t:t+nt0] + amp[idx] * templates[i]
        if vexp[idx] > float(p[15]):
            accum[i] += snippet.double()
            counts[i] += 1
        if int(p[12]):
            channels = iC[iW[i].long()].long()
            pc[idx] = wPCA[:rank] @ snippet[channels].T
    return dict(st0=st, id0=ident, x0=amp, vexp=vexp, featW=feat,
                dWU0=accum, drez=residual, nsp0=counts, featPC=pc)


def _backend(name, *args):
    try:
        from . import kernels
        fn = getattr(kernels, name, None)
    except ImportError:
        fn = None
    if fn is not None and args[1].is_cuda:
        return fn(*args)
    return {'get_spikes2': _get_spikes_reference, 'svd_small2': _svd_reference,
            'mpnu8': _mpnu_reference}[name](*args)


def _parameters(ops, nchan, nnearest, nnear, rank, rankpc, threshold, pm, flag, nt):
    th = _opt(ops, 'Th', [10, 4])
    return np.array([nt, 0, threshold, int(_opt(ops, 'nInnerIter', 60)),
                     int(_opt(ops, 'nt0', _opt(ops, 'nt', 61))), nnearest,
                     rank, float(_opt(ops, 'lam', 10)), pm, nchan, nnear,
                     int(_opt(ops, 'nt0min', 20)), flag, min(nchan, nnear, 7), rankpc,
                     float(np.atleast_1d(th)[0]), int(_opt(ops, 'useStableMode', 1))], dtype=np.float64)


def _setup(ops, X, rankpc=6):
    ops1 = dict(ops, device=X.device)
    nchan = X.shape[0]
    near = min(nchan, int(_opt(ops, 'NchanNear', 32)))
    nnearest = min(nchan, int(_opt(ops, 'Nnearest', 32)))
    iC, mask, C2C = get_closest_channels(ops1, float(_opt(ops, 'sigmaMask', 30)), near)
    Ka, Kb = get_kernels(ops1)
    return iC, mask, C2C, Ka, Kb, near, nnearest


def _progress(progress_bar, value, total):
    if progress_bar is not None:
        if callable(progress_bar):
            progress_bar(value, total)
        elif hasattr(progress_bar, 'update'):
            progress_bar.update(1)


def learn_templates(ops, bfile, iorder, progress_bar=None):
    order = np.asarray(iorder, dtype=np.int64).flatten()
    if not len(order):
        raise ValueError('iorder must contain at least one batch')
    if np.any(order < 0) or np.any(order >= bfile.n_batches):
        raise ValueError('iorder contains an invalid batch index')
    wTEMP = ops.get('wTEMP')
    wPCA = ops.get('wPCA')
    if wTEMP is None or wPCA is None or len(wPCA) < 3:
        wTEMP, wPCA = extract_templates_from_snippets(ops, bfile)
    X = bfile.padded_batch_to_torch(int(order[0]), ops)
    wTEMP = torch.as_tensor(wTEMP, device=X.device, dtype=torch.float32).contiguous()
    wPCA = torch.as_tensor(wPCA, device=X.device, dtype=torch.float32)[:3].contiguous()
    rank, nt0 = wPCA.shape
    nchan = X.shape[0]
    iC, mask, C2C, Ka, Kb, near, nnearest = _setup(ops, X)
    batch_size = int(_opt(ops, 'batch_size', 60000))
    fs = float(_opt(ops, 'fs', 30000))
    min_count = float(_opt(ops, 'minFR', 1/50)) * batch_size / fs
    max_templates = int(_opt(ops, 'nfilt_factor', 4) * nchan)
    if max_templates < 1:
        raise ValueError('nfilt_factor must allow at least one template')
    momentum = _opt(ops, 'momentum', [20, 400])
    schedule = np.exp(-1 / np.linspace(momentum[0], momentum[1], len(order)))
    threshold = float(np.atleast_1d(_opt(ops, 'Th', [10, 4]))[0])
    p = _parameters(ops, nchan, nnearest, near, rank, wTEMP.shape[0], threshold, schedule[0], 0, X.shape[1])
    basis = wPCA.double()
    ndrop = np.zeros(3)
    W = U = dWU = mu = nsp = None
    for step, batch in enumerate(order):
        if step:
            X = bfile.padded_batch_to_torch(int(batch), ops)
        p[0], p[8] = X.shape[1], schedule[step]
        if step == 0:
            candidates, _ = _backend('get_spikes2', p, X, wTEMP, iC)
            candidates = candidates[:max_templates].double()
            if not len(candidates):
                raise ValueError('No initial templates detected in the first learning batch')
            dWU = (candidates @ basis.T) @ basis
            W = basis[:, None, :].expand(-1, len(dWU), -1).clone().contiguous()
            nsp = torch.full((len(dWU),), min_count, dtype=torch.float64, device=X.device)
        iW = dWU[:, :, int(p[11]) - 1].abs().argmax(dim=1).int()
        iW, order_idx = torch.sort(iW, stable=True)
        W, dWU, nsp = W[:, order_idx].contiguous(), dWU[order_idx].contiguous(), nsp[order_idx].contiguous()
        p[1] = len(dWU)
        W, U, mu = _backend('svd_small2', p, dWU, W, iC, iW, Ka, Kb)
        utu, _ = get_me_utu(iW, iC, mask, nnearest, nchan)
        # No features in training; indices are not read by mexMPnu8.
        ilist = torch.zeros((len(dWU), nnearest), dtype=torch.int32, device=X.device)
        result = _backend('mpnu8', p, X, U.float().contiguous(), W.float().contiguous(), mu.float().contiguous(), iC, iW, utu, ilist, wPCA)
        counts = result['nsp0'].double()
        factor = torch.exp(counts * math.log(schedule[step]))
        dWU = dWU * factor[:, None, None] + (1-factor[:, None, None]) * (result['dWU0'] / counts.clamp_min(1)[:, None, None])
        nsp = .95*nsp + .05*counts
        if step < len(order)-1:
            if step % 5 == 0:
                W, U, dWU, mu, nsp, ndrop = triage_templates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop)
            p[1] = len(dWU)
            candidates, _ = _backend('get_spikes2', p, result['drez'], wTEMP, iC)
            if len(candidates):
                ndrop[2] = .9*ndrop[2] + .1*len(candidates)
                candidates = candidates[:max(0, max_templates - len(dWU))].double()
                candidates = (candidates @ basis.T) @ basis
                dWU = torch.cat((dWU, candidates)).contiguous()
                W = torch.cat((W, basis[:, None, :].expand(-1, len(candidates), -1)), dim=1).contiguous()
                nsp = torch.cat((nsp, nsp.new_full((len(candidates),), min_count)))
                mu = torch.cat((mu, mu.new_full((len(candidates),), 10)))
        if step % 100 == 0:
            logger.info('learning batch %d/%d, %d templates', step+1, len(order), len(dWU))
        _progress(progress_bar, step+1, len(order))
    W, U, dWU, mu, nsp, ndrop = triage_templates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop)
    if not len(mu):
        raise ValueError('All learned templates were removed by triage')
    wtw, ilist = get_me_wtw(W.float(), U.float(), nnearest)
    iW = dWU[:, :, int(p[11])-1].abs().argmax(dim=1).int()
    return dict(W=W, U=U, mu=mu, dWU=dWU, nsp=nsp, simScore=wtw,
                iNeigh=ilist, iNeighPC=iC[iW.long()].contiguous(), iC=iC, iW=iW,
                wPCA=wPCA, wTEMP=wTEMP, ndrop=ndrop)


def track_and_sort(ops, bfile, rez, progress_bar=None):
    W, U, mu = rez['W'], rez['U'], rez['mu']
    device = W.device
    W, U, mu = W.double().contiguous(), U.double().contiguous(), mu.double().contiguous()
    nchan = U.shape[-1]
    rank, nfilt, nt0 = W.shape
    wPCA = torch.as_tensor(rez['wPCA'], device=device, dtype=torch.float32)[:rank].contiguous()
    X = bfile.padded_batch_to_torch(0, ops)
    iC, mask, _, Ka, Kb, near, nnearest = _setup(ops, X)
    dWU = torch.einsum('rft,rfc,f->fct', W, U, mu).contiguous()
    iW = dWU[:, :, int(_opt(ops, 'nt0min', 20))-1].abs().argmax(dim=1).int()
    _, ilist = get_me_wtw(W.float(), U.float(), nnearest)
    threshold = float(np.atleast_1d(_opt(ops, 'Th', [10, 4]))[-1])
    momentum = _opt(ops, 'momentum', [20, 400])
    p = _parameters(ops, nchan, nnearest, near, rank, len(rez['wTEMP']), threshold, math.exp(-1/momentum[-1]), 2, X.shape[1])
    p[1] = nfilt
    total = dWU.clone()
    counts = torch.zeros(nfilt, dtype=torch.float64, device=device)
    rows, fw, fpc = [], [], []
    batch_size, pad = int(_opt(ops, 'batch_size', 60000)), int(_opt(ops, 'nt', nt0))
    for batch in range(bfile.n_batches):
        if batch:
            X = bfile.padded_batch_to_torch(batch, ops)
        p[0] = X.shape[1]
        W, U, mu = _backend('svd_small2', p, dWU, W, iC, iW, Ka, Kb)
        utu, _ = get_me_utu(iW, iC, mask, nnearest, nchan)
        result = _backend('mpnu8', p, X, U.float().contiguous(), W.float().contiguous(), mu.float().contiguous(), iC, iW, utu, ilist, wPCA)
        total += result['dWU0']
        counts += result['nsp0'].double()
        keep = (result['st0'] >= pad) & (result['st0'] < pad + batch_size)
        st = result['st0'][keep].cpu().numpy().astype(np.float64)
        ids = result['id0'][keep].cpu().numpy()
        amp = result['x0'][keep].cpu().numpy()
        vexp = result['vexp'][keep].cpu().numpy()
        st += int(bfile.imin) + batch*batch_size - pad + int(p[11])
        rows.append(np.column_stack((st, ids, amp, vexp, np.full(len(st), batch))))
        fw.append(result['featW'][keep].cpu().numpy())
        fpc.append(result['featPC'][keep].cpu().numpy())
        if batch % 100 == 0:
            logger.info('extracting batch %d/%d, %d spikes', batch+1, bfile.n_batches, len(st))
        _progress(progress_bar, batch+1, bfile.n_batches)
    rez.update(W=W, U=U, mu=mu, dWU=total / counts.clamp_min(1)[:, None, None], nsp=counts,
               iC=iC, iW=iW, iNeigh=ilist, iNeighPC=iC[iW.long()].contiguous())
    return np.concatenate(rows), np.concatenate(fw), np.concatenate(fpc), rez
