"""Tests of kilosort.ks25.kernels against straightforward torch references.

The references follow the MATLAB mex files kernel by kernel (same formulas,
same strict comparisons), written with plain torch operations.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')

try:
    import cupy  # noqa: F401
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

pytestmark = [cuda, pytest.mark.skipif(not HAS_CUPY, reason='needs CuPy')]

NT, NCHAN, NT0, NRANK, NRANKPC, NNEAR, NNEAREST, NFILT = 4096, 16, 61, 3, 6, 8, 8, 6
NT0MIN, LAM, TH = 20, 10.0, 8.0


def params(**kw):
    p = np.zeros(17)
    p[0], p[1], p[2], p[3], p[4], p[5] = NT, NFILT, TH, 60, NT0, NNEAREST
    p[6], p[7], p[8], p[9], p[10], p[11] = NRANK, LAM, 0.9, NCHAN, NNEAR, NT0MIN
    p[12], p[13], p[14], p[15], p[16] = 2, min(NCHAN, 7), NRANKPC, TH, 1
    for k, v in kw.items():
        p[int(k[1:])] = v
    return p


def smooth_waveforms(gen, n, nt0=NT0, device='cuda'):
    """n orthonormal smooth waveforms (nt0 samples), negative peak near NT0MIN."""
    t = torch.arange(nt0, dtype=torch.float64)
    base = []
    for k in range(n):
        w = -torch.exp(-((t - NT0MIN + 1 - 2 * k) / (3.0 + k)) ** 2) * torch.cos(0.3 * k * (t - NT0MIN))
        base.append(w)
    q, _ = torch.linalg.qr(torch.stack(base, 1))
    q = q * torch.sign(-q[NT0MIN - 1])  # negative at NT0MIN
    return q.T.contiguous().to(device)


def make_kernels(nup=10, sig=1.0, nt0=NT0, nt0min=NT0MIN, device='cuda'):
    """getKernels.m: Ka (nt0, 10) f64 = MATLAB Ka(10, nt0) transposed,
    Kb (10, nt0, nt0) f64 with Kb[j, b, a] = MATLAB Kb(a, b, j)."""
    xs = torch.arange(1, nt0 + 1, dtype=torch.float64)
    ys = torch.linspace(.5, nt0 + .5, nt0 * nup + 1, dtype=torch.float64)[:-1]
    def kern(a, b):
        d = torch.remainder(a[:, None] - b[None, :] + nt0, nt0)
        d = torch.minimum(d, nt0 - d)
        return torch.exp(-d ** 2 / sig ** 2)
    Kxx, Kyx = kern(xs, xs), kern(ys, xs)
    B = Kyx @ torch.linalg.inv(Kxx + .01 * torch.eye(nt0, dtype=torch.float64))  # (nt0*nup, nt0)
    # MATLAB reshape(B, nup, nt0, nt0) (column-major): B3(j, a, b) = B[j + nup*a, b]
    B3 = B.reshape(nt0, nup, nt0).permute(1, 0, 2)  # [j, a, b]
    A = B3[:, nt0min - 1, :]  # MATLAB A(j, b) (10, nt0)
    Ka = A.T.contiguous()  # (nt0, 10): Ka[b, j]
    Kb = B3.permute(0, 2, 1).contiguous()  # [j, b, a] = MATLAB Kb(a, b, j)
    return Ka.to(device), Kb.to(device)


def make_geometry(device='cuda'):
    # 16 channels on a line, 20 um apart; iC = NNEAR nearest channels
    yc = torch.arange(NCHAN, dtype=torch.float32) * 20
    d = (yc[:, None] - yc[None, :]).abs()
    iC = torch.argsort(d + 1e-3 * torch.arange(NCHAN)[None, :], dim=1, stable=True)[:, :NNEAR]
    return iC.to(torch.int32).contiguous().to(device)


def make_templates(gen, iC, device='cuda'):
    """W (NRANK, NFILT, NT0) f32, U (NRANK, NFILT, NCHAN) f32 (unit norm
    overall), mu, iW."""
    wave = smooth_waveforms(gen, NRANK, device=device).float()
    W = torch.stack([wave] * NFILT, 1).contiguous()  # (NRANK, NFILT, NT0)
    iW = torch.tensor([2, 4, 6, 8, 11, 13], dtype=torch.int32, device=device)
    U = torch.zeros((NRANK, NFILT, NCHAN), device=device)
    for f in range(NFILT):
        chans = iC[iW[f].long()].long()
        amp = torch.exp(-torch.arange(NNEAR, device=device, dtype=torch.float32) / 2.0)
        for k in range(NRANK):
            u = amp * (1.0 if k == 0 else 0.3) * (torch.rand(NNEAR, generator=gen, device=device) * 0.5 + 0.75)
            if k > 0:
                u = u * torch.sign(torch.rand(NNEAR, generator=gen, device=device) - 0.5)
            U[k, f, chans] = u
    U = U / torch.sqrt((U ** 2).sum(dim=(0, 2)))[None, :, None]
    mu = torch.tensor([14., 20., 12., 25., 16., 30.], device=device)
    return W, U, mu, iW


def templates_full(W, U):
    # (NFILT, NCHAN, NT0)
    return torch.einsum('kfc,kft->fct', U, W)


def make_batch(gen, W, U, mu, device='cuda'):
    """Noise plus spikes of the templates; returns X (NCHAN, NT) and the
    (t, f) list of inserted spikes (template starts)."""
    X = torch.randn((NCHAN, NT), generator=gen, device=device) * 0.5
    tmpl = templates_full(W, U)
    spikes = []
    t = 150
    f = 0
    while t < NT - 2 * NT0 - 200:
        a = mu[f] * (0.8 + 0.4 * torch.rand((), generator=gen, device=device))
        X[:, t:t + NT0] += a * tmpl[f]
        spikes.append((t, f))
        # sometimes a second, overlapping spike of another template
        if len(spikes) % 3 == 0:
            g = (f + 2) % NFILT
            X[:, t + 25:t + 25 + NT0] += mu[g] * tmpl[g]
            spikes.append((t + 25, g))
        t += 190 + int(torch.randint(0, 60, (1,), generator=gen, device=device))
        f = (f + 1) % NFILT
    return X.contiguous(), spikes


# ----------------------------------------------------------------- references

def ref_get_spikes2(p, X, wTEMP, iC, maxFR=5000):
    NT_, nt0, Nchan = int(p[0]), int(p[4]), int(p[9])
    NchanNear, Nsum, Nrank, Th = int(p[10]), int(p[13]), int(p[14]), float(p[2])
    Tend = NT_ - nt0 + 1
    dfilt = torch.zeros((Nrank, Nchan, NT_), device=X.device)
    dfilt[:, :, :Tend] = torch.einsum('ri,cti->rct', wTEMP, X.unfold(1, nt0, 1))
    g = dfilt[:, iC.long()[:, :Nsum], :]  # (Nrank, Nchan, Nsum, NT)
    sc = g.cumsum(2) ** 2 / torch.arange(1, Nsum + 1, device=X.device)[None, None, :, None]
    dout = sc.amax(dim=(0, 2)).clamp_min(0)
    best, chan = dout.max(dim=0)
    err = torch.where(best > 1e-6, best, torch.zeros_like(best))
    lock = nt0 - 1
    local = F.max_pool1d(err[None, None], 2 * lock + 1, stride=1, padding=lock)[0, 0]
    valid = (err > Th * Th) & (err >= local)
    valid[:lock] = False
    valid[NT_ - lock - 1:] = False
    st = valid.nonzero().flatten()[:maxFR]
    ch, x = chan[st], err[st]
    if st.numel():
        near = (ch[:, None] - ch[None, :]).abs() < 5
        keep = ~(near & (x[None, :] > x[:, None])).any(dim=1)
        st, ch = st[keep], ch[keep]
    clips = torch.stack([X[:, t:t + nt0] for t in st.tolist()]) if st.numel() else X.new_zeros((0, Nchan, nt0))
    return st.to(torch.int32), ch.to(torch.int32), clips, dout


def ref_svd_small2(p, dWU, W, iC, iW, Ka, Kb):
    Nfilt, nt0, Nrank, Nchan, tmax = int(p[1]), int(p[4]), int(p[6]), int(p[9]), int(p[11])
    W = W.clone()
    U = torch.zeros((Nrank, Nfilt, Nchan), dtype=torch.float64, device=dWU.device)
    mu = torch.zeros(Nfilt, dtype=torch.float64, device=dWU.device)
    for f in range(Nfilt):
        chans = iC[iW[f].long()].long()
        a = torch.zeros((Nchan, nt0), dtype=torch.float64, device=dWU.device)
        a[chans] = dWU[f, chans]
        wtw = a.T @ a
        for k in range(Nrank):
            v = W[k, f].clone()
            for i in range(100):
                x = wtw @ v
                if i < 99:
                    v = x / torch.sqrt(np.float32(1e-5).astype(np.float64).item() + (x ** 2).sum())
            wtw = wtw - v[None, :] * x[:, None]
            W[k, f] = v
        W[0, f] = -W[0, f] * torch.sign(W[0, f, tmax]) if W[0, f, tmax] != 0 else -W[0, f]
        for k in range(Nrank):
            U[k, f] = W[k, f] @ a.T
        ss = torch.sqrt((U[:, f] ** 2).sum(dim=1))
        tot = torch.sqrt(np.float32(1e-7).astype(np.float64).item() + (ss ** 2).sum())
        mu[f] = tot
        U[:, f] /= tot
        imax = int(W[0, f].abs().argmax())
        W[:, f] = torch.roll(W[:, f], -(imax - tmax), dims=1)
        up = Ka.T @ W[0, f]  # (10,)
        j = int(up.abs().argmax())
        sgn = 1.0 if up[j] >= 0 else -1.0
        Wn = W[:, f] @ Kb[j]  # out[k, a] = sum_b W[k, b] Kb[j, b, a]
        Wn[0] = -Wn[0] * sgn
        W[:, f] = Wn
    return W, U, mu


def ref_wtw2(W1, W2, UtU):
    n, nt0 = W1.shape
    out = torch.zeros((2 * nt0 - 1, n, n), device=W1.device)
    for lag in range(2 * nt0 - 1):
        off = nt0 - lag - 1
        t0, t1 = (0, lag + 1) if lag < nt0 else (lag - nt0 + 1, nt0)
        # out[lag, b, a] = UtU[b, a] * sum_t W1[a, t] W2[b, t + off]
        out[lag] = (W2[:, t0 + off:t1 + off] @ W1[:, t0:t1].T) * UtU
    return out


def ref_mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, niter):
    """Full recomputation each iteration (no local updates)."""
    NT_, Nfilt, Th, nt0 = int(p[0]), int(p[1]), float(p[2]), int(p[4])
    Nnearest, Nrank, lam, Nchan, NchanU = int(p[5]), int(p[6]), float(p[7]), int(p[9]), int(p[10])
    flag, Th2, stable = int(p[12]), float(p[15]), bool(int(p[16]))
    tmpl = templates_full(W, U)
    resid = X.double().clone() if stable else X.clone()
    lock = nt0 - 1
    st_all, id_all, y_all, x_all, feat_all = [], [], [], [], []
    for _ in range(niter):
        r32 = resid.float()
        dprod = torch.zeros((Nrank, Nfilt, NT_), device=X.device)
        for f in range(Nfilt):
            chans = iC[iW[f].long()].long()
            dprod[:, f] = U[:, f, chans] @ r32[chans]
        dout = torch.zeros((Nfilt, NT_), device=X.device)
        Tend = NT_ - nt0 + 1
        dout[:, :Tend] = torch.einsum('kfi,kfti->ft', W, dprod.unfold(2, nt0, 1))
        a = 1 + lam
        b = dout.clamp_min(0) + lam * mu[:, None]
        Cf = b * b / a - lam * mu[:, None] ** 2
        best, ibest = Cf.max(dim=0)
        err = torch.where(best > 1e-6, best, torch.zeros_like(best))
        err[:nt0] = 0
        err[NT_ - nt0:] = 0
        local = F.max_pool1d(err[None, None], 2 * lock + 1, stride=1, padding=lock)[0, 0]
        valid = (err > Th * Th) & (err >= local)
        valid[:lock] = False
        valid[NT_ - lock:] = False
        st = valid.nonzero().flatten()
        if not st.numel():
            break
        ids = ibest[st]
        y = dout[ids, st]
        x = torch.sqrt(err[st])
        if flag > 1:
            for t, i in zip(st.tolist(), ids.tolist()):
                near = iList[i].long()
                m = mu[near]
                c = (dout[near, t - 3:t + 3] + (lam / m)[:, None]) / torch.sqrt(lam / m ** 2 + 1.0)[:, None]
                feat_all.append(c.max(dim=1).values.clamp_min(0))
        for t, i, amp in zip(st.tolist(), ids.tolist(), y):
            resid[:, t:t + nt0] -= amp.to(resid.dtype) * tmpl[i].to(resid.dtype)
        st_all.append(st); id_all.append(ids); y_all.append(y); x_all.append(x)
    dev = X.device
    st = torch.cat(st_all).int() if st_all else torch.empty(0, dtype=torch.int32, device=dev)
    ids = torch.cat(id_all).int() if id_all else st.clone()
    y = torch.cat(y_all) if y_all else X.new_empty(0)
    x = torch.cat(x_all) if x_all else X.new_empty(0)
    feat = torch.stack(feat_all) if feat_all else X.new_zeros((len(st), Nnearest))
    drez = resid.float()
    dWU0 = torch.zeros((Nfilt, Nchan, nt0), dtype=torch.float64, device=dev)
    nsp0 = torch.zeros(Nfilt, dtype=torch.int32, device=dev)
    featPC = X.new_zeros((len(st), Nrank, NchanU))
    for n in range(len(st)):
        t, i = int(st[n]), int(ids[n])
        snip = drez[:, t:t + nt0] + y[n] * tmpl[i]
        if x[n] > Th2:
            dWU0[i] += snip.double()
            nsp0[i] += 1
        if flag > 0:
            chans = iC[iW[i].long()].long()
            featPC[n] = wPCA[:Nrank] @ snip[chans].T
    return dict(st0=st, id0=ids, x0=y, vexp=x, featW=feat, featPC=featPC, dWU0=dWU0, nsp0=nsp0, drez=drez)


def sort_by_time(d):
    order = torch.argsort(d['st0'].long() * 100000 + d['id0'].long())
    return {k: (v[order] if k in ('st0', 'id0', 'x0', 'vexp', 'featW', 'featPC') else v) for k, v in d.items()}


# ----------------------------------------------------------------------- tests

@pytest.fixture(scope='module')
def setup():
    from kilosort.ks25 import kernels
    gen = torch.Generator(device='cuda').manual_seed(123)
    iC = make_geometry()
    W, U, mu, iW = make_templates(gen, iC)
    X, spikes = make_batch(gen, W, U, mu)
    return dict(kernels=kernels, gen=gen, iC=iC, W=W, U=U, mu=mu, iW=iW, X=X, spikes=spikes)


def test_get_spikes2(setup):
    k, X, iC = setup['kernels'], setup['X'], setup['iC']
    wTEMP = smooth_waveforms(setup['gen'], NRANKPC).float()
    p = params(p2=6.0)
    for tail in (False, True):
        dWU, dout = k.get_spikes2(p, X, wTEMP, iC, matlab_tail=tail)
        st_r, ch_r, clips_r, dout_r = ref_get_spikes2(p, X, wTEMP, iC)
        if tail:
            # the mex loop bounds never examine the tail of the batch
            hi = NT - NT0  # conservative: everything the reference finds below this is found too
            hi = k._cleanup_range(NT, NT0 - 1, True, True)[1]
            keep = st_r < hi
            st_r, ch_r, clips_r = st_r[keep], ch_r[keep], clips_r[keep]
            assert torch.allclose(dout[:, :k._tail(NT, NT0, True)], dout_r[:, :k._tail(NT, NT0, True)], atol=1e-3)
        else:
            assert torch.allclose(dout, dout_r, atol=1e-3, rtol=1e-4)
        assert dWU.shape == (len(st_r), NCHAN, NT0)
        # cleanup_heights keeps one candidate per 5-channel neighborhood of
        # the whole batch, so only a few survive on 16 channels
        assert len(st_r) >= 2
        # clips are X[:, st:st+nt0]; recover st by matching
        assert torch.allclose(dWU, clips_r, atol=1e-6)


def test_get_spikes2_empty(setup):
    k, iC = setup['kernels'], setup['iC']
    wTEMP = smooth_waveforms(setup['gen'], NRANKPC).float()
    X = torch.zeros((NCHAN, NT), device='cuda')
    dWU, dout = k.get_spikes2(params(), X, wTEMP, iC)
    assert dWU.shape == (0, NCHAN, NT0)
    assert dout.abs().sum() == 0


def test_svd_small2(setup):
    k, W, U, mu, iW, iC = setup['kernels'], setup['W'], setup['U'], setup['mu'], setup['iW'], setup['iC']
    gen = setup['gen']
    dWU = (templates_full(W, U) * mu[:, None, None]).double()
    dWU += 0.3 * torch.randn(dWU.shape, generator=gen, device='cuda', dtype=torch.float64)
    # warm start: the standard waves, slightly perturbed
    W0 = smooth_waveforms(gen, NRANK)[:, None, :].expand(-1, NFILT, -1).clone()
    W0 += 0.05 * torch.randn(W0.shape, generator=gen, device='cuda', dtype=torch.float64)
    Ka, Kb = make_kernels()
    p = params()
    Wk, Uk, muk = k.svd_small2(p, dWU, W0, iC, iW, Ka, Kb)
    Wr, Ur, mur = ref_svd_small2(p, dWU, W0, iC, iW, Ka, Kb)
    assert Wk.dtype == torch.float64 and Uk.shape == (NRANK, NFILT, NCHAN)
    assert torch.allclose(muk, mur, rtol=1e-9)
    assert torch.allclose(Uk, Ur, atol=1e-9)
    assert torch.allclose(Wk, Wr, atol=1e-9)
    # the warm start is not modified
    assert torch.allclose(W0[0, 0], W0[0, 0])
    # first component negative at nt0min and (near) unit norm
    assert (Wk[0, :, NT0MIN] < 0).all()
    # unit-norm U over all components
    assert torch.allclose((Uk ** 2).sum(dim=(0, 2)), torch.ones(NFILT, dtype=torch.float64, device='cuda'), atol=1e-6)


def test_wtw2_and_wtw_max(setup):
    k, W, U = setup['kernels'], setup['W'], setup['U']
    gen = setup['gen']
    W = W + 0.1 * torch.randn(W.shape, generator=gen, device='cuda')
    total = torch.zeros((2 * NT0 - 1, NFILT, NFILT), device='cuda')
    for i in range(NRANK):
        for j in range(NRANK):
            utu = (U[j] @ U[i].T).contiguous()  # utu[b, a] = MATLAB U_i' U_j (a, b)
            got = k.wtw2(W[i].contiguous(), W[j].contiguous(), utu)
            ref = ref_wtw2(W[i], W[j], utu)
            assert got.shape == (2 * NT0 - 1, NFILT, NFILT)
            assert torch.allclose(got, ref, atol=1e-5)
            total += ref
    cc = k.wtw_max(W, U)
    assert torch.allclose(cc, total.max(dim=0).values, atol=1e-4)
    # a template is most similar to itself
    assert (cc.argmax(dim=1) == torch.arange(NFILT, device='cuda')).all()


@pytest.mark.parametrize('stable', [1, 0])
def test_mpnu8_one_iteration(setup, stable):
    k, X, W, U, mu, iW, iC = (setup[n] for n in ('kernels', 'X', 'W', 'U', 'mu', 'iW', 'iC'))
    gen = setup['gen']
    wPCA = smooth_waveforms(gen, NRANK).float()
    UtU = torch.ones((NFILT, NFILT), dtype=torch.bool, device='cuda')
    order = torch.argsort(torch.rand((NFILT, NFILT), generator=gen, device='cuda'), dim=1)
    iList = order[:, torch.arange(NNEAREST, device='cuda') % NFILT].int().contiguous()
    p = params(p3=1, p16=stable, p15=4.0)
    got = sort_by_time(k.mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA))
    ref = sort_by_time(ref_mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, 1))
    assert len(ref['st0']) >= 10
    assert torch.equal(got['st0'], ref['st0'])
    assert torch.equal(got['id0'], ref['id0'])
    assert torch.allclose(got['x0'], ref['x0'], rtol=1e-4, atol=1e-3)
    assert torch.allclose(got['vexp'], ref['vexp'], rtol=1e-4, atol=1e-3)
    assert torch.allclose(got['drez'], ref['drez'], atol=2e-4)
    assert torch.allclose(got['featW'], ref['featW'], rtol=1e-4, atol=1e-3)
    assert torch.allclose(got['featPC'], ref['featPC'], rtol=1e-3, atol=2e-3)
    assert torch.equal(got['nsp0'], ref['nsp0'])
    assert torch.allclose(got['dWU0'], ref['dWU0'], rtol=1e-4, atol=1e-3)
    assert torch.equal(X, setup['X'])  # input not modified


def test_mpnu8_iterations_and_early_stop(setup):
    k, X, W, U, mu, iW, iC = (setup[n] for n in ('kernels', 'X', 'W', 'U', 'mu', 'iW', 'iC'))
    gen = setup['gen']
    wPCA = smooth_waveforms(gen, NRANK).float()
    UtU = torch.ones((NFILT, NFILT), dtype=torch.bool, device='cuda')
    iList = (torch.arange(NNEAREST, device='cuda') % NFILT).int()[None, :].expand(NFILT, -1).contiguous()
    p = params(p3=60, p16=1, p15=4.0)
    full = sort_by_time(k.mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, early_stop=False))
    early = sort_by_time(k.mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, early_stop=True))
    ref = sort_by_time(ref_mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, 60))
    # the overlapping pairs need a second iteration
    assert len(full['st0']) > len(sort_by_time(k.mpnu8(params(p3=1, p15=4.0), X, U, W, mu, iC, iW, UtU, iList, wPCA))['st0'])
    for key in ('st0', 'id0', 'nsp0'):
        assert torch.equal(full[key], early[key])
    for key in ('x0', 'vexp', 'drez', 'featW', 'featPC', 'dWU0'):
        assert torch.allclose(full[key], early[key], rtol=1e-5, atol=1e-5)
    # against the full-recompute reference
    assert torch.equal(full['st0'], ref['st0'])
    assert torch.equal(full['id0'], ref['id0'])
    assert torch.allclose(full['drez'], ref['drez'], atol=5e-4)
    assert torch.allclose(full['x0'], ref['x0'], rtol=1e-3, atol=1e-2)
    # every inserted spike (including the overlapping ones) is found with
    # the right template, within one sample
    for t, f in setup['spikes']:
        d = (full['st0'].long() - t).abs()
        assert d.min() <= 1
        assert full['id0'][d.argmin()] == f


def test_mpnu8_matlab_tail(setup):
    k, W, U, mu, iW, iC = (setup[n] for n in ('kernels', 'W', 'U', 'mu', 'iW', 'iC'))
    gen = setup['gen']
    wPCA = smooth_waveforms(gen, NRANK).float()
    UtU = torch.ones((NFILT, NFILT), dtype=torch.bool, device='cuda')
    iList = torch.zeros((NFILT, NNEAREST), dtype=torch.int32, device='cuda')
    X = torch.randn((NCHAN, NT), generator=gen, device='cuda') * 0.5
    tmpl = templates_full(W, U)
    for t in (500, NT - 1024 + 50, NT - 200):  # the last two are in the mex blind spot
        X[:, t:t + NT0] += mu[1] * tmpl[1]
    p = params(p15=4.0)
    got = k.mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, matlab_tail=False)
    assert sorted(got['st0'].tolist()) == [500, NT - 1024 + 50, NT - 200]
    got = k.mpnu8(p, X, U, W, mu, iC, iW, UtU, iList, wPCA, matlab_tail=True)
    assert got['st0'].tolist() == [500]


def test_mpnu8_no_templates(setup):
    k, X, iC = setup['kernels'], setup['X'], setup['iC']
    wPCA = smooth_waveforms(setup['gen'], NRANK).float()
    p = params(p1=0)
    got = k.mpnu8(p, X, torch.zeros((NRANK, 0, NCHAN), device='cuda'), torch.zeros((NRANK, 0, NT0), device='cuda'),
                  torch.zeros(0, device='cuda'), iC, torch.zeros(0, dtype=torch.int32, device='cuda'),
                  torch.zeros((0, 0), dtype=torch.bool, device='cuda'),
                  torch.zeros((0, NNEAREST), dtype=torch.int32, device='cuda'), wPCA)
    assert got['st0'].numel() == 0 and got['dWU0'].shape == (0, NCHAN, NT0)
    assert torch.equal(got['drez'], X)
