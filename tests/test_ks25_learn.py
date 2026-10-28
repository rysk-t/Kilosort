import numpy as np
import torch
import pytest

from kilosort.ks25 import learn


def test_get_kernels_interpolates_and_layout():
    ka, kb = learn.get_kernels({'nt0': 7, 'nt0min': 3}, nup=10)
    assert ka.shape == (7, 10)
    assert kb.shape == (10, 7, 7)
    torch.testing.assert_close(ka.T, kb[:, :, 2])
    torch.testing.assert_close(torch.ones(7, dtype=torch.float64) @ kb[5],
                               torch.full((7,), 1 / (1 + .01 / (1 + 2*np.exp(-1) + 2*np.exp(-4) + 2*np.exp(-9))), dtype=torch.float64))


def test_channel_neighborhood_and_overlap():
    ops = {'xc': [0, 10, 50], 'yc': [0, 0, 0]}
    ic, mask, distances = learn.get_closest_channels(ops, 10, 2)
    assert ic.tolist() == [[0, 1], [1, 0], [2, 1]]
    torch.testing.assert_close(distances[0], torch.tensor([0., 10., 50.]))
    weights = torch.tensor([1., np.exp(-.5)], dtype=torch.float32)
    torch.testing.assert_close(mask[0], weights / (1e-3 + weights.square().sum()).sqrt())
    utu, masku = learn.get_me_utu(torch.tensor([0, 2]), ic, mask, 2, 3)
    assert utu.tolist() == [[True, True], [True, True]]
    torch.testing.assert_close(masku, mask[[0, 2]])


def test_template_correlations_include_cross_rank_and_lags():
    W = torch.tensor([[[1., 0., 0.], [0., 1., 0.]], [[0., 1., 0.], [1., 0., 0.]]])
    U = torch.tensor([[[1., 0.], [1., 1.]], [[1., 1.], [0., 1.]]])
    templates = torch.einsum('rft,rfc->fct', W, U)
    expected = torch.einsum('fct,gct->fg', templates, templates)
    score, inds = learn.get_me_wtw2(W, U, 5)
    torch.testing.assert_close(score, expected)
    assert inds.shape == (2, 5)
    spatial = (U[0] @ U[0].T)
    lagged = learn._wtw_reference(W[0], W[0], spatial)
    assert lagged.shape == (5, 2, 2)
    torch.testing.assert_close(lagged[2], (W[0] @ W[0].T) * spatial)
    maximum, neighbors = learn.get_me_wtw(W, U, 5)
    assert maximum.shape == (2, 2)
    assert torch.all(maximum >= expected - 1e-6)
    assert neighbors.shape == (2, 5)
    assert torch.equal(neighbors[:, :2], neighbors[:, 2:4])


def test_triage_removes_low_rate_and_redundant():
    W = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    U = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    dw = torch.arange(12, dtype=torch.float64).reshape(3, 2, 2)
    mu = torch.tensor([5., 5., 20.])
    nsp = torch.tensor([2., 4., .01])
    got = learn.triage_templates2({'minFR': 1., 'batch_size': 1, 'fs': 1}, None, None,
                                  W, U, dw, mu, nsp, np.zeros(3))
    assert got[3].tolist() == [5.]
    assert got[4].tolist() == [4.]
    assert np.allclose(got[5], [.1, .1, 0])
    torch.testing.assert_close(got[2], dw[1:2])


class FakeFile:
    n_batches = 2
    imin = 100
    def padded_batch_to_torch(self, batch, ops):
        return torch.zeros((2, 14))


@pytest.mark.parametrize("stable", [0, 1])
def test_final_extraction_window_and_absolute_times(monkeypatch, stable):
    ops = {'nt0': 3, 'nt0min': 1, 'nt': 2, 'batch_size': 10, 'xc': [0, 1], 'yc': [0, 0], 'ks25': {'useStableMode': stable}}
    W = torch.tensor([[[1., 0, 0]]], dtype=torch.float64)
    U = torch.tensor([[[1., 0.]]], dtype=torch.float64)
    rez = {'W': W, 'U': U, 'mu': torch.tensor([2.], dtype=torch.float64),
           'wPCA': torch.ones((1, 3)), 'wTEMP': torch.ones((1, 3))}
    def backend(name, p, *args):
        if name == 'svd_small2':
            return W, U, rez['mu']
        assert name == 'mpnu8'
        assert p[16] == stable
        return {'st0': torch.tensor([1, 2, 11, 12]), 'id0': torch.zeros(4, dtype=torch.int32),
                'x0': torch.arange(4, dtype=torch.float32), 'vexp': torch.ones(4),
                'featW': torch.arange(8).reshape(4, 2).float(),
                'featPC': torch.ones((4, 1, 2)), 'dWU0': torch.ones((1, 2, 3), dtype=torch.float64),
                'nsp0': torch.tensor([2])}
    monkeypatch.setattr(learn, '_backend', backend)
    st, fw, fpc, updated = learn.track_and_sort(ops, FakeFile(), rez)
    np.testing.assert_array_equal(st[:, 0], [101, 110, 111, 120])
    np.testing.assert_array_equal(st[:, 4], [0, 0, 1, 1])
    assert fw.shape == (4, 2) and fpc.shape == (4, 1, 2)
    assert updated['nsp'].tolist() == [4]



def test_learning_updates_average_and_uses_nested_settings(monkeypatch):
    ops = {'nt0': 3, 'nt0min': 1, 'nt': 2, 'batch_size': 10, 'fs': 10,
           'xc': [0, 1], 'yc': [0, 0],
           'ks25': {'minFR': 0., 'momentum': [2., 2.], 'Th': [7., 3.], 'nInnerIter': 4},
           'wTEMP': torch.ones((3, 3)), 'wPCA': torch.eye(3)}
    seen = []
    def backend(name, p, *args):
        if name == 'get_spikes2':
            if not seen:
                return torch.tensor([[[2., 0, 0], [0., 0, 0]]]), None
            return torch.empty((0, 2, 3)), None
        if name == 'svd_small2':
            dw, W = args[0], args[1]
            return W, torch.ones((3, len(dw), 2), dtype=torch.float64), torch.ones(len(dw), dtype=torch.float64)
        seen.append(p.copy())
        n = int(p[1])
        return {'nsp0': torch.full((n,), 2, dtype=torch.int32),
                'dWU0': torch.ones((n, 2, 3), dtype=torch.float64) * 6,
                'drez': torch.zeros((2, 14))}
    monkeypatch.setattr(learn, '_backend', backend)
    rez = learn.learn_templates(ops, FakeFile(), [1, 0])
    assert len(seen) == 2
    assert all(p[2] == 7 and p[3] == 4 and p[12] == 0 for p in seen)
    expected = torch.tensor([2., 0., 0.], dtype=torch.float64)
    factor = np.exp(-1)
    expected = expected*factor**2 + 3*(1-factor**2)
    torch.testing.assert_close(rez['dWU'][0, 0], expected)


def test_extract_templates_from_isolated_peaks():
    class Snippets:
        n_batches = 1
        def padded_batch_to_torch(self, batch, ops):
            x = torch.zeros((2, 208))
            for t in [30, 60, 90, 120, 150, 180]:
                x[0, t-1:t+2] = torch.tensor([-2., -10., -3.])
            return x
    ops = {'nt0': 3, 'nt0min': 2, 'nt': 4, 'batch_size': 200, 'nskip': 1,
           'spkTh': -6, 'loc_range': [2, 1], 'long_range': [5, 1]}
    temp, pca = learn.extract_templates_from_snippets(ops, Snippets(), 1)
    torch.testing.assert_close(temp[0], torch.tensor([-2., -10., -3.]) / np.sqrt(113))
    assert pca.shape == (1, 3)
    assert pca[0, 1] < 0


def test_reference_pursuit_detects_and_subtracts_one_template():
    X = torch.zeros((1, 30))
    X[0, 12:15] = torch.tensor([0., -5., 0.])
    W = torch.tensor([[[0., -1., 0.]]])
    U = torch.tensor([[[1.]]])
    p = np.array([30, 1, 2, 5, 3, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 2, 1], dtype=np.float64)
    got = learn._mpnu_reference(p, X, U, W, torch.ones(1), torch.tensor([[0]], dtype=torch.int32),
                                 torch.tensor([0], dtype=torch.int32), torch.ones((1, 1), dtype=torch.bool),
                                 torch.tensor([[0]], dtype=torch.int32), W[:, 0])
    assert got['st0'].tolist() == [12]
    assert got['id0'].tolist() == [0]
    torch.testing.assert_close(got['x0'], torch.tensor([5.]))
    torch.testing.assert_close(got['drez'], torch.zeros_like(X))
    torch.testing.assert_close(got['dWU0'][0, 0], X[0, 12:15].double())
    assert got['nsp0'].tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_helpers_match_cpu():
    W = torch.tensor([[[1., 0., 0.], [0., 1., 0.]], [[0., 1., 0.], [1., 0., 0.]]])
    U = torch.tensor([[[1., 0.], [1., 1.]], [[1., 1.], [0., 1.]]])
    cpu, cpu_i = learn.get_me_wtw(W, U, 3)
    gpu, gpu_i = learn.get_me_wtw(W.cuda(), U.cuda(), 3)
    torch.testing.assert_close(gpu.cpu(), cpu)
    torch.testing.assert_close(gpu_i.cpu(), cpu_i)
    a, b = learn.get_kernels({'nt0': 7, 'nt0min': 3})
    ga, gb = learn.get_kernels({'nt0': 7, 'nt0min': 3, 'device': 'cuda'})
    torch.testing.assert_close(ga.cpu(), a)
    torch.testing.assert_close(gb.cpu(), b)


def test_learning_caps_initial_templates_by_nfilt_factor(monkeypatch):
    ops = {'nt0': 3, 'nt0min': 1, 'nt': 2, 'batch_size': 10, 'fs': 10,
           'xc': [0, 1], 'yc': [0, 0], 'ks25': {'nfilt_factor': 1, 'minFR': 0.},
           'wTEMP': torch.ones((3, 3)), 'wPCA': torch.eye(3)}
    seen_counts = []
    def backend(name, p, *args):
        if name == 'get_spikes2':
            candidates = torch.zeros((5, 2, 3))
            candidates[:, 0, 0] = torch.arange(1, 6)
            return candidates, None
        if name == 'svd_small2':
            dw, W = args[:2]
            seen_counts.append(int(p[1]))
            return W, torch.ones((3, len(dw), 2), dtype=torch.float64), torch.ones(len(dw), dtype=torch.float64)
        n = int(p[1])
        return {'nsp0': torch.ones(n, dtype=torch.int32),
                'dWU0': torch.ones((n, 2, 3), dtype=torch.float64),
                'drez': torch.zeros((2, 14))}
    monkeypatch.setattr(learn, '_backend', backend)
    learn.learn_templates(ops, FakeFile(), [0])
    assert seen_counts == [2]


def test_cuda_mpnu_matches_torch_reference_on_synthetic_spikes():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import importlib.util
    if importlib.util.find_spec('kilosort.ks25.kernels') is None:
        pytest.skip('parallel CUDA kernel implementation is not available yet')
    from kilosort.ks25 import kernels
    if not hasattr(kernels, 'mpnu8'):
        pytest.skip('mpnu8 is not available yet')
    nt, nt0, nfilt, nchan, rank = 4096, 61, 2, 2, 3
    t = torch.arange(nt0, dtype=torch.float32)
    wave = -torch.exp(-((t - 20) / 2).square())
    wave /= wave.norm()
    W = torch.zeros((rank, nfilt, nt0))
    W[0, :, :] = wave
    U = torch.zeros((rank, nfilt, nchan))
    U[0, 0, 0] = U[0, 1, 1] = 1
    mu = torch.tensor([12., 14.])
    X = torch.zeros((nchan, nt))
    for when, unit, amplitude in [(400, 0, 12.), (1000, 1, 14.), (1600, 0, 13.), (2200, 1, 15.)]:
        X[:, when:when+nt0] += amplitude * torch.einsum('rc,rt->ct', U[:, unit], W[:, unit])
    ic = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    iw = torch.tensor([0, 1], dtype=torch.int32)
    utu = torch.ones((nfilt, nfilt), dtype=torch.bool)
    ilist = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    wpca = W[:, 0].contiguous()
    p = np.array([nt, nfilt, 5, 5, nt0, 2, rank, 10, .99, nchan, 2, 20, 0, 2, 3, 10, 1], dtype=np.float64)
    reference = learn._mpnu_reference(p, X, U, W, mu, ic, iw, utu, ilist, wpca)
    actual = kernels.mpnu8(p, *(v.cuda().contiguous() for v in (X, U, W, mu, ic, iw, utu, ilist, wpca)))
    def matches(result):
        return {(int(t), int(i)): float(a) for t, i, a in zip(result['st0'].cpu(), result['id0'].cpu(), result['x0'].cpu())}
    expected, observed = matches(reference), matches(actual)
    common = expected.keys() & observed.keys()
    assert expected and len(common) / len(expected.keys() | observed.keys()) >= .9
    for key in common:
        assert observed[key] == pytest.approx(expected[key], rel=.03, abs=.03)
    np.testing.assert_array_equal(actual['nsp0'].cpu().numpy(), reference['nsp0'].numpy())
    ref_dw = reference['dWU0']
    rel_error = torch.linalg.vector_norm(actual['dWU0'].cpu() - ref_dw) / torch.linalg.vector_norm(ref_dw).clamp_min(1e-12)
    assert rel_error < .1
