"""Independent numerical/behavioral checks for the KS2.5 postprocessing port."""
import copy
import math
import threading

import numpy as np
import pytest
import torch

from kilosort.ks25 import postprocess as pp


@pytest.fixture(autouse=True, scope='module')
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference_ccg(a, b, nbins=500, tbin=.001):
    """Literal 1-based MATLAB windows, brute-force histogram (no sliding window)."""
    a, b = np.asarray(a), np.asarray(b)
    K = np.zeros(2 * nbins + 1)
    for first in a:
        for second in b:
            if second - nbins * tbin < first < second + nbins * tbin:
                lag = (second - first) / tbin
                index = int(np.sign(lag) * np.floor(abs(lag) + .5))
                K[index + nbins] += 1
    far = np.r_[np.arange(2, nbins // 2 + 1), np.arange(3 * nbins // 2 + 1, 2 * nbins + 1)] - 1
    left = np.arange(nbins + 1 - 50, nbins - 10 + 1) - 1
    right = np.arange(nbins + 12, nbins + 50 + 1) - 1
    duration = max(np.max(a), np.max(b)) - min(np.min(a), np.min(b))
    rate = len(a) * len(b) * tbin / duration
    Q00, Q01 = K[far].mean() / rate, max(K[left].mean(), K[right].mean()) / rate
    R00 = max(K[far].mean(), K[left].mean(), K[right].mean())
    q, r = [], []
    for i in range(1, 11):
        total = sum(K[nbins + j] for j in range(-i, i + 1) if j != 0)
        q.append(total / (2 * i * rate))
        lam = R00 * i
        with np.errstate(invalid='ignore', divide='ignore'):
            r.append(.5 * (1 + math.erf(np.divide(total / 2 - lam, np.sqrt(2 * lam)))))
    return K, np.array(q), Q00, Q01, np.array(r)


@pytest.mark.parametrize('auto', [False, True])
def test_ccg_matches_matlab_reference(auto):
    rng = np.random.default_rng(12)
    a, b = rng.uniform(0, 2, 91), rng.uniform(0, 2, 117)
    if auto:
        b = a.copy()
    actual = pp.ccg(a[::-1], b)
    expected = reference_ccg(a, b)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, atol=1e-14, rtol=1e-14)
    if auto:
        assert actual[0][500] >= len(a)


def test_ccg_open_endpoints_and_matlab_rounding():
    a = np.array([-50., -1.5, -.5, .5, 1.5, 50.])
    K, *_ = pp.ccg(a, [0.], nbins=50, tbin=1)
    assert K.sum() == 4
    np.testing.assert_array_equal(np.flatnonzero(K), [48, 49, 51, 52])
    assert K[0] == K[-1] == K[50] == 0


def test_ccg_right_shoulder_starts_at_eleven_ms():
    a, b = [0.], [.01, .011, .05, .049]
    for actual, expected in zip(pp.ccg(a, b), reference_ccg(a, b)):
        np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize('a,b', [([], []), ([], [1]), ([1], []), ([1], [1])])
def test_ccg_empty_or_zero_duration(a, b):
    K, q, q0, q1, r = pp.ccg(a, b)
    assert K.shape == (1001,)
    assert q.shape == r.shape == (10,)
    assert np.isnan(r).all()
    assert not np.isinf(K).any()


@pytest.mark.parametrize('kwargs', [{'nbins': 49}, {'nbins': 51}, {'tbin': 0}, {'tbin': np.nan}])
def test_ccg_rejects_invalid_windows(kwargs):
    with pytest.raises(ValueError):
        pp.ccg([0], [1], **kwargs)


def merge_rez():
    labels = np.array([0, 1, 1, 2, 2, 2, 2])
    st = np.column_stack((np.arange(7) * 100, labels, np.ones((7, 3))))
    sim = np.array([[1, .8, .4], [.8, 1, .7], [.4, .7, 1]])
    features = np.arange(7 * 2 * 2, dtype=np.float32).reshape(7, 2, 2)
    return dict(st3=st, simScore=sim, cProjPC=features, iNeighPC=np.array([[0, 1], [1, 0], [0, 2]]))


@pytest.mark.parametrize('fix', [True, False])
def test_merge_chain_and_optional_pc_remap(monkeypatch, fix):
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (np.arange(1001), .1, .01))
    rez = merge_rez()
    original = rez['cProjPC'].copy()
    pp.find_merges(rez, {'ks25': {'fix_merge_pc': fix}})
    np.testing.assert_array_equal(rez['st3'][:, 1], 2)
    if fix:
        np.testing.assert_array_equal(rez['cProjPC'][0, :, 0], original[0, :, 0])
        np.testing.assert_array_equal(rez['cProjPC'][1:3, :, 0], original[1:3, :, 1])
        np.testing.assert_array_equal(rez['cProjPC'][:3, :, 1], 0)
        np.testing.assert_array_equal(rez['cProjPC'][3:], original[3:])
    else:
        np.testing.assert_array_equal(rez['cProjPC'], original)


def test_merge_diagnostics_does_not_mutate(monkeypatch):
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (np.arange(1001), .1, .01))
    rez = merge_rez()
    original = rez['st3'].copy()
    pp.find_merges(rez, {}, flag=False)
    np.testing.assert_array_equal(rez['st3'], original)
    np.testing.assert_array_equal(rez['Q_CCG'], rez['Q_CCG'].T)
    np.testing.assert_array_equal(rez['K_CCG'][0, 1], rez['K_CCG'][1, 0][::-1])
    assert np.isinf(rez['Q_CCG'][0, 2])


def test_merge_actual_refractory_trains():
    target = np.arange(0, 80, .04)
    source = np.arange(.02, 40, .04)
    st = np.zeros((len(target) + len(source), 5))
    st[:, 0] = np.r_[source, target] * 30000
    st[len(source):, 1] = 1
    rez = {'st3': st, 'simScore': np.array([[1., .8], [.8, 1.]])}
    pp.find_merges(rez, {})
    assert (rez['st3'][:, 1] == 1).all()


def split_rez(n=802):
    rng = np.random.default_rng(2)
    rank, channels, nt = 3, 2, 9
    pc = np.linalg.qr(rng.normal(size=(nt, rank)))[0].T.copy()
    features = rng.normal(0, .12, (n, rank, channels))
    features[:n // 2, 0, 0] += 8
    features[n // 2:, 1, 1] += 5
    mean_feature = features.mean(0) if n else np.zeros((rank, channels))
    waveform = np.einsum('rc,rt->ct', mean_feature, pc)
    st = np.zeros((n, 5))
    st[:, 0] = np.arange(n) * 303
    st[:, 2:4] = 12
    rez = dict(st3=st, cProjPC=features.astype(np.float32), cProj=np.ones((n, 2)),
               W=torch.tensor(pc[:, None]), U=torch.ones((rank, 1, channels), dtype=torch.float64),
               mu=torch.ones(1), dWU=torch.tensor(waveform[None]), wPCA=torch.tensor(pc),
               iNeigh=np.zeros((1, 2), dtype=np.int32), iNeighPC=np.array([[0, 1]]),
               simScore=np.ones((1, 1)), xc=np.array([0., 20.]), yc=np.array([0., 0.]))
    ops = {'fs': 30000, 'nt0min': 4, 'Nchan': channels, 'nt': nt,
           'xc': rez['xc'], 'yc': rez['yc'], 'ks25': {'NchanNear': 2, 'Nnearest': 2}}
    return rez, ops


@pytest.mark.parametrize('flag', [True, False])
@pytest.mark.parametrize('fix', [True, False])
def test_split_em_waveform_assignment_and_revisit(monkeypatch, flag, fix):
    rez, ops = split_rez()
    ops['ks25']['fix_split_swap'] = fix
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (None, 1., 1.))
    captured = {}
    def capture(rez, ops, iC, iW):
        captured['channels'] = iC[iW.long()].numpy()
    monkeypatch.setattr(pp, '_refit_templates', capture)
    pp.split_all_clusters(rez, ops, flag)
    labels = rez['st3'][:, 1].astype(int)
    assert rez['W'].shape[1] == 2
    np.testing.assert_array_equal(np.bincount(labels), [401, 401])
    for cluster in (0, 1):
        selected = cluster if fix else 1 - cluster
        expected = rez['cProjPC'][labels == selected].astype(np.float64).mean(0).T @ rez['wPCA'].numpy()
        np.testing.assert_allclose(rez['dWU'][cluster, captured['channels'][cluster]], expected, atol=2e-6)
    assert rez['simScore'][0, 1] == rez['simScore'][1, 0] == 1


@pytest.mark.parametrize('kind', ['ccg', 'similar_waveforms', 'too_small', 'constant'])
def test_split_vetoes(monkeypatch, kind):
    rez, ops = split_rez(n=600 if kind == 'too_small' else 802)
    if kind == 'constant':
        rez['cProjPC'][:] = 1
    if kind == 'similar_waveforms':
        # Two clear modes with a large common waveform: same shape/amplitude.
        rez['cProjPC'][:, 0, 0] += 200
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (None, .1, .01) if kind == 'ccg' else (None, 1., 1.))
    monkeypatch.setattr(pp, '_refit_templates', lambda *args: None)
    pp.split_all_clusters(rez, ops)
    assert rez['W'].shape[1] == 1
    assert not rez['st3'][:, 1].any()


def test_split_real_cpu_refit():
    rez, ops = split_rez(n=20)
    pp.split_all_clusters(rez, ops)
    assert rez['W'].shape == (3, 1, 9)
    assert rez['U'].shape == (3, 1, 2)
    assert rez['Wphy'].shape == (3, 1, 14)
    assert rez['simScore'].shape == rez['isplit'].shape == (1, 1)
    assert rez['iNeigh'].shape == rez['iNeighPC'].shape == (1, 2)
    assert torch.isfinite(rez['W']).all()
    assert torch.isfinite(rez['mu']).all()


@pytest.mark.parametrize('flag', [True, False])
def test_pursuit_against_numpy_equations(flag):
    rng = np.random.default_rng(83)
    raw = rng.normal(size=(750, 2, 2))
    raw[:375, 0, 0] += 4
    flat = raw.reshape(len(raw), -1)
    clp = flat - flat.mean(axis=0)
    w = np.linalg.svd(clp.T, full_matrices=False)[0][:, 0] if flag else flat.mean(axis=0)
    w /= np.linalg.norm(w)
    x = clp @ w
    upper, lower = x[x > x.mean()], x[x < x.mean()]
    means = np.array([upper.mean(), lower.mean()])
    variances = np.array([upper.var(ddof=1), lower.var(ddof=1)])
    p = len(upper) / len(x)
    for iteration in range(1, 51):
        logp = -.5 * np.log(variances) - (x[:, None] - means)**2 / (2 * variances) + np.log([p, 1-p])
        rs = np.exp(logp - logp.max(axis=1, keepdims=True))
        rs /= rs.sum(axis=1, keepdims=True)
        p = rs[:, 0].mean()
        means = (rs * x[:, None]).sum(axis=0) / rs.sum(axis=0)
        variances = (rs * (x[:, None] - means)**2).sum(axis=0) / rs.sum(axis=0)
        if iteration > 10 and iteration % 2:
            covariance = clp.T @ (clp * (rs / variances).sum(axis=1)[:, None]) / len(x)
            target = clp.T @ (rs @ (means / variances)) / len(x)
            w = np.linalg.solve(covariance, target)
            x = clp @ (w / np.linalg.norm(w))
    actual = pp._bimodal_pursuit(raw, flag, 'cpu')
    if np.linalg.norm(actual - rs) > np.linalg.norm(actual - rs[:, ::-1]):
        rs = rs[:, ::-1]
    np.testing.assert_allclose(actual, rs, atol=2e-12, rtol=2e-12)


def test_empty_templates_pass_through_postprocessing(tmp_path):
    rez, ops = export_fixture(tmp_path, empty=True)
    for key in ('W', 'U'):
        rez[key] = rez[key][:, :0]
    for key in ('mu', 'dWU', 'good', 'est_contam_rate', 'iNeigh', 'iNeighPC'):
        rez[key] = rez[key][:0]
    rez['simScore'] = np.empty((0, 0), dtype=np.float32)
    pp.find_merges(rez, ops)
    pp.split_all_clusters(rez, ops)
    pp.set_cutoff(rez, ops)
    rez['good'] = pp.get_good_units(rez, ops)
    assert rez['good'].shape == (0,)
    pp.save_to_phy25(rez, ops, tmp_path / 'empty')
    assert np.load(tmp_path / 'empty' / 'templates.npy').shape == (0, 14, 2)


def test_cutoff_threshold_strictness_good_and_feature_alignment(monkeypatch):
    # At Th=10, Q=.02 tightens tolerance to .04. Three spikes are then too
    # contaminated (.06 < default .1); revert from Th=8.5 to 9.
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (None, .02 if len(a) <= 2 else .06, .01))
    st = np.zeros((5, 5))
    st[:, 0], st[:, 3] = np.arange(5) * 100, [11, 10, 9, 8, 4]
    features = np.arange(30).reshape(5, 3, 2)
    rez = dict(st3=st, mu=np.ones(2), cProjPC=features, cProj=features[:, :, 0], st2=st.copy())
    pp.set_cutoff(rez, {'ks25': {'Th': (10, 4)}})
    np.testing.assert_array_equal(rez['Ths'], [9, 0])
    np.testing.assert_array_equal(rez['good'], [True, False])
    np.testing.assert_allclose(rez['est_contam_rate'], [.02, 1])
    np.testing.assert_array_equal(rez['cProjPC'], features[:2])
    np.testing.assert_array_equal(rez['st3'], st[:2])
    np.testing.assert_array_equal(rez['st2'], st[:2])


def test_cutoff_rejects_first_threshold_and_keeps_empty_units(monkeypatch):
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (None, .5, .1))
    st = np.array([[0., 0., 1., 10.2, 0.], [1., 1., 1., 2., 0.]])
    rez = {'st3': st, 'mu': np.ones(3), 'cProjPC': np.ones((2, 3, 2))}
    pp.set_cutoff(rez, {})
    assert len(rez['st3']) == len(rez['cProjPC']) == 0
    np.testing.assert_array_equal(rez['Ths'], [10.5, 4., 0.])
    assert not rez['good'].any()
    np.testing.assert_array_equal(rez['est_contam_rate'], 1)


def test_cutoff_threaded_matches_serial_with_real_ccgs(monkeypatch):
    rng = np.random.default_rng(42)
    spikes = []
    for cluster in range(10):
        count = 600 + cluster * 73
        times = (np.arange(count) * .03 if cluster % 2 else rng.uniform(0, 25, count))
        spikes.append(np.column_stack((times * 30000, np.full(count, cluster),
                                       np.ones(count), rng.uniform(3, 14, count), np.zeros(count))))
    st = np.concatenate(spikes)
    rng.shuffle(st)
    features = rng.normal(size=(len(st), 3, 2)).astype(np.float32)
    rez = dict(st3=st, mu=np.ones(11), cProjPC=features, cProj=features[:, :, 0], st2=st.copy())
    monkeypatch.setenv('KS_FASTEST_OFF', 'ks25_cutoff_threads')
    serial = pp.set_cutoff(copy.deepcopy(rez), {})
    monkeypatch.setenv('KS_FASTEST_OFF', '')
    parallel = pp.set_cutoff(copy.deepcopy(rez), {})
    for key in ('st3', 'st2', 'cProj', 'cProjPC', 'good', 'est_contam_rate', 'Ths'):
        np.testing.assert_array_equal(parallel[key], serial[key])


def test_cutoff_clusters_really_execute_concurrently(monkeypatch):
    monkeypatch.setenv('KS_FASTEST_OFF', '')
    rendezvous = threading.Barrier(4)
    state = threading.local()
    def refractory(a, b):
        if not getattr(state, 'started', False):
            state.started = True
            # A serial executor cannot get four clusters to this barrier.
            rendezvous.wait(timeout=10)
        return None, .01, .01
    monkeypatch.setattr(pp, '_refractory', refractory)
    st = np.zeros((80, 5))
    st[:, 0], st[:, 1], st[:, 3] = np.arange(80) * 100, np.repeat(np.arange(4), 20), 12
    rez = pp.set_cutoff({'st3': st, 'mu': np.ones(4)}, {})
    np.testing.assert_array_equal(rez['good'], True)
    np.testing.assert_array_equal(rez['Ths'], 4)


def test_split_threading_preserves_parent_revisit_and_refit(monkeypatch):
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (None, 1., 1.))
    rez, ops = split_rez()
    monkeypatch.setenv('KS_FASTEST_OFF', 'ks25_split_ccg_threads')
    serial = pp.split_all_clusters(copy.deepcopy(rez), ops)
    monkeypatch.setenv('KS_FASTEST_OFF', '')
    parallel = pp.split_all_clusters(copy.deepcopy(rez), ops)
    assert serial['W'].shape[1] == 2
    for key in ('st3', 'W', 'U', 'mu', 'dWU', 'simScore', 'iNeigh', 'iNeighPC', 'isplit'):
        np.testing.assert_array_equal(pp._numpy(parallel[key]), pp._numpy(serial[key]))


def test_split_ccg_overlaps_waveforms_on_calling_thread(monkeypatch):
    monkeypatch.setenv('KS_FASTEST_OFF', '')
    caller = threading.get_ident()
    ccg_started, waveforms_done = threading.Event(), threading.Event()
    original = pp._split_waveforms
    def refractory(a, b):
        assert threading.get_ident() != caller
        ccg_started.set()
        assert waveforms_done.wait(timeout=10)
        return None, 1., 1.
    def waveforms(*args):
        assert threading.get_ident() == caller
        assert ccg_started.wait(timeout=10)
        result = original(*args)
        waveforms_done.set()
        return result
    monkeypatch.setattr(pp, '_refractory', refractory)
    monkeypatch.setattr(pp, '_split_waveforms', waveforms)
    monkeypatch.setattr(pp, '_refit_templates', lambda *args: None)
    rez, ops = split_rez()
    pp.split_all_clusters(rez, ops)
    assert rez['W'].shape[1] == 2


def test_good_units_spatial_radius_and_zero_waveform():
    dwu = np.zeros((4, 3, 5))
    dwu[0, 0] = 1
    dwu[0, 2] = .1  # under one tenth of peak energy: ignored
    dwu[1, [0, 2]] = 1
    dwu[2, 1] = 1
    rez = dict(dWU=dwu, good=np.array([True, True, False, True]),
               xc=np.zeros(3), yc=np.array([0., 100., 300.]))
    np.testing.assert_array_equal(pp.get_good_units(rez, {}), [True, False, False, False])


def export_fixture(tmp_path, empty=False):
    rez, ops = split_rez(n=0 if empty else 3)
    rez.update(good=np.array([True]), est_contam_rate=np.array([np.nan]),
               Wrot=torch.diag(torch.tensor([2., 4.])))
    if not empty:
        rez['st3'][:, 0] = [30, 10, 10]
        rez['st3'][:, 4] = [9, 8, 7]  # batch IDs must not become clusters
    pc6 = torch.arange(54, dtype=torch.float32).reshape(6, 9)
    filename = tmp_path / 'raw data.bin'
    ops.update(filename=filename, data_dir=tmp_path, n_chan_bin=4, chanMap=np.array([3, 1]),
               wPCA=pc6, n_pcs=6, fwav=torch.ones(9), Wrot=rez['Wrot'],
               batch_size=100, Nbatches=1, dshift=np.zeros((1, 1)),
               preprocessing={'Wrot': rez['Wrot'], 'fwav': torch.ones(9)},
               settings={'filename': filename, 'data_dir': tmp_path})
    return rez, ops


@pytest.mark.parametrize('empty', [False, True])
def test_phy_export_sorted_features_raw_mapping_and_six_pc_ops(tmp_path, empty):
    rez, ops = export_fixture(tmp_path, empty)
    original_ops = copy.deepcopy(ops)
    original_st = rez['st3'].copy()
    output = tmp_path / 'phy'
    pp.save_to_phy25(rez, ops, output)
    order = np.argsort(original_st[:, 0], kind='stable')
    for name, expected in [('spike_times', original_st[order, 0]),
                           ('spike_templates', original_st[order, 1]),
                           ('spike_clusters', original_st[order, 1]),
                           ('pc_features', rez['cProjPC'][order]),
                           ('template_features', rez['cProj'][order]),
                           ('channel_map', [3, 1])]:
        np.testing.assert_array_equal(np.load(output / f'{name}.npy'), expected)
    assert np.load(output / 'spike_times.npy').dtype == np.int64
    templates = np.load(output / 'templates.npy')
    assert templates.shape == (1, 14, 2)
    np.testing.assert_array_equal(templates[:, :5], 0)
    np.testing.assert_allclose(np.load(output / 'whitening_mat.npy') @ np.load(output / 'whitening_mat_inv.npy'), np.eye(2))
    saved = np.load(output / 'ops.npy', allow_pickle=True).item()
    assert saved['n_pcs'] == 6
    np.testing.assert_array_equal(saved['wPCA'], original_ops['wPCA'])
    np.testing.assert_array_equal(saved['wPCA25'], rez['wPCA'])
    for key in ('fs', 'n_chan_bin', 'nt', 'batch_size', 'fwav', 'Wrot', 'dshift',
                'chanMap', 'n_pcs', 'Nchan', 'Nbatches', 'wPCA', 'yc'):
        assert key in saved
    params = {}
    exec((output / 'params.py').read_text(), params)
    assert params['n_channels_dat'] == 4
    assert params['hp_filtered'] is False
    assert params['dat_path'].endswith('raw data.bin')
    assert (output / 'cluster_group.tsv').read_text() == 'cluster_id\tgroup\n0\tgood\n'
    assert (output / 'cluster_ContamPct.tsv').read_text().endswith('0\t100.0\n')
    assert 'results_dir' not in ops['settings']
    np.testing.assert_array_equal(rez['st3'], original_st)


@pytest.mark.parametrize('matrix', [np.zeros((2, 2)), np.ones((2, 2))])
def test_phy_export_singular_whitening_uses_finite_pseudoinverse(tmp_path, matrix):
    rez, ops = export_fixture(tmp_path)
    rez['Wrot'] = torch.as_tensor(matrix)
    output = tmp_path / 'singular'
    pp.save_to_phy25(rez, ops, output)
    actual = np.load(output / 'whitening_mat_inv.npy')
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.linalg.pinv(matrix))
    np.testing.assert_allclose(matrix @ actual @ matrix, matrix)
    amplitude = float((output / 'cluster_Amplitude.tsv').read_text().splitlines()[1].split('\t')[1])
    assert np.isfinite(amplitude)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_split_cuda_refit_and_cpu_em_agreement(gpu, monkeypatch):
    if not gpu:
        pytest.skip('use --gpu under .bench/gpulock.py')
    # Exercise the worker CCG + calling-thread CUDA waveform reconstruction
    # on an accepted split, including parent/child revisits and final refit.
    monkeypatch.setattr(pp, '_refractory', lambda a, b: (None, 1., 1.))
    rez, ops = split_rez()
    for key in ('W', 'U', 'mu', 'dWU', 'wPCA'):
        rez[key] = rez[key].cuda()
    pp.split_all_clusters(rez, ops)
    assert rez['W'].shape[1] == 2
    assert rez['W'].is_cuda and rez['U'].is_cuda
    assert torch.isfinite(rez['W']).all() and torch.isfinite(rez['mu']).all()
    features = split_rez()[0]['cProjPC']
    cpu = pp._bimodal_pursuit(features, True, 'cpu')
    cuda = pp._bimodal_pursuit(features, True, 'cuda')
    # Eigensolver signs may reverse the two Gaussian labels.
    error = min(np.max(abs(cpu - cuda)), np.max(abs(cpu - cuda[:, ::-1])))
    assert error < 1e-6
