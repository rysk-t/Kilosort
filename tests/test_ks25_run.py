"""End-to-end KS2.5 regression on the full ZFM mini recording."""
import csv

import numpy as np
import pytest

from kilosort.ks25.run import run_kilosort25


@pytest.mark.gpu
def test_run_kilosort25_zfm_phy_outputs(gpu, request, tmp_path):
    if not gpu:
        pytest.skip('requires --gpu and the shared GPU lock')
    data_directory = request.getfixturevalue('data_directory')
    device = request.getfixturevalue('torch_device')
    binary = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    output = tmp_path / 'ks25_phy'
    ops, rez = run_kilosort25(
        settings={'n_chan_bin': 385}, filename=binary,
        probe_name='NeuroPix1_default.mat', results_dir=output,
        device=device, drift_plots=False,
    )
    required = {
        'spike_times.npy', 'spike_templates.npy', 'spike_clusters.npy',
        'amplitudes.npy', 'templates.npy', 'templates_ind.npy',
        'channel_map.npy', 'channel_positions.npy', 'template_features.npy',
        'template_feature_ind.npy', 'pc_features.npy', 'pc_feature_ind.npy',
        'whitening_mat.npy', 'whitening_mat_inv.npy', 'similar_templates.npy',
        'cluster_KSLabel.tsv', 'cluster_ContamPct.tsv', 'cluster_Amplitude.tsv',
        'cluster_group.tsv', 'params.py', 'ops.npy',
    }
    assert all((output / name).is_file() for name in required)
    times = np.load(output / 'spike_times.npy').reshape(-1)
    clusters = np.load(output / 'spike_clusters.npy').reshape(-1)
    templates = np.load(output / 'templates.npy', mmap_mode='r')
    n_samples = binary.stat().st_size // (385 * np.dtype('int16').itemsize)
    assert len(times) and len(times) == len(clusters)
    assert np.all(times[1:] >= times[:-1])
    assert np.all((times >= 0) & (times < n_samples))
    assert templates.shape[0] > 0
    assert np.all((clusters >= 0) & (clusters < templates.shape[0]))
    with (output / 'cluster_KSLabel.tsv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.reader(stream, delimiter='\t'))
    assert rows[0] == ['cluster_id', 'KSLabel']
    assert len(rows) - 1 == templates.shape[0]
    assert len(rez['st3']) == len(times)
    assert ops['n_chan_bin'] == 385
    assert np.load(output / 'ops.npy', allow_pickle=True).item()['n_chan_bin'] == 385
