import numpy as np
import torch

from kilosort.postprocessing import make_pc_features


def restore(pc_features, pc_feature_ind, clusters, n_chan):
    # Place each spike's exported features on the channel ids that
    # pc_feature_ind lists for its cluster, the way Phy reads them.
    pc_features = pc_features.numpy()
    out = np.zeros((len(clusters), n_chan, pc_features.shape[1]), np.float32)
    for s, c in enumerate(clusters):
        out[s, pc_feature_ind[c]] = pc_features[s].T
    return out


def test_pc_features_shared_template_issue_1039():
    # Reproduction from MouseLand/Kilosort#1039: two final clusters whose
    # spikes were detected with the same template. Both are larger on
    # channel 1, so both must list channel 1 first, and every exported value
    # must stay on its own channel.
    ops = {
        'iU': torch.tensor([0]),
        'iCC': torch.tensor([[0], [1]]),
        'xc': np.array([0.0, 0.0]),
        'yc': np.array([0.0, 20.0]),
        'nearest_chans': 2,
        'dmin': 20,
        'dminx': 32,
    }
    templates = np.zeros(4, dtype=np.int32)
    clusters = np.array([0, 1, 0, 1], dtype=np.int32)
    source = np.array([[1.0, 20.0], [1.0, 10.0], [2.0, 18.0], [2.0, 8.0]],
                      dtype=np.float32)

    pc_features, pc_feature_ind = make_pc_features(
        ops, templates, clusters, torch.tensor(source).unsqueeze(-1)
        )

    np.testing.assert_array_equal(pc_feature_ind, [[1, 0], [1, 0]])
    restored = restore(pc_features, pc_feature_ind, clusters, 2)
    np.testing.assert_array_equal(restored[..., 0], source)


def test_pc_features_channels_ranked_per_cluster():
    # One template centered on channel 10 of a linear probe, split into two
    # final clusters that peak on different channels. Each cluster's channel
    # ranking must come from its own spikes only.
    n_chan, n_near, n_pcs, n = 20, 10, 3, 50
    yc = 20.0 * np.arange(n_chan)
    iCC = np.stack([np.argsort(np.abs(yc - y), kind='stable')[:n_near]
                    for y in yc], axis=1)
    ops = {
        'iU': torch.tensor([10]),
        'iCC': torch.from_numpy(iCC),
        'xc': np.zeros(n_chan),
        'yc': yc,
        'nearest_chans': n_near,
        'dmin': 20,
        'dminx': 32,
    }
    templates = np.zeros(2 * n, dtype=np.int32)
    clusters = np.repeat(np.arange(2, dtype=np.int32), n)
    peaks = np.array([7, 13])

    rng = np.random.default_rng(0)
    chans = iCC[:, 10]
    amp = 10.0 / (1 + np.abs(chans[None, :] - peaks[clusters][:, None]))
    tF = amp[..., None] * (1 + 0.1 * rng.standard_normal((2 * n, n_near, n_pcs)))
    tF = tF.astype(np.float32)
    dense = np.zeros((2 * n, n_chan, n_pcs), np.float32)
    dense[:, chans] = tF

    pc_features, pc_feature_ind = make_pc_features(
        ops, templates, clusters, torch.from_numpy(tF.copy())
        )

    assert pc_feature_ind[0, 0] == 7
    assert pc_feature_ind[1, 0] == 13
    np.testing.assert_array_equal(
        restore(pc_features, pc_feature_ind, clusters, n_chan), dense
        )
