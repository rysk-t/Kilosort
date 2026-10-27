import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


COLOR_CODES = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# This is matplotlib tab10 with gray moved to the last position and all 0.5 alpha
PROBE_PLOT_COLORS = np.array([
        [0.12156863, 0.46666667, 0.70588235, 0.5],
        [1.        , 0.49803922, 0.05490196, 0.5],
        [0.17254902, 0.62745098, 0.17254902, 0.5],
        [0.83921569, 0.15294118, 0.15686275, 0.5],
        [0.58039216, 0.40392157, 0.74117647, 0.5],
        [0.54901961, 0.3372549 , 0.29411765, 0.5],
        [0.89019608, 0.46666667, 0.76078431, 0.5],
        [0.7372549 , 0.74117647, 0.13333333, 0.5],
        [0.09019608, 0.74509804, 0.81176471, 0.5],
        [0.49803922, 0.49803922, 0.49803922, 0.25]
    ])


def plot_drift_amount(ops, results_dir, tmin=0):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    dshift = ops['dshift']
    settings = ops['settings']

    fs = settings['fs']
    NT = settings['batch_size']
    t = np.arange(dshift.shape[0])*(NT/fs) + tmin
    for i in range(dshift.shape[1]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        ax.plot(t, dshift[:,i], c=color)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth shift (um)')
    fig.suptitle('Drift amount per probe section, across batches')
    fig.tight_layout()

    save_path = results_dir / 'drift_amount.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def plot_drift_scatter(st0, results_dir, tmin=0):
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    x = st0[:,0] + tmin  # spike time in seconds
    y = st0[:,1]         # depth of spike center in microns
    z = st0[:,2]         # spike amplitude (data)
    z[z < 10] = 10
    z[z > 100] = 100
    colors = np.empty((x.shape[0], 4), dtype=float)

    bin_idx = np.digitize(z, np.logspace(1, 2, 90))
    cm = matplotlib.colormaps['binary']
    for i in np.unique(bin_idx):
        # Take mean of all amplitude values within one bin, map to color
        subset = (bin_idx == i)
        a = z[subset].mean()
        colors[subset] = cm(((a-10)/90))
    
    # Scatter of spike depth over time, with color intensity proportional
    # to log of amplitude.
    ax.scatter(x, y, s=3, c=colors)
    ax.set_xlabel('Time (s)', fontsize=22)
    ax.set_ylabel('Depth (um)', fontsize=22)
    fig.suptitle('Spike amplitude across time and depth', fontsize=30)
    fig.tight_layout()

    save_path = results_dir / 'drift_scatter.png'
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _diagnostics_data(Wall0, ops):
    """The arrays drawn by `plot_diagnostics`, computed on Wall0's device."""
    wPCA = ops['wPCA'].cpu().numpy()
    features = torch.linalg.norm(Wall0, dim=2).cpu().numpy()
    mean_amp = torch.linalg.norm(Wall0, dim=(1,2)).cpu().numpy()
    return wPCA, features, mean_amp


def plot_diagnostics(Wall0, clu0, ops, results_dir):
    wPCA, features, mean_amp = _diagnostics_data(Wall0, ops)
    _draw_diagnostics(wPCA, features, mean_amp, clu0, ops['settings']['fs'],
                      results_dir)


def _draw_diagnostics(wPCA, features, mean_amp, clu0, fs, results_dir):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16,16))

    # Top left
    t = np.arange(wPCA.shape[1])/(fs/1000)
    for i in range(wPCA.shape[0]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        axes[0][0].plot(t, wPCA[i,:], c=color)
    axes[0][0].set_xlabel('Time (s)')
    axes[0][0].set_title('Temporal Features')

    # Top right
    axes[0][1].imshow(features.T, aspect='auto', vmin=0, vmax=25, cmap='binary_r')
    axes[0][1].set_xlabel('Channel Number')
    axes[0][1].set_ylabel('Unit Number')
    axes[0][1].set_title('Spatial Features')

    # Comput spike counts and mean amplitudes
    n_units = int(clu0.max()) + 1
    spike_counts = np.zeros(n_units)
    for i in range(n_units):
        spike_counts[i] = (clu0[clu0 == i]).size

    # Bottom left
    axes[1][0].plot(mean_amp)
    axes[1][0].set_xlabel('Unit Number')
    axes[1][0].set_ylabel('Amplitude (a.u.)')
    axes[1][0].set_title('Unit Amplitudes')

    # Bottom right
    axes[1][1].scatter(np.log(1 + spike_counts), mean_amp, s=3)
    axes[1][1].set_xlabel('Log(1 + Spike Count)')
    axes[1][1].set_ylabel('Amplitude (a.u.)')
    axes[1][1].set_title('Amplitude vs Spike Count')

    fig.tight_layout()
    save_path = results_dir / 'diagnostics.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def plot_spike_positions(clu, is_refractory, results_dir, positions=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    # 10 colors in palette, last one is gray for non-frefractory
    clu = clu.copy()
    bad_units = np.unique(clu)[is_refractory == 0]
    bad_idx = np.isin(clu, bad_units)
    clu = np.mod(clu, 9)
    clu[bad_idx] = 9
    colors = np.empty((clu.shape[0], 4), dtype=float)

    # Map modded cluster ids to colors
    for i in range(10):
        subset = (clu == i)
        rgba = PROBE_PLOT_COLORS[i]
        colors[subset] = rgba

    # Get x, y positions, add to scatterplot
    if positions is None:
        positions = np.load(results_dir / 'spike_positions.npy')
    xs, ys = positions[:,0], positions[:,1]
    ax.scatter(ys, xs, s=3, c=colors)
    ax.set_xlabel('Depth (um)', fontsize=22)
    ax.set_ylabel('Lateral (um)', fontsize=22)
    fig.suptitle('Spike position across probe, colored by cluster', fontsize=30)
    fig.tight_layout()

    save_path = results_dir / 'spike_positions.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


class _BackgroundPlots:
    """Call a function of this module in a separate Python process.

    The plots for long recordings take tens of seconds, during which the GPU
    would sit idle. They are drawn by a fresh interpreter (``python -c ...``,
    not multiprocessing) so that user scripts without an
    ``if __name__ == '__main__'`` guard are never re-imported. Call `wait()`
    before reading the png files.
    """

    def __init__(self, func_name, args, what):
        import pickle, subprocess, sys, tempfile
        from pathlib import Path
        self._what = what
        fd, self._args_path = tempfile.mkstemp(suffix='.pkl', prefix='ks_plot_')
        with open(fd, 'wb') as f:
            pickle.dump(args, f)
        pkg_parent = str(Path(__file__).resolve().parents[1])
        code = (f'import sys; sys.path.insert(0, {pkg_parent!r}); '
                f'from kilosort.plots import {func_name}; '
                f'{func_name}({self._args_path!r})')
        env = dict(os.environ, MPLBACKEND='Agg')
        self._proc = subprocess.Popen(
            [sys.executable, '-c', code], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )

    def wait(self):
        """Wait for the plots; raise if the plotting process failed."""
        if self._proc is None:
            return
        _, err = self._proc.communicate()
        code, self._proc = self._proc.returncode, None
        try:
            os.remove(self._args_path)
        except OSError:
            pass
        if code != 0:
            raise RuntimeError(
                f'{self._what} failed in background process:\n'
                + err.decode(errors='replace')
                )


class BackgroundDriftPlots(_BackgroundPlots):
    """`plot_drift_amount` and `plot_drift_scatter` in a separate process."""

    def __init__(self, ops, st0, results_dir, tmin=0):
        from pathlib import Path
        dshift = ops['dshift']
        if torch.is_tensor(dshift):
            dshift = dshift.cpu().numpy()
        mini_ops = {
            'dshift': np.asarray(dshift),
            'settings': {'fs': ops['settings']['fs'],
                         'batch_size': ops['settings']['batch_size']},
            }
        super().__init__('_drift_plots_from_file',
                         (mini_ops, st0, Path(results_dir), tmin), 'Drift plots')


class BackgroundDiagnostics(_BackgroundPlots):
    """`plot_diagnostics` in a separate process (norms computed here)."""

    def __init__(self, Wall0, clu0, ops, results_dir):
        from pathlib import Path
        wPCA, features, mean_amp = _diagnostics_data(Wall0, ops)
        super().__init__('_diagnostics_from_file',
                         (wPCA, features, mean_amp, np.asarray(clu0),
                          ops['settings']['fs'], Path(results_dir)),
                         'Diagnostic plots')


class BackgroundSpikePositions(_BackgroundPlots):
    """`plot_spike_positions` in a separate process."""

    def __init__(self, clu, is_refractory, positions, results_dir):
        from pathlib import Path
        super().__init__('_spike_positions_from_file',
                         (clu, is_refractory, positions, Path(results_dir)),
                         'Spike position plot')


def _drift_plots_from_file(path):
    import pickle
    with open(path, 'rb') as f:
        mini_ops, st0, results_dir, tmin = pickle.load(f)
    plot_drift_amount(mini_ops, results_dir, tmin=tmin)
    plot_drift_scatter(st0, results_dir, tmin=tmin)


def _diagnostics_from_file(path):
    import pickle
    with open(path, 'rb') as f:
        args = pickle.load(f)
    _draw_diagnostics(*args)


def _spike_positions_from_file(path):
    import pickle
    with open(path, 'rb') as f:
        clu, is_refractory, positions, results_dir = pickle.load(f)
    plot_spike_positions(clu, is_refractory, results_dir, positions=positions)
