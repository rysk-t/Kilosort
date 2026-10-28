import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# torch is imported inside the functions that need it, so that the
# background plotting processes (see _BackgroundPlots) do not load it.


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
    import torch
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


def _spike_positions_figure(clu, is_refractory, positions, corners_only=False):
    # corners_only: draw only the two corners of the positions' bounding box,
    # which gives the same data limits, axes and layout (the strips replace
    # the markers anyway) without setting up millions of markers.
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    # 10 colors in palette, last one is gray for non-frefractory.
    # Map cluster ids to palette indices through a lookup table instead of
    # scanning all spikes once per color.
    bad_units = np.unique(clu)[is_refractory == 0]
    palette_idx = np.mod(np.arange(int(clu.max()) + 1), 9)
    palette_idx[bad_units] = 9
    colors = PROBE_PLOT_COLORS[palette_idx[clu]]

    # Get x, y positions, add to scatterplot
    xs, ys = positions[:,0], positions[:,1]
    if corners_only:
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.any():
            xs = np.array([xs[ok].min(), xs[ok].max()])
            ys = np.array([ys[ok].min(), ys[ok].max()])
            points = ax.scatter(ys, xs, s=3, c=colors[:2])
        else:
            corners_only = False
    if not corners_only:
        points = ax.scatter(ys, xs, s=3, c=colors)
    ax.set_xlabel('Depth (um)', fontsize=22)
    ax.set_ylabel('Lateral (um)', fontsize=22)
    fig.suptitle('Spike position across probe, colored by cluster', fontsize=30)
    fig.tight_layout()

    return fig, ax, points, colors


def plot_spike_positions(clu, is_refractory, results_dir, positions=None):
    if positions is None:
        positions = np.load(results_dir / 'spike_positions.npy')
    fig, _, _, _ = _spike_positions_figure(clu, is_refractory, positions)

    save_path = results_dir / 'spike_positions.png'
    fig.savefig(save_path, dpi=_POSITIONS_DPI)
    plt.style.use('default')
    plt.close(fig)


_POSITIONS_DPI = 300
# Pixels added on each side of a strip when selecting the spikes to draw in
# it; must exceed the marker radius plus antialiasing (about 7 px at dpi 300).
_STRIP_MARGIN = 20


def _spike_positions_strip(clu, is_refractory, positions, k, n_strips, out):
    """Pixel columns of strip `k` of the spike position plot.

    The figure is set up exactly as in `plot_spike_positions` (all spikes,
    so the axis limits and layout are the same), then only the markers that
    reach into this strip are kept, in their original order. Every pixel of
    the strip is covered by the same markers, blended in the same order, as
    in the full plot, so the strips put side by side give the same image.
    """
    import io as _io
    # This file is loaded on its own in the plotting process (no kilosort
    # package, see _BackgroundPlots), so read KS_FASTEST_OFF directly.
    corners = 'plot_corners' not in os.environ.get('KS_FASTEST_OFF', '').split(',')
    fig, ax, points, colors = _spike_positions_figure(
        clu, is_refractory, positions, corners_only=corners
        )
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    W = int(round(fig.get_figwidth() * _POSITIONS_DPI))
    H = int(round(fig.get_figheight() * _POSITIONS_DPI))
    xy = np.column_stack((positions[:,1], positions[:,0]))
    px = (ax.transData + fig.transFigure.inverted()).transform(xy)[:,0] * W

    # Strip edges at quantiles of the marker columns, so that each strip
    # draws about the same number of markers.
    finite = px[np.isfinite(px)]
    inner = np.quantile(finite, np.arange(1, n_strips) / n_strips) if finite.size else []
    edges = np.concatenate(([0], np.clip(np.round(inner), 0, W), [W])).astype(int)
    edges = np.maximum.accumulate(edges)
    c0, c1 = edges[k], edges[k+1]

    keep = (px >= c0 - _STRIP_MARGIN) & (px < c1 + _STRIP_MARGIN)
    if 'plot_bycolor' in os.environ.get('KS_FASTEST_OFF', '').split(','):
        points.set_offsets(xy[keep])
        points.set_facecolor(colors[keep])
    else:
        # Single-color scatters (one per palette entry, in 16 consecutive
        # chunks of spikes so that later spikes still mostly end up on top):
        # matplotlib then draws them with its fast marker path instead of one
        # path per spike (~15 us each). Only the stacking of overlapping
        # markers of different colors within a chunk differs from the plot
        # drawn in spike order.
        bad_units = np.unique(clu)[is_refractory == 0]
        palette_idx = np.mod(np.arange(int(clu.max()) + 1), 9)
        palette_idx[bad_units] = 9
        pal = palette_idx[clu]
        points.remove()
        idx = np.nonzero(keep)[0]
        for chunk in np.array_split(idx, 16):
            for g in range(10):
                sel = chunk[pal[chunk] == g]
                if sel.size:
                    ax.scatter(xy[sel, 0], xy[sel, 1], s=3, c=PROBE_PLOT_COLORS[g:g+1])
    buf = _io.BytesIO()
    fig.savefig(buf, format='raw', dpi=_POSITIONS_DPI)
    plt.style.use('default')
    plt.close(fig)
    image = np.frombuffer(buf.getvalue(), np.uint8).reshape(H, W, 4)
    np.save(out, image[:, c0:c1])


class _BackgroundPlots:
    """Call a function of this module in a separate Python process.

    The plots for long recordings take tens of seconds, during which the GPU
    would sit idle. They are drawn by a fresh interpreter (``python -c ...``,
    not multiprocessing) so that user scripts without an
    ``if __name__ == '__main__'`` guard are never re-imported. The
    interpreter loads this file directly rather than the kilosort package,
    whose import pulls in torch and CUDA (several seconds on Windows) that
    the plots don't need. Call `wait()` before reading the png files.
    """

    def __init__(self, func_name, args, what, args_path=None, call_args=()):
        import pickle, subprocess, sys, tempfile
        self._what = what
        # With `args_path`, the arguments were pickled by the caller, who
        # also removes the file.
        self._owns_args = args_path is None
        if args_path is None:
            fd, args_path = tempfile.mkstemp(suffix='.pkl', prefix='ks_plot_')
            with open(fd, 'wb') as f:
                pickle.dump(args, f)
        self._args_path = args_path
        call = ', '.join(repr(a) for a in (args_path,) + tuple(call_args))
        code = ('import importlib.util as u; '
                f's = u.spec_from_file_location("kilosort_plots", {__file__!r}); '
                'm = u.module_from_spec(s); s.loader.exec_module(m); '
                f'm.{func_name}({call})')
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
        if self._owns_args:
            try:
                os.remove(self._args_path)
            except OSError:
                pass
        if code != 0:
            raise RuntimeError(
                f'{self._what} failed in background process:\n'
                + err.decode(errors='replace')
                )

    def cancel(self):
        """Stop the plotting process (e.g. after an error) and clean up."""
        if self._proc is not None:
            self._proc.kill()
            self._proc.communicate()
            self._proc = None
            if self._owns_args:
                try:
                    os.remove(self._args_path)
                except OSError:
                    pass


class BackgroundDriftPlots(_BackgroundPlots):
    """`plot_drift_amount` and `plot_drift_scatter` in a separate process."""

    def __init__(self, ops, st0, results_dir, tmin=0):
        from pathlib import Path
        import torch
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


class BackgroundSpikePositions:
    """`plot_spike_positions` in separate processes.

    For many spikes, drawing the markers takes most of the time (~28 s for
    2.7M spikes), so the image is drawn as `n_strips` vertical strips in
    parallel (see `_spike_positions_strip`) and put together in `wait()`.
    The png is the same as the one `plot_spike_positions` writes.
    """

    def __init__(self, clu, is_refractory, positions, results_dir,
                 n_strips=None):
        import pickle, tempfile
        from pathlib import Path
        self._results_dir = Path(results_dir)
        if n_strips is None:
            n_strips = 1 if len(clu) < 300_000 else min(
                12 if len(clu) > 3_000_000 else 6, os.cpu_count() or 1)
        self._tmpdir = None
        if n_strips <= 1:
            self._procs = [_BackgroundPlots(
                '_spike_positions_from_file',
                (clu, is_refractory, positions, self._results_dir),
                'Spike position plot'
                )]
            return
        self._tmpdir = tempfile.mkdtemp(prefix='ks_positions_')
        args_path = os.path.join(self._tmpdir, 'args.pkl')
        with open(args_path, 'wb') as f:
            pickle.dump((clu, is_refractory, positions), f)
        self._strips = [os.path.join(self._tmpdir, f'strip_{k}.npy')
                        for k in range(n_strips)]
        self._procs = [
            _BackgroundPlots('_spike_positions_strip_from_file', None,
                             'Spike position plot', args_path=args_path,
                             call_args=(k, n_strips, self._strips[k]))
            for k in range(n_strips)
            ]

    def wait(self):
        """Wait for the plot; raise if a plotting process failed."""
        import shutil
        from matplotlib import image
        try:
            # Wait for all processes before raising, so that none of them
            # still has a file open when the temporary folder is removed.
            errors = []
            for p in self._procs:
                try:
                    p.wait()
                except RuntimeError as e:
                    errors.append(e)
            if errors:
                raise errors[0]
            if self._tmpdir is not None and self._strips:
                strips = [np.load(s) for s in self._strips]
                self._strips = []
                # Same call as savefig's png writer, so the file is the same.
                image.imsave(self._results_dir / 'spike_positions.png',
                             np.concatenate(strips, axis=1), format='png',
                             origin='upper', dpi=_POSITIONS_DPI)
        finally:
            if self._tmpdir is not None:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
                self._tmpdir = None

    def cancel(self):
        """Stop the plotting processes (e.g. after an error) and clean up."""
        import shutil
        for p in self._procs:
            p.cancel()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


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


def _spike_positions_strip_from_file(path, k, n_strips, out):
    import pickle
    with open(path, 'rb') as f:
        clu, is_refractory, positions = pickle.load(f)
    _spike_positions_strip(clu, is_refractory, positions, k, n_strips, out)
