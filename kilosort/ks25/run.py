"""main_kilosort.m: the Kilosort 2.5 pipeline.

preprocessing and drift estimation are the Kilosort 4 steps (the same
algorithm: `preprocessDataSub` / `datashift2`), the rest follows the MATLAB
code (`learnAndSolve8b`, `find_merges`, `splitAllClusters`, `set_cutoff`,
`get_good_units`, `rezToPhy`).
"""
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import kilosort
from kilosort import io
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.run_kilosort import (
    set_files, setup_logger, close_logger, initialize_ops,
    compute_preprocessing, compute_drift_correction,
)
from kilosort.utils import log_performance, get_performance, ops_as_string
import kilosort.plots as kplots

from kilosort.ks25.params import DEFAULT_SETTINGS25, TUNED_SETTINGS25
from kilosort.ks25 import learn, postprocess

logger = logging.getLogger(__name__)


def split_settings(settings):
    """Kilosort 4 settings (for the shared preprocessing / drift steps) and
    Kilosort 2.5 settings (`DEFAULT_SETTINGS25`), each with their defaults."""
    settings = dict(settings)
    s25 = dict(DEFAULT_SETTINGS25)
    if settings.get('preset', s25['preset']) == 'tuned':
        s25.update(TUNED_SETTINGS25)
    for k in DEFAULT_SETTINGS25:
        if k in settings:
            s25[k] = settings.pop(k)
    s4 = {**DEFAULT_SETTINGS, **settings}
    return s4, s25


def batch_order(ops):
    """`rez.iorig` of datashift2: batches sorted by their mean drift."""
    dshift = ops.get('dshift', None)
    n = ops['Nbatches']
    if dshift is None:
        return np.arange(n)
    return np.argsort(np.asarray(dshift).reshape(n, -1).mean(1), kind='stable')


def run_kilosort25(settings, probe=None, probe_name=None, filename=None,
                   data_dir=None, results_dir=None, data_dtype=None,
                   do_CAR=True, invert_sign=False, device=None,
                   progress_bar=None, bad_channels=None, shank_idx=None,
                   verbose_console=False, verbose_log=False,
                   drift_plots=True):
    """Sort with the Kilosort 2.5 algorithm. Arguments as `run_kilosort`.

    `settings` takes the Kilosort 4 keys (`n_chan_bin` is required) and the
    Kilosort 2.5 keys of `kilosort.ks25.params.DEFAULT_SETTINGS25`.

    Returns
    -------
    ops : dict
        Kilosort 4 style ops with the stage times in `runtime_*`, plus the
        Kilosort 2.5 results (`W`, `U`, `mu`, `simScore`, ...).
    rez : dict
        The MATLAB `rez` structure (st3, cProj, cProjPC, W, U, mu, dWU, ...).
    """
    settings, settings25 = split_settings(settings)
    if settings.get('n_chan_bin') is None:
        raise ValueError('`n_chan_bin` must be specified in settings')
    filename, data_dir, results_dir, probe = set_files(
        settings, filename, probe, probe_name, data_dir, results_dir,
        bad_channels, shank_idx)
    setup_logger(results_dir, verbose_console)
    background_plots = []
    try:
        logger.info(f"Kilosort version {kilosort.__version__} (ks25: Kilosort 2.5 algorithm)")
        logger.info(f"Python version {platform.python_version()}")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logger.info(f'Using CUDA device: {torch.cuda.get_device_name(device)}')
        if data_dtype is None:
            data_dtype = 'int16'
        logger.info(f"Sorting {filename}")

        tic0 = time.time()
        ops, settings = initialize_ops(settings, probe, data_dtype, do_CAR,
                                       invert_sign, device, False)
        ops['ks25'] = settings25
        logger.debug(f"Initial ops:\n\n{ops_as_string(ops)}\n")

        # --- shared with Kilosort 4: filters, whitening, drift -------------
        ops = compute_preprocessing(ops, device, tic0=tic0)
        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1)
        ops, bfile, st0 = compute_drift_correction(
            ops, device, tic0=tic0, progress_bar=progress_bar,
            verbose=verbose_log)
        if st0 is not None and drift_plots:
            background_plots.append(kplots.BackgroundDriftPlots(
                ops, st0, results_dir, tmin=settings['tmin']))

        # --- Kilosort 2.5 ----------------------------------------------------
        tic = time.time()
        logger.info(' ')
        logger.info('Extracting 1D templates and PCs (extractTemplatesfromSnippets).')
        logger.info('-'*40)
        wTEMP, wPCA = learn.extract_templates_from_snippets(
            ops, bfile, nPCs=settings25['NrankPC'])
        ops['wTEMP'] = wTEMP
        ops['wPCA'] = wPCA
        ops['n_pcs'] = int(wPCA.shape[0])  # bench.avg_wav expects this many rows
        ops['runtime_wpca'] = time.time() - tic
        logger.info(f'templates extracted in {ops["runtime_wpca"]:.2f}s; '
                    f'total {time.time()-tic0:.2f}s')

        tic = time.time()
        logger.info(' ')
        logger.info('Learning templates (learnTemplates).')
        logger.info('-'*40)
        iorder = batch_order(ops)
        nmax = settings25.get('learn_batches_max')
        if nmax is not None and len(iorder) > nmax:
            # speed knob: a regular subset of the drift-sorted batches
            iorder = iorder[np.round(np.linspace(0, len(iorder)-1, nmax)).astype(int)]
        if hasattr(bfile, 'set_prefetch_order'):
            bfile.set_prefetch_order(iorder)
        rez = learn.learn_templates(ops, bfile, iorder, progress_bar=progress_bar)
        if hasattr(bfile, 'set_prefetch_order'):
            bfile.set_prefetch_order(None)
        ops['runtime_learn'] = time.time() - tic
        logger.info(f'{rez["W"].shape[1]} templates learned in '
                    f'{ops["runtime_learn"]:.2f}s; total {time.time()-tic0:.2f}s')
        log_performance(logger, 'info', 'Resource usage after learning', reset=True)

        tic = time.time()
        logger.info(' ')
        logger.info('Extracting spikes (trackAndSort).')
        logger.info('-'*40)
        st3, fW, fWpc, rez = learn.track_and_sort(ops, bfile, rez,
                                                  progress_bar=progress_bar)
        rez['st3'] = st3
        rez['cProj'] = fW
        rez['cProjPC'] = fWpc
        rez['xc'] = ops['xc']
        rez['yc'] = ops['yc']
        rez['Wrot'] = ops['Wrot']
        ops['runtime_track'] = time.time() - tic
        dump = os.environ.get('KS25_DUMP_REZ')
        if dump:
            # development aid: the state before the post-processing, to run
            # merges / splits / cutoff on their own (torch.load(dump))
            torch.save({k: v for k, v in rez.items()}, dump)
            logger.info(f'rez after trackAndSort saved to {dump}')
        logger.info(f'{len(st3)} spikes extracted in {ops["runtime_track"]:.2f}s; '
                    f'total {time.time()-tic0:.2f}s')
        log_performance(logger, 'info', 'Resource usage after extraction', reset=True)

        tic = time.time()
        logger.info(' ')
        logger.info('Merging (find_merges).')
        logger.info('-'*40)
        rez = postprocess.find_merges(rez, ops, True)
        ops['runtime_merge'] = time.time() - tic
        logger.info(f'merges done in {ops["runtime_merge"]:.2f}s; total {time.time()-tic0:.2f}s')

        tic = time.time()
        logger.info(' ')
        logger.info('Splitting (splitAllClusters).')
        logger.info('-'*40)
        rez = postprocess.split_all_clusters(rez, ops, True)
        ops['runtime_split'] = time.time() - tic
        logger.info(f'splits done in {ops["runtime_split"]:.2f}s; total {time.time()-tic0:.2f}s')

        tic = time.time()
        logger.info(' ')
        logger.info('Amplitude cutoff (set_cutoff) and good units.')
        logger.info('-'*40)
        rez = postprocess.set_cutoff(rez, ops)
        rez['good'] = postprocess.get_good_units(rez, ops)
        ops['runtime_cutoff'] = time.time() - tic
        logger.info(f'found {int(np.sum(np.asarray(rez["good"]) > 0))} good units; '
                    f'cutoff in {ops["runtime_cutoff"]:.2f}s; total {time.time()-tic0:.2f}s')

        tic = time.time()
        logger.info(' ')
        logger.info('Saving to phy (rezToPhy).')
        logger.info('-'*40)
        for k in ('W', 'U', 'mu', 'simScore', 'iNeigh', 'iNeighPC', 'good',
                  'est_contam_rate'):
            if k in rez:
                v = rez[k]
                ops[k] = v.cpu().numpy() if torch.is_tensor(v) else v
        ops['runtime'] = time.time() - tic0  # provisional, for ops.npy
        postprocess.save_to_phy25(rez, ops, results_dir)
        ops['runtime_postproc'] = time.time() - tic
        ops['runtime'] = time.time() - tic0
        ops['usage_postproc'] = get_performance()
        logger.info(f'saved in {ops["runtime_postproc"]:.2f}s; total {ops["runtime"]:.2f}s')

        while background_plots:
            background_plots[0].wait()
            background_plots.pop(0)
        logger.info('Sorting finished.')
    except Exception:
        logger.exception('Encountered error in `run_kilosort25`:')
        raise
    finally:
        for p in background_plots:
            try:
                p.cancel()
            except Exception:
                pass
        close_logger()
    return ops, rez
