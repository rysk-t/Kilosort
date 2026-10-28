"""Kilosort 2.5 algorithm (MATLAB + CUDA, branch `kilosort25`) in Python.

The preprocessing and drift estimation are shared with Kilosort 4 (same
algorithm, see `run_kilosort.compute_preprocessing` and
`compute_drift_correction`). Template learning, the final extraction pass,
merges, splits, and the amplitude cutoff follow the MATLAB code:

    kernels.py      CUDA mex functions (mexGetSpikes2, mexSVDsmall2,
                    mexMPnu8, mexWtW2) as CuPy RawModule kernels
    learn.py        extractTemplatesfromSnippets, learnTemplates, trackAndSort
    postprocess.py  find_merges, splitAllClusters, set_cutoff, get_good_units,
                    rezToPhy
    run.py          main_kilosort.m: the whole pipeline
    params.py       Kilosort 2.5 settings (configFile384.m and constants)
"""
