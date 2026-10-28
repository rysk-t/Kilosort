"""Kilosort 2.5 parameters: configFiles/configFile384.m plus the constants
hard-coded in the MATLAB functions. The Kilosort 4 settings that the shared
preprocessing / drift steps use (batch_size, nt, nt0min, nblocks,
highpass_cutoff, whitening_range, nskip, ...) come from
`kilosort.parameters.DEFAULT_SETTINGS`."""

DEFAULT_SETTINGS25 = {
    # configFile384.m
    'Th': (10.0, 4.0),       # threshold on projections [learning, final pass]
    'lam': 10.0,             # amplitude penalty
    'AUCsplit': 0.9,         # isolation needed for a split
    'minFR': 1/50,           # minimum spike rate (Hz) of a template
    'momentum': (20.0, 400.0),  # number of spikes averaged into a template
    'sigmaMask': 30.0,       # spatial constant (um) of the channel mask
    'ThPre': 8.0,            # threshold crossings for pre-clustering
    'spkTh': -6.0,           # single-channel spike threshold (SD)
    'nfilt_factor': 4,       # max templates per channel
    'useStableMode': 1,      # deterministic subtraction / averaging in mexMPnu8
    # constants of the MATLAB functions
    'NchanNear': 32,         # channels per template (learnTemplates)
    'Nnearest': 32,          # neighbor templates for the template features
    'Nrank': 3,              # rank of the templates
    'NrankPC': 6,            # rank of the 1D PCs / number of wTEMP prototypes
    'nInnerIter': 60,        # iterations of the matching pursuit per batch
    'Nsum': 7,               # channels summed in mexGetSpikes2
    'loc_range': (5, 4),     # isolated_peaks_new
    'long_range': (30, 6),
    # speed knobs of this implementation (not in the MATLAB code)
    'learn_batches_max': None,  # None: learn on every batch like the MATLAB code
    # known quirks of the MATLAB code; True fixes them, False reproduces MATLAB
    'fix_split_swap': True,    # splitAllClusters assigns c1/c2 to the wrong half
    'fix_merge_pc': True,      # find_merges does not remap the PC feature channels
    'matlab_tail': False,      # True: skip the last ~Nthreads samples of a batch like the CUDA loops
}
