import os
import gc
import logging
import warnings
logger = logging.getLogger(__name__)

from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from kilosort import fused_kernels
from kilosort.utils import template_path, log_performance, trying


def my_max2d(X, dt):
    Xmax = max_pool2d(
        X.unsqueeze(0), [2*dt[0]+1, 2*dt[1]+1],
        stride=[1,1], padding=[dt[0],dt[1]]
        )    
    return Xmax[0]

def my_sum2d(X, dt):
    Xsum = avg_pool2d(
        X.unsqueeze(0), [2*dt[0]+1, 2*dt[1]+1],
        stride=[1,1], padding=[dt[0],dt[1]]
        )    
    Xsum *= (2*dt[0]+1) * (2*dt[1]+1)
    return Xsum[0]

def extract_snippets(X, nt, twav_min, Th_single_ch, loc_range=[4,5],
                     long_range=[6,30], device=torch.device('cuda')):
    Xabs   = X.abs()
    Xmax   = my_max2d(Xabs, loc_range)
    ispeak = torch.logical_and(Xmax==Xabs, Xabs > Th_single_ch).float()

    ispeak_sum  = my_sum2d(ispeak, long_range)
    is_peak_iso = ((ispeak_sum==1) * (ispeak==1))

    is_peak_iso[:, :nt] = 0
    is_peak_iso[:, -nt:] = 0

    xy = is_peak_iso.nonzero()

    clips = X[xy[:,:1], xy[:,1:2] - twav_min + torch.arange(nt, device=device)]

    return clips

def extract_wPCA_wTEMP(ops, bfile, nt=61, twav_min=20, Th_single_ch=6, nskip=25,
                       device=torch.device('cuda')):

    clips = np.zeros((500000,nt), 'float32')
    i = 0
    for j in range(0, bfile.n_batches, nskip):
        X = bfile.padded_batch_to_torch(j, ops)
        
        clips_new = extract_snippets(X, nt=nt, twav_min=twav_min,
                                     Th_single_ch=Th_single_ch, device=device)

        nnew = len(clips_new)

        if i+nnew>clips.shape[0]:
            break

        clips[i:i+nnew] = clips_new.cpu().numpy()
        i+= nnew 

    clips = clips[:i]
    clips /= (clips**2).sum(1, keepdims=True)**.5

    model = TruncatedSVD(n_components=ops['settings']['n_pcs']).fit(clips)
    wPCA = torch.from_numpy(model.components_).to(device).float()

    with warnings.catch_warnings():
        msg = 'KMeans is known to have a memory leak on Windows with MKL'
        warnings.filterwarnings("ignore", message=msg)
        # Prevents memory leak for KMeans when using MKL on Windows
        nthread = os.environ.get('OMP_NUM_THREADS')
        new_nthread = 7
        if nthread is not None:
            new_nthread = min(int(nthread), new_nthread)
        os.environ['OMP_NUM_THREADS'] = str(new_nthread)
        # sklearn's multithreaded KMeans reduces the per-thread partial sums
        # in thread completion order, so the centers (and with them the
        # sorting) differ in the last bits from run to run. One OpenMP thread
        # makes it deterministic (about 0.5 s more per call).
        with threadpool_limits(limits=1, user_api='openmp'):
            model = KMeans(n_clusters=ops['settings']['n_templates'], n_init = 10).fit(clips)
        wTEMP = torch.from_numpy(model.cluster_centers_).to(device).float()
        wTEMP = wTEMP / (wTEMP**2).sum(1).unsqueeze(1)**.5
        if nthread is not None:
            os.environ['OMP_NUM_THREADS'] = nthread
        else:
            os.environ.pop('OMP_NUM_THREADS')

    return wPCA, wTEMP

def get_waves(ops, device=torch.device('cuda')):
    dd = np.load(template_path())
    wTEMP = torch.from_numpy(dd['wTEMP']).to(device)
    wPCA = torch.from_numpy(dd['wPCA']).to(device)
    return wPCA, wTEMP

def template_centers(ops):
    shank_idx = ops['kcoords']
    xc = ops['xc']
    yc = ops['yc']
    dmin = ops['settings']['dmin']
    if dmin is None:
        # Try to determine a good value automatically based on contact positions.
        y_uniq = np.unique(yc)
        if y_uniq.size == 1:
            dmin = 1
        else:
            dmin = np.median(np.diff(np.unique(y_uniq)))
    ops['dmin'] = dmin
    ops['dminx'] = dminx = ops['settings']['dminx']

    # Iteratively determine template placement for each shank separately.
    yup = np.array([])
    xup = np.array([])
    for i in np.unique(shank_idx):
        xc_i = xc[shank_idx == i]
        yc_i = yc[shank_idx == i]
        xmin, xmax, ymin, ymax = xc_i.min(), xc_i.max(), yc_i.min(), yc_i.max()

        yup = np.concatenate([yup, np.arange(ymin, ymax+.01, dmin/2)])
        nx = np.round((xmax - xmin) / (dminx/2)) + 1
        xup = np.concatenate([xup, np.linspace(xmin, xmax, int(nx))])

    ops['yup'] = np.unique(yup)
    ops['xup'] = np.unique(xup)

    return ops


def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    if device.type == 'cuda' and fused_kernels.enabled():
        return fused_kernels.verified_call(
            'template_match', _template_match_torch, fused_kernels.template_match,
            X, ops, iC, iC2, weigh, device=device
            )
    return _template_match_torch(X, ops, iC, iC2, weigh, device=device)


def _template_match_torch(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    niter = 40
    nb = (NT-1)//niter+1

    W = ops['wTEMP'].unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=nt//2)
    As    = torch.zeros((Nfilt, NT), device=device)
    Amaxs = torch.zeros((Nfilt, NT), device=device)
    imaxs = torch.zeros((Nfilt, NT), dtype = torch.int64, device=device)
    ti = torch.arange(Nfilt, device = device)
    tj = torch.arange(nb, device = device)

    for t in range(niter):
        A = torch.einsum('ijk, jklm-> iklm', weigh, B[iC,:, nb*t:nb*(t+1)])        
        A = A.transpose(1,2)
        A = A.reshape(-1, Nfilt, A.shape[-1])
        
        #Aa, imax = torch.max(A, 0) 
        Aa, imax = torch.max(A.abs(), 0)
        imax = (1+imax) * A[imax, ti.unsqueeze(-1), tj[:A.shape[-1]]].sign()

        As[:, nb*t:nb*(t+1)] = Aa
        imaxs[:, nb*t:nb*(t+1)] = imax
        Amax = torch.max(Aa[iC2], 0)[0]
        Amaxs[:, nb*t:nb*(t+1)] = Amax

    Amaxs[:,:nt] = 0
    Amaxs[:,-nt:] = 0
    Amaxs  = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride = 1, padding = nt0).squeeze(0)
    xy = torch.logical_and(Amaxs==As, As > ops['Th_universal']).nonzero()
    imax = imaxs[xy[:,0], xy[:,1]]
    amp = As[xy[:,0], xy[:,1]]

    ssign = imax.sign()
    imax = imax.abs()-1
    adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] * ssign

    #adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] 
    
    #xy[:,1] -= nt
    return xy, imax, amp, adist


def nearest_chans(ys, yc, xs, xc, nC, device=torch.device('cuda')):
    ds = (ys - yc[:,np.newaxis])**2 + (xs - xc[:,np.newaxis])**2
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(device)
    ds = np.sort(ds, 0)[:nC]

    return iC, ds


def yweighted(yc, iC, adist, xy, device=torch.device('cuda')):    

    yy = torch.from_numpy(yc).to(device)[iC]
    cF0 = torch.nn.functional.relu(adist)
    cF0 = cF0/cF0.sum(0)

    yct = (cF0 * yy[:,xy[:,0]]).sum(0)
    return yct

def grow_buffer(a, needed, ibatch, n_batches):
    """Grow the spike buffer `a` (numpy or CPU torch) to hold `needed` rows:
    at least double it, and go straight to the size that the batches seen so
    far project for the whole recording (plus 10%), so that it is not copied
    again and again."""
    projected = int(needed / (ibatch + 1) * n_batches * 1.1) + 1
    new_len = max(2 * len(a), projected, needed)
    shape = (new_len - len(a),) + tuple(a.shape[1:])
    if isinstance(a, np.ndarray):
        return np.concatenate((a, np.zeros(shape, a.dtype)), 0)
    return torch.cat((a, torch.zeros(shape, dtype=a.dtype)), 0)


def run(ops, bfile, device=torch.device('cuda'), progress_bar=None,
        clear_cache=False, verbose=False, return_features=True):
    """Detect spikes with the universal templates.

    With `return_features=False` the PC features (tF) are not computed and
    None is returned in their place (used for the drift estimate, which only
    needs the spike times, positions and amplitudes).
    """
    sig = ops['settings']['min_template_size']
    nsizes = ops['settings']['template_sizes']
    nb = ops['Nbatches']

    if (ops['settings']['templates_from_data'] and trying('wpca_reuse')
            and ops.pop('_templates_from_data', False)):
        # Computed from this recording in the drift pass (before drift
        # correction) instead of a second time. Off by default: ~6 s faster
        # on the 45 min sim, but fewer good units on real data.
        logger.info('Reusing universal templates computed from data.')
    elif ops['settings']['templates_from_data']:
        logger.info('Re-computing universal templates from data.')
        # Determine templates and PC features from data.
        ops['wPCA'], ops['wTEMP'] = extract_wPCA_wTEMP(
            ops, bfile, nt=ops['nt'], twav_min=ops['nt0min'], 
            Th_single_ch=ops['settings']['Th_single_ch'], nskip=25,
            device=device
            )
        if not return_features and trying('wpca_reuse'):
            # drift pass: the next pass (detect_univ) may reuse them
            ops['_templates_from_data'] = True
    else:
        logger.info('Using built-in universal templates.')
        # Use pre-computed templates.
        ops['wPCA'], ops['wTEMP'] = get_waves(ops, device=device)

    ops = template_centers(ops)
    [ys, xs] = np.meshgrid(ops['yup'], ops['xup'])
    ys, xs = ys.flatten(), xs.flatten()
    logger.info(f'Number of universal templates: {ys.size}')
    xc, yc = ops['xc'], ops['yc']

    nC = ops['settings']['nearest_chans']
    nC2 = ops['settings']['nearest_templates']
    iC, ds = nearest_chans(ys, yc, xs, xc, nC, device=device)

    # Don't use templates that are too far away from nearest channel
    # (use square of max distance since ds are squared distances)
    igood = ds[0,:] <= ops['max_channel_distance']**2
    iC = iC[:,igood]
    ds = ds[:,igood]
    ys = ys[igood]
    xs = xs[igood]
    ops['ycup'], ops['xcup'] = ys, xs

    iC2, _ = nearest_chans(ys, ys, xs, xs, nC2, device=device)

    ds_torch = torch.from_numpy(ds).to(device).float()
    template_sizes = sig * (1+torch.arange(nsizes, device=device))
    weigh = torch.exp(-ds_torch.unsqueeze(-1) / template_sizes**2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh**2).sum(1).unsqueeze(1)**.5

    st = np.zeros((10**6, 6), 'float64')
    if return_features:
        tF = np.zeros((10**6, nC , ops['settings']['n_pcs']), 'float32')
    else:
        tF = None

    k = 0
    nt = ops['nt']
    tarange = torch.arange(-(nt//2),nt//2+1, device = device)
    logger.info('Detecting spikes...')
    prog = tqdm(np.arange(bfile.n_batches), miniters=200 if progress_bar else None, 
                mininterval=60 if progress_bar else None)
    # repeat performance log after every 10 minutes of data
    log_skip = int(600 / (ops['batch_size'] / ops['fs']))
    try:
        for ibatch in prog:
            if ibatch % log_skip == 0:
                log_performance(logger, 'debug', f'Batch {ibatch} of {nb-1} ({100*(ibatch/nb):.1f}%)')

            X = bfile.padded_batch_to_torch(ibatch, ops)
            xy, imax, amp, adist = template_match(X, ops, iC, iC2, weigh, device=device)
            yct = yweighted(yc, iC, adist, xy, device=device)
            nsp = len(xy)

            if k+nsp>st.shape[0]:
                st = grow_buffer(st, k+nsp, ibatch, bfile.n_batches)
                if return_features:
                    tF = grow_buffer(tF, k+nsp, ibatch, bfile.n_batches)

            if return_features:
                xsub = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
                xfeat = xsub @ ops['wPCA'].T
                tF[k:k+nsp] = xfeat.transpose(0,1).cpu().numpy()

            # The six columns are assembled on the GPU (float64, the same
            # operations as before) and copied back in one go.
            t_shift = ibatch * bfile.batch_downsampling * (ops['batch_size']/ops['fs'])
            cols = torch.stack([
                (xy[:,1] - nt).double() / ops['fs'] + t_shift,
                yct.double(),
                amp.double(),
                imax.double(),
                torch.full((nsp,), float(ibatch), dtype=torch.float64, device=device),
                xy[:,0].double(),
                ], 1)
            st[k:k+nsp] = cols.cpu().numpy()

            k = k + nsp
            if clear_cache:
                gc.collect()
                torch.cuda.empty_cache()

            if progress_bar is not None:
                progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
    except:
        logger.exception(f'Error in spikedetect.run on batch {ibatch}')
        try:
            logger.debug(f'X shape: {X.shape}')
            logger.debug(f'xy shape: {xy.shape}')
        except UnboundLocalError:
            # Error happened before one or both of these was assigned,
            # no need to raise an additional error for this.
            pass
        raise
            
    log_performance(logger, 'debug', f'Batch {ibatch} of {nb-1} ({100*(ibatch/nb):.1f}%)')

    st = st[:k]
    if return_features:
        tF = tF[:k]
    ops['iC'] = iC
    ops['iC2'] = iC2
    ops['weigh'] = weigh
    return st, tF, ops
