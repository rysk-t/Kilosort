import numpy as np
import torch
from numba import njit
import math
from kilosort.CCG import compute_CCG, CCG_metrics

def count_elements(kk, iclust, my_clus, xtree):
    n1 = np.isin(iclust, my_clus[xtree[kk, 0]]).sum()
    n2 = np.isin(iclust, my_clus[xtree[kk, 1]]).sum()
    return n1, n2

def _members(iclust, labels, nlab):
    """Same as np.isin(iclust, labels) for labels in [0, nlab), with a
    lookup table instead of sorting."""
    lut = np.zeros(max(nlab, max(labels) + 1), bool)
    lut[labels] = True
    return lut[iclust]


def check_split(Xd, kk, xtree, iclust, my_clus, ix1=None, ix2=None):
    if ix1 is None or ix2 is None:
        ixy = np.isin(iclust, my_clus[xtree[kk, 2]])
        iclu = iclust[ixy]
        labels = 2*np.isin(iclu, my_clus[xtree[kk, 0]]) - 1
    else:
        # The two children partition the parent. Reuse their masks while
        # preserving the original spike order in the regression inputs.
        ixy = ix1 | ix2
        labels = 2*ix1[ixy] - 1

    Xs = Xd[ixy]
    Xs[:,-1] = 1

    w = np.ones((Xs.shape[0],1))
    w[labels>0] = np.mean(labels<0)
    w[labels<0] = np.mean(labels>0)

    CC = Xs.T @ (Xs * w)
    CC = CC + .01 * np.eye(CC.shape[0])
    b = np.linalg.solve(CC, labels @ (Xs * w))
    xproj = Xs @ b

    score = bimod_score(xproj)
    return xproj, score

def check_split_device(Xg, ix1, ix2):
    """`check_split` (with ix1, ix2) on the device of Xg, the features of
    the same spikes as Xd. Same float64 arithmetic, so only the order of the
    sums differs from the CPU version."""
    ixy = ix1 | ix2
    sel = torch.from_numpy(np.nonzero(ixy)[0]).to(Xg.device)
    labels = torch.from_numpy(2*ix1[ixy] - 1).to(Xg.device, torch.float64)
    Xs = Xg.index_select(0, sel).double()
    Xs[:,-1] = 1

    npos = (labels > 0).double().mean()
    nneg = (labels < 0).double().mean()
    w = torch.where(labels > 0, nneg, npos).unsqueeze(1)

    Xw = Xs * w
    CC = Xs.T @ Xw
    CC = CC + .01 * torch.eye(CC.shape[0], dtype=CC.dtype, device=CC.device)
    b = torch.linalg.solve(CC, labels @ Xw)
    xproj = (Xs @ b).cpu().numpy()

    score = bimod_score(xproj)
    return xproj, score

def clean_tree(valid_merge, xtree, inode):
    ix = (xtree[:,2]==inode).nonzero()[0]
    if len(ix)==0:
        return
    valid_merge[ix] = 0
    clean_tree(valid_merge, xtree, xtree[ix, 0])
    clean_tree(valid_merge, xtree, xtree[ix, 1])
    return

def bimod_score(xproj):
    from scipy.ndimage import gaussian_filter1d
    xbin, _ = np.histogram(xproj, np.linspace(-2,2,400))
    xbin = gaussian_filter1d(xbin.astype('float32'), 4)

    imin = np.argmin(xbin[175:225])
    xmin = np.min(xbin[175:225])
    xm1  = np.max(xbin[:imin+175])
    xm2  = np.max(xbin[imin+175:])

    score = 1 - np.maximum(xmin/xm1, xmin/xm2)
    return score

def check_CCG(st1, st2=None, nbins = 500, tbin  = 1/1000):
    if st2 is None:
        st2 = st1.copy()
    # Empty trains must be checked before compute_CCG (it calls .max() on
    # both arrays). len(st2) == 0, not len(st2 == 0): the latter is the size
    # of a boolean array, which is truthy for every non-empty st2 and made
    # this guard fire on all real inputs, disabling the refractoriness gate.
    if len(st1) == 0 or len(st2) == 0:
        return False, False
    K , T = compute_CCG(st1, st2, nbins = nbins, tbin = tbin)
    if T == 0:
        return False, False
    R12, Q12, Q00 = CCG_metrics(st1, st2, K, T,  nbins = nbins, tbin = tbin)
    is_refractory    = R12<.1  and (Q12<.2  or Q00<.25)
    cross_refractory = R12<.25 and (Q12<.05 or Q00<.25)
    return is_refractory, cross_refractory

def refractoriness(st1, st2):
    # compute goodness of st1, st2, and both

    is_refractory = check_CCG(st1, st2)[1]
    if is_refractory:
        criterion = 1 # never split
        #print('this is refractory')
    else:
        criterion = 0
        #good_0 = check_CCG(np.hstack((st1,st2)))[0]
        #good_1 = check_CCG(st1)[0]
        #good_2 = check_CCG(st2)[0]
        #print(good_0, good_1, good_2)
        #if (good_0==1) and (good_1==0) and (good_2==0):
        #    criterion = 1 # don't split
        #    print('good cluster becomes bad')
    return criterion

def split(Xd, xtree, tstat, iclust, my_clus, verbose = True, meta = None,
          Xg = None):
    # Xg: optional copy of Xd on the GPU, used for the bimodality regression.
    xtree = np.array(xtree)

    kk = xtree.shape[0]-1
    nc = xtree.shape[0] + 1
    valid_merge = np.ones((nc-1,), 'bool')
    nlab = int(iclust.max()) + 1 if iclust.size else 0


    for kk in range(nc-2,-1,-1):
        if not valid_merge[kk]:
            continue;

        # This criterion cannot merge the node, so no spike masks or other
        # tests are needed. The old verbose-only mask sums were unused.
        if tstat[kk,0] < 0.2:
            continue

        ix1 = _members(iclust, my_clus[xtree[kk, 0]], nlab)
        ix2 = _members(iclust, my_clus[xtree[kk, 1]], nlab)

        criterion = 0
        score = np.nan
        if meta is not None and criterion==0:
            # second mutation is based on meta_data
            criterion = refractoriness(meta[ix1],meta[ix2])
            #criterion = 0
        
        if criterion==0:
            if Xg is not None:
                xproj, score = check_split_device(Xg, ix1, ix2)
            else:
                xproj, score = check_split(Xd, kk, xtree, iclust, my_clus, ix1, ix2)
            # third mutation is bimodality
            #xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            criterion = 2 * (score <  .6) - 1

        if criterion==0:
            # fourth mutation is local modularity (not reachable)
            score = tstat[kk,-1]
            criterion = score > .15

        if criterion==1:
            valid_merge[kk] = 0
            clean_tree(valid_merge, xtree, xtree[kk,0])
            clean_tree(valid_merge, xtree, xtree[kk,1])

    tstat = tstat[valid_merge]
    xtree = xtree[valid_merge]

    return xtree, tstat


def new_clusters(iclust, my_clus, xtree, tstat):

    if len(xtree)==0:
        return np.zeros_like(iclust)
         

    nc = xtree.max() + 1

    isleaf = np.zeros(2*nc-1,)
    isleaf[xtree[:,0]] = 1
    isleaf[xtree[:,1]] = 1
    isleaf[xtree[:,2]] = 0

    ind = np.nonzero(isleaf)[0]
    iclust1 = iclust.copy()
    nlab = int(iclust.max()) + 1 if iclust.size else 0
    for j in range(len(ind)):
        ix = _members(iclust, my_clus[ind[j]], nlab)
        iclust1[ix] = j
        xtree[xtree[:,0] == ind[j], 0] = j
        xtree[xtree[:,1] == ind[j], 1] = j


    return iclust1
