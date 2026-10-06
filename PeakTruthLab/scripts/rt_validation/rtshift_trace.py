"""Batch the existing nearest-centroid EIC rule and verify against the original extractor."""
import numpy as np
from recovery_helpers import _extract_trace

def extract_batch(spectra,mzs):
    mzs=np.asarray(mzs,dtype=np.float64)
    rt=np.asarray([s.rt_min for s in spectra],dtype=np.float64)
    ys=np.zeros((len(mzs),len(rt)),dtype=np.float64)
    tol=mzs*10.*1e-6
    for i,s in enumerate(spectra):
        if not len(s.mz):continue
        pos=np.searchsorted(s.mz,mzs,side='left')
        left=np.clip(pos-1,0,len(s.mz)-1);right=np.clip(pos,0,len(s.mz)-1)
        near=np.where((s.mz[right]-mzs)<(mzs-s.mz[left]),right,left)
        ys[:,i]=np.where(np.abs(s.mz[near]-mzs)<=tol,s.intensity[near],0.)
    for k in np.unique(np.linspace(0,len(mzs)-1,min(5,len(mzs))).round().astype(int)):
        legacy_rt,legacy_y,_=_extract_trace(spectra,float(mzs[k]),10.,'ppm','nearest')
        assert np.array_equal(rt,legacy_rt) and np.array_equal(ys[k],legacy_y),'batch EIC differs from project extractor'
    return rt,ys
