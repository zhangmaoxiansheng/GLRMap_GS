"""Sparse feature identities and deterministic source sampling support."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sqlite3
import numpy as np

PAIR_BASE = 2147483647

def pair_seed(a,b,seed):
    return (int.from_bytes(hashlib.sha256((a+'\0'+b).encode()).digest()[:4],'little')+seed)%(2**31-1)


@contextmanager
def offline_sparse_weights():
    import torch
    original=torch.hub.load_state_dict_from_url
    from ..runtime import SPARSE_WEIGHTS
    cache=SPARSE_WEIGHTS
    def load(url,*args,**kwargs):
        name=kwargs.get('file_name',url.rsplit('/',1)[1])
        if name not in ['superpoint_v1.pth','superpoint_lightglue_v0-1_arxiv.pth']:
            raise RuntimeError(f'Unexpected sparse weight request: {name}')
        return torch.load(cache/name,map_location='cpu',weights_only=True)
    torch.hub.load_state_dict_from_url=load
    try:yield
    finally:torch.hub.load_state_dict_from_url=original


def sift_data(model,images,database):
    keypoints={};existing={}
    with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
        for im in images:
            rows,cols,blob=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(im.image_id,)).fetchone()
            xy=np.frombuffer(blob,np.float32).reshape(rows,cols)[:,:2].copy()
            assert len(xy)==len(im.points2D)
            assert np.allclose(xy,np.array([p.xy for p in im.points2D]),atol=1e-6,rtol=0)
            keypoints[im.name]=xy
            ids=np.array([p.point3D_id if p.has_point3D() else 0 for p in im.points2D],dtype=np.int64)
            existing[im.name]=ids
    return keypoints,existing


def valid_mask(xy,mask):
    finite=np.isfinite(xy).all(1)
    pixel=np.floor(np.nan_to_num(xy)).astype(int)
    inside=finite&(pixel[:,0]>=0)&(pixel[:,0]<mask.shape[1])&(pixel[:,1]>=0)&(pixel[:,1]<mask.shape[0])
    keep=np.zeros(len(xy),bool);keep[inside]=mask[pixel[inside,1],pixel[inside,0]]>0
    return keep
