"""Paired sparse evidence; canonical source identity and coordinates stay fixed."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

def precision_to_original(precision, matcher_wh, original_wh):
    # RoMa v2 refiner.py declares its precision parameters in matcher PIXELS.
    # Cached V1 keys ending in "normalized" were misnamed; warps are normalized,
    # precision matrices are not. Same transform as upstream prec_map_coordinates.
    ratio=np.asarray(matcher_wh,dtype=float)/np.asarray(original_wh,dtype=float)
    return np.asarray(precision)*ratio[None,:,None]*ratio[None,None,:]


def snap_two_endpoints(raw0,raw1,anchors,tau=2.):
    raw0=np.asarray(raw0);raw1=np.asarray(raw1);n=len(raw0)
    chosen=np.full(n,-1,np.int32);status=np.zeros(n,np.uint8);dist=np.full((n,2),np.nan,np.float32)
    final0=raw0.copy();final1=raw1.copy()
    finite=np.isfinite(raw0).all(1)&np.isfinite(raw1).all(1);status[~finite]=5
    if not len(anchors['p0']):return dict(chosen=chosen,status=status,dist=dist,p0=final0,p1=final1)
    tree=cKDTree(np.column_stack((anchors['p0'],anchors['p1'])))
    valid_indices=np.flatnonzero(finite)
    neighborhoods=tree.query_ball_point(np.column_stack((raw0[finite],raw1[finite])),np.sqrt(2)*tau)
    for i,indices in zip(valid_indices,neighborhoods):
        if not indices:continue
        ids=np.asarray(indices,int)
        d0=np.linalg.norm(anchors['p0'][ids]-raw0[i],axis=1);d1=np.linalg.norm(anchors['p1'][ids]-raw1[i],axis=1)
        ids=ids[(d0<=tau)&(d1<=tau)]
        if not len(ids):status[i]=4;continue
        conflict=(anchors['old0'][ids]>0)&(anchors['old1'][ids]>0)&(anchors['old0'][ids]!=anchors['old1'][ids])
        if conflict.any():status[i]=3;continue
        groups={}
        for a in ids:
            old=int(anchors['old0'][a])
            same_old=old>0 and old==int(anchors['old1'][a])
            identity=('existing',old) if same_old else (int(anchors['kind'][a]),int(anchors['key0'][a]),int(anchors['key1'][a]))
            groups.setdefault(identity,[]).append(int(a))
        existing={int(anchors['old0'][a]) for a in ids if anchors['old0'][a]>0 and anchors['old0'][a]==anchors['old1'][a]}
        if len(existing)>1:status[i]=3;continue
        if len(groups)!=1:status[i]=2;continue
        # Priority operates only WITHIN one proven identity, never overwrites conflicts.
        a=min(ids,key=lambda k:(0 if anchors['old0'][k]>0 and anchors['old0'][k]==anchors['old1'][k] else 1,int(anchors['kind'][k]),int(k)))
        chosen[i]=a;status[i]=1;final0[i]=anchors['p0'][a];final1[i]=anchors['p1'][a]
        dist[i]=[np.linalg.norm(raw0[i]-final0[i]),np.linalg.norm(raw1[i]-final1[i])]
    return dict(chosen=chosen,status=status,dist=dist,p0=final0,p1=final1)


def identity_snap(raw_src,raw_dst,seed_kind,seed_key,anchors,tau=2.):
    """Full paired evidence only, with exactly the canonical source identity."""
    result=raw_dst.copy();status=np.zeros(len(raw_src),np.uint8);chosen=np.full(len(raw_src),-1,np.int32)
    groups=defaultdict(list)
    for a,(kind,key) in enumerate(zip(anchors['kind'],anchors['key0'])):groups[(int(kind),int(key))].append(a)
    for r,(kind,key) in enumerate(zip(seed_kind,seed_key)):
        ids=groups.get((int(kind),int(key)),[])
        if not ids:continue
        ids=np.array(ids,int);ids=ids[np.linalg.norm(anchors['p0'][ids]-raw_src[r],axis=1)<1e-4]
        if not len(ids):continue
        local=snap_two_endpoints(raw_src[r:r+1],raw_dst[r:r+1],{k:v[ids] for k,v in anchors.items()},tau)
        status[r]=local['status'][0]
        if local['chosen'][0]>=0:
            anchor=int(ids[local['chosen'][0]])
            if not np.array_equal(anchors['p0'][anchor],raw_src[r]):
                # Tolerance was for representation checking, not coordinate drift.
                status[r]=4;continue
            chosen[r]=anchor;result[r]=anchors['p1'][anchor]
    return result,status,chosen
