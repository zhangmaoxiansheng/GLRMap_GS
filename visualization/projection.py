"""Unchanged XYZ/RGB at one pinhole camera; nearest positive Z per native pixel."""
import numpy as np


def raster(points,pose,K,width,height,multi=False):
    selected=points['track_length']>=3 if multi else np.ones(len(points['xyz']),bool)
    pose=np.asarray(pose)[:3]
    xyz=points['xyz'][selected].astype(float);rgb=points['rgb'][selected]
    local=xyz@pose[:,:3].T+pose[:,3];z=local[:,2]
    positive=np.isfinite(local).all(1)&(z>0)
    ids=np.flatnonzero(positive);uv=local[ids,:2]/z[ids,None]
    uv=uv*np.array([K[0,0],K[1,1]])+K[:2,2]
    inside=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    ids=ids[inside];pixel=np.floor(uv[inside]).astype(int);flat=pixel[:,1]*width+pixel[:,0]
    # Group by destination pixel, then take the nearest positive camera-Z.
    order=np.lexsort((z[ids],flat));ordered=flat[order]
    first=np.r_[True,ordered[1:]!=ordered[:-1]] if len(order) else np.empty(0,bool)
    pick=order[first];dest=flat[pick];src=ids[pick]
    depth=np.full(width*height,np.inf);depth[dest]=z[src]
    colors=np.zeros((width*height,3),np.uint8);colors[dest]=rgb[src]
    return dict(depth=depth.reshape(height,width),rgb=colors.reshape(height,width,3),
        projected_points=len(ids),occupied_pixels=len(dest),input_points=int(selected.sum()))


def coverage(hit,roi,tile=12,min_hits=3):
    h,w=roi.shape;yy,xx=np.indices((h,w));cols=(w+tile-1)//tile
    cell=(yy//tile)*cols+xx//tile
    weights=np.bincount(cell[roi],minlength=int(cell.max())+1)
    hits=np.bincount(cell[hit&roi],minlength=len(weights))
    supported=hits>=min_hits
    return dict(roi_pixels=int(roi.sum()),hit_pixels=int((hit&roi).sum()),
        pixel_fraction=float((hit&roi).sum()/max(1,roi.sum())),
        supported_tile_area_fraction=float(weights[supported].sum()/max(1,weights.sum())),
        tile_pixels=tile,min_unique_hit_pixels=min_hits)
