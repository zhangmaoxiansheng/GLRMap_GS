"""Sample stored camera-Z at true Track pixels using the processed K."""
import numpy as np
from ..matching.sparse import valid_mask


def processed_grid(field, local, params):
    """Stored integer processed pixels → native pixels; support distinct fx/fy."""
    h, w = field['conf'].shape
    yy, xx = np.indices((h, w))
    rays = np.stack([xx, yy, np.ones_like(xx)], -1)@np.linalg.inv(field['K']).T
    focal = np.array([params[0], params[4] if len(params) == 5 else params[0]])
    native = rays[..., :2]/rays[..., 2, None]*focal+params[1:3]
    static = valid_mask(native.reshape(-1, 2), field['mask']).reshape(h, w)
    valid = static & np.isfinite(local).all(-1) & (local[..., 2] > 0) & np.isfinite(field['conf'])
    return rays, native, valid

def bilinear(field, xy):
    """Coordinates use the SDK's pixel/K resize convention, with no crop."""
    h, w = field.shape
    xy = np.asarray(xy)
    inside = np.isfinite(xy).all(-1) & (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    q = np.nan_to_num(xy, nan=0., posinf=0., neginf=0.)
    x = np.clip(q[..., 0], 0, w-1); y = np.clip(q[..., 1], 0, h-1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0+1, w-1); y1 = np.minimum(y0+1, h-1)
    a = x-x0; b = y-y0
    value = np.zeros_like(x, dtype=np.result_type(field.dtype, float))
    # Zero-weight neighbors must not invalidate an exact valid pixel (0 * NaN).
    # A nonfinite neighbor with positive weight still invalidates interpolation.
    for weight, corner in [((1-a)*(1-b), field[y0,x0]), (a*(1-b), field[y0,x1]),
                           ((1-a)*b, field[y1,x0]), (a*b, field[y1,x1])]:
        value += weight*np.where(weight > 0, corner, 0.)
    return value, inside


def rays_from_pixels(xy,params):
    fy=params[:,:,4] if params.shape[-1]==5 else params[:,:,0]
    return np.stack(((xy[:,:,0]-params[:,:,1])/params[:,:,0],
        (xy[:,:,1]-params[:,:,2])/fy,np.ones(xy.shape[:2])),-1)


def sample(fields,im,local,native_xy,boundary_log_jump=0.):
    shape=im.shape;d=np.full(shape,np.nan);conf=d.copy();spread=d.copy();usable=np.zeros(shape,bool)
    for index in np.unique(im):
        select=im==index;f=fields[int(index)];cam=local[select]
        ray=cam[:,:2]/np.maximum(cam[:,2,None],1e-12)
        uv=ray@f['K'][:2,:2].T+f['K'][:2,2]
        depth,inside=bilinear(f['depth'],uv);c,_=bilinear(f['conf'],uv);jump,_=bilinear(f['spread'],uv)
        static=valid_mask(native_xy[select],f['mask'])
        good=inside&static&(cam[:,2]>0)&np.isfinite(depth)&(depth>0)&np.isfinite(c)&(c>=f['threshold'])
        if boundary_log_jump>0:good&=jump<=boundary_log_jump
        d[select]=depth;conf[select]=c;spread[select]=jump;usable[select]=good
    return d,conf,spread,usable
