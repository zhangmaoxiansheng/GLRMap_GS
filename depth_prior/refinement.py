"""Robust reprojection + log-depth XYZ fit with fixed cameras and observations."""
import numpy as np
from ..triangulation.geometry import project_batch
from .sampling import rays_from_pixels

def robust_cost(xyz,poses,params,xy,depth,weight,depth_sigma):
    active=(weight>0)&np.isfinite(depth)&(depth>0)
    weight=np.where(active,weight,0.);depth=np.where(active,depth,1.)
    pixel,z,_=project_batch(xyz,poses,params)
    obs=np.linalg.norm(pixel-xy,axis=2)
    dz=(np.log(np.maximum(z,1e-12))-np.log(np.maximum(depth,1e-12)))/depth_sigma
    # Huber(delta=1 px) on actual image residuals, delta=2 sigma on depth.
    oc=np.where(obs<=1,.5*obs*obs,obs-.5)
    dc=np.where(np.abs(dz)<=2,.5*dz*dz,2*np.abs(dz)-2)
    cost=(oc+weight*dc).sum(1)
    cost[(z<=0).any(1)|~np.isfinite(cost)]=np.inf
    return cost


def optimize(xyz,poses,params,xy,depth,weight,depth_sigma,iterations):
    active=(weight>0)&np.isfinite(depth)&(depth>0)
    weight=np.where(active,weight,0.);depth=np.where(active,depth,1.)
    result=xyz.astype(float).copy()
    cost=robust_cost(result,poses,params,xy,depth,weight,depth_sigma)
    for _ in range(iterations):
        pixel,z,jac=project_batch(result,poses,params)
        residual=pixel-xy
        ow=1/np.maximum(np.linalg.norm(residual,axis=2),1.)
        dr=(np.log(np.maximum(z,1e-12))-np.log(np.maximum(depth,1e-12)))/depth_sigma
        dj=poses[:,:,2,:3]/np.maximum(z[:,:,None],1e-12)/depth_sigma
        dw=weight*np.minimum(1.,2/np.maximum(np.abs(dr),1e-12))
        h=np.einsum('nkai,nkaj,nk->nij',jac,jac,ow)+np.einsum('nki,nkj,nk->nij',dj,dj,dw)
        grad=np.einsum('nkai,nka,nk->ni',jac,residual,ow)+np.einsum('nki,nk,nk->ni',dj,dr,dw)
        damping=np.maximum(np.trace(h,axis1=1,axis2=2)*1e-8,1e-10)
        delta=np.linalg.solve(h+damping[:,None,None]*np.eye(3),grad)
        delta=np.nan_to_num(delta)
        # Bound each step in camera-depth units, not distance to world origin.
        limit=np.maximum(np.median(z,axis=1),1e-6)*.1
        delta*=np.minimum(1,limit/np.maximum(np.linalg.norm(delta,axis=1),1e-12))[:,None]
        accepted=np.zeros(len(result),bool)
        for factor in [1.,.25,.0625]:
            trial=result-delta*factor
            trial_cost=robust_cost(trial,poses,params,xy,depth,weight,depth_sigma)
            good=(trial_cost<cost)&~accepted
            result[good]=trial[good];cost[good]=trial_cost[good];accepted|=good
    # Exact fallback: disabling all depth preserves original XYZ, rather than
    # silently switching the reconstruction's geometry-only robust loss.
    result[~(weight>0).any(1)]=xyz[~(weight>0).any(1)]
    return result


def refine(initial, poses, params, xy, depth, confidence, valid, sigma, iterations):
    """Soft fit uses all trusted observations; replacement uses the strongest one."""
    eligible=valid.any(1)
    best=np.argmax(np.where(valid,confidence,-np.inf),axis=1)
    row=np.arange(len(initial))
    ray=rays_from_pixels(xy,params)[row,best]
    pp=poses[row,best]
    target=np.einsum('nji,nj->ni',pp[:,:,:3],ray*depth[row,best,None]-pp[:,:,3])
    target[~eligible]=initial[~eligible]
    weight=valid.astype(float)
    seed=initial.astype(float).copy()
    better=eligible&(robust_cost(target,poses,params,xy,depth,weight,sigma)<robust_cost(seed,poses,params,xy,depth,weight,sigma))
    seed[better]=target[better]
    fitted=initial.astype(float).copy()
    if eligible.any():
        fitted[eligible]=optimize(seed[eligible],poses[eligible],params[eligible],xy[eligible],
            depth[eligible],weight[eligible],sigma,iterations)
    return fitted,target,best,eligible
