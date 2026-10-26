"""Existing low-angle DA3 constraint kernels; separate from direct error screening."""
import numpy as np
from .sampling import sample, rays_from_pixels
from .refinement import optimize, robust_cost

def constraints(fields,im,xy,poses,params,boundary=.2,tolerance=.2):
    ray=rays_from_pixels(xy,params)
    depth,conf,spread,usable=sample(fields,im,ray,xy,boundary)
    rotation=poses[:,:,:,:3];translation=poses[:,:,:,3]
    surface=np.einsum('nkji,nkj->nki',rotation,ray*depth[:,:,None]-translation)
    support=np.zeros(im.shape,np.int16);occlusion=support.copy();conflict=support.copy()
    n,k=im.shape
    # One peer for two views, both adjacent peers for multi-view tracks.
    for shift in sorted(set([1,k-1])):
        target=(np.arange(k)+shift)%k
        target_pose=poses[:,target];target_params=params[:,target]
        cam=np.einsum('nkij,nkj->nki',target_pose[:,:,:,:3],surface)+target_pose[:,:,:,3]
        z=cam[:,:,2]
        focal=np.stack([target_params[:,:,0],target_params[:,:,4] if target_params.shape[-1]==5 else target_params[:,:,0]],-1)
        pixel=cam[:,:,:2]/np.maximum(z[:,:,None],1e-12)*focal+target_params[:,:,1:3]
        d,c,s,target_usable=sample(fields,im[:,target],cam,pixel,boundary)
        basic=usable&(z>0)&np.isfinite(d)&(d>0)
        relative=(z-d)/np.maximum(d,1e-12)
        # Raw target depth is read before confidence filtering. Low confidence
        # foreground does not expose a more distant high-confidence layer.
        blocked=basic&(relative>tolerance)
        compatible=basic&target_usable&(np.abs(relative)<=tolerance)
        ahead=basic&target_usable&(relative < -tolerance)
        support+=compatible;occlusion+=blocked;conflict+=ahead
        # Corresponding TRUE observations must also predict compatible camera
        # depths. This disables competing depth layers without averaging them.
        peer_good=usable&usable[:,target]
        observed_relative=np.abs(z-depth[:,target])/np.maximum(depth[:,target],1e-12)
        conflict+=peer_good&(observed_relative>tolerance)
    valid=usable&(support>=1)&(occlusion==0)&(conflict==0)
    return dict(depth=depth,valid=valid,surface=surface,conf=conf,
        support=support,occluded=occlusion,conflict=conflict)


def depth_fit(initial,poses,params,xy,prior,angle,depth_sigma=.15,iterations=6,
              reliable_angle=2.,weak_weight=.02,strong_weight=1.):
    # Geometric observability modulates strength; it is not a correctness score.
    g=np.clip((reliable_angle-angle)/max(reliable_angle-.5,1e-6),0,1)
    strength=weak_weight+(strong_weight-weak_weight)*g
    weight=prior['valid']*strength[:,None]
    eligible=prior['valid'].sum(1)>=2
    weight[~eligible]=0
    seed=initial.copy();cost=robust_cost(seed,poses,params,xy,prior['depth'],weight,depth_sigma)
    # A single depth ray can initialize a poorly conditioned DLT estimate.
    # Choose by the SAME joint objective; never average competing surfaces.
    for view in range(xy.shape[1]):
        candidate=prior['surface'][:,view]
        candidate_cost=robust_cost(candidate,poses,params,xy,prior['depth'],weight,depth_sigma)
        better=eligible&prior['valid'][:,view]&(candidate_cost<cost)
        seed[better]=candidate[better];cost[better]=candidate_cost[better]
    result=optimize(seed,poses,params,xy,prior['depth'],weight,depth_sigma,iterations)
    result[~eligible]=initial[~eligible]
    return result,weight,eligible
