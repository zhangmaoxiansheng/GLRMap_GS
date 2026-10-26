"""Fixed-camera XYZ fitting, robust initialization and evidence-bound repair."""
import itertools
import numpy as np

def project_batch(xyz,poses,params):
    """N tracks, K cameras: SIMPLE_RADIAL/PINHOLE pixels and XYZ Jacobians."""
    local=np.einsum('nkij,nj->nki',poses[:,:,:,:3],xyz)+poses[:,:,:,3]
    z=local[:,:,2];safe=np.where(np.abs(z)>1e-12,z,1e-12)
    x=local[:,:,0]/safe;y=local[:,:,1]/safe
    f,cx,cy,k=np.moveaxis(params[...,:4],-1,0)
    fy=params[...,4] if params.shape[-1]==5 else f
    radial=1+k*(x*x+y*y)
    pixels=np.stack((f*x*radial+cx,fy*y*radial+cy),-1)
    normal=np.zeros((*x.shape,2,3));normal[:,:,0,0]=1/safe;normal[:,:,0,2]=-x/safe;normal[:,:,1,1]=1/safe;normal[:,:,1,2]=-y/safe
    distort=np.zeros((*x.shape,2,2));distort[:,:,0,0]=f*(radial+2*k*x*x);distort[:,:,1,1]=fy*(radial+2*k*y*y)
    distort[:,:,0,1]=2*f*k*x*y;distort[:,:,1,0]=2*fy*k*x*y
    jac=np.einsum('nkab,nkbc,nkcd->nkad',distort,normal,poses[:,:,:,:3])
    return pixels,z,jac


def dlt_batch(poses,rays):
    rows=np.stack((rays[:,:,0,None]*poses[:,:,2,:]-poses[:,:,0,:],rays[:,:,1,None]*poses[:,:,2,:]-poses[:,:,1,:]),2)
    rows=rows.reshape(len(poses),-1,4)
    _,_,v=np.linalg.svd(rows,full_matrices=False);h=v[:,-1]
    denominator=np.where(np.abs(h[:,3])>1e-15,h[:,3],np.nan)
    return h[:,:3]/denominator[:,None]


def fit_xyz(xyz,poses,params,xy,keep,iterations=5):
    xyz=xyz.copy()
    for _ in range(iterations):
        pixels,depth,jac=project_batch(xyz,poses,params)
        residual=pixels-xy
        weights=keep.astype(float)
        h=np.einsum('nkai,nkaj,nk->nij',jac,jac,weights)
        g=np.einsum('nkai,nka,nk->ni',jac,residual,weights)
        damping=np.maximum(np.trace(h,axis1=1,axis2=2)*1e-10,1e-10)
        h+=damping[:,None,None]*np.eye(3)
        delta=np.linalg.solve(h,g);delta=np.nan_to_num(delta)
        limit=np.maximum(np.linalg.norm(xyz,axis=1),1.)*.25
        delta*=np.minimum(1,limit/np.maximum(np.linalg.norm(delta,axis=1),1e-12))[:,None]
        xyz-=delta
    return xyz


def maximum_angles(xyz,centers,keep):
    rays=xyz[:,None]-centers;rays/=np.maximum(np.linalg.norm(rays,axis=2,keepdims=True),1e-12)
    result=np.zeros(len(xyz))
    for i,j in itertools.combinations(range(centers.shape[1]),2):
        angle=np.degrees(np.arccos(np.clip((rays[:,i]*rays[:,j]).sum(1),-1,1)))
        result=np.maximum(result,np.where(keep[:,i]&keep[:,j],angle,0))
    return result


def robust_initial(poses,params,xy,rays,centers,threshold=2.,max_hypotheses=8):
    n,k=xy.shape[:2]
    if k==2:
        xyz=fit_xyz(dlt_batch(poses,rays),poses,params,xy,np.ones((n,k),bool))
        pixel,depth,_=project_batch(xyz,poses,params)
        errors=np.linalg.norm(pixel-xy,axis=2)
        return xyz,(errors<=threshold)&(depth>0),errors
    pairs=list(itertools.combinations(range(k),2))
    # Choose diverse hypotheses by baseline length. Actual angle is checked later.
    distances=np.stack([np.linalg.norm(centers[:,i]-centers[:,j],axis=1) for i,j in pairs],1)
    order=np.argsort(-distances,axis=1)[:,:min(max_hypotheses,len(pairs))]
    best_xyz=np.zeros((n,3));best_count=np.full(n,-1);best_cost=np.full(n,np.inf)
    index=np.arange(n)
    for column in range(order.shape[1]):
        selected=np.array(pairs)[order[:,column]]
        p=poses[index[:,None],selected];r=rays[index[:,None],selected]
        xyz=dlt_batch(p,r)
        pixel,depth,_=project_batch(xyz,poses,params);error=np.linalg.norm(pixel-xy,axis=2)
        good=(error<=threshold)&(depth>0)&np.isfinite(xyz).all(1)[:,None]
        count=good.sum(1);cost=np.minimum(error,threshold*2).sum(1)
        better=(count>best_count)|((count==best_count)&(cost<best_cost))
        best_xyz[better]=xyz[better];best_count[better]=count[better];best_cost[better]=cost[better]
    pixel,depth,_=project_batch(best_xyz,poses,params);error=np.linalg.norm(pixel-xy,axis=2)
    consensus=(error<=threshold)&(depth>0)
    safe=consensus.sum(1)>=3
    # Without independent 3-view consensus, retain all observations for diagnosis.
    fit_mask=np.where(safe[:,None],consensus,True)
    best_xyz=fit_xyz(best_xyz,poses,params,xy,fit_mask)
    pixel,depth,_=project_batch(best_xyz,poses,params);error=np.linalg.norm(pixel-xy,axis=2)
    return best_xyz,(error<=threshold)&(depth>0),error


def repair_track(xyz,poses,params,xy,rays,centers,alternatives,track_id,threshold=2.,min_angle=2.,repair_enabled=True,max_outliers=2,max_alternatives=4):
    """One bounded repair pass; only evidence-bound candidate coordinates are tested."""
    xy=xy.copy();k=len(xy);keep=np.ones(k,bool);history=[];repaired=[];dropped=[]
    pixel,depth,_=project_batch(xyz[None],poses[None],params[None]);errors=np.linalg.norm(pixel[0]-xy,axis=1)
    trusted=(errors<=threshold)&(depth[0]>0);bad=np.flatnonzero(~trusted)
    if not len(bad):return dict(xyz=xyz,xy=xy,keep=keep,history=history,repaired=repaired,dropped=dropped,status='unchanged')
    if len(bad)>max_outliers or (k>2 and trusted.sum()<3):
        return dict(xyz=xyz,xy=xy,keep=keep,history=[dict(reason='insufficient_independent_consensus',errors=errors.tolist())],repaired=[],dropped=[],status='rejected_no_consensus')
    if repair_enabled:
        for observation in bad:
            reference=fit_xyz(xyz[None],poses[None],params[None],xy[None],trusted[None])[0] if trusted.sum()>=3 else xyz
            predicted,_,_=project_batch(reference[None],poses[None],params[None])
            candidates=alternatives.get(int(observation),[])[:max_alternatives]
            for candidate in candidates:
                record=dict(observation=int(observation),from_xy=xy[observation].tolist(),to_xy=np.asarray(candidate['xy']).tolist(),source=candidate.get('source'),edge_id=candidate.get('edge_id'),accepted=False)
                allowed_sources={'original_roma_of_unique_snapped_pair','unique_close_observation_with_real_pair_edge','verified_sift_same_track','verified_superpoint_lightglue_same_track'}
                if candidate.get('source') not in allowed_sources or not candidate.get('identity_verified') or candidate.get('track_id',track_id)!=track_id or candidate.get('conflict',False) or candidate.get('existing_track_id',0)>0:
                    record['reason']='unproven_or_conflicting_identity';history.append(record);continue
                if k==2 and not candidate.get('independent_match',False):
                    record['reason']='two_view_repair_requires_independent_match';history.append(record);continue
                if trusted.sum()>=3 and np.linalg.norm(predicted[0,observation]-candidate['xy'])>threshold:
                    record['reason']='candidate_inconsistent_with_consensus';history.append(record);continue
                proposed=xy.copy();proposed[observation]=candidate['xy']
                trial=fit_xyz(reference[None],poses[None],params[None],proposed[None],np.ones((1,k),bool))[0]
                q,z,_=project_batch(trial[None],poses[None],params[None]);after=np.linalg.norm(q[0]-proposed,axis=1)
                angle=maximum_angles(trial[None],centers[None],np.ones((1,k),bool))[0]
                good=np.isfinite(trial).all() and (z>0).all() and angle>=min_angle and after.max()<=threshold and after[observation]<errors[observation]*.75
                record.update(errors_before=errors.tolist(),errors_after=after.tolist(),reason='improved_all_view_geometry' if good else 'replacement_failed_geometry')
                if good:
                    record['accepted']=True;xyz=trial;xy=proposed;errors=after;trusted=(after<=threshold)&(z[0]>0);repaired.append(int(observation));history.append(record);break
                history.append(record)
    bad=np.flatnonzero(~trusted)
    if len(bad):
        if trusted.sum()<3 or len(bad)>max_outliers:
            return dict(xyz=xyz,xy=xy,keep=keep,history=history,repaired=repaired,dropped=[],status='rejected_no_consensus')
        keep=trusted.copy();dropped=bad.tolist()
        xyz=fit_xyz(xyz[None],poses[None],params[None],xy[None],keep[None])[0]
        history.append(dict(reason='replacement_failed_drop_only_outliers',dropped=dropped))
    return dict(xyz=xyz,xy=xy,keep=keep,history=history,repaired=repaired,dropped=dropped,status='repaired' if repaired else ('outliers_removed' if dropped else 'unchanged'))
