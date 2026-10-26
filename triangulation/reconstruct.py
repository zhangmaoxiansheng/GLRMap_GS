"""Triangulate real shared Tracks; preserve observation evidence and strict two-view gates."""
from collections import Counter
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
from ..io_utils import save_npz, export_ply, write_json
from ..sfm.geometry import rays, fixed_camera_parameters
from ..matching.filters import shared_filter_settings
from .geometry import robust_initial, project_batch, fit_xyz, dlt_batch, maximum_angles, repair_track

def percentiles(x):
    x = np.asarray(x); finite = x[np.isfinite(x)]
    return np.percentile(finite,[50,90,99]).tolist() if len(finite) else []


def reconstruct(root, policy, built):
    mode = 'shared'
    start = time.monotonic(); dest = root/f'{policy}_{mode}'; dest.mkdir()
    images = built['images']; model = built['model']; tracks = built['tracks']
    filters=shared_filter_settings({'filter_settings':built.get('filter_settings',{})})
    fit_threshold=filters['multiview_max_reprojection_px']
    poses=np.stack([im.cam_from_world().matrix() for im in images]); centers=np.array([im.projection_center() for im in images])
    params=np.array([fixed_camera_parameters(model.cameras[im.camera_id]) for im in images])
    ni=built['node_image']; nxy=built['node_xy']; nr=np.empty((len(ni),3))
    for i in np.unique(ni):
        im=images[i]
        ids=np.flatnonzero(ni==i);nr[ids]=rays(model.cameras[im.camera_id],nxy[ids])
    tids=np.arange(len(tracks)); lens=np.array([len(t) for t in tracks]); kept=[]; rejects=Counter(); anomalies=[]
    for k in sorted(set(lens)):
        group=tids[lens==k]
        for first in range(0,len(group),8192):
            ids=group[first:first+8192];nodes=np.array([tracks[t] for t in ids]); im=ni[nodes]; xy=nxy[nodes]; p=poses[im]; par=params[im]; cc=centers[im]; rr=nr[nodes]
            xyz,consensus,error=robust_initial(p,par,xy,rr,cc,threshold=fit_threshold)
            keep=np.ones((len(ids),k),bool)
            for row in np.flatnonzero(~consensus.all(1)):
                alts={obs:built.get('alternatives',{}).get((int(ids[row]),obs),[]) for obs in range(k)}
                result=repair_track(xyz[row],p[row],par[row],xy[row],rr[row],cc[row],alts,int(ids[row]),threshold=fit_threshold,min_angle=filters['multiview_min_angle_deg'],repair_enabled=built.get('repair_enabled',False),max_outliers=2)
                xyz[row]=result['xyz'];xy[row]=result['xy'];keep[row]=result['keep']
                if result['status'].startswith('rejected'):keep[row]=False
                anomalies.append(dict(track_id=int(ids[row]),status=result['status'],errors_before=error[row].tolist(),removed=result['dropped'],repaired=result['repaired'],history=result['history']))
            pix,depth,_=project_batch(xyz,p,par);error=np.linalg.norm(pix-xy,axis=2);counts=keep.sum(1)
            maxerr=np.max(np.where(keep,error,0),1);angles=maximum_angles(xyz,cc,keep)
            # Use RETAINED count after repair: a 3-view track reduced to 2 views
            # must satisfy the stricter two-view geometry/evidence thresholds.
            min_angle=np.where(counts==2,filters['two_view_min_angle_deg'],filters['multiview_min_angle_deg'])
            max_error=np.where(counts==2,filters['two_view_max_reprojection_px'],fit_threshold)
            good=np.isfinite(xyz).all(1)&np.isfinite(error).all(1)&(counts>=2)&((depth>0)|~keep).all(1)&(maxerr<=max_error)&(angles>=min_angle)
            rejects['geometric_rejected']+=int((~good).sum())
            # The retained star must include the original source observation.
            rejects['source_observation_rejected']+=int((good&~keep[:,0]).sum());good&=keep[:,0]
            evidence=(built['node_conf'][nodes]>=filters['two_view_min_overlap'])&(built['node_cycle'][nodes]<=filters['two_view_max_cycle_px'])&(built['node_sigma'][nodes]<=filters['two_view_max_sigma_px'])&built['node_joint'][nodes]
            b_pass=(evidence|~keep).all(1)
            rejects['two_view_evidence_rejected']+=int((good&(counts==2)&~b_pass).sum());good&=(counts>=3)|b_pass
            for row in np.flatnonzero(good):
                active=np.flatnonzero(keep[row]); kept.append((int(ids[row]),xyz[row].copy(),im[row,active],xy[row,active],error[row,active],float(angles[row])))
        print(f'{policy}/{mode}: {k}-view proposed tracks {len(group)}',flush=True)
    xyz=np.array([r[1] for r in kept],np.float32).reshape(-1,3);length=np.array([len(r[2]) for r in kept],np.int64); rgb=np.zeros((len(kept),3),np.uint8)
    if kept:
        source = Path(json.loads((root/'manifest.json').read_text())['source'])
        source_images = np.array([r[2][0] for r in kept])
        source_pixels = np.floor(np.stack([r[3][0] for r in kept])).astype(int)
        for i in np.unique(source_images):
            rows = np.flatnonzero(source_images == i)
            with Image.open(source/'images'/images[i].name) as photo:
                pixel = np.asarray(photo.convert('RGB'))
            x, y = source_pixels[rows].T
            rgb[rows] = pixel[y, x]
    offsets=np.r_[0,np.cumsum(length)]; oi=np.concatenate([r[2] for r in kept]) if kept else np.empty(0,np.int16)
    oxy=np.concatenate([r[3] for r in kept]) if kept else np.empty((0,2)); err=np.concatenate([r[4] for r in kept]) if kept else np.empty(0)
    loo=np.full(len(kept),np.nan); stability=np.full(len(kept),np.nan); all_loo=np.full(len(err),np.nan); all_stability=np.full(len(err),np.nan); all_remaining_angle=np.full(len(err),np.nan)
    compute_holdouts=built.get('compute_holdouts',True)
    for k in sorted(set(length[length>=3])) if compute_holdouts else []:
        group=np.flatnonzero(length==k)
        for first in range(0,len(group),4096):
            ids=group[first:first+4096]; positions=offsets[ids,None]+np.arange(k); im=oi[positions]; xy=oxy[positions]; pp=poses[im]; par=params[im]
            rr=np.empty((len(ids),k,3))
            for i,image in enumerate(images):
                sel=im==i;rr[sel]=rays(model.cameras[image.camera_id],xy[sel])
            for held in range(k):
                other=np.arange(k)!=held; p=pp[:,other]; q=par[:,other]; r=rr[:,other]; observation=xy[:,other]
                ref=fit_xyz(dlt_batch(p,r),p,q,observation,np.ones((len(ids),k-1),bool))
                pixel,depth,_=project_batch(ref,pp,par); valid=np.isfinite(ref).all(1)&(depth>0).all(1)
                e=np.linalg.norm(pixel[:,held]-xy[:,held],axis=1);e[~valid]=np.inf
                relative=np.linalg.norm(ref-xyz[ids],axis=1)/np.maximum(np.median(np.linalg.norm(xyz[ids,None]-centers[im],axis=2),axis=1),1e-12);relative[~valid]=np.inf
                all_loo[positions[:,held]]=e;all_stability[positions[:,held]]=relative
                all_remaining_angle[positions[:,held]]=maximum_angles(ref,centers[im[:,other]],np.ones((len(ids),k-1),bool))
                select=(np.array([kept[t][0] for t in ids])%k)==held
                loo[ids[select]]=e[select];stability[ids[select]]=relative[select]
    quality=np.where((length>=3)&(np.array([max(r[4]) for r in kept])<=1)&(loo<=2)&(stability<=.02),1,2).astype(np.uint8)
    if not compute_holdouts:quality[:]=0  # No A/B validation claim without holdout evidence.
    means=np.array([np.mean(r[4]) for r in kept]); maximum=np.array([np.max(r[4]) for r in kept]);angles=np.array([r[5] for r in kept]); tids=np.array([r[0] for r in kept],np.int64)
    save_npz(dest/'new_points.npz',xyz=xyz,rgb=rgb,quality=quality,track_ids=tids,track_length=length,reprojection=means,max_reprojection=maximum,angle=angles,leave_one_out_px=loo,relative_xyz_stability=stability,source=np.full(len(kept),3,np.uint8))
    save_npz(dest/'point_observations.npz',offsets=offsets,image_index=oi,xy=oxy,errors=err)
    save_npz(dest/'all_holdouts.npz',offsets=offsets,errors=all_loo,relative_xyz_change=all_stability,remaining_angle_deg=all_remaining_angle)
    save_npz(dest/'tracks.npz',offsets=built['offsets'],image_index=ni[built['nodes']],xy_original=nxy[built['nodes']],track_keys=built.get('track_keys',np.empty((0,2),np.int32)),observation_evidence=built.get('observation_evidence',np.empty((0,2),np.int32)))
    export_ply(dest/'new_points.ply',xyz,rgb)
    write_json(dest/'track_report.json',built['report']);write_json(dest/'anomalies.json',anomalies)
    multi=length>=3; hist=dict(Counter(map(int,length)))
    finite_loo=np.isfinite(all_loo)&np.repeat(multi,length)
    requested=int(length[multi].sum()) if compute_holdouts else 0; pass_count=int((all_loo[finite_loo]<=2).sum())
    metric=dict(input_pairs=built['report']['input_pairs'],mode=mode,accepted_new_points=len(kept),filter_settings=filters,
                accepted_two_views=int((length==2).sum()),accepted_at_least_3_views=int(multi.sum()),accepted_at_least_4_views=int((length>=4).sum()),track_length_histogram=hist,
                total_retained_observations=len(err),grade_A=int((quality==1).sum()),grade_B=int((quality==2).sum()),rejections=dict(rejects),
                reprojection_mean_px=float(means.mean()) if len(means) else None,reprojection_percentiles_px=percentiles(means),
                multiview_observation_reprojection_mean_px=float(err[np.repeat(multi,length)].mean()) if multi.any() else None,
                multiview_observation_reprojection_percentiles_px=percentiles(err[np.repeat(multi,length)]),
                leave_one_out_percentiles_px=percentiles(loo),all_holdout_percentiles_px=percentiles(all_loo[np.repeat(multi,length)]),
                holdout_computed=compute_holdouts,holdout_observations=requested,holdout_failed_fits=requested-int(finite_loo.sum()),holdout_at_most_2px_fraction=pass_count/max(1,requested) if compute_holdouts else None,
                all_holdout_relative_xyz_percentiles=percentiles(all_stability[np.repeat(multi,length)]),
                angle_percentiles_deg=percentiles(angles),build_seconds=built['report']['seconds'],triangulation_and_validation_seconds=time.monotonic()-start,
                camera_poses_unchanged=True,intrinsics_unchanged=True,
                snap_enabled=bool(built.get('snap_enabled',False)),
                limitation='Internal consistency, no independent 3D GT. Different reference stars may describe overlapping surfaces and are not spatially deduplicated. '+('Identity-preserving paired SIFT/SP Snap with real Raw repair candidates.' if built.get('snap_enabled',False) else 'No sparse anchors or Snap; source identity reuse is isolated.'))
    write_json(dest/'metrics.json',metric)
    return metric
