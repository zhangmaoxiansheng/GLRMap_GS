"""GLOMAP and immutable-overlap BA with explicit batch inputs; no experiment globals."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import numpy as np
from ..io_utils import sha, write_json

def rigid_rotation(matrix):
    u,_,vt=np.linalg.svd(matrix)
    sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    return u@np.diag(sign)@vt


def oriented_similarity(moving_centers,reference_centers,moving_c2w,reference_c2w):
    """Camera rotations resolve roll ambiguity of a nearly straight trajectory."""
    rotation=rigid_rotation(np.einsum('nij,nkj->ik',reference_c2w,moving_c2w))
    rotated=moving_centers@rotation.T
    a=rotated-rotated.mean(0);b=reference_centers-reference_centers.mean(0)
    scale=float(np.sum(a*b)/np.sum(a*a))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid alignment scale')
    translation=reference_centers.mean(0)-scale*rotated.mean(0)
    return scale,rotation,translation


def run_logged(command,folder,label,repo_root):
    log=folder/(label+'.log');done=folder/(label+'.done.json')
    if done.exists():
        if json.loads(done.read_text())['command']!=list(map(str,command)):raise ValueError('Changed completed command: '+label)
        return
    env=os.environ.copy();env.update(QT_QPA_PLATFORM='offscreen',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    command=list(map(str,command));print(label,flush=True)
    start=time.monotonic()
    with log.open('a') as f:
        result=subprocess.run(command,cwd=repo_root,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{label} failed ({result.returncode}): {log}')
    write_json(done,dict(command=command,elapsed_seconds=time.monotonic()-start))


def static_masks(output,index,block):
    """Keep dynamic and sky masks separate; this intersection is only for SfM."""
    from PIL import Image
    first,stop,_=block
    (output/'masks/sfm_static').mkdir(parents=True,exist_ok=True)
    records=[]
    for frame in range(first,stop):
        name=f'{frame:06d}'
        dynamic=(output/'masks')/'vehicle_valid'/(name+'.png')
        sky=output/'sky_all/foreground_valid_hard'/(name+'.png')
        a=np.array(Image.open(dynamic));b=np.array(Image.open(sky))
        if a.shape!=(370,666) or b.shape!=a.shape:raise ValueError(name+' mask shape')
        valid=np.minimum(a,b)
        dest=(output/'masks/sfm_static')/(name+'.jpg.png')
        if dest.exists():
            if not np.array_equal(np.array(Image.open(dest)),valid):raise ValueError('Changed static mask '+name)
        else:Image.fromarray(valid).save(dest)
        records.append(dict(name=name,dynamic_sha256=sha(dynamic),sky_sha256=sha(sky),
            sfm_sha256=sha(dest),valid_fraction=float((valid>0).mean())))
    write_json(output/'sfm'/f'block_{index:02d}'/'static_masks.json',dict(
        method='pixelwise minimum of dynamic-valid and exact-zero DA-V2 hard foreground; no sky morphology',
        records=records))


def features(output,index,block,colmap,repo_root):
    first,stop,anchor_stop=block;folder=output/'sfm'/f'block_{index:02d}'
    names=[f'{n:06d}.jpg' for n in range(first,stop)]
    expected={r['image']:r['sha256'] for r in json.loads((output/'frames_manifest.json').read_text())}
    for name in names:
        if sha(folder/'images'/name)!=expected[name]:raise ValueError(name)
        if not ((output/'masks/sfm_static')/(name+'.png')).exists():raise FileNotFoundError(name+' mask')
    inputs=dict(count=len(names),first=first,stop=stop,
         anchor_stop=anchor_stop,images_sha256={n:expected[n] for n in names},
         masks_sha256={n:sha((output/'masks/sfm_static')/(n+'.png')) for n in names})
    audit_path=folder/'input_audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text())!=inputs:
        raise ValueError('Existing SfM database input images/masks changed')
    if not audit_path.exists():write_json(audit_path,inputs)
    run_logged([colmap,'feature_extractor','--database_path',folder/'database.db',
        '--image_path',folder/'images','--ImageReader.mask_path',(output/'masks/sfm_static'),
        '--ImageReader.single_camera','1','--ImageReader.camera_model','PINHOLE',
        '--ImageReader.camera_params','545.6095244714393,545.6095244714393,333,185',
        '--SiftExtraction.use_gpu','1','--SiftExtraction.gpu_index','0',
        '--SiftExtraction.num_threads','8','--SiftExtraction.max_num_features','8192'],folder,'features',repo_root)


def matching(folder,colmap,repo_root):
    run_logged([colmap,'sequential_matcher','--database_path',folder/'database.db',
        '--SequentialMatching.overlap','20','--SequentialMatching.loop_detection','0',
        '--SiftMatching.use_gpu','1','--SiftMatching.gpu_index','0',
        '--SiftMatching.num_threads','8','--SiftMatching.guided_matching','1'],folder,'matching',repo_root)
    con=sqlite3.connect(f'file:{folder/"database.db"}?mode=ro',uri=True)
    counts={table:con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in ['images','keypoints','matches','two_view_geometries']}
    counts['verified_nonempty_pairs']=con.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>=30').fetchone()[0]
    con.close();write_json(folder/'matching_report.json',counts)


def global_sfm(folder,glomap,repo_root):
    (folder/'global').mkdir(exist_ok=True)
    run_logged([glomap,'mapper','--database_path',folder/'database.db',
        '--image_path',folder/'images','--output_path',folder/'global','--output_format','bin',
        '--constraint_type','ONLY_POINTS','--skip_view_graph_calibration','1',
        '--RelPoseEstimation.max_epipolar_error','1',
        '--GlobalPositioning.use_gpu','1','--GlobalPositioning.gpu_index','0',
        '--BundleAdjustment.use_gpu','1','--BundleAdjustment.gpu_index','0',
        '--BundleAdjustment.optimize_intrinsics','0','--BundleAdjustment.optimize_principal_point','0',
        '--BundleAdjustment.max_num_iterations','300','--ba_iteration_num','4',
        '--retriangulation_iteration_num','2'],folder,'glomap',repo_root)


def anchored(folder,block,refs):
    import pycolmap
    if (folder/'anchored.done.json').exists():return
    first,stop,anchor_stop=block
    model=pycolmap.Reconstruction(str(folder/'global/0'))
    names=[f'{n:06d}.jpg' for n in range(first,stop)]
    lookup={im.name:im for im in model.images.values() if im.has_pose}
    if set(lookup)!=set(names):raise ValueError(f'GLOMAP missing: {sorted(set(names)-set(lookup))}')
    anchors=[f'{n:06d}.jpg' for n in range(first,anchor_stop)]
    E=np.stack([refs[n][0] for n in anchors]);R=E[:,:3,:3];t=E[:,:3,3]
    C=-np.einsum('nji,nj->ni',R,t)
    Cm=np.stack([lookup[n].projection_center() for n in anchors])
    Rm=np.stack([lookup[n].cam_from_world().rotation.matrix().T for n in anchors])
    scale,rotation,translation=oriented_similarity(Cm,C,Rm,R.transpose(0,2,1))
    aligned=scale*(Cm@rotation.T)+translation
    initial_error=np.linalg.norm(aligned-C,axis=1)
    model.transform(pycolmap.Sim3d(scale,pycolmap.Rotation3d(rotation),translation))
    initial_rotation=[]
    for n in anchors:
        A=lookup[n].cam_from_world().rotation.matrix();B=refs[n][0][:3,:3]
        initial_rotation.append(float(np.degrees(np.arccos(np.clip((np.trace(A@B.T)-1)/2,-1,1)))))
    step=float(np.median(np.linalg.norm(np.diff(C,axis=0),axis=1)))
    init_report=dict(scale=scale,rotation=rotation.tolist(),translation=translation.tolist(),
         anchor_center_rmse=float(np.sqrt(np.mean(initial_error**2))),
         anchor_center_max=float(initial_error.max()),reference_median_step=step,
         anchor_rotation_max_deg=max(initial_rotation),anchor_count=len(anchors))
    write_json(folder/'initial_alignment.json',init_report)
    if init_report['anchor_center_rmse']>step or max(initial_rotation)>2:
        raise ValueError('Global initialization does not agree sufficiently with immutable overlap; inspect initial_alignment.json')
    config=pycolmap.BundleAdjustmentConfig()
    for im in lookup.values():config.add_image(im.image_id)
    for camera in model.cameras.values():
        expected=refs[anchors[0]][1]
        if not np.allclose(camera.calibration_matrix(),expected,rtol=0,atol=1e-10):raise ValueError('GLOMAP changed K')
        config.set_constant_cam_intrinsics(camera.camera_id)
    for n in anchors:
        im=lookup[n]
        model.frames[im.frame_id].set_cam_from_world(im.camera_id,pycolmap.Rigid3d(refs[n][0][:3]))
        config.set_constant_rig_from_world_pose(im.frame_id)
    before_fixed={n:lookup[n].cam_from_world().matrix().copy() for n in anchors}
    options=pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length=False;options.refine_principal_point=False;options.refine_extra_params=False
    options.refine_sensor_from_rig=False
    options.ceres.loss_function_type=pycolmap.LossFunctionType.SOFT_L1
    options.ceres.loss_function_scale=1.
    options.ceres.use_gpu=False  # This installed pycolmap wheel is CPU-only; glomap above uses GPU.
    options.ceres.solver_options.max_num_iterations=150
    options.ceres.solver_options.num_threads=8
    options.ceres.solver_options.function_tolerance=1e-6
    reports=[]
    for round_index in range(2):
        adjuster=pycolmap.create_default_bundle_adjuster(options,config,model)
        summary=adjuster.solve()
        print(summary.brief_report(),flush=True)
        if not summary.is_solution_usable():raise RuntimeError('Anchored BA failed')
        manager=pycolmap.ObservationManager(model)
        removed=manager.filter_all_points3D(4. if round_index==0 else 2.,1.)
        reports.append(dict(round=round_index,summary=summary.todict(),filtered_observations=removed))
    drift=max(float(np.max(abs(lookup[n].cam_from_world().matrix()-before_fixed[n]))) for n in anchors)
    if drift!=0:raise ValueError(f'Fixed anchor moved in BA: {drift}')
    dest=folder/'anchored/sparse/0';dest.mkdir(parents=True,exist_ok=True)
    model.write(str(dest))
    # JSON conversion is deliberately limited to the human-readable solver report.
    write_json(folder/'anchored.done.json',dict(fixed_pose_drift=drift,anchors=len(anchors),
         rounds=[dict(round=r['round'],filtered_observations=r['filtered_observations'],summary=str(r['summary'])) for r in reports]))


def audit(folder,block,refs):
    import pycolmap
    first,stop,anchor_stop=block
    model=pycolmap.Reconstruction(str(folder/'anchored/sparse/0'))
    names=[f'{n:06d}.jpg' for n in range(first,stop)]
    lookup={im.name:im for im in model.images.values() if im.has_pose}
    if set(lookup)!=set(names):raise ValueError('Incomplete registered model')
    E=np.repeat(np.eye(4)[None],len(names),axis=0)
    K=np.empty((len(names),3,3));observations=[];means=[];p95s=[];behind=[]
    for i,n in enumerate(names):
        im=lookup[n];E[i,:3]=im.cam_from_world().matrix();K[i]=model.cameras[im.camera_id].calibration_matrix()
        obs=[p for p in im.points2D if p.has_point3D()]
        xyz=np.array([model.points3D[p.point3D_id].xyz for p in obs]);xy=np.array([p.xy for p in obs])
        if not len(obs):raise ValueError(f'No observations: {n}')
        pc=xyz@E[i,:3,:3].T+E[i,:3,3];uv=pc@K[i].T
        err=np.linalg.norm(uv[:,:2]/uv[:,2:]-xy,axis=1)
        observations.append(len(obs));means.append(float(err.mean()));p95s.append(float(np.percentile(err,95)));behind.append(int((pc[:,2]<=0).sum()))
    anchor_diff=max(float(np.max(abs(E[i]-refs[n][0]))) for i,n in enumerate(names[:anchor_stop-first]))
    k_diff=max(float(np.max(abs(K[i]-refs[n][1]))) for i,n in enumerate(names[:anchor_stop-first]))
    C=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3]);steps=np.linalg.norm(np.diff(C,axis=0),axis=1)
    local=np.array([np.median(steps[max(0,i-10):min(len(steps),i+11)]) for i in range(len(steps))])
    rotations=E[:,:3,:3];cos=(np.einsum('nij,nij->n',rotations[1:],rotations[:-1])-1)/2
    angular=np.degrees(np.arccos(np.clip(cos,-1,1)))
    bad_step=np.flatnonzero(steps>4*np.maximum(local,np.median(steps)*.1))
    bad_rotation=np.flatnonzero(angular>5)
    report=dict(first=first,stop=stop,registered=len(names),points=model.num_points3D(),
         fixed_anchor_pose_max_difference=anchor_diff,fixed_K_max_difference=k_diff,
         mean_reprojection=float(model.compute_mean_reprojection_error()),minimum_observations=min(observations),
         max_per_frame_mean_reprojection=max(means),max_per_frame_p95_reprojection=max(p95s),
         behind_camera_observations=sum(behind),bad_step_frames=(bad_step+first+1).tolist(),
         bad_rotation_frames=(bad_rotation+first+1).tolist(),step_quantiles=np.percentile(steps,[0,50,95,100]).tolist(),
         per_frame=[dict(name=n,observations=o,mean=m,p95=p) for n,o,m,p in zip(names,observations,means,p95s)])
    passed=(anchor_diff<1e-10 and k_diff<1e-10 and min(observations)>=30 and max(means)<=2 and
            max(p95s)<=4.1 and not sum(behind) and not len(bad_step) and not len(bad_rotation))
    report['passed']=bool(passed);write_json(folder/'audit.json',report)
    np.savez_compressed(folder/'trajectory_diagnostics.npz',centers=C,steps=steps,rotation_step_deg=angular)
    if not passed:raise ValueError(f'SfM quality gate failed: {folder/"audit.json"}')
    np.savez_compressed(folder/'cameras.npz',extrinsics=E,intrinsics=K,image_names=np.array(names))
    print(json.dumps({k:v for k,v in report.items() if k!='per_frame'},indent=2),flush=True)


def run_block(output, index, block, reference_cameras, *, repo_root, colmap, glomap):
    """Use prepared output/sfm/block_XX/images and masks; block=(first, stop, anchor_stop)."""
    output, repo_root = Path(output), Path(repo_root)
    folder = output/'sfm'/f'block_{index:02d}'
    static_masks(output, index, block)
    features(output, index, block, colmap, repo_root)
    matching(folder, colmap, repo_root)
    global_sfm(folder, glomap, repo_root)
    with np.load(reference_cameras) as arrays:
        refs = {str(name):(pose.copy(), intrinsics.copy()) for name,pose,intrinsics in
                zip(arrays['image_names'], arrays['extrinsics'], arrays['intrinsics'])}
    anchored(folder, block, refs)
    audit(folder, block, refs)
