"""Native ReSplat world conversion, source-pixel removal and fixed-road initializer."""
from pathlib import Path
import shutil
import cv2
import numpy as np
from .state import ROOT, GS_ROOT, cameras, code_files, gpu, verify, write
from ..io_utils import sha


def source_masks(dynamic_masks, covered_road_masks, names):
    """One native export block per context; any ignored source pixel removes its latent cell."""
    dynamic, roads, per_source = [], [], []
    for valid, covered, name in zip(dynamic_masks, covered_road_masks, names):
        d = cv2.resize((valid == 0).astype(np.float32), (240, 128), interpolation=cv2.INTER_AREA) > 0
        r = cv2.resize((covered > 127).astype(np.float32), (240, 128), interpolation=cv2.INTER_AREA) > 0
        dynamic.append(d.ravel())
        roads.append(r.ravel())
        per_source.append(dict(image=str(name), dynamic=int(d.sum()), road=int(r.sum())))
    return np.concatenate(dynamic), np.concatenate(roads), per_source


def road_targets(path, names, shape, coverage_folder):
    """The curve-supported replacement mask and positive camera-Z targets must agree."""
    with np.load(path) as arrays:
        lookup = {str(name): i for i, name in enumerate(arrays['image_names'])}
        indices = [lookup[name] for name in names]
        depth, mask = arrays['depth'][indices].copy(), arrays['correction_mask'][indices].copy()
    if depth.shape != (len(names), *shape) or mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError('Road depth/mask shape or dtype mismatch')
    if not np.isfinite(depth[mask]).all() or (depth[mask] <= 0).any():
        raise ValueError('Road targets must contain positive, finite camera-Z')
    for name, supported in zip(names, mask):
        replacement = cv2.imread(str(Path(coverage_folder)/f'{Path(name).stem}.png'), 0)
        if replacement is None or not np.array_equal(replacement > 127, supported):
            raise ValueError('road_masks must be actual curve coverage, not semantic road: '+name)
    return depth, mask


def prepare(c,fingerprint):
    from plyfile import PlyData,PlyElement
    import torch
    # Read the standalone camera utilities without importing CUDA-only Scene.
    import importlib.util
    spec=importlib.util.spec_from_file_location('pipeline_colmap_loader',GS_ROOT/'scene/colmap_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    rotmat2qvec,qvec2rotmat=loader.rotmat2qvec,loader.qvec2rotmat
    from .conversion import conversion
    from .rendering import gpu_model,camera_render
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'input_manifest.json').exists():verify(out,c,fingerprint)
    E,K,names=cameras(c);n=len(names)
    for d in ['inputs/road','inputs/vehicle_valid','inputs/sky_diagnostic','data/images','data/sparse/0']:(out/d).mkdir(parents=True,exist_ok=True)
    source_hashes={};snapshots={}
    def capture(src,dest=None):
        src=Path(src);h=sha(src);source_hashes[str(src)]=h
        if dest:
            dst=out/dest
            if dst.exists():
                if sha(dst)!=h:raise ValueError('Existing snapshot differs: '+str(dst))
            else:shutil.copy2(src,dst)
            snapshots[str(dst)]=h
    for key in ['cameras','reference_cameras','resplat_ply','resplat_video','road_ply','road_constraints','road_depth']:
        if key in c:capture(c[key])
    for name in names:
        capture(Path(c['images'])/name,'data/images/'+name)
        for key,folder in [('road_masks','road'),('dynamic_masks','vehicle_valid'),('sky_diagnostic_masks','sky_diagnostic')]:
            if key in c:capture(Path(c[key])/(Path(name).stem+'.png'),f'inputs/{folder}/{Path(name).stem}.png')
    # Verify image crop and centered projection required by Graphdeco cameras.
    image=cv2.imread(str(out/'data/images'/names[0]));h,w=image.shape[:2]
    if (w%2 or h%2) or abs(K[0,0,2]-w/2)>1e-6 or abs(K[0,1,2]-h/2)>1e-6:
        raise ValueError('This Graphdeco pipeline requires even image size and centered principal point')
    rows=[]
    for i,(name,e) in enumerate(zip(names,E)):
        img=cv2.imread(str(out/'data/images'/name));assert img.shape[:2]==(h,w)
        for kind in ['road','vehicle_valid']:
            mask=cv2.imread(str(out/f'inputs/{kind}/{Path(name).stem}.png'),0)
            if mask is None or mask.shape!=(h,w) or not set(np.unique(mask)).issubset({0,255}):raise ValueError('Mask/crop convention failed: '+name)
        q=rotmat2qvec(e[:3,:3]);assert abs(qvec2rotmat(q)-e[:3,:3]).max()<1e-10
        rows.append(f'{i+1} '+' '.join(f'{x:.17g}' for x in [*q,*e[:3,3]])+f' 1 {name}\n\n')
    sparse=out/'data/sparse/0'
    (sparse/'images.txt').write_text(''.join(rows))
    (sparse/'cameras.txt').write_text(f'1 PINHOLE {w} {h} '+' '.join(f'{v:.17g}' for v in [K[0,0,0],K[0,1,1],K[0,0,2],K[0,1,2]])+'\n')
    (sparse/'points3D.txt').write_text('# Camera-only container; initialization uses explicitGS PLY\n')
    dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','u1'),('green','u1'),('blue','u1')]
    PlyData([PlyElement.describe(np.empty(0,dtype=dtype),'vertex')],text=False).write(sparse/'points3D.ply')
    np.savez_compressed(out/'data/input_colmap_cameras.npz',extrinsics=E,intrinsics=K,image_names=names)
    ctx=np.array(c['context_indices']);assert len(ctx)==16 and len(np.unique(ctx))==16 and ctx.min()>=0 and ctx.max()<n
    depth,mask=road_targets(c['road_depth'],names,(h,w),out/'inputs/road')
    np.savez_compressed(out/'inputs/road_depth.npz',depth=depth,correction_mask=mask,image_names=names)
    raw=PlyData.read(c['resplat_ply'],mmap='r')['vertex'].data.copy()
    assert len(raw)==16*128*240 and all(np.isfinite(raw[k]).all() for k in raw.dtype.names)
    # Match native exporter float32 pose normalization exactly, including
    # source-view covariance rotations and world SH3 transformation.
    scene=dict(c2w=np.linalg.inv(E.astype(np.float32)).astype(np.float32))
    with gpu(c):
        world,centered,pivot,conversion_audit=conversion(raw,scene,ctx)
        wm,cm=gpu_model(world),gpu_model(centered);checks=[]
        cap=cv2.VideoCapture(c['resplat_video']) if c.get('resplat_video') else None
        for i in sorted(set([0,n//2,n-1])):
            a,_=camera_render(wm,E[i],K[i],w,h);b,_=camera_render(cm,E[i]@pivot,K[i],w,h)
            psnr=float(-10*np.log10(max(float(np.mean((a-b)**2)),1e-12)))
            delta=float(abs(a-b).mean()*255);assert psnr>40 and delta<.3,(psnr,delta)
            item=dict(local_frame=i,world_vs_centered_PSNR=psnr,mean_gray_delta=delta)
            if cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,ref=cap.read();assert ok
                rh,rw=ref.shape[:2];hk=K[i].copy();hk[0]*=rw/w;hk[1]*=rh/h
                rgb,_=camera_render(wm,E[i],hk,rw,rh);truth=cv2.cvtColor(ref,cv2.COLOR_BGR2RGB)/255.
                item['PSNR_vs_native_encoded_video']=float(-10*np.log10(max(float(np.mean((rgb-truth)**2)),1e-12)))
                assert item['PSNR_vs_native_encoded_video']>28,item
            checks.append(item)
        if cap:cap.release()
        conversion_audit.update(passed=True,checks=checks)
        # Do not retain the two audit models after releasing GPU ownership.
        del wm,cm
        torch.cuda.empty_cache()
    PlyData([PlyElement.describe(world,'vertex')],text=False).write(out/'resplat_colmap_world.ply')
    write(out/'world_conversion_audit.json',conversion_audit)
    with np.load(c['road_constraints']) as z:road_count=int(z['count'])-int(z['road_start'])
    accepted=PlyData.read(c['road_ply'],mmap='r')['vertex'].data
    first=int(c.get('road_start',len(accepted)-road_count));assert 0<=first and first+road_count<=len(accepted)
    road=accepted[first:first+road_count].copy()
    padded=np.zeros(road_count,dtype=world.dtype)
    for field in road.dtype.names:
        if field not in world.dtype.names:
            if field not in ('nx','ny','nz'):raise ValueError('Unsupported roadPLY attribute: '+field)
        else:padded[field]=road[field]
    dynamic_masks=[cv2.imread(str(out/f'inputs/vehicle_valid/{Path(names[i]).stem}.png'),0) for i in ctx]
    road_masks=[cv2.imread(str(out/f'inputs/road/{Path(names[i]).stem}.png'),0) for i in ctx]
    dynamic,roads,per_source=source_masks(dynamic_masks,road_masks,names[ctx])
    keep=~(dynamic|roads);start=int(keep.sum());initial=np.concatenate([world[keep],padded])
    assert not dynamic[keep].any() and not roads[keep].any()
    PlyData([PlyElement.describe(initial,'vertex')],text=False).write(out/'static_road_init.ply')
    np.savez_compressed(out/'initialization_lineage.npz',resplat_kept_rows=np.flatnonzero(keep),
        dynamic_removed_rows=np.flatnonzero(dynamic),road_removed_rows=np.flatnonzero(roads),
        road_start=start,road_count=road_count,total_count=len(initial),context_indices=ctx,
        accepted_road_source_rows=np.arange(first,first+road_count))
    for field in road.dtype.names:
        if field in initial.dtype.names:assert road[field].tobytes()==initial[field][start:].tobytes()
    saved=PlyData.read(out/'static_road_init.ply',mmap='r')['vertex'].data
    assert all(saved[k][start:].tobytes()==padded[k].tobytes() for k in padded.dtype.names)
    write(out/'initialization_audit.json',dict(passed=True,views=n,road_count=road_count,background_count=start,total_count=len(initial),
        removed_dynamic_count=int(dynamic.sum()),removed_road_count=int(roads.sum()),removed_union_count=int((dynamic|roads).sum()),
        all_original_road_attributes_preserved=True,dynamic_source_rows_remaining=0,per_source=per_source,
        initializer_sha256=sha(out/'static_road_init.ply'),source_mask_not_dilated=True,
        warning='Source classification does not prove every learned-offset ghost removed'))
    files=code_files()
    if any(sha(path)!=digest for path,digest in source_hashes.items()):
        raise ValueError('GS inputs changed while preparing initialization')
    write(out/'input_manifest.json',dict(config_fingerprint=fingerprint,source_hashes=source_hashes,snapshot_hashes=snapshots,
        code_hashes={str(p):sha(p) for p in files},width=w,height=h,views=n,phase_steps=n*c['steps_per_view_per_phase'],
        real_sky_RGB_supervised=True,road_depth='RoMa fitted surface support only',road_freeze='all6 tensors',
        Gaussian_densification=False,Gaussian_pruning=False))
    for p in files:
        relative=Path('GLRMap_GS')/p.relative_to(ROOT) if ROOT in p.parents else Path('gs_backend')/p.relative_to(GS_ROOT)
        dest=out/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    print('Initializer ready:',len(initial),'GS; roads:',road_count,'dynamic removed:',int(dynamic.sum()),flush=True)
