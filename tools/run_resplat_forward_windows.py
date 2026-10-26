"""Reproducible16-context native-camera windows; default100 frames, short validation supported."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import subprocess

from GLRMap_GS.tools import resplat_runtime as pilot

ROOT=pilot.ROOT
sha,save,probe=pilot.sha,pilot.save,pilot.probe


def update(out,**values):
    data=json.loads((out/'manifest.json').read_text());data.update(values);save(out/'manifest.json',data)


def prepare(args):
    import numpy as np
    from PIL import Image
    out=args.out
    if out.exists():raise FileExistsError(out)
    assert args.segment_frames>=20 and args.stop>args.start and (args.stop-args.start)%args.segment_frames==0
    (out/'inputs/images').mkdir(parents=True);(out/'source_snapshot').mkdir();(out/'segments').mkdir()
    files=[]
    def copy(source,destination):
        shutil.copy2(source,destination);h=sha(source);assert sha(destination)==h
        files.append(dict(source=str(source),snapshot=str(destination),sha256=h))
    copy(args.cameras,out/'inputs/full_original_cameras.npz')
    copy(args.frame_manifest,out/'inputs/original_frames_manifest.json')
    for source in [Path(__file__),Path(pilot.__file__),pilot.RESPLAT/'scripts/infer_colmap.py',pilot.RESPLAT/'src/model/ply_export.py']:
        copy(source,out/'source_snapshot'/source.name)
    reader_spec=importlib.util.spec_from_file_location('colmap_reader',pilot.GS_ROOT/'scene/colmap_loader.py')
    reader=importlib.util.module_from_spec(reader_spec);reader_spec.loader.exec_module(reader)
    records={r['image']:r for r in json.loads(args.frame_manifest.read_text())}
    with np.load(args.cameras) as archive:
        lookup={str(name):i for i,name in enumerate(archive['image_names'])}
        names=np.array([f'{i:06d}.jpg' for i in range(args.start,args.stop)])
        indices=[lookup[str(name)] for name in names]
        E=archive['extrinsics'][indices].copy();K=archive['intrinsics'][indices].copy()
    assert np.isfinite(E).all() and np.isfinite(K).all() and np.allclose(K,K[0],rtol=0,atol=1e-10)
    selected=[]
    for name in names:
        src=args.images/str(name);assert Image.open(src).size==(666,370)
        copy(src,out/'inputs/images'/str(name));assert files[-1]['sha256']==records[str(name)]['sha256']
        selected.append(records[str(name)])
    np.savez_compressed(out/'inputs/selected_cameras.npz',image_names=names,extrinsics=E,intrinsics=K)
    save(out/'inputs/selected_frames.json',selected)
    reuse={}
    if args.reuse:
        for s in json.loads((args.reuse/'segments.json').read_text()):reuse[tuple(s['global_frames'])]=s
    segments=[]
    for index,first in enumerate(range(0,len(names),args.segment_frames)):
        last=first+args.segment_frames;name=f'segment_{index:02d}';dataset=out/'inputs/segments'/name
        (dataset/'images').mkdir(parents=True);(dataset/'sparse/0').mkdir(parents=True)
        k=K[first]
        (dataset/'sparse/0/cameras.txt').write_text('1 PINHOLE 666 370 '+' '.join(f'{v:.17g}' for v in [k[0,0],k[1,1],k[0,2],k[1,2]])+'\n')
        rows=[]
        for j in range(first,last):
            q=reader.rotmat2qvec(E[j,:3,:3]);assert np.max(abs(reader.qvec2rotmat(q)-E[j,:3,:3]))<1e-10
            rows.append(f'{args.start+j+1} '+' '.join(f'{v:.17g}' for v in [*q,*E[j,:3,3]])+f' 1 {names[j]}\n\n')
            (dataset/'images'/str(names[j])).symlink_to(out/'inputs/images'/str(names[j]))
        (dataset/'sparse/0/images.txt').write_text(''.join(rows));(dataset/'sparse/0/points3D.txt').write_text('# Camera-only container\n')
        frame_range=[args.start+first,args.start+last];context=np.linspace(0,args.segment_frames-1,16,dtype=int)
        destination=out/'segments'/name
        original=reuse.get(tuple(frame_range))
        if original:
            source=Path(original['output']);done=json.loads((source/'complete.json').read_text())
            assert done['passed'] and original['context_local_indices']==context.tolist()
            with np.load(args.reuse/'inputs/selected_cameras.npz') as old:
                old_lookup={str(n):i for i,n in enumerate(old['image_names'])};oi=[old_lookup[str(n)] for n in names[first:last]]
                assert np.array_equal(old['extrinsics'][oi],E[first:last]) and np.array_equal(old['intrinsics'][oi],K[first:last])
            assert sha(source/'gaussians.ply')==done['ply_sha256'] and sha(source/'video.mp4')==done['video_sha256']
            destination.symlink_to(source,target_is_directory=True)
            for path in [source/'complete.json',source/'gaussians.ply',source/'video.mp4',source/'metrics.json']:
                files.append(dict(source=str(path),snapshot=None,sha256=sha(path),reused_read_only=True))
        segments.append(dict(index=index,name=name,dataset=str(dataset),output=str(destination),global_frames=frame_range,
            context_local_indices=context.tolist(),context_image_names=[str(names[first+i]) for i in context],
            reused_from=str(Path(original['output'])) if original else None))
    save(out/'segments.json',segments)
    save(out/'input_manifest.json',dict(files=files,checkpoint=str(pilot.CHECKPOINT),checkpoint_sha256=sha(pilot.CHECKPOINT)))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};save(out/'config.json',config)
    save(out/'manifest.json',dict(status='prepared',label=args.label,date='2026-10-09',global_frames=[args.start,args.stop],
        frames=len(names),fps=10,duration_seconds=len(names)/10,segments=len(segments),reused_segments=sum(s['reused_from'] is not None for s in segments),
        source_video_start_seconds=selected[0]['video_time_nominal'],source_video_end_exclusive_seconds=selected[-1]['video_time_nominal']+.1,
        source_resolution=[666,370],inference_resolution=[960,512],model='dl3dv_16v_540x960',num_context=16,num_refine=2,
        context_selection='Uniform, including both endpoints',camera_optimization=False,camera_smoothing_kernel=1,
        scene_gradient_training_steps=0,composition='Video concatenation only; no Gaussian fusion or seam blending',
        RGB='Original sky and dynamic RGB retained',completed_segments=[]))
    print(f'Prepared {len(segments)} windows; reused {sum(s["reused_from"] is not None for s in segments)}',flush=True)


def verify_inputs(out):
    manifest=json.loads((out/'input_manifest.json').read_text())
    for item in manifest['files']:
        assert sha(item['source'])==item['sha256'],item['source']
        if item['snapshot']:assert sha(item['snapshot'])==item['sha256'],item['snapshot']
    assert sha(pilot.CHECKPOINT)==manifest['checkpoint_sha256']


def run(args):
    import numpy as np
    import torch
    from plyfile import PlyData
    out=args.out;verify_inputs(out)
    sys.path.insert(0,str(pilot.RESPLAT/'scripts'));import infer_colmap as inference
    argv=[str(pilot.RESPLAT/'scripts/infer_colmap.py'),'--model_preset','dl3dv_16v_540x960',
        '--checkpoint',str(pilot.CHECKPOINT),'--images_dir','images','--sparse_dir','sparse/0','--start_frame','0',
        '--frame_distance',str(args.segment_frames),'--context_selection','uniform','--num_context','16','--num_refine','2',
        '--image_shape','512','960','--render_chunk_size','2','--smooth_video_kernel','1','--smooth_video_fps','10',
        '--no_save_images','--save_video','--save_ply']
    original=sys.argv
    try:sys.argv=argv;options=inference.parse_args()
    finally:sys.argv=original
    preset=inference.MODEL_PRESETS[options.model_preset];options.max_resolution=preset['max_resolution'];options.overrides=preset['overrides']
    save(out/'effective_config.json',vars(options))
    segments=json.loads((out/'segments.json').read_text())
    with np.load(out/'inputs/selected_cameras.npz') as archive:E=archive['extrinsics'].copy();K=archive['intrinsics'].copy()
    encoder=decoder=shim=None;completed=[];started=time.time()
    with pilot.gpu_slot():
        os.chdir(pilot.RESPLAT)
        for segment in segments:
            destination=Path(segment['output']);done=destination/'complete.json'
            if done.exists():
                record=json.loads(done.read_text());assert record['passed'] and record['global_frames']==segment['global_frames']
                probe(destination/'video.mp4',args.segment_frames);assert sha(destination/'gaussians.ply')==record['ply_sha256']
                completed.append(segment['index']);update(out,completed_segments=completed);continue
            destination.mkdir(parents=True,exist_ok=True)
            if any(destination.iterdir()):raise FileExistsError(f'Partial window requires review: {destination}')
            first=segment['index']*args.segment_frames;last=first+args.segment_frames
            scene=inference.load_colmap_scene(segment['dataset'],sparse_dir='sparse/0',images_dir='images')
            assert scene['image_names']==[f'{i:06d}.jpg' for i in range(*segment['global_frames'])]
            normalized=K[first:last].copy();normalized[:,0]/=666;normalized[:,1]/=370
            pose_error=float(np.max(abs(np.linalg.inv(scene['c2w'].astype(np.float64))-E[first:last])))
            K_error=float(np.max(abs(scene['intrinsics']-normalized)))
            assert pose_error<64*np.finfo(np.float32).eps*max(1,float(np.max(abs(E[first:last])))) and K_error<1e-6
            context=inference.select_context_views(scene['c2w'],16,'uniform');assert context.tolist()==segment['context_local_indices']
            smooth_error=float(np.max(abs(np.array(inference.render_stabilization_path(scene['c2w'][:,:3],k_size=1))-scene['c2w'][:,:3])))
            assert smooth_error<2e-6
            audit=dict(passed=True,max_E_float32_error=pose_error,max_normalized_K_error=K_error,kernel1_pose_error=smooth_error)
            save(Path(segment['dataset'])/'camera_audit.json',audit)
            update(out,status='running',stage='ReSplat_native_interpolation',current_segment=segment['index'],completed_segments=completed)
            print(f'Window {segment["index"]+1}/{len(segments)}: global {segment["global_frames"]}',flush=True);t=time.time()
            log=out/(segment['name']+'.log')
            with log.open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                encoder,decoder,shim=inference.run_scene(options,segment['dataset'],segment['name'],str(destination),encoder,decoder,shim)
            vertices=PlyData.read(destination/'gaussians.ply')['vertex'].data
            assert len(vertices)>0 and all(np.isfinite(vertices[n]).all() for n in vertices.dtype.names)
            assert sum(n.startswith('f_rest_') for n in vertices.dtype.names)==45
            metrics=json.loads((destination/'metrics.json').read_text());assert len(metrics['per_view'])==args.segment_frames-16
            withheld={scene['image_names'][i] for i in range(args.segment_frames) if i not in set(context.tolist())}
            assert {m['name'] for m in metrics['per_view']}==withheld
            save(done,dict(passed=True,global_frames=segment['global_frames'],gaussians=len(vertices),sh_degree=3,
                ply_coordinate_system='Native export; source-local rotations, not a fused/world-corrected PLY',
                ply_sha256=sha(destination/'gaussians.ply'),video_sha256=sha(destination/'video.mp4'),
                ffprobe=probe(destination/'video.mp4',args.segment_frames),withheld_metrics=metrics['mean'],camera_audit=audit,elapsed_seconds=time.time()-t))
            completed.append(segment['index']);update(out,completed_segments=completed)
            print(f'Complete {len(completed)}/{len(segments)}: {metrics["mean"]}, {time.time()-t:.1f}s',flush=True);torch.cuda.empty_cache()
    os.chdir(ROOT);update(out,status='running',stage='concatenate_and_compare',inference_elapsed_seconds=time.time()-started)
    publish(args);verify_inputs(out);update(out,status='complete',stage='complete',original_inputs_verified=True,elapsed_seconds=time.time()-started)


def publish(args):
    import numpy as np
    out=args.out;segments=json.loads((out/'segments.json').read_text());count=args.stop-args.start
    videos=[Path(s['output'])/'video.mp4' for s in segments]
    for path in videos:probe(path,args.segment_frames)
    concat=out/'concat.txt';concat.write_text(''.join("file '"+str(p)+"'\n" for p in videos))
    (ROOT/'output_video').mkdir(exist_ok=True)
    solo=ROOT/'output_video'/(args.label+'.mp4');compare=ROOT/'output_video'/(args.label+'_vs_original.mp4')
    if not solo.exists():subprocess.run(['ffmpeg','-n','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(solo)],check=True)
    probe(solo,count)
    if not compare.exists():
        left=f"scale=960:512:flags=lanczos,pad=960:552:0:40:black,drawtext=text='Original RGB | global {args.start}-{args.stop-1}':x=10:y=12:fontsize=19:fontcolor=white"
        right="pad=968:552:8:40:black,drawtext=text='ReSplat | 16 inputs per window | fixed SfM interpolation':x=18:y=12:fontsize=19:fontcolor=white"
        subprocess.run(['ffmpeg','-n','-v','error','-threads','2','-framerate','10','-start_number',str(args.start),
            '-i',str(out/'inputs/images/%06d.jpg'),'-i',str(solo),'-filter_complex_threads','2','-filter_complex',
            f'[0:v]{left}[a];[1:v]{right}[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v',str(count),
            '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(compare)],check=True)
    probes={str(p):probe(p,count) for p in [solo,compare]}
    metrics={s['name']:json.loads((Path(s['output'])/'metrics.json').read_text())['mean'] for s in segments}
    mean={k:round(float(np.mean([v[k] for v in metrics.values()])),4) for k in ['psnr','ssim','lpips']}
    save(out/'aggregate_metrics.json',dict(withheld_views=(args.segment_frames-16)*len(segments),mean=mean,per_segment=metrics))
    update(out,published_video=str(solo),published_comparison_video=str(compare),ffprobe=probes,withheld_metrics=mean,
        frame_coverage=dict(first=args.start,last=args.stop-1,count=count,gaps=0,duplicates=0))
    print(f'Published {count/10:g}s: {solo}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','publish'])
    for name in ['images','cameras','frame-manifest','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True)
    p.add_argument('--segment-frames',type=int,default=100)
    p.add_argument('--label',required=True);p.add_argument('--reuse',type=Path)
    args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    try:globals()[args.stage](args)
    except BaseException as error:
        if (args.out/'manifest.json').exists():update(args.out,status='failed',error=repr(error))
        raise
