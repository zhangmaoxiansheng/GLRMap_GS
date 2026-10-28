"""Current fixed-road GS native/lateral/lane-change rendering and offline video report."""
import argparse
import json
from pathlib import Path
from ..gs.state import effective, verify, gpu, write
from ..io_utils import sha, save_npz
from .trajectory import lateral as lateral_camera
from .video import encode, verify_video
from .html import media_report


def review(c,fingerprint):
    import cv2
    import numpy as np
    import torch
    from plyfile import PlyData
    from ..gs.rendering import gpu_model,camera_render

    out=Path(c['output']);manifest=verify(out,c,fingerprint);w,h=manifest['width'],manifest['height']
    r=json.loads((out/'training_report.json').read_text());assert sha(r['final_PLY'])==r['final_sha256']
    (out/'review').mkdir(exist_ok=True)
    def label(rgb,title):
        image=np.zeros((h+40,w,3),np.uint8);image[40:]=np.round(np.clip(rgb,0,1)*255).astype(np.uint8)
        cv2.putText(image,title,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1,cv2.LINE_AA);return image
    with gpu(c):
        torch.set_num_threads(8)
        paths=[out/'resplat_colmap_world.ply',out/'static_road_init.ply',Path(r['final_PLY'])]
        models=[gpu_model(PlyData.read(p,mmap='r')['vertex'].data) for p in paths]
        with np.load(out/'data/input_colmap_cameras.npz') as z:E,K,names=z['extrinsics'],z['intrinsics'],z['image_names']
        n=len(names);fps=int(c['fps']);comparison=out/'original_resplat_init_trained_forward.mp4';single=out/'trained_forward.mp4'
        writers=[encode(comparison,w*4,h+40,fps),encode(single,w,h,fps)];metrics=[]
        try:
            for i,(e,k) in enumerate(zip(E,K)):
                truth=cv2.cvtColor(cv2.imread(str(out/'data/images'/names[i])),cv2.COLOR_BGR2RGB)/255.
                valid=cv2.imread(str(out/f'inputs/vehicle_valid/{Path(names[i]).stem}.png'),0)>127
                road=cv2.imread(str(out/f'inputs/road/{Path(names[i]).stem}.png'),0)>127
                sp=out/f'inputs/sky_diagnostic/{Path(names[i]).stem}.png'
                sky=cv2.imread(str(sp),0)==0 if sp.exists() else np.zeros((h,w),bool)
                images=[camera_render(model,e,k,w,h)[0] for model in models]
                row=np.hstack([label(truth,f'Original / {names[i]}'),label(images[0],'Raw ReSplat / vehicles'),
                    label(images[1],'Static initialization / fixed road'),label(images[2],f'After{r["iterations"]} steps')])
                writers[0].stdin.write(row.tobytes());writers[1].stdin.write(np.round(images[-1]*255).astype(np.uint8).tobytes())
                record=dict(image=names[i])
                for tag,rgb in zip(['raw','init','trained'],images):
                    for region,mask in [('static',valid),('road',road&valid),('sky',sky&valid),('nonroad',valid&~road&~sky)]:
                        if mask.any():record[tag+'_'+region+'_L1']=float(abs(rgb-truth)[mask].mean()*255)
                metrics.append(record)
                if i%10==0 or i==n-1:cv2.imwrite(str(out/f'review/native_{i:06d}.jpg'),cv2.cvtColor(row,cv2.COLOR_RGB2BGR))
        finally:
            for writer in writers:writer.stdin.close();assert writer.wait()==0
        lateral=out/'init_vs_trained_lateral.mp4';writer=encode(lateral,w*2,h+40,24);audits=[];lateral_frames=0
        try:
            for anchor in c.get('lateral_anchors',[30,60,85]):
                if not 0<=anchor<n:raise ValueError('Lateral anchor outside segment')
                trajectory,audit=lateral_camera(E,K,c.get('lateral_amplitude_world',.06891966864338281),anchor=anchor,frames=96);audits.append(audit)
                for j,(e,k) in enumerate(zip(trajectory['extrinsics'],trajectory['intrinsics'])):
                    images=[camera_render(model,e,k,w,h)[0] for model in models[1:]]
                    row=np.hstack([label(images[0],f'Init / lateral at local{anchor}'),label(images[1],'Trained / same pose')])
                    writer.stdin.write(row.tobytes());lateral_frames+=1
                    if j in [0,24,72,95]:cv2.imwrite(str(out/f'review/lateral_{anchor:04d}_{j:03d}.jpg'),cv2.cvtColor(row,cv2.COLOR_RGB2BGR))
        finally:writer.stdin.close();assert writer.wait()==0
        keys=set().union(*(row.keys() for row in metrics))-{'image'}
        means={k:float(np.mean([row[k] for row in metrics if k in row])) for k in sorted(keys)}
        write(out/'render_metrics.json',dict(means=means,frames=metrics,lateral_camera_audits=audits,
            caveat='Training-view fitting metrics, no staticGT in removed-vehicle pixels or lateralGT',
            renderers='Common gsplatAA eps2d0.1 comparison; GraphdecoAAoff training'))
        del models
        torch.cuda.empty_cache()
    videos={}
    for p,count,rate in [(comparison,n,fps),(single,n,fps),(lateral,lateral_frames,24)]:
        videos[str(p)]=verify_video(p,count,rate)
    verify(out,c,fingerprint)
    write(out/'render_complete.json',dict(complete=True,config_fingerprint=fingerprint,iterations=r['iterations'],views=n,
        source_hashes_unchanged=True,road_fields_bit_exact=r['road_fields_bit_exact'],
        road_geometry_bit_exact=r.get('road_geometry_bit_exact',r['road_fields_bit_exact']),
        means=means,videos=videos,final_PLY=r['final_PLY'],
        final_sha256=r['final_sha256'],manual_visual_review_complete=False,user_acceptance_pending=True))


def lane_change(config, fingerprint):
    import cv2
    import numpy as np
    import torch
    from plyfile import PlyData
    from ..gs.rendering import gpu_model, camera_render
    out = Path(config['output'])
    verify(out, config, fingerprint)
    report = json.loads((out/'training_report.json').read_text())
    if sha(report['final_PLY']) != report['final_sha256']:
        raise ValueError('Final PLY changed before rendering')
    with np.load(out/'data/input_colmap_cameras.npz') as arrays:
        E, K, names = arrays['extrinsics'], arrays['intrinsics'], arrays['image_names']
    manifest = json.loads((out/'input_manifest.json').read_text())
    width, height = manifest['width'], manifest['height']
    trajectory, audit = lateral_camera(E, K, config.get('lateral_amplitude_world', .06891966864338281))
    save_npz(out/'lane_change_cameras.npz', **trajectory, source_extrinsics=E, image_names=names)
    destination = out/'forward_lane_change.mp4'
    with gpu(config):
        torch.set_num_threads(8)
        model = gpu_model(PlyData.read(report['final_PLY'], mmap='r')['vertex'].data)
        writer = encode(destination, width*2, height+40, 10)
        try:
            for index, (pose, intrinsics) in enumerate(zip(trajectory['extrinsics'], K)):
                row = np.zeros((height+40, width*2, 3), np.uint8)
                for col, camera in enumerate([E[index], pose]):
                    image = camera_render(model, camera, intrinsics, width, height)[0]
                    row[40:, col*width:(col+1)*width] = np.round(image*255).astype(np.uint8)
                for col, title in enumerate(['Native trajectory', 'Forward lane change / SfM units']):
                    cv2.putText(row, title, (col*width+8, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1, cv2.LINE_AA)
                writer.stdin.write(row.tobytes())
                if index in [len(E)//4, len(E)//2, 3*len(E)//4]:
                    cv2.imwrite(str(out/f'review/lane_change_{index:03d}.jpg'), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        finally:
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError('ffmpeg lane-change encoding failed')
        del model
        torch.cuda.empty_cache()
    audit['video'] = verify_video(destination, len(E), 10)
    write(out/'lane_change_audit.json', audit)
    verify(out, config, fingerprint)
    return destination


def render(config, fingerprint, png_only=False):
    # Both use short-lived locks. Models from the first phase are released before lane change.
    review(config, fingerprint)
    lane_change(config, fingerprint)
    out = Path(config['output'])
    videos = [('原轨迹', out/'trained_forward.mp4'),
              ('原图／ReSplat／混合初始化／训练后', out/'original_resplat_init_trained_forward.mp4'),
              ('定点左右横移：初始化／训练后', out/'init_vs_trained_lateral.mp4'),
              ('前进中变道：原轨迹／左右横移', out/'forward_lane_change.mp4')]
    previews = []
    if png_only:
        from PIL import Image
        for source in sorted((out/'review').glob('*.jpg')):
            destination = source.with_suffix('.png')
            with Image.open(source) as preview:
                preview.save(destination)
            previews.append(destination)
    if not png_only:
        media_report(out/'render_comparison.html', '固定道路＋ReSplat背景训练',
                 '相同相机对照，真实天空监督；横移单位为SfM单位，尚未标定为米，新视角无真值。', videos)
    reports = ['render_metrics.json', 'render_complete.json', 'lane_change_audit.json', 'lane_change_cameras.npz']
    if not png_only:
        reports.append('render_comparison.html')
    write(out/'visualization_complete.json', dict(complete=True, config_fingerprint=fingerprint,
          artifact_hashes={str(path): sha(path) for _, path in videos} | {
              str(out/name): sha(out/name) for name in reports} | {str(path): sha(path) for path in previews},
          manual_visual_review_complete=False))
    return out/('visualization_complete.json' if png_only else 'render_comparison.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--png-only', action='store_true', help='Keep video/image renders without HTML')
    args = parser.parse_args()
    config, fingerprint = effective(args.config.resolve())
    print(render(config, fingerprint, png_only=args.png_only))


if __name__ == '__main__':
    main()
