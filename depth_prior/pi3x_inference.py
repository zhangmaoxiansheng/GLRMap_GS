"""Pi3X inference with every existing Pose/K active and no input depth."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from ..io_utils import sha


def run(input_path, destination, checkpoint):
    import torch
    from safetensors.torch import load_file
    from ..runtime import ROOT, path
    sdk = path('pi3x_source', ROOT.parent/'gluemap/thirdparty/pi3')
    sys.path.insert(0, str(sdk))
    from pi3.models.pi3x import Pi3X
    input_path, destination, checkpoint = Path(input_path), Path(destination), Path(checkpoint)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    with np.load(input_path) as data:
        images=data['image'];poses=data['poses'];K=data['intrinsics'];names=data['image_names']
    n,h,w,_=images.shape
    assert n>=2 and poses.shape==(n,4,4) and K.shape==(n,3,3)
    assert h%14==0 and w%14==0
    image=torch.from_numpy(images).permute(0,3,1,2).float()[None].to('cuda')/255.
    pose=torch.from_numpy(poses).float()[None].to('cuda')
    intrinsic=torch.from_numpy(K).float()[None].to('cuda')
    active=torch.ones((1,n),dtype=torch.bool,device='cuda')
    print(f'Pi3X input: {n} frames {w}x{h}; Pose=all True; K=all True; depth=None',flush=True)
    start=time.monotonic();network=Pi3X(use_multimodal=True).eval()
    network.load_state_dict(load_file(str(checkpoint)),strict=True)
    network=network.to('cuda');torch.cuda.synchronize();load_seconds=time.monotonic()-start
    print(f'Model loaded in {load_seconds:.2f}s',flush=True)
    capture={};original=network.encode
    def capture_conditions(*a,**kw):
        output=original(*a,**kw)
        capture['normalized_input_poses']=output[1].detach().float().cpu().numpy()[0]
        capture['depth_condition_mask']=output[2].detach().cpu().numpy()[0]
        capture['pose_condition_mask']=output[3].detach().cpu().numpy()[0]
        capture['normalization_factor']=output[4].detach().float().cpu().numpy()
        return output
    network.encode=capture_conditions
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
        result=network(imgs=image,poses=pose,intrinsics=intrinsic,depths=None,
            mask_add_pose=active.clone(),mask_add_ray=active.clone(),mask_add_depth=torch.zeros_like(active))
    torch.cuda.synchronize();seconds=time.monotonic()-start
    assert capture['pose_condition_mask'].all() and not capture['depth_condition_mask'].any()
    output={key:result[key].detach().float().cpu().numpy()[0] for key in ['local_points','camera_poses','conf']}
    output['conf_logits']=output.pop('conf')[...,0]
    output['metric']=result['metric'].detach().float().cpu().numpy()
    assert all(np.isfinite(v).all() for v in output.values())
    print(f'Inference complete: {seconds:.2f}s; saving raw predictions',flush=True)
    np.savez_compressed(destination/'prediction.npz',**output,image_names=names,**capture)
    report=dict(status='complete',model='yyfz233/Pi3X',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        input=str(input_path),input_sha256=sha(input_path),frame_count=n,input_resolution=[w,h],
        pose_conditioned=True,intrinsics_conditioned=True,depth_conditioned=False,
        pose_condition_count=int(capture['pose_condition_mask'].sum()),depth_condition_count=0,
        inference_mode='One joint batch containing all frames',pose_input_convention='OpenCV camera-to-world',
        load_seconds=load_seconds,inference_seconds=seconds,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
        output_pose_hard_locked=False,script_sha256=sha(__file__),
        sdk_sha256=sha(Path(__import__('pi3.models.pi3x',fromlist=['__file__']).__file__)))
    (destination/'inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.ckpt)


if __name__ == '__main__':
    main()
