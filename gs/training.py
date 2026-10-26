"""Two-phase background fitting with bit-exact fixed road and full optimizer/RNG recovery."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import signal
import time
from .state import FIELDS, gpu, verify, write
from ..io_utils import sha


def train(c,fingerprint,stop_after=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from plyfile import PlyData
    import sys
    from ..runtime import GS_ROOT
    sys.path.insert(0,str(GS_ROOT))
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from scene import Scene,GaussianModel
    from scene.dataset_readers import getNerfppNorm
    from types import SimpleNamespace
    from gaussian_renderer import render
    from utils.general_utils import safe_state,get_expon_lr_func
    from utils.loss_utils import masked_l1_loss,masked_ssim
    from utils.sky_mask_utils import ValidMaskSupervisor
    out=Path(c['output']);m=verify(out,c,fingerprint);n=m['views'];phase_steps=m['phase_steps'];total=phase_steps*2
    if (out/'training_report.json').exists():
        r=json.loads((out/'training_report.json').read_text());assert sha(r['final_PLY'])==r['final_sha256'];return
    path=out/'trained';path.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();lp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args(['-s',str(out/'data'),'-m',str(path),'--sh_degree','3','--iterations',str(total),
        '--position_lr_max_steps',str(phase_steps),'--densify_until_iter','0'])
    dataset,options,pipeline=lp.extract(args),op.extract(args),pp.extract(args)
    (path/'cfg_args').write_text(str(argparse.Namespace(**vars(dataset))))
    write(out/'training_config.json',dict(**vars(args),phase_steps=phase_steps,RGB_mask='dynamic only; sky included',
        road_freeze='all6 tensors',phase_optimizer_state='continuous',depth_schedule=[['exponential',.5,.05],['constant',.05]]))
    interrupted=[False]
    def stop(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with gpu(c):
        safe_state(False);torch.set_num_threads(8);started=time.monotonic()
        model=GaussianModel(3,'default');scene=Scene(dataset,model,init_ply=str(out/'static_road_init.ply'))
        with np.load(c.get('reference_cameras',out/'data/input_colmap_cameras.npz')) as z:
            ref=z['extrinsics']
        # Optional reference cameras describe only this road reconstruction,
        # avoiding an extent multiplier from an entire long route.
        model.spatial_lr_scale=getNerfppNorm([SimpleNamespace(R=e[:3,:3].T,T=e[:3,3]) for e in ref])['radius']
        model.training_setup(options);cams=scene.getTrainCameras();assert len(cams)==n
        cam_lookup={cam.image_name:cam for cam in cams}
        with np.load(out/'initialization_lineage.npz') as z:start=int(z['road_start']);count=int(z['total_count'])
        pinned={k:getattr(model,k)[start:].detach().clone() for k in FIELDS}
        counts=Counter();phase_counts=[Counter(),Counter()];stack=[];begin=1;losses=[];previous_seconds=0.;resumed_from=0
        checkpoint=out/'checkpoint_latest.pt'
        if checkpoint.exists():
            state=torch.load(checkpoint,weights_only=False)
            assert state['config_fingerprint']==fingerprint
            model.restore(state['model'],options);counts=Counter(state['counts']);phase_counts=[Counter(x) for x in state['phase_counts']]
            stack=[cam_lookup[name] for name in state['stack']];random.setstate(state['random_state'])
            torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
            begin=state['iteration']+1;resumed_from=state['iteration'];losses=state['losses'];previous_seconds=state['elapsed_seconds']
            assert all(torch.equal(getattr(model,k)[start:],v) for k,v in pinned.items())
            print('Resume exact optimizer/sampling state at iteration',resumed_from,flush=True)
        with np.load(out/'inputs/road_depth.npz') as z:
            targets=z['depth'];mask=z['correction_mask'];lookup={str(name):i for i,name in enumerate(z['image_names'])}
        masks=ValidMaskSupervisor(out/'inputs/vehicle_valid');cached={}
        for cam in cams:
            i=lookup[cam.image_name];d=targets[i];good=mask[i]&np.isfinite(d)&(d>0);inv=np.zeros_like(d);inv[good]=1/d[good]
            valid=masks.target_for(cam);size=(cam.image_height,cam.image_width)
            den=F.interpolate(torch.tensor(good.astype(np.float32),device='cuda')[None,None],size=size,mode='bilinear',align_corners=False)[0]
            target=F.interpolate(torch.tensor(inv,device='cuda')[None,None],size=size,mode='bilinear',align_corners=False)[0]/den.clamp_min(1e-8)
            dv=(den>.999)&(valid>.5);assert dv.any(),cam.image_name;cached[cam.image_name]=(valid,target,dv)
        def persist(iteration):
            assert all(torch.equal(getattr(model,k)[start:],v) for k,v in pinned.items())
            state=dict(config_fingerprint=fingerprint,iteration=iteration,model=model.capture(),counts=dict(counts),
                phase_counts=[dict(x) for x in phase_counts],stack=[cam.image_name for cam in stack],random_state=random.getstate(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),losses=losses,
                elapsed_seconds=previous_seconds+time.monotonic()-started)
            tmp=checkpoint.with_suffix('.partial');torch.save(state,tmp);tmp.replace(checkpoint)
            write(out/'progress.json',dict(stage='training',iteration=iteration,total_steps=total,resumed_from=resumed_from,
                elapsed_seconds=state['elapsed_seconds'],checkpoint=str(checkpoint)))
        bg=torch.zeros(3,device='cuda');weight_fn=get_expon_lr_func(.5,.05,max_steps=phase_steps)
        for iteration in range(begin,total+1):
            phase=(iteration-1)//phase_steps;relative=(iteration-1)%phase_steps+1
            model.update_learning_rate(relative)
            if not stack:stack=cams.copy()
            cam=stack.pop(random.randrange(len(stack)));valid,target,dv=cached[cam.image_name]
            package=render(cam,model,pipeline,bg,use_trained_exp=False,separate_sh=False)
            prediction=package['render'];truth=cam.original_image.cuda()
            rgb=.8*masked_l1_loss(prediction,truth,valid)+.2*(1-masked_ssim(prediction,truth,valid))
            depth=abs(package['depth'][dv]-target[dv]).mean();weight=weight_fn(relative) if phase==0 else .05
            loss=rgb+weight*depth
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            for k in FIELDS:
                p=getattr(model,k)
                if p.grad is not None:p.grad[start:].zero_()
            model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for k in FIELDS:getattr(model,k)[start:].copy_(pinned[k])
            counts[cam.image_name]+=1;phase_counts[phase][cam.image_name]+=1
            if iteration%100==0:
                item=dict(iteration=iteration,RGB=float(rgb),road_inverse_depth=float(depth),depth_weight=float(weight));losses.append(item)
                print(item,'elapsed',round(previous_seconds+time.monotonic()-started,2),flush=True)
            del package,prediction,truth,rgb,depth,loss
            if iteration%int(c['checkpoint_every'])==0 or iteration in (phase_steps,total):persist(iteration)
            if iteration in (phase_steps,total):scene.save(iteration)
            if interrupted[0] or (stop_after is not None and iteration>=stop_after):
                persist(iteration);write(out/'status.json',dict(status='interrupted_checkpoint_saved',iteration=iteration))
                return 75
        assert all(torch.isfinite(getattr(model,k)).all() for k in FIELDS)
        final=path/f'point_cloud/iteration_{total}/point_cloud.ply'
        # Recover a crash after the final checkpoint but before PLY export.
        if not final.exists():scene.save(total)
        original=PlyData.read(out/'static_road_init.ply',mmap='r')['vertex'].data
        saved=PlyData.read(final,mmap='r')['vertex'].data
        assert len(saved)==count and all(saved[k][start:].tobytes()==original[k][start:].tobytes() for k in original.dtype.names)
        assert all(len(x)==n and set(x.values())=={c['steps_per_view_per_phase']} for x in phase_counts)
        write(out/'training_report.json',dict(complete=True,iterations=total,updates=total,views=n,Gaussian_count=count,
            road_count=count-start,road_fields_bit_exact=True,RGB_view_counts=dict(counts),phase_counts=[dict(x) for x in phase_counts],
            elapsed_seconds=previous_seconds+time.monotonic()-started,resumed_from=resumed_from,
            peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,final_PLY=str(final),final_sha256=sha(final),losses=losses))
