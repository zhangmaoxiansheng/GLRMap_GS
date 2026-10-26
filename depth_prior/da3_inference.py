"""Unfiltered DA3 prior conditioned on fixed pinhole SfM cameras."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from ..runtime import ROOT as BASE, path
SDK = path('da3_source', BASE.parent/'Depth-Anything-3')/'src'
MODEL = path('da3_model', BASE/'models/DA3')


def run(source, output, process_res=504):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('New output directory required')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/superdim-mpl-cache')
    sys.path.insert(0,str(SDK))
    import torch
    import pycolmap
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    torch.set_num_threads(8);torch.manual_seed(42)
    model=pycolmap.Reconstruction(str(source/'sparse/0'));images=sorted(model.images.values(),key=lambda x:x.name)
    ex=[];intr=[]
    for im in images:
        c=model.cameras[im.camera_id]
        name=str(c.model).split('.')[-1]
        if name=='SIMPLE_PINHOLE':f,cx,cy=c.params;fx=fy=f
        elif name=='PINHOLE':fx,fy,cx,cy=c.params
        else:raise ValueError(f'Pinhole source required: {name}')
        e=np.eye(4);e[:3]=im.cam_from_world().matrix();ex.append(e);intr.append([[fx,0,cx],[0,fy,cy],[0,0,1]])
    ex=np.array(ex,np.float32);intr=np.array(intr,np.float32)
    output.mkdir(parents=True);start=time.monotonic()
    network=DepthAnything3.from_pretrained(str(MODEL)).to('cuda').eval();torch.cuda.synchronize();load_seconds=time.monotonic()-start
    alignment={};raw={};original=network._align_to_input_extrinsics_intrinsics
    def capture(extrinsics,intrinsics,prediction,align_to_input_ext_scale=True):
        raw['predicted_extrinsics']=prediction.extrinsics.copy()
        _,_,scale,aligned=align_poses_umeyama(prediction.extrinsics,extrinsics.numpy(),ransac=len(extrinsics)>=10,return_aligned=True,random_state=42)
        alignment['sdk_pose_scale']=float(scale);raw['aligned_predicted_extrinsics']=aligned.copy()
        return original(extrinsics,intrinsics,prediction,align_to_input_ext_scale)
    network._align_to_input_extrinsics_intrinsics=capture
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    prediction=network.inference([str(source/'images'/im.name) for im in images],extrinsics=ex,intrinsics=intr,
                align_to_input_ext_scale=True,infer_gs=False,process_res=process_res,process_res_method='upper_bound_resize',ref_view_strategy='saddle_balanced')
    torch.cuda.synchronize();inference_seconds=time.monotonic()-start
    assert np.allclose(prediction.extrinsics,ex[:,:3],atol=1e-6)
    depth=prediction.depth.astype(np.float32);conf=prediction.conf.astype(np.float32)
    assert depth.shape==conf.shape and len(depth)==len(images)
    np.savez_compressed(output/'prior.npz',depth=depth,conf=conf,intrinsics=prediction.intrinsics,extrinsics=prediction.extrinsics,
                        image=prediction.processed_images,image_names=np.array([im.name for im in images]),image_ids=np.array([im.image_id for im in images]),
                        original_intrinsics=intr,original_extrinsics=ex,**raw)
    report=dict(status='complete',source=str(source),model=str(MODEL),sdk_path=str(SDK),frame_count=len(images),depth_shape=list(depth.shape),
                convention='camera_z; input GLOMAP w2c and scale; processed K stored explicitly',full_valid_geometry=True,
                confidence_filtered=False,depth_percentile_filtered=False,gs_generated=False,load_seconds=load_seconds,
                inference_seconds=inference_seconds,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,alignment=alignment,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                api_sha256=hashlib.sha256((SDK/'depth_anything_3/api.py').read_bytes()).hexdigest())
    (output/'inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-res', type=int, default=504)
    args = parser.parse_args()
    run(args.source, args.output, args.process_res)


if __name__ == '__main__':
    main()
