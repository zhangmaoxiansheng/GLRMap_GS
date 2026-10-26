"""Rescale the private SfM model and SIFT DB together for native-crop RoMa input."""
from pathlib import Path
import json
import shutil
import sqlite3
import subprocess
import cv2
import numpy as np
from PIL import Image
from ..io_utils import write_json
from ..matching.sparse import sift_data
from .geometry import model_data

def prepare_source(root,low_source):
    source=root/'source';source.mkdir()
    model,images,_=model_data(low_source/'sparse/0')
    sift_data(model,images,low_source/'database.db')
    signature=json.loads((low_source/'manifest.json').read_text())['signature']
    ratio=900/666
    assert ratio==500/370 and len(images)>0
    folders=['images','static_valid_masks','road_masks','sparse/0']
    for name in folders:(source/name).mkdir(parents=True)
    # Same 10FPS extraction/filter and crop as the original EXP54 inputs, but
    # omit the 666x370 downscale. No upsampling of the old JPEGs.
    command=['ffmpeg','-n','-hide_banner','-loglevel','warning','-ss',str(signature.get('video_start_seconds',signature['start_seconds'])),
        '-i',str(signature['video_path']),'-t',str(signature['duration_seconds']),'-an',
        '-vf','fps=10,crop=900:500:190:0','-q:v','2','-start_number',str(int(Path(images[0].name).stem)),
        str(source/'images/%06d.jpg')]
    with (root/'extraction.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    assert sorted(p.name for p in (source/'images').glob('*.jpg'))==[im.name for im in images]
    errors=[]
    for im in images:
        high=cv2.imread(str(source/'images'/im.name));old=cv2.imread(str(low_source/'images'/im.name))
        assert high.shape[:2]==(500,900)
        down=cv2.resize(high,(666,370),interpolation=cv2.INTER_LANCZOS4)
        errors.append(float(np.abs(down.astype(float)-old.astype(float)).mean()))
        # Essential same-frame sanity check before reusing fixed camera poses.
        if errors[-1]>5:raise ValueError(f'Frame correspondence mismatch: {im.name}, RGB MAE={errors[-1]:.3f}')
        for folder in ['static_valid_masks','road_masks']:
            mask=Image.open(low_source/folder/(Path(im.name).stem+'.png'))
            mask.resize((900,500),Image.Resampling.NEAREST).save(source/folder/(Path(im.name).stem+'.png'))
    shutil.copy2(low_source/'database.db',source/'database.db')
    db=sqlite3.connect(source/'database.db')
    # Retain feature indices and verified pair identities; transform coordinates
    # in the private DB/model together. No SIFT rematching or camera adjustment.
    for image_id,rows,cols,blob in db.execute('SELECT image_id,rows,cols,data FROM keypoints').fetchall():
        points=np.frombuffer(blob,np.float32).reshape(rows,cols).copy()
        points[:,:2]*=ratio
        if cols==4:points[:,2]*=ratio
        elif cols==6:points[:,2:]*=ratio
        db.execute('UPDATE keypoints SET data=? WHERE image_id=?',(points.tobytes(),image_id))
        if image_id in model.images:
            image=model.images[image_id]
            assert len(image.points2D)==rows
            for point,xy in zip(image.points2D,points[:,:2]):point.xy=xy.astype(float)
    scale=np.diag([ratio,ratio,1.]);inverse=np.linalg.inv(scale)
    for pair_id,fblob,hblob in db.execute('SELECT pair_id,F,H FROM two_view_geometries').fetchall():
        f=None if not fblob else (inverse.T@np.frombuffer(fblob,np.float64).reshape(3,3)@inverse).tobytes()
        h=None if not hblob else (scale@np.frombuffer(hblob,np.float64).reshape(3,3)@inverse).tobytes()
        db.execute('UPDATE two_view_geometries SET F=?,H=? WHERE pair_id=?',(f,h,pair_id))
    for camera in model.cameras.values():
        assert camera.model_name in ('SIMPLE_PINHOLE','PINHOLE')
        camera.rescale(900,500)
        db.execute('UPDATE cameras SET width=?,height=?,params=? WHERE camera_id=?',
            (900,500,np.asarray(camera.params,np.float64).tobytes(),camera.camera_id))
    db.commit();db.close();model.write(str(source/'sparse/0'))
    new_model,new_images,_=model_data(source/'sparse/0');sift_data(new_model,new_images,source/'database.db')
    old_model,old_images,_=model_data(low_source/'sparse/0')
    assert np.array_equal(np.stack([im.cam_from_world().matrix() for im in new_images]),np.stack([im.cam_from_world().matrix() for im in old_images]))
    assert all(np.array_equal(p.xyz,old_model.points3D[pid].xyz) for pid,p in new_model.points3D.items())
    low_manifest=json.loads((low_source/'manifest.json').read_text())
    write_json(source/'manifest.json',dict(status='complete',signature=low_manifest['signature'],
        source_kind='Original COLMAP poses and XYZ; private 900x500 image-coordinate rescale',
        resolution=[900,500],crop_xywh=[190,0,900,500],coordinate_scale=ratio,
        original_source=str(low_source),static_mask_method='Nearest resize of existing DA-V2 hard foreground AND original vehicle masks; no morphology',
        extraction_command=command,frame_rgb_mae_mean=float(np.mean(errors)),frame_rgb_mae_max=max(errors),
        original_camera_poses_unchanged=True,original_sparse_xyz_unchanged=True))
    print(f'Source ready: {len(images)} native 900x500 frames, fixed poses, frame MAE mean/max {np.mean(errors):.3f}/{max(errors):.3f}',flush=True)
    return source
