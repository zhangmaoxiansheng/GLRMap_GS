"""Camera-only global scale and strict image/Pose/K handover."""
from pathlib import Path
import numpy as np
from PIL import Image
from ..io_utils import save_npz
from ..sfm.geometry import model_data, fixed_camera_parameters


def prepare_conditions(source, cached_rgb, output):
    """Reuse the exact processed RGB/Pose/K cache; never infer a crop or old experiment path."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    images, poses, _, _ = load_geometry(source)
    model, _, _ = model_data(source/'sparse/0')
    with np.load(cached_rgb) as arrays:
        rgb, intrinsics = arrays['image'], arrays['intrinsics']
        names, cached_pose = arrays['image_names'], arrays['extrinsics']
    assert names.tolist() == [image.name for image in images]
    np.testing.assert_allclose(cached_pose[:, :3], poses, rtol=0, atol=1e-6)
    n, h, w, channels = rgb.shape
    assert n >= 2 and channels == 3 and h % 14 == 0 and w % 14 == 0
    assert rgb.dtype == np.uint8
    native = np.stack([model.cameras[image.camera_id].calibration_matrix() for image in images])
    with Image.open(source/'images'/images[0].name) as photo:
        width, height = photo.size
    expected = native.copy()
    expected[:, 0] *= w/width
    expected[:, 1] *= h/height
    np.testing.assert_allclose(intrinsics, expected, rtol=1e-6, atol=1e-4)
    world_to_camera = np.tile(np.eye(4), (n, 1, 1))
    world_to_camera[:, :3] = poses
    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output, image=rgb, poses=np.linalg.inv(world_to_camera), intrinsics=intrinsics,
             image_names=names, native_intrinsics=native, world_to_camera=world_to_camera)

def load_geometry(source):
    model,images,_=model_data(source/'sparse/0')
    assert all(model.cameras[im.camera_id].model_name in ('SIMPLE_PINHOLE','PINHOLE') for im in images)
    poses=np.stack([im.cam_from_world().matrix() for im in images])
    params=np.array([fixed_camera_parameters(model.cameras[im.camera_id]) for im in images])
    centers=np.array([im.projection_center() for im in images])
    return images,poses,params,centers


def camera_similarity(predicted,target):
    """One camera-only global similarity, with orientation-fixed rotation.

    Camera rotations determine rotation even for a straight/collinear route;
    centers then determine positive scale and translation by least squares.
    No depth or triangulated points are involved in this alignment.
    """
    u,_,vt=np.linalg.svd(np.einsum('nij,nkj->ik',target[:,:3,:3],predicted[:,:3,:3]))
    rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    pc=predicted[:,:3,3];tc=target[:,:3,3]
    rotated=(pc-pc.mean(0))@rotation.T;centered=tc-tc.mean(0)
    scale=float((rotated*centered).sum()/np.square(rotated).sum())
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid predicted camera baseline for scale alignment')
    translation=tc.mean(0)-scale*rotation@pc.mean(0)
    aligned=scale*pc@rotation.T+translation
    residual=np.linalg.norm(aligned-tc,axis=1)
    angle=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',target[:,:3,:3],rotation@predicted[:,:3,:3])-1)/2,-1,1)))
    return scale,rotation,translation,dict(method='Mean camera orientation + positive least-squares center similarity',
        scale=scale,rotation=rotation.tolist(),translation=translation.tolist(),
        camera_center_rmse=float(np.sqrt(np.square(residual).mean())),
        camera_center_error_p50_p90=np.percentile(residual,[50,90]).tolist(),
        camera_rotation_error_deg_p50_p90=np.percentile(angle,[50,90]).tolist(),
        input_camera_span=float(np.linalg.norm(tc[-1]-tc[0])),
        depth_or_point_alignment_used=False,per_image_scale_fitted=False)


def check_mapping(data, predicted, names, poses, native_k, resolution):
    """Fail on mismatched cache/crop/Pose; never silently re-register it."""
    assert data['image_names'].tolist() == predicted['image_names'].tolist() == names
    np.testing.assert_allclose(data['world_to_camera'][:, :3], poses, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.inv(data['poses'])[:, :3], poses, rtol=0, atol=1e-7)
    np.testing.assert_allclose(data['native_intrinsics'], native_k, rtol=0, atol=1e-7)
    h, w = predicted['conf_logits'].shape[1:]
    assert data['image'].shape[:3] == (len(names), h, w)
    assert predicted['local_points'].shape == (len(names), h, w, 3)
    expected = native_k.copy()
    expected[:, 0] *= w / resolution[0]
    expected[:, 1] *= h / resolution[1]
    np.testing.assert_allclose(data['intrinsics'], expected, rtol=1e-6, atol=1e-4)
