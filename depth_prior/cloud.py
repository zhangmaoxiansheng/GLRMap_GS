"""Camera-Z unprojection and explicitly untracked depth supplements."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from ..io_utils import save_npz, export_ply, write_json
from .cache import read_arrays, fields_from_pi3x, guard_hashes, verify_hashes
from .mapping import camera_similarity, load_geometry
from .sampling import processed_grid


def write_cloud(root, name, xyz, rgb, image_ids, pixels, confidence, metric):
    destination = Path(root)/name
    destination.mkdir()
    n = len(xyz)
    cloud = dict(xyz=xyz.astype(np.float32), rgb=rgb, source_image_index=image_ids,
                 source_pixel=pixels, confidence=confidence, quality=np.zeros(n, np.uint8),
                 track_length=np.zeros(n, np.int16))
    save_npz(destination/'new_points.npz', **cloud)
    export_ply(destination/'new_points.ply', cloud['xyz'], rgb)
    write_json(destination/'metrics.json', dict(metric, accepted_new_points=n, input_pairs=0,
               has_track_observations=False, observation_label='Pose条件化深度预测', unfused=True, voxel_downsample=False))
    return cloud


def unproject_clouds(data, predicted, fields, params, scale, rotation, translation):
    chunks = {key: [] for key in ['fixed', 'native', 'rgb', 'source_image_index', 'source_pixel', 'confidence']}
    for index, field in enumerate(fields):
        local = predicted['local_points'][index].astype(float)
        rays, _, basic = processed_grid(field, local, params[index])
        selected = basic & (field['conf'] >= field['threshold'])
        yy, xx = np.nonzero(selected)
        depth = local[..., 2]*scale
        points = rays[selected]*(depth[selected]/rays[selected][:, 2])[:, None]
        pose = data['poses'][index]
        chunks['fixed'].append(points@pose[:3, :3].T+pose[:3, 3])
        predicted_pose = predicted['camera_poses'][index]
        native = local[selected]@predicted_pose[:3, :3].T+predicted_pose[:3, 3]
        chunks['native'].append(scale*native@rotation.T+translation)
        chunks['rgb'].append(data['image'][index][selected])
        chunks['source_image_index'].append(np.full(selected.sum(), index, np.int16))
        chunks['source_pixel'].append(np.column_stack([xx, yy]).astype(np.int16))
        chunks['confidence'].append(field['conf'][selected])
    return {key: np.concatenate(values) for key, values in chunks.items()}


def export_prior(prior, source, output, quantile=.7):
    """Re-export cached predictions into a fresh prior; fit one camera-only similarity."""
    prior, source, output = Path(prior).resolve(), Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    hashes = guard_hashes([prior/'input_conditions.npz', prior/'pi3x/prediction.npz', prior/'pi3x/inference.json',
                          *source.glob('sparse/0/*.bin'), *source.glob('static_valid_masks/*.png')])
    data = read_arrays(prior/'input_conditions.npz')
    predicted = read_arrays(prior/'pi3x/prediction.npz')
    inference = json.loads((prior/'pi3x/inference.json').read_text())
    assert inference['pose_conditioned'] and inference['intrinsics_conditioned'] and not inference['depth_conditioned']
    images, poses, params, _ = load_geometry(source)
    scale, rotation, translation, alignment = camera_similarity(predicted['camera_poses'], data['poses'])
    fields = fields_from_pi3x(data, predicted, source, images, poses, params, scale, quantile)
    clouds = unproject_clouds(data, predicted, fields, params, scale, rotation, translation)
    output.mkdir(parents=True)
    (output/'input_conditions.npz').symlink_to(prior/'input_conditions.npz')
    (output/'pi3x').symlink_to(prior/'pi3x', target_is_directory=True)
    write_json(output/'camera_alignment.json', alignment)
    metric = dict(model='yyfz233/Pi3X', pose_conditioned=True, intrinsics_conditioned=True,
                  depth_conditioned=False, confidence_quantile=quantile, global_scale=scale,
                  confidence_representation='Raw confidence logits, ranked within static valid pixels',
                  per_frame_confidence_threshold=[field['threshold'] for field in fields],
                  per_frame_points=np.bincount(clouds['source_image_index'], minlength=len(images)).tolist(),
                  input_resolution=list(data['image'].shape[1:3][::-1]), static_valid_only=True,
                  depth_fitted_to_sfm_points=False, model_output_pose_hard_locked=False)
    for name, key in [('PI3X_FIXED', 'fixed'), ('PI3X_NATIVE', 'native')]:
        write_cloud(output, name, clouds[key], clouds['rgb'], clouds['source_image_index'],
                    clouds['source_pixel'], clouds['confidence'], dict(metric, coordinates=(
                        'Input SfM c2w and processed K; predicted Z scaled once by camera baseline similarity'
                        if key == 'fixed' else 'Raw model local points and predicted c2w; global camera-only similarity')))
    verify_hashes(hashes)
    write_json(output/'manifest.json', dict(status='complete', source=str(source), prior=str(prior), input_hashes=hashes,
               confidence_quantile=quantile, frame_count=len(images), input_resolution=list(data['image'].shape[1:3][::-1]),
               viewer_branches=['PI3X_FIXED', 'PI3X_NATIVE'],
               viewer_labels={'PI3X_FIXED': 'Pi3X：固定SfM相机', 'PI3X_NATIVE': 'Pi3X：原生相机＋全局对齐'}))
    return clouds


def uncovered_prior_points(existing_xyz, prior_xyz, radius):
    """Coverage only: a depth-pixel footprint is not an accuracy measurement."""
    distance, _ = cKDTree(existing_xyz).query(prior_xyz, workers=1)
    return distance > radius, distance


def footprint_radii(cloud, fields, radius_px=1.):
    radius = np.empty(len(cloud['xyz']))
    for index in np.unique(cloud['source_image_index']):
        pick = np.flatnonzero(cloud['source_image_index'] == index)
        xy = cloud['source_pixel'][pick]
        field = fields[int(index)]
        radius[pick] = radius_px*field['depth'][xy[:, 1], xy[:, 0]]/min(field['K'][0, 0], field['K'][1, 1])
    return radius


def merge_supplement(replaced, added):
    n, m = len(replaced['xyz']), len(added['xyz'])
    return dict(xyz=np.concatenate([replaced['xyz'], added['xyz']]), rgb=np.concatenate([replaced['rgb'], added['rgb']]),
                quality=np.zeros(n+m, np.uint8), track_length=np.r_[replaced['track_length'], np.zeros(m, np.int16)],
                track_ids=np.r_[replaced['track_ids'], np.full(m, -1, np.int64)],
                point_origin=np.r_[np.zeros(n, np.uint8), np.ones(m, np.uint8)],
                source_image_index=np.r_[np.full(n, -1, np.int16), added['source_image_index']],
                source_pixel=np.concatenate([np.full((n, 2), -1, np.int16), added['source_pixel']]),
                confidence=np.r_[np.full(n, np.nan), added['confidence']])
