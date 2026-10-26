"""Anchor diagnostics shared by top30 assessment and the stricter 30–60% band."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from ..io_utils import save_npz, write_json
from ..matching.sparse import valid_mask
from ..sfm.geometry import model_data
from .anchors import AnchorConfig, geometry_anchors, local_status
from .cache import read_arrays, pi3x_fields, native_intrinsics, pi3x_inputs, guard_hashes, verify_hashes
from .mapping import load_geometry
from .sampling import processed_grid


def camera_trust(data, predicted, alignment, config):
    rotation, translation = np.array(alignment['rotation']), np.array(alignment['translation'])
    centers = alignment['scale']*predicted['camera_poses'][:, :3, 3]@rotation.T+translation
    error = np.linalg.norm(centers-data['poses'][:, :3, 3], axis=1)/alignment['input_camera_span']
    angle = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', data['poses'][:, :3, :3],
                         rotation@predicted['camera_poses'][:, :3, :3])-1)/2, -1, 1)))
    return (error <= config.max_center_error_fraction) & (angle <= config.max_pose_angle_deg), error, angle


def assess_anchors(reference, prior, output, config=None, combined=False):
    """Sparse-only top30 diagnostics, or SfM+raw-RoMa top60 maps for band gating."""
    reference, prior, output = Path(reference).resolve(), Path(prior).resolve(), Path(output).resolve()
    config = config or AnchorConfig()
    source = Path(json.loads((reference/'manifest.json').read_text())['source'])
    paths = [*pi3x_inputs(prior, source), reference/'manifest.json']
    if combined:
        paths += [reference/'B1_shared/new_points.npz', reference/'B1_shared/point_observations.npz']
    hashes = guard_hashes(paths)
    images, poses, params, _ = load_geometry(source)
    model, _, _ = model_data(source/'sparse/0')
    fields, _ = pi3x_fields(prior, source, images, poses, params)
    data, predicted = read_arrays(prior/'input_conditions.npz'), read_arrays(prior/'pi3x/prediction.npz')
    alignment = json.loads((prior/'camera_alignment.json').read_text())
    trusted, center_error, angle = camera_trust(data, predicted, alignment, config)
    sparse, stats = geometry_anchors(model, images, poses, params, [field['mask'] for field in fields], config)
    if combined:
        points = read_arrays(reference/'B1_shared/new_points.npz')
        obs = read_arrays(reference/'B1_shared/point_observations.npz')
        qualified = ((points['track_length'] >= 3) & (points['angle'] >= 3)
                     & (points['max_reprojection'] <= 1) & np.isfinite(points['xyz']).all(1))
        owner = np.repeat(np.arange(len(points['xyz'])), np.diff(obs['offsets']))
        slots = np.flatnonzero(qualified[owner])
        slots = slots[np.argsort(obs['image_index'][slots], kind='stable')]
        cut = np.searchsorted(obs['image_index'][slots], np.arange(len(images)+1))
    output.mkdir(parents=True, exist_ok=False)
    (output/'maps').mkdir()
    intrinsics = native_intrinsics(params)
    records = []
    for index, (image, field, anchors) in enumerate(zip(images, fields, sparse)):
        _, _, valid = processed_grid(field, predicted['local_points'][index], params[index])
        top30 = valid & (field['conf'] >= field['threshold'])
        candidate = valid & (field['conf'] >= np.quantile(field['conf'][valid], .4)) if combined and valid.any() else top30
        xy, ids, depth, error = anchors[:, 1:3], anchors[:, 0].astype(np.int64), anchors[:, 3], anchors[:, 4]
        kind = np.zeros(len(ids), int)
        if combined:
            selected = slots[cut[index]:cut[index+1]]
            point = owner[selected]
            xy = np.concatenate([xy, obs['xy'][selected]])
            ids = np.r_[ids*2, points['track_ids'][point]*2+1].astype(np.int64)
            depth = np.r_[depth, points['xyz'][point]@poses[index, 2, :3]+poses[index, 2, 3]]
            error = np.r_[error, obs['errors'][selected]]
            kind = np.r_[kind, np.ones(len(selected), int)]
        rays = np.c_[xy, np.ones(len(xy))]@np.linalg.inv(intrinsics[index]).T
        uv = rays@field['K'].T
        uv = uv[:, :2]/uv[:, 2, None]
        if combined:
            h, w = field['depth'].shape
            good = (valid_mask(xy, field['mask']) & (depth > 0) & np.isfinite(uv).all(1)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
            uv, ids, depth, error, kind = (value[good] for value in [uv, ids, depth, error, kind])
            cell = (uv[:, 1]//4).astype(int)*((w+3)//4)+(uv[:, 0]//4).astype(int)
            order = np.lexsort((ids, error, kind, cell))
            _, first = np.unique(cell[order], return_index=True)
            pick = order[first]
            uv, ids, depth, kind = (value[pick] for value in [uv, ids, depth, kind])
        maps = local_status(field['depth'], valid, candidate, uv, depth, ids, config, trusted=bool(trusted[index]))
        save_npz(output/'maps'/f'{Path(image.name).stem}.npz', **maps, top30_original=top30, candidate_mask=candidate,
                 anchor_ids=ids, anchor_xy_processed=uv, anchor_z=depth, anchor_kind=kind,
                 aligned_camera_z=field['depth'].astype(np.float32), processed_K=field['K'], world_to_camera=poses[index],
                 pose_scale_trusted=np.array(trusted[index]))
        records.append(dict(image=image.name, pose_scale_trusted=bool(trusted[index]),
                            camera_rotation_error_deg=float(angle[index]), camera_center_error_fraction=float(center_error[index]),
                            anchors=len(ids), status_counts=np.bincount(maps['prior_status'].ravel(), minlength=4).tolist()))
    verify_hashes(hashes)
    write_json(output/'summary.json', dict(anchor_selection=stats, frame_stats=records, combined=combined, input_hashes=hashes,
               parameters=asdict(config), xyz_modified=False, heldout_rule='Anchor ID modulo 5 == 0 never votes'))
    return records
