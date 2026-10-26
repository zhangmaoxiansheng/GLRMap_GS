"""Fixed-camera Track refinement and optional DA3 screening on fresh outputs."""
import json
from pathlib import Path
import numpy as np
from ..io_utils import save_npz, export_ply, write_json
from ..triangulation.geometry import project_batch, maximum_angles
from .cache import read_arrays, pi3x_fields, da3_fields, pi3x_inputs, guard_hashes, verify_hashes
from .mapping import load_geometry
from .sampling import sample, rays_from_pixels
from .refinement import refine
from .cloud import write_cloud, footprint_radii, uncovered_prior_points, merge_supplement
from .filter import FilterConfig, geometry_protected_mask, select_supported_depth_conflicts


def track_batches(obs, batch_size=4096):
    lengths = np.diff(obs['offsets'])
    for count in sorted(set(lengths)):
        indices = np.flatnonzero(lengths == count)
        for first in range(0, len(indices), batch_size):
            ids = indices[first:first+batch_size]
            slots = obs['offsets'][ids, None]+np.arange(count)
            yield ids, slots, obs['image_index'][slots], obs['xy'][slots]


def write_tracks(output, name, points, obs, metric):
    destination = Path(output)/name
    destination.mkdir()
    save_npz(destination/'new_points.npz', **points)
    save_npz(destination/'point_observations.npz', **obs)
    export_ply(destination/'new_points.ply', points['xyz'], points['rgb'])
    write_json(destination/'metrics.json', dict(metric, accepted_new_points=len(points['xyz']),
               accepted_two_views=int((points['track_length'] == 2).sum()),
               accepted_at_least_3_views=int((points['track_length'] >= 3).sum())))


def refine_tracks(reference, prior, output, quantile=.7, depth_sigma=.02, iterations=8):
    """Existing top30 production path: soft constraint, strongest-depth replacement, coverage fill."""
    reference, prior, output = Path(reference).resolve(), Path(prior).resolve(), Path(output).resolve()
    if not .7 <= quantile < 1 or depth_sigma <= 0 or not np.isfinite(depth_sigma) or iterations < 1:
        raise ValueError('Invalid top30 refinement settings')
    old = json.loads((reference/'manifest.json').read_text())
    source = Path(old['source'])
    hashes = guard_hashes([*pi3x_inputs(prior, source), reference/'manifest.json',
                          reference/'B1_shared/new_points.npz', reference/'B1_shared/point_observations.npz',
                          prior/'PI3X_FIXED/new_points.npz', prior/'PI3X_FIXED/metrics.json'])
    images, poses, params, centers = load_geometry(source)
    fields, scale = pi3x_fields(prior, source, images, poses, params, quantile)
    points = read_arrays(reference/'B1_shared/new_points.npz')
    obs = read_arrays(reference/'B1_shared/point_observations.npz')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', dict(old, status='refining', reference=str(reference), prior=str(prior), input_hashes=hashes))
    outputs = [{key: value.copy() for key, value in points.items()} for _ in range(2)]
    errors = [obs['errors'].copy() for _ in range(2)]
    eligible = np.zeros(len(points['xyz']), bool)
    support = np.zeros(len(eligible), np.int16)
    nonpositive = np.zeros((2, len(eligible)), bool)
    for ids, slots, im, xy in track_batches(obs):
        pp, qq = poses[im], params[im]
        initial = points['xyz'][ids].astype(float)
        depth, confidence, _, valid = sample(fields, im, rays_from_pixels(xy, qq), xy, 0)
        fitted, target, _, active = refine(initial, pp, qq, xy, depth, confidence, valid, depth_sigma, iterations)
        eligible[ids], support[ids] = active, valid.sum(1)
        for index, xyz in enumerate([fitted, target]):
            xyz = xyz.astype(np.float32)
            pixel, camera_z, _ = project_batch(xyz, pp, qq)
            error = np.linalg.norm(pixel-xy, axis=2)
            cloud = outputs[index]
            cloud['xyz'][ids[active]] = xyz[active]
            cloud['reprojection'][ids[active]] = error[active].mean(1)
            cloud['max_reprojection'][ids[active]] = error[active].max(1)
            cloud['angle'][ids[active]] = maximum_angles(xyz[active], centers[im[active]], np.ones(im[active].shape, bool))
            cloud['quality'][ids[active]] = 0
            cloud['leave_one_out_px'][ids[active]] = np.nan
            cloud['relative_xyz_stability'][ids[active]] = np.nan
            errors[index][slots[active]] = error[active]
            nonpositive[index, ids] = (camera_z <= 0).any(1)
    for index, (name, cloud, error) in enumerate(zip(['CONSTRAINED', 'REPLACED'], outputs, errors)):
        assert np.array_equal(cloud['xyz'][~eligible], points['xyz'][~eligible])
        assert np.isfinite(cloud['xyz']).all()
        write_tracks(output, name, cloud, dict(obs, errors=error), dict(
            input_pairs=old['pair_budget'], has_track_observations=True, xyz_modified=True,
            pose_unchanged=True, track_coordinates_unchanged=True, geometry_revalidated=False,
            adjusted_tracks=int(eligible.sum()), nonpositive_tracks=int(nonpositive[index].sum()),
            adjusted_tracks_max_reprojection_above_2px=int((cloud['max_reprojection'][eligible] > 2).sum()),
            observation_label='原真实Track；XYZ已调整，未重新验收'))
    high = read_arrays(prior/'PI3X_FIXED/new_points.npz')
    thresholds = np.array([field['threshold'] for field in fields])
    selected = high['confidence'] >= thresholds[high['source_image_index']].astype(high['confidence'].dtype)
    if quantile == .7:
        np.testing.assert_allclose(thresholds, json.loads((prior/'PI3X_FIXED/metrics.json').read_text())['per_frame_confidence_threshold'], rtol=0, atol=1e-12)
        assert selected.all(), 'The original top30 must remain intact'
    high = {key: value[selected] for key, value in high.items()}
    radius = footprint_radii(high, fields)
    selected, distance = uncovered_prior_points(outputs[1]['xyz'], high['xyz'], radius)
    added = write_cloud(output, 'PI3X_ADDED', high['xyz'][selected], high['rgb'][selected],
                        high['source_image_index'][selected], high['source_pixel'][selected], high['confidence'][selected],
                        dict(coverage_reference='REPLACED', coverage_radius_depth_pixels=1.))
    merged = merge_supplement(outputs[1], added)
    destination = output/'REPLACED_PLUS'
    destination.mkdir()
    save_npz(destination/'new_points.npz', **merged)
    export_ply(destination/'new_points.ply', merged['xyz'], merged['rgb'])
    write_json(destination/'metrics.json', dict(accepted_new_points=len(merged['xyz']), input_pairs=old['pair_budget'],
               has_track_observations=True, mixed_depth_points=True, roma_tracks=len(points['xyz']),
               pi3x_added_points=len(added['xyz']), observation_reference=str(output/'REPLACED/point_observations.npz'),
               point_origin='0: original Track with possibly replaced XYZ; 1: untracked Pi3X depth point'))
    save_npz(output/'track_adjustments.npz', eligible=eligible, high_confidence_views=support)
    save_npz(output/'supplement_selection.npz', selected=selected, nearest_reference_distance=distance, radius=radius,
             source_image_index=high['source_image_index'], source_pixel=high['source_pixel'])
    verify_hashes(hashes)
    write_json(output/'manifest.json', dict(old, status='complete', reference=str(reference), prior=str(prior), input_hashes=hashes,
               confidence_quantile=quantile, global_scale=scale, depth_sigma=depth_sigma, iterations=iterations,
               viewer_branches=['CONSTRAINED', 'REPLACED', 'REPLACED_PLUS', 'PI3X_ADDED']))
    return outputs, merged


def filter_da3(reference, prior, output, relative_threshold=.2):
    """Latest DA3 policy: protect strong geometry; direct symmetric depth error on the rest."""
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if not 0 < relative_threshold < 1:
        raise ValueError('Relative depth threshold must be between zero and one')
    old = json.loads((reference/'manifest.json').read_text())
    source = Path(old['source'])
    hashes = guard_hashes([Path(prior), reference/'manifest.json', reference/'B1_shared/new_points.npz',
                          reference/'B1_shared/point_observations.npz', *source.glob('sparse/0/*.bin'),
                          *source.glob('static_valid_masks/*.png')])
    images, poses, params, _ = load_geometry(source)
    fields = da3_fields(prior, source, images, poses, params, .7, 0)
    points = read_arrays(reference/'B1_shared/new_points.npz')
    obs = read_arrays(reference/'B1_shared/point_observations.npz')
    protected = geometry_protected_mask(points)
    rejected = np.zeros(len(points['xyz']), bool)
    for ids, slots, im, xy in track_batches(obs):
        active = ~protected[ids]
        if not active.any():
            continue
        ids, im, xy = ids[active], im[active], xy[active]
        pp, qq = poses[im], params[im]
        depth, _, _, valid = sample(fields, im, rays_from_pixels(xy, qq), xy, 0)
        _, camera_z, _ = project_batch(points['xyz'][ids], pp, qq)
        rejected[ids] = select_supported_depth_conflicts(camera_z, depth, valid, FilterConfig(relative_threshold))['reject']
    output.mkdir(parents=True, exist_ok=False)
    for name, keep in [('KEPT', ~rejected), ('REJECTED', rejected)]:
        cloud = {key: value[keep] for key, value in points.items()}
        observation_keep = np.repeat(keep, np.diff(obs['offsets']))
        retained = {key: obs[key][observation_keep] for key in ['image_index', 'xy', 'errors']}
        retained['offsets'] = np.r_[0, np.cumsum(cloud['track_length'])]
        write_tracks(output, name, cloud, retained, dict(input_pairs=old['pair_budget'], has_track_observations=True,
                     xyz_modified=False, observation_label='DA3直接误差筛选；XYZ与真实观测未修改'))
    save_npz(output/'selection.npz', protected=protected, rejected=rejected)
    verify_hashes(hashes)
    write_json(output/'manifest.json', dict(old, status='complete', reference=str(reference), prior=str(prior), input_hashes=hashes,
               relative_depth_threshold=relative_threshold, confidence_quantile=.7, evidence_mode='track-depth',
               geometry_protection=True, boundary_check=False, viewer_branches=['KEPT', 'REJECTED']))
    return rejected
