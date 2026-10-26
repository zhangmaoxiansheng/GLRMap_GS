"""Preserve the top30 fusion; add only locally verified rank30–60 depth points."""
import json
from pathlib import Path
import numpy as np
from ..io_utils import save_npz, export_ply, write_json
from .anchors import AnchorConfig, local_error_p90, select_band
from .anchor_maps import assess_anchors
from .cache import read_arrays, pi3x_fields, pi3x_inputs, guard_hashes, verify_hashes
from .mapping import load_geometry
from .cloud import unproject_clouds, write_cloud, footprint_radii, uncovered_prior_points


def relax_band(reference, prior, baseline, output, median_error=.05, p90_error=.10):
    if not 0 < median_error <= p90_error:
        raise ValueError('Require 0 < median error <= P90 error')
    reference, prior, baseline, output = (Path(path).resolve() for path in [reference, prior, baseline, output])
    old = json.loads((reference/'manifest.json').read_text())
    source = Path(old['source'])
    hashes = guard_hashes([*pi3x_inputs(prior, source), reference/'manifest.json',
                          reference/'B1_shared/new_points.npz', reference/'B1_shared/point_observations.npz',
                          prior/'PI3X_FIXED/new_points.npz', baseline/'REPLACED_PLUS/new_points.npz',
                          baseline/'REPLACED/point_observations.npz'])
    config = AnchorConfig(max_reprojection_px=1., good_log_error=np.log(1+median_error),
                          max_log_mad=.03, unknown_strength=0.)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', dict(old, status='band_running', baseline=str(baseline)))
    assess_anchors(reference, prior, output/'anchor_maps', config, combined=True)
    images, poses, params, _ = load_geometry(source)
    fields, scale = pi3x_fields(prior, source, images, poses, params, .4)
    data, predicted = read_arrays(prior/'input_conditions.npz'), read_arrays(prior/'pi3x/prediction.npz')
    alignment = json.loads((prior/'camera_alignment.json').read_text())
    raw = unproject_clouds(data, predicted, fields, params, scale,
                          np.array(alignment['rotation']), np.array(alignment['translation']))
    accepted = np.zeros(len(raw['fixed']), bool)
    high = read_arrays(prior/'PI3X_FIXED/new_points.npz')
    for index, image in enumerate(images):
        maps = read_arrays(output/'anchor_maps/maps'/f'{Path(image.name).stem}.npz')
        p90 = local_error_p90(maps)
        adopted, band = select_band(maps['top30_original'], maps['candidate_mask'], maps['prior_status'], p90, np.log(1+p90_error))
        # Preserve the established top30 mask, including confidence ties.
        rows = np.flatnonzero(high['source_image_index'] == index)
        uv = high['source_pixel'][rows]
        expected = np.zeros_like(band)
        expected[uv[:, 1], uv[:, 0]] = True
        np.testing.assert_array_equal(expected, maps['top30_original'])
        rows = np.flatnonzero(raw['source_image_index'] == index)
        uv = raw['source_pixel'][rows]
        accepted[rows] = band[uv[:, 1], uv[:, 0]]
        save_npz(output/'anchor_maps/maps'/f'{Path(image.name).stem}.npz', **maps,
                 adopted=adopted, accepted_band=band, anchor_error_p90=p90)
    selected = dict(xyz=raw['fixed'][accepted].astype(np.float32),
                    **{key: raw[key][accepted] for key in ['rgb', 'source_image_index', 'source_pixel', 'confidence']})
    base = read_arrays(baseline/'REPLACED_PLUS/new_points.npz')
    radius = footprint_radii(selected, fields)
    keep, distance = uncovered_prior_points(base['xyz'], selected['xyz'], radius)
    extra = write_cloud(output, 'NEW_FILL', selected['xyz'][keep], selected['rgb'][keep],
                        selected['source_image_index'][keep], selected['source_pixel'][keep], selected['confidence'][keep],
                        dict(anchor_gated=True, coverage_reference='Original top30 fused cloud'))
    n = len(extra['xyz'])
    merged = {key: np.concatenate([base[key], extra[key]]) for key in
              ['xyz', 'rgb', 'source_image_index', 'source_pixel', 'confidence']}
    merged.update(track_ids=np.r_[base['track_ids'], np.full(n, -1, np.int64)],
                  track_length=np.r_[base['track_length'], np.zeros(n, np.int16)],
                  quality=np.r_[base['quality'], np.zeros(n, np.uint8)],
                  point_origin=np.r_[base['point_origin'], np.full(n, 2, np.uint8)])
    destination = output/'FUSED_AFTER'
    destination.mkdir()
    save_npz(destination/'new_points.npz', **merged)
    export_ply(destination/'new_points.ply', merged['xyz'], merged['rgb'])
    save_npz(output/'band_selection.npz', accepted=accepted, coverage_selected=keep, nearest_reference_distance=distance)
    write_json(destination/'metrics.json', dict(accepted_new_points=len(merged['xyz']), input_pairs=old['pair_budget'],
               has_track_observations=True, mixed_depth_points=True,
               observation_reference=str(baseline/'REPLACED/point_observations.npz'),
               point_origin='0: original RoMa; 1: top30 depth; 2: untracked verified rank30–60 depth'))
    verify_hashes(hashes)
    write_json(output/'manifest.json', dict(old, status='complete', prior=str(prior), baseline=str(baseline), input_hashes=hashes,
               median_error=median_error, p90_error=p90_error, top30_unchanged=True,
               band_candidates=int(accepted.sum()), added_points=n, viewer_branches=['NEW_FILL', 'FUSED_AFTER']))
    return merged
