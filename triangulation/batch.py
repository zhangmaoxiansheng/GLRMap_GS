"""Bound memory by reference-image chunks while preserving all selected neighbors."""
import gc
import json
from pathlib import Path
import numpy as np
from ..io_utils import save_npz, export_ply, write_json
from ..sfm.geometry import model_data
from .tracks import build_shared
from .reconstruct import reconstruct


def triangulate(root, policy='B1', chunk_size=100, first_frame=None, stop_frame=None):
    root = Path(root)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if (root/f'{policy}_shared').exists():
        raise FileExistsError(f'Output already exists: {root/f"{policy}_shared"}')
    manifest = json.loads((root/'manifest.json').read_text())
    pairs = manifest['policies'][policy]
    _, images, _ = model_data(Path(manifest['source'])/'sparse/0')
    if (first_frame is None) != (stop_frame is None):
        raise ValueError('Provide both first_frame and stop_frame')
    if first_frame is not None:
        images = [im for im in images if first_frame <= int(Path(im.name).stem) < stop_frame]
        if not images or [im.name for im in images] != [f'{n:06d}.jpg' for n in range(first_frame, stop_frame)]:
            raise ValueError('Requested reference window is incomplete')
    parts = []
    for chunk, start in enumerate(range(0, len(images), chunk_size)):
        label = f'{policy}_chunk{chunk:02d}'
        part = root/f'{label}_shared'
        if not (part/'metrics.json').exists():
            built = build_shared(root, pairs, anchor_names={im.name for im in images[start:start+chunk_size]})
            built.update(repair_enabled=True, snap_enabled=True, compute_holdouts=False)
            reconstruct(root, label, built)
            del built
            gc.collect()
        parts.append(part)
    return merge_chunks(root/f'{policy}_shared', parts, len(pairs), manifest['filter_settings'])


def merge_chunks(destination, parts, pair_count, filters):
    """Track IDs index proposed Tracks, so shift by proposed count, not accepted count."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    arrays, observations, offsets = {}, {}, [0]
    track_shift = 0
    for part in parts:
        with np.load(part/'new_points.npz') as points:
            for key in points.files:
                arrays.setdefault(key, []).append(points[key]+track_shift if key == 'track_ids' else points[key])
        with np.load(part/'tracks.npz') as tracks:
            track_shift += len(tracks['offsets'])-1
        with np.load(part/'point_observations.npz') as obs:
            offsets.extend((obs['offsets'][1:]+offsets[-1]).tolist())
            for key in ['image_index', 'xy', 'errors']:
                observations.setdefault(key, []).append(obs[key])
    # Release each field's chunk buffers as it is merged, rather than retaining
    # two complete clouds until a dictionary comprehension finishes.
    for key in arrays:
        arrays[key] = np.concatenate(arrays[key])
    for key in observations:
        observations[key] = np.concatenate(observations[key])
    # Missing/corrupt input chunks must not leave a destination that blocks retry.
    destination.mkdir()
    save_npz(destination/'new_points.npz', **arrays)
    save_npz(destination/'point_observations.npz', offsets=np.asarray(offsets, np.int64), **observations)
    export_ply(destination/'new_points.ply', arrays['xyz'], arrays['rgb'])
    metric = dict(input_pairs=pair_count, accepted_new_points=len(arrays['xyz']),
                  accepted_two_views=int((arrays['track_length'] == 2).sum()),
                  accepted_at_least_3_views=int((arrays['track_length'] >= 3).sum()),
                  reference_chunks=len(parts), global_track_ids_unique=True,
                  camera_poses_unchanged=True, filter_settings=filters,
                  track_sources=[str(part) for part in parts])
    write_json(destination/'metrics.json', metric)
    return metric
