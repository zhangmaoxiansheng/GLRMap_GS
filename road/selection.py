"""Select raw road Tracks from their actual retained image observations."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..io_utils import save_npz, export_ply, write_json
from ..matching.sparse import valid_mask
from ..sfm.geometry import model_data


def extract_road(root, destination, branch='B1_shared'):
    """Keep at least two static-road observations and road fraction >=75%.

    Return selected point arrays, original full-cloud rows and reference views.
    The input experiment stays read-only; destination is a new road branch.
    """
    root, destination = Path(root), Path(destination)
    manifest = json.loads((root/'manifest.json').read_text())
    source = Path(manifest['source'])
    with np.load(root/branch/'new_points.npz') as arrays:
        points = {key: arrays[key] for key in arrays.files}
    with np.load(root/branch/'point_observations.npz') as arrays:
        obs = {key: arrays[key] for key in arrays.files}
    _, images, _ = model_data(source/'sparse/0')
    is_road = np.zeros(len(obs['xy']), bool)
    for index, image in enumerate(images):
        ids = np.flatnonzero(obs['image_index'] == index)
        stem = Path(image.name).stem+'.png'
        road = np.array(Image.open(source/'road_masks'/stem)) > 128
        static = np.array(Image.open(source/'static_valid_masks'/stem)) > 0
        is_road[ids] = valid_mask(obs['xy'][ids], road & static)
    counts = np.bincount(np.repeat(np.arange(len(points['xyz'])), np.diff(obs['offsets'])),
                         weights=is_road, minlength=len(points['xyz'])).astype(int)
    lengths = points['track_length']
    selected = (counts >= 2) & (counts/np.maximum(lengths, 1) >= .75)
    full_rows = np.flatnonzero(selected)
    reference_views = obs['image_index'][obs['offsets'][:-1]][full_rows]
    road = {key: value[selected] for key, value in points.items()}
    destination.mkdir()
    save_npz(destination/'new_points.npz', **road)
    export_ply(destination/'new_points.ply', road['xyz'], road['rgb'])
    save_npz(destination/'road_point_selection.npz', source_point_index=full_rows,
             road_observations=counts[selected])
    write_json(destination/'metrics.json', dict(
        accepted_new_points=len(full_rows),
        input_pairs=json.loads((root/branch/'metrics.json').read_text())['input_pairs'],
        accepted_two_views=int((lengths[selected] == 2).sum()),
        accepted_at_least_3_views=int((lengths[selected] >= 3).sum()),
        definition='At least two retained static road observations and road fraction >=75%; raw triangulated XYZ, no flattening'))
    return road, full_rows, reference_views
