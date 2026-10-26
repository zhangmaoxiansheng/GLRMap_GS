"""Adapt photographic, curve-densified road points into immutable flattened Gaussians."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from ..io_utils import save_npz, sha, write_json
from ..road.fitting import load_patches
from ..road.sampling import query_surface
from ..road.surface import evaluate


def road_gaussians(xyz, rgb, source_views, patches, spacing, tangent_patch_index=2):
    """Common rows use the same patch normals/tangent in every overlapping GS window."""
    normals = np.zeros_like(xyz, dtype=float)
    weights = np.zeros(len(xyz))
    for patch in patches:
        first, stop = patch['first'], patch['stop']
        ids = np.flatnonzero((source_views >= first) & (source_views < stop))
        local = (xyz[ids]-patch['origin']) @ patch['basis'].T
        _, normal = evaluate(patch['surface'], local[:, :2])
        weight = np.minimum(1., np.minimum((source_views[ids]-first+1)/20, (stop-source_views[ids])/20))
        normals[ids] += weight[:, None]*(normal @ patch['basis'])
        weights[ids] += weight
    if not (weights > 0).all():
        raise ValueError('Road source rows have no corresponding fitted patch')
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # The original 1000-view road uses one shared reference, not each window's forward axis.
    forward = patches[tangent_patch_index]['basis'][0]
    tangent = forward[None]-normals*(normals @ forward)[:, None]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    matrices = np.stack((tangent, np.cross(normals, tangent), normals), axis=2)
    rotation = Rotation.from_matrix(matrices).as_quat()[:, [3, 0, 1, 2]]
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
              'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    road = np.zeros(len(xyz), dtype=[(key, 'f4') for key in fields])
    for axis, name in enumerate('xyz'):
        road[name] = xyz[:, axis]
        road[f'f_dc_{axis}'] = (rgb[:, axis]/255.-.5)/.28209479177387814
        road[f'scale_{axis}'] = np.log(spacing*(.12 if axis == 2 else .7))
    road['opacity'] = np.log(.6/.4)
    for axis in range(4):
        road[f'rot_{axis}'] = rotation[:, axis]
    if not all(np.isfinite(road[key]).all() for key in fields):
        raise ValueError('Nonfinite road Gaussian attributes')
    assert np.array_equal(np.column_stack([road[key] for key in 'xyz']), xyz)
    return road, normals.astype(np.float32)


def prepare_assets(dense, cameras, road_masks, dynamic_masks, first_frame, output, num_frames=100, tangent_patch_index=2):
    """Read organized road outputs and native cameras; write a fresh window's assets.

    Patch bounds and source_image_index refer to the full ordered camera archive,
    not to this local window. No inference or training runs here.
    """
    from plyfile import PlyData, PlyElement
    dense, output = Path(dense), Path(output)
    road_masks, dynamic_masks = Path(road_masks), Path(dynamic_masks)
    camera_hash = sha(cameras)
    with np.load(cameras) as arrays:
        poses, intrinsics = arrays['extrinsics'], arrays['intrinsics']
        names = arrays['image_names'].astype(str)
    selected = np.flatnonzero([first_frame <= int(Path(name).stem) < first_frame+num_frames for name in names])
    if (num_frames<20 or names[selected].tolist() != [f'{i:06d}.jpg' for i in range(first_frame, first_frame+num_frames)]
            or not np.array_equal(selected, np.arange(selected[0], selected[0]+num_frames))):
        raise ValueError('Road assets require the ordered native camera window with at least20 views')
    inputs = [Path(cameras), dense/'ROAD_fitted/new_points.npz', dense/'dense_report.json',
              *sorted((dense/'surfaces').glob('patch_*.npz'))]
    inputs += [folder/f'{Path(name).stem}.png' for name in names[selected]
               for folder in [road_masks, dynamic_masks]]
    input_hashes = {str(path.resolve()): sha(path) for path in inputs}
    if input_hashes[str(Path(cameras).resolve())] != camera_hash:
        raise ValueError('Cameras changed while preparing the window')
    patches = load_patches(dense/'surfaces')
    if not 0 <= tangent_patch_index < len(patches):
        raise ValueError('Requested shared tangent patch is missing')
    with np.load(dense/'ROAD_fitted/new_points.npz') as arrays:
        xyz, rgb, source_views = arrays['xyz'], arrays['rgb'], arrays['source_image_index']
    height, width = 370, 666
    keep = (source_views >= selected[0]) & (source_views < selected[-1]+101)
    for index in selected[np.linspace(0, num_frames-1, 16, dtype=int)]:
        points = xyz @ poses[index, :3, :3].T+poses[index, :3, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = (points @ intrinsics[index].T)[:, :2]/points[:, 2:]
        inside = (points[:, 2] > 0) & np.isfinite(xy).all(1)
        inside &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        rows = np.flatnonzero(inside)
        pixels = np.floor(xy[rows]).astype(int)
        mask = cv2.imread(str(road_masks/f'{Path(names[index]).stem}.png'), 0)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f'Missing native 666x370 road mask: {names[index]}')
        keep[rows[mask[pixels[:, 1], pixels[:, 0]] > 127]] = True
    rows = np.flatnonzero(keep)
    spacing = json.loads((dense/'dense_report.json').read_text())['cell_size_SfM']
    road, normals = road_gaussians(xyz[rows], rgb[rows], source_views[rows], patches, spacing, tangent_patch_index)
    output.mkdir(parents=True, exist_ok=False)
    covered = output/'covered_road_masks'
    covered.mkdir()
    PlyData([PlyElement.describe(road, 'vertex')], text=False).write(output/'road_gaussians.ply')
    save_npz(output/'road_constraints.npz', road_start=0, count=len(road))
    save_npz(output/'source_rows.npz', dense_road_source_rows=rows, source_image_index=source_views[rows], normals=normals)
    save_npz(output/'reference_cameras.npz', extrinsics=poses[selected], intrinsics=intrinsics[selected], image_names=names[selected])
    depth = np.zeros((num_frames, height, width), np.float32)
    correction = np.zeros(depth.shape, bool)
    for local, index in enumerate(selected):
        stem = Path(names[index]).stem
        paths = [road_masks/f'{stem}.png', dynamic_masks/f'{stem}.png']
        masks = [cv2.imread(str(path), 0) for path in paths]
        if any(mask is None or mask.shape != (height, width) for mask in masks):
            raise ValueError(f'Missing native road/static masks: {stem}')
        ys, xs = np.nonzero((masks[0] > 127) & (masks[1] > 127))
        query = query_surface(patches, index, np.column_stack([xs, ys]), poses[index], intrinsics[index])
        good = query['valid']
        depth[local, ys[good], xs[good]] = query['depth'][good]
        correction[local, ys[good], xs[good]] = True
        cv2.imwrite(str(covered/f'{stem}.png'), correction[local].astype(np.uint8)*255)
    if not correction.reshape(num_frames, -1).any(1).all():
        raise ValueError('Some cameras have no supported road depth')
    save_npz(output/'road_depth_targets.npz', depth=depth, correction_mask=correction, image_names=names[selected])
    report = dict(road_count=len(rows), input_road_count=len(xyz),
                  supported_road_pixels_per_view=correction.reshape(num_frames, -1).sum(1).tolist(),
                  views=num_frames, tangent_patch_index=tangent_patch_index,
                  road_replacement='Only actual surface coverage; uncovered ReSplat road stays trainable',
                  original_xyz_rgb_unchanged=True, input_sha256=input_hashes)
    if any(sha(path) != digest for path, digest in input_hashes.items()):
        raise ValueError('Road asset inputs changed while preparing the window')
    write_json(output/'audit.json', report)
    return report
