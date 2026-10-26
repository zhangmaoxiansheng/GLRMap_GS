"""Shared ray/surface query for dense road points and native-resolution road depth."""
from pathlib import Path
import cv2
import numpy as np
from .surface import GRID, evaluate, intersect


def query_surface(patches, view_index, xy, pose, intrinsics):
    """Return full-length XYZ, camera-Z, incidence cosine and valid mask.

    Pose may be 3x4 SfM or 4x4 GS world-to-camera. Coordinates are used
    exactly as supplied: no implicit half-pixel shift.
    Invalid depth/XYZ/cosine are NaN, so unsupported road is never a fabricated target.
    """
    inverse_rotation = np.linalg.inv(pose[:3, :3])
    center = -inverse_rotation @ pose[:3, 3]
    rays = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(intrinsics).T @ inverse_rotation.T
    total = np.zeros(len(xy))
    weighted_z = np.zeros(len(xy))
    weighted_cosine = np.zeros(len(xy))
    for patch in patches:
        if not patch['first'] <= view_index < patch['stop']:
            continue
        surface, basis, origin = patch['surface'], patch['basis'], patch['origin']
        points, depth, good = intersect(surface, center, rays, origin, basis)
        safe = np.where(np.isfinite(points), points, origin)
        _, normals = evaluate(surface, ((safe-origin) @ basis.T)[:, :2])
        cosine = abs((rays*(normals @ basis)).sum(1))/np.maximum(np.linalg.norm(rays, axis=1), 1e-12)
        good &= (cosine >= .03) & np.isfinite(depth) & (depth > 0)
        weight = min(1., (view_index-patch['first']+1)/20, (patch['stop']-view_index)/20)
        total[good] += weight
        weighted_z[good] += depth[good]*weight
        weighted_cosine[good] += cosine[good]*weight
    valid = total > 0
    depth, cosine = np.full(len(xy), np.nan), np.full(len(xy), np.nan)
    points = np.full((len(xy), 3), np.nan)
    depth[valid] = weighted_z[valid]/total[valid]
    cosine[valid] = weighted_cosine[valid]/total[valid]
    points[valid] = center+depth[valid, None]*rays[valid]
    return dict(xyz=points, depth=depth, cosine=cosine, valid=valid)


def select_cells(xyz, score, origin, spacing):
    """Keep the best actual ray intersection per cell; never move it to a voxel center."""
    if not len(xyz):
        return np.empty(0, np.int64)
    cells = np.floor((xyz.astype(float)-origin)/spacing).astype(np.int32)
    keys = np.ascontiguousarray(cells).view([('x', 'i4'), ('y', 'i4'), ('z', 'i4')]).ravel()
    order = np.lexsort((score, keys))
    sorted_keys = keys[order]
    return order[np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]]


def sample_road(source, model, images, patches):
    """Photographic road pixels at stride 2, then GRID/2 cell selection."""
    source = Path(source)
    chunks, views = [], []
    for index, image in enumerate(images):
        photo = cv2.imread(str(source/'images'/image.name))
        road = cv2.imread(str(source/'road_masks'/f'{Path(image.name).stem}.png'), 0)
        static = cv2.imread(str(source/'static_valid_masks'/f'{Path(image.name).stem}.png'), 0)
        camera = model.cameras[image.camera_id]
        shape = (camera.height, camera.width)
        if photo is None or road is None or static is None:
            raise FileNotFoundError(f'Missing road image or masks: {image.name}')
        if photo.shape[:2] != shape or road.shape != shape or static.shape != shape:
            raise ValueError(f'Camera/image/mask resolution mismatch: {image.name}')
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        yy, xx = np.indices(shape)
        ys, xs = np.nonzero((road > 127) & (static > 127) & (yy % 2 == 0) & (xx % 2 == 0))
        query = query_surface(patches, index, np.column_stack([xs, ys]),
                              image.cam_from_world().matrix(), camera.calibration_matrix())
        good = query['valid']
        score = query['depth'][good]/np.sqrt(query['cosine'][good])
        chunks.append(dict(xyz=query['xyz'][good].astype(np.float32), rgb=photo[ys[good], xs[good]],
                           score=score.astype(np.float32), source_image_index=np.full(good.sum(), index, np.int16),
                           source_xy=np.column_stack([xs[good], ys[good]]).astype(np.int16)))
        views.append(dict(name=image.name, road_sampled_pixels=len(xs), valid_surface_intersections=int(good.sum())))
        if index % 100 == 0:
            print('DENSE_ROAD', index, len(images), 'rays', sum(len(c['xyz']) for c in chunks), flush=True)
    samples = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    count = len(samples['xyz'])
    if not count:
        raise ValueError('No supported static-road ray intersections')
    chosen = select_cells(samples['xyz'], samples['score'], images[0].projection_center(), GRID/2)
    return {key: value[chosen] for key, value in samples.items() if key != 'score'}, views, count
