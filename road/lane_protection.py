"""Keep matched marking geometry separate from synthetic road ray intersections."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from ..matching.sparse import valid_mask


def marking_votes(observations, images, mask_dir):
    """At least two retained observations and >=50% support, not an RGB-only label."""
    lengths = np.diff(observations['offsets'])
    hit = np.zeros(len(observations['xy']), bool)
    for index in np.unique(observations['image_index']):
        rows = np.flatnonzero(observations['image_index'] == index)
        path = Path(mask_dir)/(Path(images[index].name).stem+'.png')
        mask = np.asarray(Image.open(path)) > 0
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        hit[rows] = valid_mask(observations['xy'][rows], mask)
    point = np.repeat(np.arange(len(lengths)), lengths)
    votes = np.bincount(point, weights=hit, minlength=len(lengths)).astype(np.int32)
    return (votes >= 2) & (votes*2 >= lengths), votes


def clear_surface(dense, protected_xyz, images, mask_dir, origin, basis, radius):
    """Remove synthetic points in source marking bands or matched-lane UV tubes.

    Only the returned selection changes. The real matched points never enter a
    surface projection, interpolation or voxel averaging operation.
    UV distance deliberately ignores fitted height errors. Use a local road
    frame/window; this is not a global rule for overlapping roads or overpasses.
    """
    if radius <= 0:
        raise ValueError('Lane protection radius must be positive in SfM units')
    source_band = np.zeros(len(dense['xyz']), bool)
    for index in np.unique(dense['source_image_index']):
        rows = np.flatnonzero(dense['source_image_index'] == index)
        path = Path(mask_dir)/(Path(images[index].name).stem+'.png')
        mask = np.asarray(Image.open(path)) > 0
        band = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        source_band[rows] = valid_mask(dense['source_xy'][rows], band)
    near_lane = np.zeros(len(dense['xyz']), bool)
    if len(protected_xyz):
        uv = ((protected_xyz-origin) @ basis.T)[:, :2]
        tree = cKDTree(uv)
        for start in range(0, len(near_lane), 200000):
            end = min(start+200000, len(near_lane))
            query = ((dense['xyz'][start:end]-origin) @ basis.T)[:, :2]
            distance = tree.query(query, distance_upper_bound=radius)[0]
            near_lane[start:end] = np.isfinite(distance)
    return ~(source_band | near_lane), dict(
        fitted_input=len(source_band), source_marking_band=int(source_band.sum()),
        matched_lane_footprint=int(near_lane.sum()),
        fitted_removed=int((source_band | near_lane).sum()),
        radius_SfM=float(radius), source_band_radius_pixels=3,
        mode='Keep real XYZ/RGB; remove synthetic competitors, including wrong-height competitors')
