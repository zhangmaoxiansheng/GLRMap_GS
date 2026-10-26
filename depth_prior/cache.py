"""Read explicit Pi3X/DA3 cache formats into the same camera-Z sampling fields."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from ..io_utils import sha
from ..matching.sparse import valid_mask
from .mapping import check_mapping


def guard_hashes(paths):
    return {str(path): sha(path) for path in paths}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise RuntimeError(f'Input changed during depth processing: {path}')


def pi3x_inputs(prior, source):
    prior, source = Path(prior), Path(source)
    return [prior/'input_conditions.npz', prior/'pi3x/prediction.npz', prior/'pi3x/inference.json',
            prior/'camera_alignment.json', *source.glob('sparse/0/*.bin'), *source.glob('static_valid_masks/*.png')]


def read_arrays(path):
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def native_intrinsics(params):
    matrix = np.zeros((len(params), 3, 3))
    matrix[:, 0, 0] = params[:, 0]
    matrix[:, 1, 1] = params[:, 4] if params.shape[-1] == 5 else params[:, 0]
    matrix[:, 0, 2], matrix[:, 1, 2], matrix[:, 2, 2] = params[:, 1], params[:, 2], 1.
    return matrix


def fields_from_pi3x(data, predicted, source, images, poses, params, scale, quantile=.7, min_confidence=0.):
    if not 0 <= quantile < 1 or not 0 <= min_confidence < 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid confidence selection or camera-only scale')
    source = Path(source)
    with Image.open(source/'images'/images[0].name) as photo:
        resolution = photo.size
    check_mapping(data, predicted, [image.name for image in images], poses, native_intrinsics(params), resolution)
    minimum = np.log(min_confidence/(1-min_confidence)) if min_confidence > 0 else -np.inf
    fields = []
    for i, image in enumerate(images):
        local, confidence = predicted['local_points'][i], predicted['conf_logits'][i]
        h, w = confidence.shape
        mask = np.asarray(Image.open(source/'static_valid_masks'/f'{Path(image.name).stem}.png')) > 0
        intrinsics = data['intrinsics'][i]
        yy, xx = np.indices((h, w))
        ray = np.stack([xx, yy, np.ones_like(xx)], -1)@np.linalg.inv(intrinsics).T
        xy = ray[..., :2]/ray[..., 2, None]*data['native_intrinsics'][i, [0, 1], [0, 1]]+params[i, 1:3]
        static = valid_mask(xy.reshape(-1, 2), mask).reshape(h, w)
        good = static & np.isfinite(local).all(-1) & (local[..., 2] > 0) & np.isfinite(confidence)
        threshold = max(minimum, float(np.quantile(confidence[good], quantile))) if good.any() else np.inf
        fields.append(dict(depth=local[..., 2].astype(float)*scale, conf=confidence, K=intrinsics,
                           mask=mask, spread=np.zeros((h, w)), threshold=threshold))
    return fields


def pi3x_fields(prior, source, images, poses, params, quantile=.7, min_confidence=0.):
    prior = Path(prior)
    data = read_arrays(prior/'input_conditions.npz')
    predicted = read_arrays(prior/'pi3x/prediction.npz')
    inference = json.loads((prior/'pi3x/inference.json').read_text())
    assert inference['pose_conditioned'] and inference['intrinsics_conditioned'] and not inference['depth_conditioned']
    scale = json.loads((prior/'camera_alignment.json').read_text())['scale']
    return fields_from_pi3x(data, predicted, source, images, poses, params, scale, quantile, min_confidence), scale


def da3_fields(prior, source, images, poses, params, quantile=.4, boundary_log_jump=0.):
    """Already SfM-conditioned DA3 Z; no point/depth scale fitting, no fixed EXP54 dimensions."""
    with np.load(prior) as arrays:
        names = arrays['image_names'].tolist()
        depth, confidence, intrinsics, extrinsics = (arrays[key] for key in ['depth', 'conf', 'intrinsics', 'extrinsics'])
    assert names == [image.name for image in images]
    np.testing.assert_allclose(extrinsics[:, :3], poses, rtol=0, atol=1e-6)
    source = Path(source)
    with Image.open(source/'images'/images[0].name) as photo:
        width, height = photo.size
    hd, wd = depth.shape[1:]
    expected = native_intrinsics(params)
    expected[:, 0] *= wd/width
    expected[:, 1] *= hd/height
    np.testing.assert_allclose(intrinsics, expected, rtol=1e-6, atol=1e-4)
    fields = []
    for index, image in enumerate(images):
        mask = np.asarray(Image.open(source/'static_valid_masks'/f'{Path(image.name).stem}.png')) > 0
        static = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((wd, hd), Image.Resampling.NEAREST)) > 0
        basic = np.isfinite(depth[index]) & (depth[index] > 0) & np.isfinite(confidence[index]) & static
        threshold = float(np.quantile(confidence[index][basic], quantile)) if basic.any() else np.inf
        logdepth = np.log(np.maximum(np.nan_to_num(depth[index]), 1e-12))
        spread = maximum_filter(logdepth, size=3)-minimum_filter(logdepth, size=3)
        fields.append(dict(depth=depth[index], conf=confidence[index], K=intrinsics[index], mask=mask,
                           spread=spread, threshold=threshold, boundary_log_jump=boundary_log_jump))
    return fields
