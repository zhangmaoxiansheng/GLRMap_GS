"""Offline native WebGL packing and camera navigation for explicit experiment branches."""
import base64
import json
from pathlib import Path
import numpy as np
from ..io_utils import sha, write_json

TEMPLATE = Path(__file__).with_name('pointcloud_viewer.html')


def branch_inputs(root, branches=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    branches = branches or manifest.get('viewer_branches')
    if not branches or len(set(branches)) != len(branches):
        raise ValueError('Explicit, unique viewer branches required')
    labels = manifest.get('viewer_labels', {})
    return [dict(name=name, label=labels.get(name, name), path=root/name/'new_points.npz',
                 metrics_path=root/name/'metrics.json') for name in branches]


def load_cloud(path):
    with np.load(path, allow_pickle=False) as arrays:
        points = {key: arrays[key] for key in arrays.files if key in ['xyz', 'rgb', 'quality', 'track_length']}
    xyz, rgb = points['xyz'], points['rgb']
    if xyz.shape != (len(xyz), 3) or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError(f'Expected finite Nx3 XYZ/RGB: {path}')
    if rgb.dtype != np.uint8:
        raise ValueError(f'Expected original uint8 photographic RGB: {path}')
    for key in ['quality', 'track_length']:
        if key in points and points[key].shape != (len(xyz),):
            raise ValueError(f'Invalid {key} shape: {path}')
    return points


def camera_settings(model, images, signature):
    centers = np.array([image.projection_center() for image in images])
    origin = centers[(len(images)-1)//2]
    step = max(float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1))), 1e-6) if len(images)>1 else 1e-6
    cameras = []
    for index, image in enumerate(images):
        rotation = image.cam_from_world().rotation.matrix()
        camera = model.cameras[image.camera_id]
        seconds = (signature['frame_times_seconds'][index] if 'frame_times_seconds' in signature
                   else signature['start_seconds']+index/signature['fps'])
        cameras.append(dict(name=image.name, eye=(image.projection_center()-origin).tolist(),
                            forward=rotation[2].tolist(), up=(-rotation[1]).tolist(),
                            fov=float(2*np.arctan(camera.height/(2*camera.mean_focal_length()))), seconds=seconds))
    return centers, origin, step, cameras


def viewer(root, output, model, images, signature, branches=None, max_points=250000):
    """Sample only presentation buffers; source arrays and projection density are untouched."""
    root, output = Path(root), Path(output)
    if output.stem.lower() == 'index' or output.suffix.lower() != '.html':
        raise ValueError('Use a standalone .html report filename')
    if max_points <= 0:
        raise ValueError('Positive presentation point cap required')
    manifest = json.loads((root/'manifest.json').read_text())
    specs = branch_inputs(root, branches)
    centers, origin, step, cameras = camera_settings(model, images, signature)
    records, bounds, payload = [], [], []
    support = manifest.get('viewer_filter_mode') == 'track_support'
    dtype = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,)), ('pad', 'u1')])
    for index, spec in enumerate(specs):
        metric = json.loads(spec['metrics_path'].read_text())
        points = load_cloud(spec['path'])
        xyz, rgb = points['xyz'], points['rgb']
        quality = points.get('quality', np.zeros(len(xyz), np.uint8))
        if support and metric.get('has_track_observations', True):
            quality = np.where(points['track_length'] >= 3, 1, 2).astype(np.uint8)
        original_count = len(xyz)
        if original_count != metric['accepted_new_points']:
            raise ValueError(f'Cloud/metric count mismatch: {spec["name"]}')
        cap = int(manifest.get('portable_max_points', {}).get(spec['name'], max_points))
        if cap <= 0:
            raise ValueError('Positive per-branch point cap required')
        if original_count > cap:
            selected = np.sort(np.random.default_rng(20261015).choice(original_count, cap, replace=False))
            xyz, rgb, quality = xyz[selected], rgb[selected], quality[selected]
        packed = np.zeros(len(xyz), dtype=dtype)
        packed['xyz'], packed['rgb'], packed['pad'] = xyz.astype(float)-origin, rgb, quality
        if len(xyz):
            bounds.append(np.percentile(packed['xyz'], [2, 98], axis=0))
        payload += [f'<script type="application/octet-stream" data-cloud="{index}">'
                    +base64.b64encode(packed[start:start+1000000].tobytes()).decode('ascii')+'</script>'
                    for start in range(0, len(packed), 1000000)]
        records.append(dict(name=spec['name'], label=spec['label'], count=len(xyz), originalCount=original_count,
                            pairs=metric.get('input_pairs', 0), observationLabel=metric.get('observation_label',
                                '匹配Track' if metric.get('has_track_observations', True) else '非匹配采样（无Track）'),
                            hasTrackObservations=metric.get('has_track_observations', True), source_sha256=sha(spec['path']),
                            bytes=packed.nbytes, gradeA=int((quality == 1).sum()), gradeB=int((quality == 2).sum()),
                            qualityAvailable=bool((quality > 0).any())))
    limits = np.stack(bounds) if bounds else np.array([[[0, 0, 0], [1, 1, 1]]])
    low = np.minimum(limits[:, 0].min(0), (centers-origin).min(0))
    high = np.maximum(limits[:, 1].max(0), (centers-origin).max(0))
    names = [spec['name'] for spec in specs]
    split = manifest.get('viewer_split_branches') or names[:min(3, len(names))]
    settings = dict(title=manifest.get('viewer_title', root.name), clouds=records, cameras=cameras,
                    origin=origin.tolist(), step=step, bounds=[low.tolist(), high.tolist()], pointSize=.8,
                    brightness=1.5, displayFraction=1, renderer='offline native WebGL', supportFilter=support,
                    comparisonNote=manifest.get('portable_note', '展示降采样；原始点云未修改。图像投影与BEV使用全量点云。'),
                    defaultSplit=len(split)>1, splitIndices=[names.index(name) for name in split],
                    qualityAvailable=any(item['qualityAvailable'] for item in records))
    html = TEMPLATE.read_text().replace('__SETTINGS_JSON__', json.dumps(settings, ensure_ascii=False).replace('</', '<\\/'))
    html = html.replace('__CLOUD_PAYLOAD__', '\n'.join(payload))
    if output.exists():
        history = output.parent/'viewer_history'
        history.mkdir(exist_ok=True)
        (history/f'{output.stem}_{sha(output)[:12]}.html').write_bytes(output.read_bytes())
    temporary = output.with_suffix('.tmp')
    temporary.write_text(html)
    temporary.replace(output)
    write_json(output.with_name(output.stem+'_revision.json'), dict(
        renderer=settings['renderer'], packed_stride_bytes=dtype.itemsize, clouds=records,
        html_sha256=sha(output), template_sha256=sha(TEMPLATE), presentation_only=True))
    return records

