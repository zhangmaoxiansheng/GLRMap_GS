"""Raw road selection → checked local surfaces → photographic ray densification."""
import gc
import json
from pathlib import Path
import numpy as np
from ..io_utils import save_npz, export_ply, write_json, sha
from ..sfm.geometry import model_data
from .selection import extract_road
from .fitting import fit_patches, save_patch
from .sampling import sample_road
from .surface import GRID


def run(root, output, branch='B1_shared', first_frame=None, stop_frame=None):
    """Read an existing triangulation; write every road artifact to a fresh output."""
    root, output = Path(root).resolve(), Path(output).resolve()
    previous = json.loads((root/'manifest.json').read_text())
    source = Path(previous['source'])
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(previous, status='road_running', road_source_branch=branch,
                    road_input_experiment=str(root),
                    viewer_branches=['FUSED_BEFORE', 'FUSED_AFTER', 'ROAD_shared', 'ROAD_fitted'],
                    viewer_labels={'FUSED_BEFORE': '原始RoMa完整点云', 'FUSED_AFTER': '非道路＋稠密拟合道路',
                                   'ROAD_shared': '原始RoMa道路', 'ROAD_fitted': '像素射线生成的稠密道路'})
    write_json(output/'manifest.json', manifest)
    try:
        model, images, _ = model_data(source/'sparse/0')
        if any(camera.model_name not in ('PINHOLE', 'SIMPLE_PINHOLE') for camera in model.cameras.values()):
            raise ValueError('This road ray pipeline requires pinhole cameras')
        selected = np.arange(len(images))
        if (first_frame is None) != (stop_frame is None):
            raise ValueError('Provide both first_frame and stop_frame')
        if first_frame is not None:
            selected = np.array([i for i, im in enumerate(images) if first_frame <= int(Path(im.name).stem) < stop_frame])
            if (len(selected) < 2 or [images[i].name for i in selected] != [f'{n:06d}.jpg' for n in range(first_frame, stop_frame)]
                    or not np.array_equal(selected, np.arange(selected[0], selected[-1]+1))):
                raise ValueError('Requested road window is incomplete')
        local_images = [images[i] for i in selected]
        poses = np.stack([image.cam_from_world().matrix() for image in local_images])
        road, full_rows, frames = extract_road(root, output/'ROAD_shared', branch)
        raw_count = len(road['xyz'])
        if not np.isin(frames, selected).all():
            raise ValueError('Triangulation contains road reference frames outside the requested window')
        patches = fit_patches(road, full_rows, frames-selected[0], poses, output)
        del road, frames
        gc.collect()
        dense, views, candidates = sample_road(source, model, local_images, patches)
        # Sampling uses local indices; persisted lineage/windows remain global.
        dense['source_image_index'] += selected[0]
        if selected[0]:
            for index, patch in enumerate(patches):
                save_patch(output/'surfaces'/f'patch_{index:02d}.npz',
                           patch['first']+selected[0], patch['stop']+selected[0],
                           patch['surface'], patch['origin'], patch['basis'])
        manifest['reference_image_names'] = [im.name for im in local_images]
        destination = output/'ROAD_fitted'
        destination.mkdir()
        save_npz(destination/'new_points.npz', **dense)
        export_ply(destination/'new_points.ply', dense['xyz'], dense['rgb'])
        metrics = json.loads((root/branch/'metrics.json').read_text())
        write_json(destination/'metrics.json', dict(
            input_pairs=metrics['input_pairs'], accepted_new_points=len(dense['xyz']), has_track_observations=False,
            observation_label='道路像素与RoMa拟合曲面求交；无独立匹配Track'))
        (output/'FUSED_BEFORE').symlink_to(root/branch, target_is_directory=True)
        with np.load(root/branch/'new_points.npz') as points:
            full_xyz, full_rgb = points['xyz'], points['rgb']
        nonroad = np.ones(len(full_xyz), bool)
        nonroad[full_rows] = False
        fused_xyz = np.concatenate([full_xyz[nonroad], dense['xyz']])
        fused_rgb = np.concatenate([full_rgb[nonroad], dense['rgb']])
        assert np.array_equal(fused_xyz[:nonroad.sum()], full_xyz[nonroad])
        assert np.array_equal(fused_rgb[:nonroad.sum()], full_rgb[nonroad])
        destination = output/'FUSED_AFTER'
        destination.mkdir()
        save_npz(destination/'new_points.npz', xyz=fused_xyz, rgb=fused_rgb)
        export_ply(destination/'new_points.ply', fused_xyz, fused_rgb)
        write_json(destination/'metrics.json', dict(
            input_pairs=metrics['input_pairs'], accepted_new_points=len(fused_xyz), has_track_observations=False,
            observation_label='原始非道路点＋射线曲面稠密道路，非道路XYZ/RGB不变'))
        report = dict(
            raw_road_points=raw_count, ray_intersections=candidates, dense_road_points=len(dense['xyz']),
            pixel_stride=2, cell_size_SfM=GRID/2, nonroad_points_preserved=int(nonroad.sum()), views=views,
            geometry_source='RoMa Track fitted surfaces',
            color_source='Same source-camera road pixel RGB; nearest/most frontal candidate retained per spatial cell',
            new_point_tracks=False, original_xyz_rgb_unchanged=True,
            input_sha256={str(path): sha(path) for path in [
                root/branch/'new_points.npz', root/branch/'point_observations.npz',
                source/'sparse/0/images.bin', source/'sparse/0/cameras.bin']})
        write_json(output/'dense_report.json', report)
        manifest.update(status='complete',
                        viewer_title=previous.get('viewer_title', '三环RoMa点云')+' · 稠密道路',
                        portable_note='原始非道路点保持不变；新增道路来自像素射线与拟合曲面，不是新增匹配Track。')
        write_json(output/'manifest.json', manifest)
        return report
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'manifest.json', manifest)
        raise
