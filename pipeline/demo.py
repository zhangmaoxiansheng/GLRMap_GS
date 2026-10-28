"""100 RGB images → SAM/SfM/RoMa road/ReSplat → trained GS and videos."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..io_utils import sha, write_json
from ..runtime import ROOT, GS_ROOT, GPU_LOCK, path, python_for


def semantic_masks(out):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    import importlib.util
    spec = importlib.util.spec_from_file_location('road_mask_merge', GS_ROOT/'complete_sanhuan_sam31_road_masks.py')
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    sys.path.insert(0, str(path('sam3_source', GS_ROOT/'third_party/sam3')))
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    checkpoint = path('sam3_checkpoint', GS_ROOT/'checkpoints/sam3.1/sam3.1_multiplex.pt')
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint),
        max_num_objects=16, multiplex_count=16, use_fa3=False, compile=False,
        async_loading_frames=False, default_output_prob_thresh=.45)
    predictor.model.detector.eval()
    processor = Sam3Processor(predictor.model.detector, confidence_threshold=.45)
    for folder in ['vehicle_valid', 'road', 'lane', 'sfm_static']:
        (out/'masks'/folder).mkdir(parents=True, exist_ok=True)
    records = []
    for image in sorted((out/'images').glob('*.jpg')):
        rgb = Image.open(image).convert('RGB')
        masks = {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(rgb)
            for key, prompt in [('vehicle', 'vehicle'), ('person', 'person'),
                                ('road', 'road'), ('lane', 'lane markings')]:
                processor.reset_all_prompts(state)
                result = processor.set_text_prompt(state=state, prompt=prompt)
                instances = result['masks'][:, 0].detach().cpu().numpy().astype(bool)
                masks[key] = instances.any(axis=0) if len(instances) else np.zeros((370, 666), bool)
        dynamic = Image.fromarray((masks['vehicle'] | masks['person']).astype('uint8')*255)
        valid = np.asarray(dynamic.filter(ImageFilter.MaxFilter(21))) == 0
        road, _ = merge.merge_masks(masks['road'], masks['lane'], valid, maximum_gap=24)
        for key, mask in [('vehicle_valid', valid), ('road', road), ('lane', masks['lane'])]:
            Image.fromarray(mask.astype('uint8')*255).save(out/'masks'/key/(image.stem+'.png'))
        shutil.copy2(out/'masks/vehicle_valid'/(image.stem+'.png'), out/'masks/sfm_static'/(image.name+'.png'))
        records.append(dict(image=image.name, static_fraction=float(valid.mean()), road_fraction=float(road.mean())))
        print('SAM', len(records), image.name, flush=True)
    predictor.shutdown()
    write_json(out/'masks/sam.json', dict(checkpoint_sha256=sha(checkpoint), threshold=.45,
        prompts=['vehicle', 'person', 'road', 'lane markings'], dynamic_dilation=10, records=records))


def sfm(out):
    import numpy as np
    import pycolmap
    from ..sfm.reconstruction import run_logged
    folder = out/'sfm'
    folder.mkdir(exist_ok=True)
    (folder/'sparse').mkdir(exist_ok=True)
    colmap = str(path('colmap', '/usr/local/bin/colmap'))
    # Existing image-only GLOMAP recipe: shared pinhole camera, self-calibration.
    run_logged([colmap, 'feature_extractor', '--database_path', folder/'database.db',
        '--image_path', out/'images', '--ImageReader.mask_path', out/'masks/sfm_static',
        '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE',
        '--SiftExtraction.use_gpu', '1', '--SiftExtraction.gpu_index', '0',
        '--SiftExtraction.num_threads', '8', '--SiftExtraction.max_num_features', '8192'],
        folder, 'features', ROOT)
    run_logged([colmap, 'sequential_matcher', '--database_path', folder/'database.db',
        '--SequentialMatching.overlap', '20', '--SequentialMatching.quadratic_overlap', '0',
        '--SequentialMatching.loop_detection', '0', '--SiftMatching.use_gpu', '1',
        '--SiftMatching.gpu_index', '0', '--SiftMatching.num_threads', '8',
        '--SiftMatching.guided_matching', '1'], folder, 'matching', ROOT)
    run_logged([path('glomap', ROOT.parent/'glomap/build/glomap/glomap'), 'mapper',
        '--database_path', folder/'database.db', '--image_path', out/'images',
        '--output_path', folder/'sparse',
        '--GlobalPositioning.use_gpu', '1',
        '--BundleAdjustment.use_gpu', '1', '--BundleAdjustment.optimize_intrinsics', '1',
        '--BundleAdjustment.optimize_principal_point', '0', '--ba_iteration_num', '4',
        '--retriangulation_iteration_num', '2'], folder, 'glomap', ROOT)
    model = pycolmap.Reconstruction(str(folder/'sparse/0'))
    images = sorted(model.images.values(), key=lambda im: im.name)
    expected = [p.name for p in sorted((out/'images').glob('*.jpg'))]
    if [im.name for im in images] != expected or any(not im.has_pose for im in images):
        raise ValueError('SfM must register all 100 input images')
    centers = np.stack([im.projection_center() for im in images])
    median_step = float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    if not np.isfinite(median_step) or median_step <= 0:
        raise ValueError('Degenerate SfM trajectory')
    model.export_PLY(str(folder/'sparse.ply'))
    E = np.repeat(np.eye(4)[None], len(images), axis=0)
    for index, image in enumerate(images):
        E[index, :3] = image.cam_from_world().matrix()
    K = np.stack([model.cameras[im.camera_id].calibration_matrix() for im in images])
    error = float(model.compute_mean_reprojection_error())
    observations = [image.num_points3D for image in images]
    rotations = E[:, :3, :3]
    angular = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', rotations[1:], rotations[:-1])-1)/2, -1, 1)))
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    local = np.array([np.median(steps[max(0, i-10):min(len(steps), i+11)]) for i in range(len(steps))])
    bad_steps = np.flatnonzero(steps > 4*np.maximum(local, median_step*.1))
    passed = bool(np.isfinite(error) and error <= 2 and min(observations) >= 30
                  and not len(bad_steps) and angular.max() <= 5)
    write_json(folder/'audit.json', dict(registered=len(images), points=model.num_points3D(),
        mean_reprojection=error, intrinsics=K[0].tolist(), input_calibration=False,
        input_poses=False, median_camera_step=median_step, minimum_observations=min(observations),
        bad_step_indices=bad_steps.tolist(), maximum_rotation_step_deg=float(angular.max()),
        coordinate_transform_applied=False, passed=passed,
        coordinates='Unmodified GLOMAP world coordinates and scale'))
    if not passed:
        raise ValueError(f'SfM quality gate failed: {folder/"audit.json"}')
    np.savez_compressed(out/'cameras.npz', extrinsics=E, intrinsics=K, image_names=np.array(expected))


def matching_inputs(out):
    import sqlite3
    from ..sfm.geometry import model_data
    from ..matching.pairs import select_pairs
    roma = out/'roma'
    source = roma/'source'
    source.mkdir(parents=True)
    shutil.copytree(out/'sfm/sparse', source/'sparse')
    with sqlite3.connect(f'file:{out/"sfm/database.db"}?mode=ro', uri=True) as src, sqlite3.connect(source/'database.db') as dst:
        src.backup(dst)
    (source/'images').symlink_to(out/'images', target_is_directory=True)
    for folder, origin in [('static_valid_masks', out/'masks/sfm_static'), ('road_masks', out/'masks/road')]:
        (source/folder).mkdir()
        for image in sorted((out/'images').glob('*.jpg')):
            name = image.name+'.png' if folder == 'static_valid_masks' else image.stem+'.png'
            shutil.copy2(origin/name, source/folder/(image.stem+'.png'))
    model, images, _ = model_data(source/'sparse/0')
    pairs, scores = select_pairs(model, images, 180, 360)
    write_json(roma/'manifest.json', dict(status='prepared', source=str(source), policies={'B1': pairs},
        pair_budget=len(pairs), seeds_per_image=20000, per_pair_cap=40000,
        matcher_overrides={}, filter_settings=dict(match_min_overlap=.35, match_max_cycle_px=3.,
            match_max_epipolar_px=3., multiview_min_angle_deg=1., multiview_max_reprojection_px=3.,
            two_view_min_angle_deg=2., two_view_min_overlap=.5, two_view_max_cycle_px=1.5,
            two_view_max_sigma_px=10., two_view_max_epipolar_px=1.5, two_view_max_reprojection_px=1.5),
        valid_region='SAM dynamic-valid only; sky segmentation disabled', holdout_validation=False,
        viewer_branches=['FUSED_BEFORE', 'FUSED_AFTER', 'ROAD_shared', 'ROAD_fitted']))
    write_json(roma/'selection.json', scores)
    (roma/'pairs.txt').write_text(''.join(' '.join(pair)+'\n' for pair in pairs))
    for folder in ['direct_pairs', 'seeds', 'superpoint', 'sky_masks/foreground_valid_hard']:
        (roma/'shared'/folder).mkdir(parents=True)
    for mask in (source/'static_valid_masks').glob('*.png'):
        shutil.copy2(mask, roma/'shared/sky_masks/foreground_valid_hard'/mask.name)


def demo_images(folder):
    from PIL import Image
    images = sorted(folder.glob('*.jpg'))
    if len(images) != 100 or any(Image.open(image).size != (666, 370) for image in images):
        raise ValueError('This demo requires exactly 100 native 666x370 JPEGs')
    start = int(images[0].stem)
    if [image.name for image in images] != [f'{i:06d}.jpg' for i in range(start, start+100)]:
        raise ValueError('Use consecutive six-digit JPEG names at 10FPS')
    return images


def check_road_rendering(out):
    import numpy as np
    # The pinned Graphdeco CUDA rasterizer rejects camera-Z <= 0.2.
    # Diagnose incompatible SfM units; never rescale geometry or change clipping.
    with np.load(out/'road_assets/road_depth_targets.npz') as arrays:
        depth = arrays['depth'][arrays['correction_mask']]
    clipped = int(np.count_nonzero(depth <= .2))
    passed = bool(len(depth) and np.isfinite(depth).all() and clipped == 0)
    write_json(out/'road_assets/renderer_audit.json', dict(passed=passed,
        backend_camera_z_cutoff=.2, supervised_pixels=len(depth), clipped_pixels=clipped,
        minimum_depth=float(depth.min()) if len(depth) else None,
        coordinate_transform_applied=False))
    if not passed:
        raise ValueError('Road depth is incompatible with the pinned renderer; '
                         'see road_assets/renderer_audit.json. No automatic rescaling was applied.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'run', 'semantic', 'sfm', 'matching-inputs', 'road-renderer-check'])
    parser.add_argument('--images', type=Path, default=ROOT/'demo/input/images')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/demo_100_images')
    parser.add_argument('--resume', action='store_true', help='Continue this same run with unchanged input JPEGs')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage in ['semantic', 'sfm', 'matching-inputs', 'road-renderer-check']:
        return {'semantic': semantic_masks, 'sfm': sfm,
                'matching-inputs': matching_inputs, 'road-renderer-check': check_road_rendering}[args.stage](out)
    images = demo_images(args.images.resolve())
    first = int(images[0].stem)
    plan = dict(images=str(args.images.resolve()), output=str(out), sfm_views=100, views=100,
        first_frame=first, stop_frame=first+100,
        scene_input='RGB images only', input_masks=False, input_cameras=False,
        input_point_clouds=False, input_resplat=False, sky_segmentation=False,
        coordinate_normalization=False, training_steps=2000, road_train_appearance=True,
        matching='180 to 360 covisibility-selected pairs within the 100 input images',
        triangulation='All 100 input images; no external context frames',
        stages=['semantic', 'sfm', 'matching-inputs', 'match', 'triangulate',
                'road', 'road_assets', 'road-renderer-check', 'resplat_prepare', 'resplat_run', 'gs_train', 'render'])
    if args.stage == 'plan':
        print(json.dumps(plan, indent=2)); return
    run(out, images, first, plan, args.resume)


def run(out, images, first, plan, resume=False):
    if resume:
        if sha(out/'demo_source.py') != sha(Path(__file__)):
            raise ValueError('Resume implementation changed; use a fresh output directory')
        if json.loads((out/'plan.json').read_text()) != plan:
            raise ValueError('Resume plan changed')
        previous = json.loads((out/'frames_manifest.json').read_text())
        if len(previous) != len(images) or any(sha(image) != row['sha256'] or sha(out/'images'/image.name) != row['sha256']
               for image, row in zip(images, previous)):
            raise ValueError('Resume images changed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/'images').mkdir()
        shutil.copy2(Path(__file__), out/'demo_source.py')
    records = []
    for index, image in enumerate(images):
        if not resume:
            shutil.copy2(image, out/'images'/image.name)
        records.append(dict(image=image.name, sha256=sha(out/'images'/image.name),
                            video_time_nominal=index/10))
    write_json(out/'frames_manifest.json', records)
    write_json(out/'plan.json', plan)
    state = json.loads((out/'progress.json').read_text()) if resume else dict(completed_stages=[])
    state.update(status='running')
    state.pop('error', None)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', QT_QPA_PLATFORM='offscreen')
    def step(name, command):
        if any(stage['stage'] == name for stage in state['completed_stages']):
            print('Completed earlier:', name, flush=True)
            return
        state['stage'] = name
        write_json(out/'progress.json', state)
        tick = time.monotonic()
        print('Start', name, flush=True)
        with (out/(name+'.log')).open('a') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['completed_stages'].append(dict(stage=name, seconds=time.monotonic()-tick))
    pr = python_for('resplat')
    pm = python_for('matching')
    ps = python_for('sfm')
    try:
        for stage, python in [('semantic', python_for('sam')),
                              ('sfm', ps), ('matching-inputs', pm)]:
            step(stage, [python, '-u', '-m', 'GLRMap_GS.pipeline.demo', stage, '--output', out])
        for stage in ['match', 'triangulate', 'road']:
            command = [pm, '-u', '-m', 'GLRMap_GS.pipeline.reconstruct', stage, out/'roma']
            if stage == 'road': command += ['--output', out/'dense_road']
            step(stage, command)
        step('road_assets', [pr, '-m', 'GLRMap_GS.pipeline.gs', 'road-assets', '--dense', out/'dense_road',
            '--cameras', out/'cameras.npz', '--road-masks', out/'masks/road', '--dynamic-masks', out/'masks/vehicle_valid',
            '--first-frame', first, '--num-frames', 100, '--tangent-patch-index', 0, '--output', out/'road_assets'])
        step('road-renderer-check', [pr, '-m', 'GLRMap_GS.pipeline.demo', 'road-renderer-check', '--output', out])
        shared = ['--images', out/'images', '--cameras', out/'cameras.npz', '--frame-manifest', out/'frames_manifest.json',
                  '--out', out/'resplat', '--start', first, '--stop', first+100, '--segment-frames', 100,
                  '--label', 'images_demo_'+out.name]
        for stage in ['prepare', 'run']:
            step('resplat_'+stage, [pr, '-u', ROOT/'tools/run_resplat_forward_windows.py', stage, *shared])
        assets, native = out/'road_assets', out/'resplat/segments/segment_00'
        config = dict(output=str(out/'gs'), first_frame=first, stop_frame=first+100, fps=10,
            images=str(out/'images'), cameras=str(out/'cameras.npz'), reference_cameras=str(assets/'reference_cameras.npz'),
            resplat_ply=str(native/'gaussians.ply'), resplat_video=str(native/'video.mp4'),
            road_ply=str(assets/'road_gaussians.ply'), road_constraints=str(assets/'road_constraints.npz'),
            road_depth=str(assets/'road_depth_targets.npz'), road_masks=str(assets/'covered_road_masks'),
            dynamic_masks=str(out/'masks/vehicle_valid'),
            gpu_lock=str(GPU_LOCK), steps_per_view_per_phase=10, checkpoint_every=1000,
            road_train_appearance=True,
            lateral_anchors=[25, 50, 75], python_resplat=pr, python_gs=python_for('gs'))
        write_json(out/'gs_config.json', config)
        step('gs_train', [pr, '-u', '-m', 'GLRMap_GS.pipeline.gs', 'run', '--config', out/'gs_config.json'])
        step('render', [pr, '-u', '-m', 'GLRMap_GS.visualization.gs', '--config', out/'gs_config.json', '--png-only'])
        report = json.loads((out/'gs/training_report.json').read_text())
        rendered = json.loads((out/'gs/visualization_complete.json').read_text())
        if not (report['complete'] and report['views'] == 100 and report['iterations'] == 2000
                and report['road_geometry_bit_exact'] and rendered['complete']
                and report['road_trainable_fields'] == ['_features_dc', '_features_rest', '_opacity']
                and set(report['RGB_view_counts']) == {f'{i:06d}.jpg' for i in range(first, first+100)}):
            raise ValueError('End-to-end execution checks failed')
        if any(sha(out/'images'/row['image']) != row['sha256'] for row in records):
            raise ValueError('Input snapshot changed during execution')
        state.update(status='complete', stage='complete', execution_checks_passed=True,
                     visual_quality_requires_review=True, final_ply=report['final_PLY'])
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(out/'progress.json', state)


if __name__ == '__main__':
    main()
