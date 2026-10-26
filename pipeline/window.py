"""One selected road window: prepared batch -> SfM -> RoMa -> road -> ReSplat -> GS -> PNG/MP4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from ..runtime import ROOT, GPU_LOCK, python_for
from ..io_utils import sha, write_json
from .cloud import plan as cloud_plan


def plan(args):
    source = args.input.resolve()
    batch = cloud_plan(source)
    if not 0 <= args.block < len(batch['blocks']):
        raise ValueError('Block outside input plan')
    block = batch['blocks'][args.block]
    count = args.stop_frame-args.first_frame
    if not 20 <= count <= 100 or not block[0] <= args.first_frame < args.stop_frame <= block[1]:
        raise ValueError('Select20–100 consecutive frames within the chosen SfM block')
    settings = (args.settings or source/'matching_settings.json').resolve()
    reference = Path(batch['first_anchor']) if args.block == 0 else source/'sfm'/f'block_{args.block-1:02d}'/'cameras.npz'
    for path in [settings, reference, Path(batch['source_video']), source/'frames_manifest.json']:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.reuse_sfm:
        audit = json.loads((source/'sfm'/f'block_{args.block:02d}'/'audit.json').read_text())
        if not audit['passed'] or (audit['first'], audit['stop']) != tuple(block[:2]):
            raise ValueError('Requested reused SfM has not passed its batch audit')
    return dict(input=str(source), output=str(args.output.resolve()), block=block,
                source_block_index=args.block, reference=str(reference.resolve()), settings=str(settings),
                source_video=batch['source_video'], first_frame=args.first_frame, stop_frame=args.stop_frame,
                fps=10, views=count, total_training_steps=count*20, reuse_sfm=args.reuse_sfm,
                matching='Full-batch geometry selects pairs; infer only pairs touching the reference window',
                triangulation='Only selected reference frames; retain their direct neighbor observations',
                stages=['inputs', 'reuse_sfm' if args.reuse_sfm else 'sfm', 'matching_inputs',
                        'match', 'triangulate', 'road', 'point_cloud_png', 'road_assets',
                        'resplat_prepare', 'resplat_run', 'gs_train', 'render'])


def prepare_inputs(p):
    """Snapshot masks/metadata, link immutable RGB; never adopt reconstruction outputs implicitly."""
    from PIL import Image
    source, out = Path(p['input']), Path(p['output'])/'cloud'
    folders = ['frames', 'masks/vehicle_valid', 'masks/road', 'sky_all/foreground_valid_hard', 'sfm/block_00/images']
    for folder in folders:
        (out/folder).mkdir(parents=True)
    records = {r['image']: r for r in json.loads((source/'frames_manifest.json').read_text())}
    selected = []
    hashes = {p['reference']: sha(p['reference']), p['settings']: sha(p['settings'])}
    for frame in range(*p['block'][:2]):
        name = f'{frame:06d}.jpg'
        image = source/'frames'/name
        if sha(image) != records[name]['sha256'] or Image.open(image).size != (666, 370):
            raise ValueError(f'Input image/hash/crop mismatch:{name}')
        for folder in ['frames', 'sfm/block_00/images']:
            (out/folder/name).symlink_to(image.resolve())
        for folder in folders[1:4]:
            mask = source/folder/f'{frame:06d}.png'
            if Image.open(mask).size != (666, 370):
                raise ValueError(f'Mask crop mismatch:{mask}')
            shutil.copy2(mask, out/folder/mask.name)
        selected.append(records[name])
    write_json(out/'frames_manifest.json', selected)
    write_json(out/'plan.json', dict(fps=10, blocks=[p['block']], first_anchor=p['reference'], source_video=p['source_video']))
    shutil.copy2(p['settings'], out/'matching_settings.json')
    write_json(out/'upstream_inputs.json', dict(reference_and_settings_hashes=hashes, input=str(source)))


def reuse_sfm(p):
    """Explicit read-only reuse after matching current RGB/static masks to the SfM input audit."""
    from ..sfm.reconstruction import static_masks
    source = Path(p['input'])/'sfm'/f'block_{p["source_block_index"]:02d}'
    cloud = Path(p['output'])/'cloud'
    destination = cloud/'sfm/block_00'
    static_masks(cloud, 0, p['block'])
    expected = json.loads((source/'input_audit.json').read_text())
    records = json.loads((cloud/'frames_manifest.json').read_text())
    current = dict(count=len(records), first=p['block'][0], stop=p['block'][1], anchor_stop=p['block'][2],
                   images_sha256={r['image']:r['sha256'] for r in records},
                   masks_sha256={r['image']:sha(cloud/'masks/sfm_static'/(r['image']+'.png')) for r in records})
    if current != expected:
        raise ValueError('Reused SfM image/static-mask inputs differ; recompute without --reuse-sfm')
    shutil.copytree(source/'anchored/sparse', destination/'anchored/sparse')
    with sqlite3.connect(f'file:{source/"database.db"}?mode=ro', uri=True) as src, sqlite3.connect(destination/'database.db') as dst:
        src.backup(dst)
    for name in ['input_audit.json', 'audit.json', 'cameras.npz']:
        shutil.copy2(source/name, destination/name)
    paths = [destination/'cameras.npz', destination/'database.db', *sorted((destination/'anchored/sparse/0').glob('*.bin'))]
    write_json(destination/'reuse.json', dict(source=str(source), input_audit_matches=True,
                                             copied_artifacts={str(path):sha(path) for path in paths}))


def select_window_pairs(roma, first, stop):
    """Restrict expensive matching, not the SfM geometry used to score pair candidates."""
    path = roma/'manifest.json'
    manifest = json.loads(path.read_text())
    original = manifest['policies']['B1']
    names = {f'{n:06d}.jpg' for n in range(first, stop)}
    pairs = [pair for pair in original if any(name in names for name in pair)]
    if not pairs or not names <= {name for pair in pairs for name in pair}:
        raise ValueError('Some selected reference frames have no geometry-selected neighbors')
    manifest.update(full_batch_pair_budget=len(original), pair_budget=len(pairs), policies={'B1':pairs},
                    reference_image_names=sorted(names))
    write_json(path, manifest)
    (roma/'pairs.txt').write_text(''.join(' '.join(pair)+'\n' for pair in pairs))


def run(args, p):
    out = Path(p['output'])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'plan.json', p)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR='/tmp/sanhuan-window-mpl', HF_HUB_OFFLINE='1', QT_QPA_PLATFORM='offscreen')
    status = dict(status='running', completed_stages=[], started=time.strftime('%F %T'))
    started = time.monotonic()

    def step(name, command):
        status['stage'] = name
        write_json(out/'progress.json', status)
        tick = time.monotonic()
        with (out/(name+'.log')).open('x') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed:{result.returncode}; see {out/(name+".log")}')
        status['completed_stages'].append(dict(stage=name, seconds=time.monotonic()-tick))

    cloud, roma, dense = out/'cloud', out/'cloud/block_00/roma', out/'dense_road'
    cameras, config_path = cloud/'sfm/block_00/cameras.npz', out/'gs_config.json'
    count, first, stop = p['views'], p['first_frame'], p['stop_frame']
    interval = ['--first-frame', first, '--stop-frame', stop]
    python = sys.executable
    try:
        status['stage'] = 'inputs';write_json(out/'progress.json', status)
        prepare_inputs(p)
        if args.reuse_sfm:
            status['stage'] = 'reuse_sfm';write_json(out/'progress.json', status)
            reuse_sfm(p)
        else:
            step('sfm', [args.python_sfm, '-u', '-m', 'GLRMap_GS.pipeline.cloud', 'sfm', cloud])
        step('matching_inputs', [python, '-u', '-m', 'GLRMap_GS.pipeline.cloud',
                                'matching-inputs', cloud, '--settings', cloud/'matching_settings.json'])
        select_window_pairs(roma, first, stop)
        for stage in ['match', 'triangulate', 'road']:
            command = [python, '-u', '-m', 'GLRMap_GS.pipeline.reconstruct', stage, roma]
            if stage != 'match':command += interval
            if stage == 'road':command += ['--output', dense]
            step(stage, command)
        step('point_cloud_png', [python, '-m', 'GLRMap_GS.visualization.report', dense,
                                '--image-name', f'{first+count//2:06d}.jpg', '--png-only'])
        step('road_assets', [args.python_resplat, '-m', 'GLRMap_GS.pipeline.gs', 'road-assets',
                            '--dense', dense, '--cameras', cameras, '--road-masks', cloud/'masks/road',
                            '--dynamic-masks', cloud/'masks/vehicle_valid', '--first-frame', first,
                            '--num-frames', count, '--tangent-patch-index', 0, '--output', out/'road_assets'])
        shared = ['--images', cloud/'frames', '--cameras', cameras, '--frame-manifest', cloud/'frames_manifest.json',
                  '--out', out/'resplat', '--start', first, '--stop', stop, '--segment-frames', count,
                  '--label', 'window_'+hashlib.sha256(str(out).encode()).hexdigest()[:16]]
        for stage in ['prepare', 'run']:
            step('resplat_'+stage, [args.python_resplat, '-u', ROOT/'tools/run_resplat_forward_windows.py', stage, *shared])
        assets, native = out/'road_assets', out/'resplat/segments/segment_00'
        config = dict(output=str(out/'gs'), first_frame=first, stop_frame=stop, fps=10,
            images=str(cloud/'frames'), cameras=str(cameras), reference_cameras=str(assets/'reference_cameras.npz'),
            resplat_ply=str(native/'gaussians.ply'), resplat_video=str(native/'video.mp4'),
            road_ply=str(assets/'road_gaussians.ply'), road_constraints=str(assets/'road_constraints.npz'),
            road_depth=str(assets/'road_depth_targets.npz'), road_masks=str(assets/'covered_road_masks'),
            dynamic_masks=str(cloud/'masks/vehicle_valid'), sky_diagnostic_masks=str(cloud/'sky_all/foreground_valid_hard'),
            gpu_lock=str(GPU_LOCK), steps_per_view_per_phase=10,
            checkpoint_every=100, lateral_anchors=[count//4, count//2, 3*count//4],
            python_resplat=args.python_resplat, python_gs=args.python_gs)
        write_json(config_path, config)
        step('gs_train', [args.python_resplat, '-u', '-m', 'GLRMap_GS.pipeline.gs',
                          'run', '--config', config_path])
        step('render', [args.python_resplat, '-u', '-m', 'GLRMap_GS.visualization.gs',
                        '--config', config_path, '--png-only'])
        training = json.loads((out/'gs/training_report.json').read_text())
        rendered = json.loads((out/'gs/visualization_complete.json').read_text())
        if not (training['complete'] and training['views']==count and training['iterations']==count*20
                and training['road_fields_bit_exact'] and rendered['complete']):
            raise ValueError('Final GS execution checks failed')
        status.update(status='complete', stage='complete', elapsed_seconds=time.monotonic()-started,
                      execution_checks_passed=True, visual_quality_requires_review=True, final_ply=training['final_PLY'])
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}:{error}')
        raise
    finally:
        write_json(out/'progress.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'run'])
    parser.add_argument('--input', type=Path, required=True, help='Prepared10FPS batch images/masks/metadata')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite an existing experiment')
    parser.add_argument('--first-frame', type=int, required=True)
    parser.add_argument('--stop-frame', type=int, required=True, help='Exclusive')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--reuse-sfm', action='store_true', help='Explicitly reuse audited input SfM; recompute local matching onward')
    parser.add_argument('--python-sfm', default=python_for('sfm'))
    parser.add_argument('--python-resplat', default=python_for('resplat'))
    parser.add_argument('--python-gs', default=python_for('gs'))
    args = parser.parse_args()
    p = plan(args)
    if args.stage == 'plan':print(json.dumps(p, ensure_ascii=False, indent=2))
    else:run(args, p)


if __name__ == '__main__':main()
