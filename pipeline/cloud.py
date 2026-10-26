"""Prepared 1000-view blocks → SfM → native RoMa input → tracks → dense road → report.

Image extraction and mask inference belong to data_preparation. This queue requires
their complete inputs and never imports an experiment module or changes its globals.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from ..runtime import ROOT, path, python_for
from ..io_utils import sha, write_json

# This queue's fixed experiment budget. plan() and execution share one definition;
# --settings supplies matcher/filter thresholds, not a second copy of these counts.
INITIAL_PAIR_BUDGET = 1800
PAIR_BUDGET = 3600
SEEDS_PER_IMAGE = 20000
REFERENCE_CHUNK = 100


def plan(root):
    """Read prepared metadata only; this does not check files or acquire a GPU."""
    root = Path(root)
    source = json.loads((root/'plan.json').read_text())
    if source['fps'] != 10:
        raise ValueError('Prepared road inputs must use the established10FPS timeline')
    blocks = source['blocks']
    for index, (first, stop, anchor_stop) in enumerate(blocks):
        if stop-first != 1000 or anchor_stop-first != 100:
            raise ValueError('Current cloud queue uses 1000-view blocks and 100-view anchors')
        if index and first != blocks[index-1][1]-100:
            raise ValueError('Neighboring SfM blocks require 100-view overlap')
    if not blocks:
        raise ValueError('No prepared blocks')
    return dict(output=str(root), blocks=blocks, unique_views=blocks[-1][1]-blocks[0][0], fps=10,
                initial_pair_budget=INITIAL_PAIR_BUDGET, pair_budget=PAIR_BUDGET, seeds_per_image=SEEDS_PER_IMAGE,
                reference_chunk=REFERENCE_CHUNK, source_video=source['source_video'], first_anchor=source['first_anchor'],
                stages=['sfm', 'matching-inputs', 'match', 'triangulate', 'road', 'report'],
                mask_inference=False, GS_training=False, feedforward_depth=False)


def matching_inputs(root, index, block, settings):
    """Create private low/native inputs, then select pairs; includes FFmpeg extraction.

    root is the prepared queue, not roma/. This stage requires fresh low_source/
    and roma/ directories; completed-stage recovery is owned by run().
    """
    from ..sfm.source import prepare_source
    from ..sfm.geometry import model_data
    from ..matching.pairs import select_pairs
    first, stop, _ = block
    destination = root/f'block_{index:02d}'
    destination.mkdir(exist_ok=True)
    low = destination/'low_source'
    low.mkdir()
    sfm = root/'sfm'/f'block_{index:02d}'
    shutil.copytree(sfm/'anchored/sparse', low/'sparse')
    with sqlite3.connect(f'file:{sfm/"database.db"}?mode=ro', uri=True) as src, sqlite3.connect(low/'database.db') as dst:
        src.backup(dst)
    for folder in ['images', 'static_valid_masks', 'road_masks']:
        (low/folder).mkdir()
    for frame in range(first, stop):
        stem = f'{frame:06d}'
        (low/'images'/f'{stem}.jpg').symlink_to((root/'frames'/f'{stem}.jpg').resolve())
        for folder, source in [('static_valid_masks', root/f'masks/sfm_static/{stem}.jpg.png'),
                               ('road_masks', root/f'masks/road/{stem}.png')]:
            shutil.copy2(source, low/folder/f'{stem}.png')
    records = {row['image']: row for row in json.loads((root/'frames_manifest.json').read_text())}
    metadata = json.loads((root/'plan.json').read_text())
    # Sanhuan frame 0 is at10s on the experiment timeline; video_time_nominal
    # is the independent extraction timestamp in the original long source file.
    signature = dict(start_seconds=10+first/10, duration_seconds=100,
                     fps=10, frame_count=1000, video_start_seconds=records[f'{first:06d}.jpg']['video_time_nominal'],
                     video_path=metadata['source_video'])
    write_json(low/'manifest.json', dict(status='complete', signature=signature))
    roma = destination/'roma'
    roma.mkdir()
    source = prepare_source(roma, low)
    model, images, _ = model_data(source/'sparse/0')
    pairs, scores = select_pairs(model, images, INITIAL_PAIR_BUDGET, PAIR_BUDGET)
    reference = json.loads(Path(settings).read_text())
    manifest = dict(status='prepared', source=str(source), source_segment=signature, policies={'B1': pairs},
                    pair_budget=len(pairs), seeds_per_image=SEEDS_PER_IMAGE, per_pair_cap=2*SEEDS_PER_IMAGE,
                    matcher_overrides=reference['matcher_overrides'], filter_settings=reference['filter_settings'],
                    holdout_validation=False, viewer_branches=['FUSED_BEFORE', 'FUSED_AFTER', 'ROAD_shared', 'ROAD_fitted'],
                    viewer_labels={'FUSED_BEFORE': '原始RoMa完整点云', 'FUSED_AFTER': '非道路＋稠密曲面道路',
                                   'ROAD_shared': '原始RoMa道路', 'ROAD_fitted': '射线曲面稠密道路'},
                    viewer_title=f'三环 {first:06d}–{stop:06d} · 1000视角',
                    portable_note='展示点云最多25万点／组；投影与BEV使用全量点云。曲面道路没有独立匹配Track。')
    write_json(roma/'manifest.json', manifest)
    write_json(destination/'selection.json', scores)
    (roma/'pairs.txt').write_text(''.join(' '.join(pair)+'\n' for pair in pairs))
    for folder in ['direct_pairs', 'seeds', 'superpoint', 'sky_masks/foreground_valid_hard']:
        (roma/'shared'/folder).mkdir(parents=True)
    for mask in (source/'static_valid_masks').glob('*.png'):
        shutil.copy2(mask, roma/'shared/sky_masks/foreground_valid_hard'/mask.name)


def matching_cache_sizes(roma):
    """Required cache files, not merely a report; report-only retired caches cannot resume."""
    manifest = json.loads((roma/'manifest.json').read_text())
    paths = [roma/'shared/matching_report.json']
    names = set()
    for a, b in manifest['policies']['B1']:
        names.update([a, b])
        path = roma/'shared/direct_pairs'/f'{Path(a).stem}_{Path(b).stem}.npz'
        paths += [path, path.with_suffix('.json')]
    for name in names:
        paths += [roma/'shared/seeds'/f'{Path(name).stem}.npz', roma/'shared/superpoint'/f'{Path(name).stem}.pt']
    result = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing matching cache; explicitly rebuild it before triangulation: {path}')
        result[str(path)] = path.stat().st_size
    return result


def artifacts(root, index, stage):
    destination = root/f'block_{index:02d}'
    roma, sfm, road = destination/'roma', root/'sfm'/f'block_{index:02d}', destination/'dense_road'
    if stage == 'sfm':
        return [sfm/name for name in ['audit.json', 'cameras.npz', 'database.db',
                'anchored/sparse/0/cameras.bin', 'anchored/sparse/0/images.bin', 'anchored/sparse/0/points3D.bin']]
    if stage == 'matching-inputs':
        return [roma/name for name in ['manifest.json', 'pairs.txt', 'source/manifest.json', 'source/database.db',
                'source/sparse/0/cameras.bin', 'source/sparse/0/images.bin', 'source/sparse/0/points3D.bin']]+[
                path for folder in ['source/images', 'source/static_valid_masks', 'source/road_masks',
                                    'shared/sky_masks/foreground_valid_hard'] for path in sorted((roma/folder).iterdir())]
    if stage == 'match':
        matching_cache_sizes(roma)
        return [roma/'shared/matching_report.json']
    if stage == 'triangulate':
        return [roma/'B1_shared'/name for name in ['metrics.json', 'new_points.npz', 'new_points.ply', 'point_observations.npz']]
    folder = road/'comparison' if stage == 'report' else road
    return sorted(path for path in folder.rglob('*') if path.is_file())


def execute(args, block):
    root, index = args.root, args.block
    destination = root/f'block_{index:02d}'
    roma = destination/'roma'
    if args.stage == 'sfm':
        from ..sfm.reconstruction import run_block
        reference = Path(plan(root)['first_anchor']) if index == 0 else root/'sfm'/f'block_{index-1:02d}/cameras.npz'
        run_block(root, index, block, reference, repo_root=ROOT, colmap=args.colmap, glomap=args.glomap)
    elif args.stage == 'matching-inputs':
        matching_inputs(root, index, block, args.settings)
    elif args.stage == 'match':
        from ..matching.geometry_guided import match_all
        match_all(roma)
    elif args.stage == 'triangulate':
        from ..triangulation.batch import triangulate
        matching_cache_sizes(roma)
        triangulate(roma, chunk_size=REFERENCE_CHUNK)
    elif args.stage == 'road':
        from ..road.reconstruct import run
        run(roma, destination/'dense_road')
    else:
        from ..visualization.report import run
        run(destination/'dense_road', png_only=args.png_only)


def run(args):
    root = args.root
    queue = plan(root)
    # Prepared images/masks are owned upstream. Hash masks and metadata, while the
    # SfM feature boundary checks each RGB against frames_manifest.json.
    inputs = [root/'plan.json', root/'frames_manifest.json', args.settings, Path(queue['first_anchor'])]
    for first, stop, _ in queue['blocks']:
        for frame in range(first, stop):
            for folder in ['masks/vehicle_valid', 'masks/road', 'sky_all/foreground_valid_hard']:
                inputs.append(root/folder/f'{frame:06d}.png')
    code = list(ROOT.glob('*.py')) + [ROOT/'config/romav2.json']
    for folder in ['sfm', 'matching', 'triangulation', 'road', 'depth_prior', 'gs', 'visualization', 'pipeline', 'tools']:
        code.extend(p for p in (ROOT/folder).iterdir()
                    if p.suffix in ['.py', '.html'] and not p.name.startswith('test_'))
    manifest = dict(inputs={str(path): sha(path) for path in set(inputs)},
                    code={str(path): sha(path) for path in code},
                    environments=[args.python_sfm, args.python_matching], colmap=args.colmap, glomap=str(args.glomap))
    if args.png_only:
        manifest['png_only'] = True
    with (root/'cloud_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root/'cloud_queue_manifest.json'
        if marker.exists():
            if not args.resume or json.loads(marker.read_text()) != manifest:
                raise ValueError('Existing/changed queue: use --resume with identical inputs and code')
        else:
            if any((root/f'block_{i:02d}/roma').exists() or (root/'sfm'/f'block_{i:02d}/database.db').exists()
                   for i in range(len(queue['blocks']))):
                raise ValueError('Use a fresh prepared root; historical reconstruction outputs are not adopted')
            write_json(marker, manifest)
        for index, block in enumerate(queue['blocks']):
            for stage in queue['stages']:
                done = root/f'block_{index:02d}_{stage}.done.json'
                if done.exists():
                    record = json.loads(done.read_text())
                    if any(sha(path) != digest for path, digest in record['artifacts'].items()):
                        raise ValueError(f'Changed completed artifact: {done}')
                    if stage == 'match' and matching_cache_sizes(root/f'block_{index:02d}/roma') != record['cache_sizes']:
                        raise ValueError('Matching cache inventory changed; rebuild explicitly')
                    continue
                write_json(root/'progress.json', dict(stage=stage, block=index, updated_at=time.strftime('%F %T')))
                python = args.python_sfm if stage == 'sfm' else args.python_matching
                command = [python, '-u', '-m', 'GLRMap_GS.pipeline.cloud', stage,
                           str(root), '--block', str(index), '--settings', str(args.settings),
                           '--colmap', args.colmap, '--glomap', str(args.glomap)]
                if args.png_only:
                    command.append('--png-only')
                with (root/f'block_{index:02d}_{stage}.log').open('a') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write_json(root/'progress.json', dict(stage='failed', failed_stage=stage, block=index, returncode=result.returncode))
                    return result.returncode
                record = dict(artifacts={str(path): sha(path) for path in artifacts(root, index, stage)})
                if stage == 'match':
                    record['cache_sizes'] = matching_cache_sizes(root/f'block_{index:02d}/roma')
                write_json(done, record)
        write_json(root/'progress.json', dict(stage='complete', blocks=len(queue['blocks']), GS_training=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'run', 'sfm', 'matching-inputs', 'match', 'triangulate', 'road', 'report'])
    parser.add_argument('root', type=Path, help='Fresh prepared root: plan.json, frames_manifest.json, native images/masks/SfM image folders')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--settings', type=Path, help='Existing RoMa matcher_overrides/filter_settings manifest')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--png-only', action='store_true', help='Generate PNG reports without HTML')
    parser.add_argument('--colmap', default=str(path('colmap','/usr/local/bin/colmap')))
    parser.add_argument('--glomap', type=Path, default=path('glomap', ROOT.parent/'glomap/build/glomap/glomap'))
    parser.add_argument('--python-sfm', default=python_for('sfm'))
    parser.add_argument('--python-matching', default=python_for('matching'))
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.settings:
        args.settings = args.settings.resolve()
    queue = plan(args.root)
    if args.stage == 'plan':
        print(json.dumps(queue, ensure_ascii=False, indent=2))
    elif args.stage == 'run':
        if args.settings is None:
            parser.error('run requires --settings')
        raise SystemExit(run(args))
    else:
        if not 0 <= args.block < len(queue['blocks']):
            parser.error('--block outside prepared blocks')
        if args.stage == 'matching-inputs' and args.settings is None:
            parser.error('matching-inputs requires --settings')
        execute(args, queue['blocks'][args.block])


if __name__ == '__main__':
    main()
