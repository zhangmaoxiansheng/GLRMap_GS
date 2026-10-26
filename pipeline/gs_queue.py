"""Ordered 100-view fixed-road GS jobs, 25-view overlap, standalone render reports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
from ..gs.state import ROOT, cameras, code_files, effective, verify, write
from ..io_utils import sha
from ..visualization.html import media_report
from ..visualization.video import join_forward
from .gs import plan as group_plan


def windows(first, stop):
    if stop-first < 100 or (stop-first-100) % 75:
        raise ValueError('Use 100-view windows at stride75; a 1000-view interval gives 13 jobs')
    return [(start, start+100) for start in range(first, stop-99, 75)]


def overlap(older, newer):
    """Same 25 cameras plus byte-identical Gaussian fields on shared dense source rows."""
    from plyfile import PlyData
    ae, ak, an = cameras(older)
    be, bk, bn = cameras(newer)
    count = older['stop_frame']-newer['first_frame']
    if count != 25 or not (np.array_equal(an[-count:], bn[:count])
            and np.array_equal(ae[-count:], be[:count]) and np.array_equal(ak[-count:], bk[:count])):
        raise ValueError('GS neighbors must share exactly25 identical camera names, poses and K')
    with np.load(Path(older['road_ply']).parent/'source_rows.npz') as a, np.load(Path(newer['road_ply']).parent/'source_rows.npz') as b:
        common, ia, ib = np.intersect1d(a['dense_road_source_rows'], b['dense_road_source_rows'], return_indices=True)
    ga = PlyData.read(older['road_ply'], mmap='r')['vertex'].data
    gb = PlyData.read(newer['road_ply'], mmap='r')['vertex'].data
    if not len(common) or ga.dtype != gb.dtype or any(not np.array_equal(ga[key][ia], gb[key][ib]) for key in ga.dtype.names):
        raise ValueError('Overlapping road source rows must have identical Gaussian attributes')
    return dict(training_overlap_views=count, pose_K_names_bit_exact=True, common_road_rows=len(common),
                common_road_all_fields_bit_exact=True, global_frames=[newer['first_frame'], older['stop_frame']])


def read(path):
    """Read prepared window configs; relative paths resolve against the repository root.

    This validates ranges/cameras, but the heavier common-road-row audit belongs
    to run(). No ReSplat inference, road-asset preparation or training happens here.
    """
    queue = json.loads(Path(path).read_text())
    queue['output'] = str((ROOT/queue['output']).resolve())
    queue['configs'] = [str((ROOT/path).resolve()) for path in queue['configs']]
    jobs = [effective(path) for path in queue['configs']]
    if not jobs or len({config['output'] for config, _ in jobs}) != len(jobs):
        raise ValueError('Unique prepared GS outputs required')
    intervals = [(config['first_frame'], config['stop_frame']) for config, _ in jobs]
    if intervals != windows(intervals[0][0], intervals[-1][1]):
        raise ValueError('GS jobs must follow100-view / stride75 intervals')
    for config, fingerprint in jobs:
        group_plan(config, fingerprint)
    return queue, jobs


def plan(queue, jobs):
    plans = [group_plan(config, fingerprint) for config, fingerprint in jobs]
    return dict(output=queue['output'], groups=len(jobs), fps=10, training_overlap_views=25,
                unique_views=jobs[-1][0]['stop_frame']-jobs[0][0]['first_frame'],
                intervals=[[c['first_frame'], c['stop_frame']] for c, _ in jobs],
                iterations_per_group=[p['total_steps'] for p in plans],
                total_iterations=sum(p['total_steps'] for p in plans), model_fusion=False,
                minimum_free_disk_GiB=queue.get('minimum_free_disk_GiB', 20))


def verify_group(config, fingerprint):
    out = Path(config['output'])
    verify(out, config, fingerprint)
    report = json.loads((out/'training_report.json').read_text())
    budget = config['steps_per_view_per_phase']
    if (report['iterations'] != 200*budget or not report['road_fields_bit_exact'] or len(report['phase_counts']) != 2
            or any(len(counts) != 100 or set(counts.values()) != {budget} for counts in report['phase_counts'])
            or sha(report['final_PLY']) != report['final_sha256']):
        raise ValueError('GS training report/final PLY failed verification')
    with np.load(config['reference_cameras']) as arrays:
        E, K, names = cameras(config)
        if not (np.array_equal(arrays['extrinsics'], E) and np.array_equal(arrays['intrinsics'], K)
                and np.array_equal(arrays['image_names'], names)):
            raise ValueError('Window reference cameras changed')
    rendering = json.loads((out/'visualization_complete.json').read_text())
    if rendering['config_fingerprint'] != fingerprint or any(sha(path) != digest for path, digest in rendering['artifact_hashes'].items()):
        raise ValueError('Rendering configuration/artifact changed')
    return dict(final_PLY=report['final_PLY'], final_sha256=report['final_sha256'],
                road_count=report['road_count'], background_count=report['Gaussian_count']-report['road_count'],
                training_seconds=report['elapsed_seconds'], html=str(out/'render_comparison.html'))


def run(queue, jobs, path, resume):
    out = Path(queue['output'])
    out.parent.mkdir(parents=True, exist_ok=True)
    files = code_files()+[Path(__file__)]+[p for p in Path(__file__).parents[1].joinpath('visualization').iterdir()
                 if p.suffix in ['.py', '.html'] and not p.name.startswith('test_')]
    manifest = dict(queue_sha256=sha(path), configs={path: sha(path) for path in queue['configs']},
                    code={str(path): sha(path) for path in files})
    with (out.parent/(out.name+'.queue.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out/'queue_manifest.json').exists() and out.exists() and any(out.iterdir()):
            raise ValueError('Use a fresh queue output; prepared per-window assets/configs remain separate')
        out.mkdir(exist_ok=True)
        marker = out/'queue_manifest.json'
        if marker.exists() and (not resume or json.loads(marker.read_text()) != manifest):
            raise ValueError('Existing/changed queue: use --resume with identical configuration and code')
        write(marker, manifest)
        write(out/'plan.json', plan(queue, jobs))
        overlaps = [overlap(a[0], b[0]) for a, b in zip(jobs, jobs[1:])]
        write(out/'overlap_audit.json', dict(passed=True, overlaps=overlaps))
        child, interrupted = [None], [False]
        def stop(signum, frame):
            interrupted[0] = True
            if child[0] is not None and child[0].poll() is None:
                try:
                    os.killpg(child[0].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        old_handlers = {sig: signal.signal(sig, stop) for sig in [signal.SIGTERM, signal.SIGINT]}
        results, started = [], time.monotonic()
        try:
            (out/'html').mkdir(exist_ok=True)
            for index, ((config, fingerprint), config_path) in enumerate(zip(jobs, queue['configs'])):
                if shutil.disk_usage(out).free/2**30 < queue.get('minimum_free_disk_GiB', 20)+2:
                    write(out/'progress.json', dict(stage='stopped_for_disk_reserve', group=index))
                    return 76
                for module, extra in [('pipeline.gs', ['run', '--resume']), ('visualization.gs', [])]:
                    if interrupted[0]:
                        write(out/'progress.json', dict(stage='interrupted', group=index))
                        return 75
                    render_done = Path(config['output'])/'visualization_complete.json'
                    if module == 'visualization.gs' and render_done.exists():
                        verify_group(config, fingerprint)
                        continue
                    write(out/'progress.json', dict(stage=module, group=index, completed_groups=len(results)))
                    command = [config['python_resplat'], '-u', '-m', 'GLRMap_GS.'+module,
                               *extra, '--config', config_path]
                    with (out/f'group_{index:02d}_{module}.log').open('a') as log:
                        child[0] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        code = child[0].wait()
                        child[0] = None
                    if interrupted[0] or code:
                        write(out/'progress.json', dict(stage='interrupted' if interrupted[0] or code == 75 else 'failed',
                                                       group=index, returncode=code))
                        return 75 if interrupted[0] else code
                result = verify_group(config, fingerprint)
                results.append(result)
                shutil.copy2(result['html'], out/'html'/f'group_{index:02d}.html')
                write(out/'group_results.json', results)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        if interrupted[0]:
            write(out/'progress.json', dict(stage='interrupted', completed_groups=len(results)))
            return 75
        video = out/'joined_forward.mp4'
        join_forward([dict(first_frame=c['first_frame'], stop_frame=c['stop_frame'],
                           video=Path(c['output'])/'trained_forward.mp4') for c, _ in jobs],
                     video, queue.get('blend_frames', 10))
        media_report(out/'html/full_forward.html', '重叠窗口完整轨迹预览',
                     '相邻25帧共同监督，重叠区同时间溶解；这是视频预览，未融合高斯模型。', [('全轨迹', video)])
        write(out/'summary.json', dict(complete=True, groups=len(results), model_fusion=False,
              elapsed_seconds=time.monotonic()-started, standalone_html=str(out/'html/full_forward.html')))
        write(out/'progress.json', dict(stage='complete', completed_groups=len(results)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'run', 'status'])
    parser.add_argument('--queue', type=Path, required=True, help='JSON with output and ordered prepared configs')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    queue, jobs = read(args.queue.resolve())
    if args.stage == 'plan':
        print(json.dumps(plan(queue, jobs), ensure_ascii=False, indent=2))
    elif args.stage == 'status':
        path = Path(queue['output'])/'progress.json'
        print(path.read_text() if path.exists() else 'Not started')
    else:
        raise SystemExit(run(queue, jobs, args.queue.resolve(), args.resume))


if __name__ == '__main__':
    main()
