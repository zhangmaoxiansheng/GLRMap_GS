"""Explicit fixed-road GS stages. Native ReSplat inference and visualization are separate."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
from ..gs.state import ROOT, cameras, effective, verify, write
from ..io_utils import sha


def plan(config, fingerprint):
    """Summarize a validated config using its cameras; never initialize or render GS."""
    _, _, names = cameras(config)
    steps = len(names)*config['steps_per_view_per_phase']
    return dict(config_fingerprint=fingerprint, views=len(names), fps=10, phase_steps=steps,
                total_steps=steps*2, output=config['output'], stages=['prepare', 'train'],
                road_freeze='all six tensors; no pose/K optimization',
                Gaussian_densification=False, Gaussian_pruning=False,
                RGB='dynamic valid mask only; real sky supervised',
                input_boundary='Prepared RoMa road assets and native ReSplat16v PLY; no model inference')


def stage_files(config, stage):
    out = Path(config['output'])
    if stage == 'prepare':
        return [out/name for name in ['static_road_init.ply', 'resplat_colmap_world.ply',
                'initialization_lineage.npz', 'inputs/road_depth.npz', 'data/input_colmap_cameras.npz',
                'initialization_audit.json', 'world_conversion_audit.json']]
    report = out/'training_report.json'
    return [report, Path(json.loads(report.read_text())['final_PLY'])]


def run(config, fingerprint, config_path, resume):
    """Own prepare/train subprocesses and committed stage markers; rendering is separate."""
    out = Path(config['output'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent/(out.name+'.pipeline.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists() and any(out.iterdir()) and not resume:
            raise ValueError('Existing output: pass --resume or choose a new directory')
        if (out/'input_manifest.json').exists():
            verify(out, config, fingerprint)
        out.mkdir(exist_ok=True)
        write(out/'plan.json', plan(config, fingerprint))
        write(out/'effective_config.json', config)
        for stage, python in [('prepare', config['python_resplat']), ('train', config['python_gs'])]:
            marker = out/(stage+'.done.json')
            if marker.exists():
                verify(out, config, fingerprint)
                record = json.loads(marker.read_text())
                if record['config_fingerprint'] != fingerprint:
                    raise ValueError('Stage configuration changed')
                if any(sha(path) != digest for path, digest in record['artifact_hashes'].items()):
                    raise ValueError('Stage artifact changed')
                continue
            write(out/'status.json', dict(status='running', stage=stage))
            with (out/(stage+'.log')).open('a') as log:
                result = subprocess.run([python, '-u', '-m', 'GLRMap_GS.pipeline.gs',
                                         stage, '--config', str(config_path)], cwd=ROOT,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write(out/'status.json', dict(status='interrupted' if result.returncode == 75 else 'failed',
                                             stage=stage, returncode=result.returncode))
                return result.returncode
            write(marker, dict(config_fingerprint=fingerprint,
                               artifact_hashes={str(path): sha(path) for path in stage_files(config, stage)}))
        write(out/'status.json', dict(status='training_complete', rendering_not_run=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'prepare', 'train', 'run', 'status', 'road-assets'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Save full optimizer/RNG state and exit75 at this training iteration')
    parser.add_argument('--dense', type=Path)
    parser.add_argument('--cameras', type=Path)
    parser.add_argument('--road-masks', type=Path)
    parser.add_argument('--dynamic-masks', type=Path)
    parser.add_argument('--first-frame', type=int)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--tangent-patch-index', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'road-assets':
        keys = ['dense', 'cameras', 'road_masks', 'dynamic_masks', 'first_frame', 'output']
        if any(getattr(args, key) is None for key in keys):
            parser.error('road-assets requires --dense --cameras --road-masks --dynamic-masks --first-frame --output')
        from ..gs.road import prepare_assets
        print(json.dumps(prepare_assets(**{key: getattr(args, key) for key in keys},
                         num_frames=args.num_frames, tangent_patch_index=args.tangent_patch_index), ensure_ascii=False, indent=2))
        return
    if args.config is None:
        parser.error('--config is required')
    config, fingerprint = effective(args.config.resolve())
    if args.stage == 'plan':
        print(json.dumps(plan(config, fingerprint), ensure_ascii=False, indent=2))
    elif args.stage == 'status':
        for name in ['status.json', 'progress.json', 'training_report.json']:
            path = Path(config['output'])/name
            if path.exists():
                print(name, path.read_text())
    elif args.stage == 'run':
        raise SystemExit(run(config, fingerprint, args.config.resolve(), args.resume))
    elif args.stage == 'prepare':
        from ..gs.initialization import prepare
        prepare(config, fingerprint)
    else:
        from ..gs.training import train
        raise SystemExit(train(config, fingerprint, args.stop_after) or 0)


if __name__ == '__main__':
    main()
