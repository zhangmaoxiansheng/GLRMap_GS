"""Explicit optional cached-depth stages; no automatic activation in reconstruction/GS."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    prepare = stages.add_parser('prepare', help='Validate and reuse exact processed RGB/Pose/K')
    prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--cached-rgb', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True, help='New input_conditions.npz file')
    export = stages.add_parser('export', help='Camera-only scale alignment and confidence-ranked clouds')
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--prior', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--confidence-quantile', type=float, default=.7)
    refine = stages.add_parser('refine', help='Top30 fixed-camera XYZ constraint/replacement and coverage fill')
    refine.add_argument('--reference', type=Path, required=True)
    refine.add_argument('--prior', type=Path, required=True)
    refine.add_argument('--output', type=Path, required=True)
    refine.add_argument('--depth-sigma', type=float, default=.02)
    refine.add_argument('--iterations', type=int, default=8)
    anchors = stages.add_parser('anchors', help='Read-only sparse-anchor assessment of top30')
    anchors.add_argument('--reference', type=Path, required=True)
    anchors.add_argument('--prior', type=Path, required=True)
    anchors.add_argument('--output', type=Path, required=True)
    band = stages.add_parser('band', help='Add anchor-approved rank30–60 points to the original top30 fusion')
    band.add_argument('--reference', type=Path, required=True)
    band.add_argument('--prior', type=Path, required=True)
    band.add_argument('--baseline', type=Path, required=True)
    band.add_argument('--output', type=Path, required=True)
    band.add_argument('--median-error', type=float, default=.05)
    band.add_argument('--p90-error', type=float, default=.10)
    screen = stages.add_parser('filter-da3', help='Protected geometry + direct 20 percent depth error screening')
    screen.add_argument('--reference', type=Path, required=True)
    screen.add_argument('--prior', type=Path, required=True, help='Existing conditioned prior.npz')
    screen.add_argument('--output', type=Path, required=True)
    screen.add_argument('--relative-depth-threshold', type=float, default=.2)
    args = parser.parse_args()
    if args.stage == 'prepare':
        from ..depth_prior.mapping import prepare_conditions
        prepare_conditions(args.source, args.cached_rgb, args.output)
    elif args.stage == 'export':
        from ..depth_prior.cloud import export_prior
        export_prior(args.prior, args.source, args.output, args.confidence_quantile)
    elif args.stage == 'refine':
        from ..depth_prior.track_pipeline import refine_tracks
        refine_tracks(args.reference, args.prior, args.output, depth_sigma=args.depth_sigma, iterations=args.iterations)
    elif args.stage == 'anchors':
        from ..depth_prior.anchor_maps import assess_anchors
        assess_anchors(args.reference, args.prior, args.output)
    elif args.stage == 'band':
        from ..depth_prior.band_pipeline import relax_band
        relax_band(args.reference, args.prior, args.baseline, args.output, args.median_error, args.p90_error)
    else:
        from ..depth_prior.track_pipeline import filter_da3
        filter_da3(args.reference, args.prior, args.output, args.relative_depth_threshold)


if __name__ == '__main__':
    main()
