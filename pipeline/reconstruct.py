"""Run organized matching, triangulation or road reconstruction on prepared inputs."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['match', 'triangulate', 'road'])
    parser.add_argument('root', type=Path, help='Prepared experiment with manifest.json and shared/ inputs')
    parser.add_argument('--policy', default='B1', help='Triangulation pair policy in manifest.json')
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--output', type=Path, help='Required fresh output directory for road')
    parser.add_argument('--first-frame', type=int)
    parser.add_argument('--stop-frame', type=int)
    args = parser.parse_args()
    if args.stage == 'match':
        from ..matching.geometry_guided import match_all
        match_all(args.root.resolve())
    elif args.stage == 'triangulate':
        from ..triangulation.batch import triangulate
        triangulate(args.root.resolve(), args.policy, args.chunk_size, args.first_frame, args.stop_frame)
    else:
        if args.output is None:
            parser.error('road requires --output; input triangulation remains read-only')
        from ..road.reconstruct import run
        run(args.root.resolve(), args.output.resolve(), branch=f'{args.policy}_shared',
            first_frame=args.first_frame, stop_frame=args.stop_frame)


if __name__ == '__main__':
    main()
