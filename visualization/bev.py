"""Equal-scale photographic BEV and side view; never flatten or resample source XYZ."""
import numpy as np
from ..road.surface import camera_basis
from ..io_utils import sha, write_json
from .clouds import load_cloud


def compare(specs, poses, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin, forward, lateral, up, _ = camera_basis(poses)
    basis = np.stack([forward, lateral, up])
    bounds, records = [], []
    for spec in specs:
        points = load_cloud(spec['path'])
        local = (points['xyz'].astype(float)-origin) @ basis.T
        if len(local):
            bounds.append(np.percentile(local, [.2, 99.8], axis=0))
        records.append(dict(name=spec['name'], input_count=len(local), source_sha256=sha(spec['path'])))
    if not bounds:
        raise ValueError('No points for BEV comparison')
    low, high = np.min(np.array(bounds)[:, 0], axis=0), np.max(np.array(bounds)[:, 1], axis=0)
    margin = np.maximum((high-low)*.025, .01)
    low, high = low-margin, high+margin
    fig, axes = plt.subplots(len(specs), 2, figsize=(15, 4*len(specs)), squeeze=False, layout='constrained')
    for row, (spec, record) in enumerate(zip(specs, records)):
        points = load_cloud(spec['path'])
        local = (points['xyz'].astype(float)-origin) @ basis.T
        for column, axis in enumerate([1, 2]):
            ax = axes[row, column]
            ax.scatter(local[:, 0], local[:, axis], s=.7, c=points['rgb']/255., linewidths=0, rasterized=True)
            ax.set_facecolor('#111820')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[axis], high[axis])
            ax.set_xlabel('Forward / SfM units')
            ax.set_ylabel(('Lateral' if axis == 1 else 'Height')+' / SfM units')
            ax.set_title(f'{spec["label"]}: {record["input_count"]:,} points')
    fig.suptitle('Original XYZ/RGB; common frame, bounds and point size; no road flattening')
    fig.savefig(output, dpi=180)
    plt.close(fig)
    report = dict(origin=origin.tolist(), basis=basis.tolist(), bounds=[low.tolist(), high.tolist()],
                  point_size_pt_squared=.7, display_bounds_percentiles=[.2, 99.8], all_points_used=True,
                  xyz_modified=False, world_units_are_not_meters=True, clouds=records)
    write_json(output.with_suffix('.json'), report)
    return report

