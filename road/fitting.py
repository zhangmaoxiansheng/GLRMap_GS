"""Overlapping RoMa road fits with a deterministic source-row holdout."""
from pathlib import Path
import numpy as np
from ..io_utils import write_json
from .surface import GRID, camera_basis, fit_surface, evaluate


def fit_patches(road, full_rows, reference_views, poses, output):
    output = Path(output)
    (output/'surfaces').mkdir()
    xyz = road['xyz'].astype(float)
    strict = ((road['track_length'] >= 4) & (road['angle'] >= 3)
              & (road['reprojection'] <= .5) & (road['max_reprojection'] <= 1))
    patches, reports = [], []
    for core in range(0, len(poses), 100):
        first, stop = max(0, core-20), min(len(poses), core+120)
        origin, forward, lateral, up, _ = camera_basis(poses[first:stop])
        basis = np.stack([forward, lateral, up])
        rows = np.flatnonzero((reference_views >= first) & (reference_views < stop) & strict)
        local = (xyz[rows]-origin) @ basis.T
        hold = full_rows[rows] % 5 == 0
        surface, _, _ = fit_surface(local[~hold], GRID, seed=54+core)
        height, normals = evaluate(surface, local[:, :2])
        error = abs(height-local[:, 2])*normals[:, 2]
        training = (error <= GRID*3) & ~hold
        if training.sum() < 1000:
            raise ValueError(f'Patch {core//100}: insufficient training consensus ({training.sum()})')
        surface['support_min'] = np.percentile(local[training, :2], .25, axis=0)
        surface['support_max'] = np.percentile(local[training, :2], 99.75, axis=0)
        supported = ((local[:, :2] >= surface['support_min']-.05).all(1)
                     & (local[:, :2] <= surface['support_max']+.05).all(1))
        held = error[hold & supported]
        quantiles = np.percentile(held, [50, 90, 99]).tolist() if len(held) else []
        passed = len(held) >= 500 and quantiles[1] <= GRID*5
        report = dict(first=first, stop=stop, strict_tracks=len(rows), training_consensus=int(training.sum()),
                      heldout_supported=len(held), heldout_p50_p90_p99=quantiles, passed=bool(passed))
        reports.append(report)
        write_json(output/'fit_report.json', dict(
            patches=reports, grid_SfM=GRID,
            protocol='>=1000 training consensus, >=500 supported heldout, P90<=5GRID; '
                     'support from training consensus only, no heldout height selection.'))
        if not passed:
            raise ValueError(f'Patch {core//100} failed road fit: {report}')
        # Persist the actual temporal support; downstream readers must not infer it from a filename.
        save_patch(output/'surfaces'/f'patch_{core//100:02d}.npz', first, stop, surface, origin, basis)
        patches.append(dict(first=first, stop=stop, surface=surface, origin=origin, basis=basis))
    return patches


def save_patch(path, first, stop, surface, origin, basis):
    np.savez_compressed(path, **surface, origin=origin, basis=basis, first=first, stop=stop)


def load_patches(folder):
    """Read this branch's explicit-window, five-coefficient road artifacts."""
    paths = sorted(Path(folder).glob('patch_*.npz'))
    if not paths:
        raise FileNotFoundError(f'No road patches in {folder}')
    patches = []
    for path in paths:
        with np.load(path) as arrays:
            first, stop = int(arrays['first']), int(arrays['stop'])
            surface = {key: arrays[key] for key in
                       ['coefficients', 'offset', 'scale', 'support_min', 'support_max']}
            origin, basis = arrays['origin'], arrays['basis']
        if first < 0 or stop <= first or surface['coefficients'].shape != (5,):
            raise ValueError(f'Invalid quadratic road patch/window: {path}')
        patches.append(dict(first=first, stop=stop, surface=surface, origin=origin, basis=basis))
    return patches
