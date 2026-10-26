"""Configuration, immutable inputs and exclusive GPU ownership for the fixed-road GS stage."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
from ..io_utils import sha, write_json

from ..runtime import ROOT, GS_ROOT, GPU_LOCK, python_for
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)


def effective(path):
    c=json.loads(Path(path).read_text())
    for name in ['output','images','cameras','reference_cameras','resplat_ply','resplat_video','road_ply',
        'road_constraints','road_depth','road_masks','dynamic_masks','sky_diagnostic_masks','gpu_lock']:
        if name in c:c[name]=str((ROOT/c[name]).resolve())
    n=c['stop_frame']-c['first_frame']
    c.setdefault('fps',10);c.setdefault('context_indices',np.linspace(0,n-1,16,dtype=int).tolist())
    c.setdefault('steps_per_view_per_phase',10);c.setdefault('checkpoint_every',250)
    c.setdefault('python_resplat',python_for('resplat'))
    c.setdefault('python_gs',python_for('gs'))
    c.setdefault('gpu_lock',str(GPU_LOCK))
    if c['fps']!=10:raise ValueError('This road pipeline defaults to10FPS; alternate rates require a separately reviewed configuration')
    if not isinstance(c['steps_per_view_per_phase'],int) or c['steps_per_view_per_phase']<=0:raise ValueError('Positive integer per-view training budget required')
    if not isinstance(c['checkpoint_every'],int) or c['checkpoint_every']<=0:raise ValueError('Positive checkpoint interval required')
    c['implementation']='GLRMap_GS.gs'
    for name in ['images','cameras','resplat_ply','road_ply','road_constraints','road_depth','road_masks','dynamic_masks']:
        if not Path(c[name]).exists():raise FileNotFoundError(c[name])
    fingerprint=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    return c,fingerprint


@contextmanager
def gpu(c):
    lock=Path(c['gpu_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        print('Waiting for GPU lock',lock,flush=True);fcntl.flock(f,fcntl.LOCK_EX)
        yield


def cameras(c):
    with np.load(c['cameras']) as z:
        names=z['image_names'].astype(str);ids=[i for i,n in enumerate(names) if c['first_frame']<=int(Path(n).stem)<c['stop_frame']]
        E,K= z['extrinsics'][ids].copy(),z['intrinsics'][ids].copy();names=names[ids]
    expected=[f'{i:06d}.jpg' for i in range(c['first_frame'],c['stop_frame'])]
    n=len(expected)
    if names.tolist()!=expected or n<20:
        raise ValueError('Native ReSplat requires ordered contiguous JPEG cameras,16 contexts and heldout views')
    if not np.isfinite(E).all() or not np.isfinite(K).all():raise ValueError('Nonfinite cameras')
    if E.shape!=(n,4,4) or K.shape!=(n,3,3):raise ValueError('Expected 4x4 world-to-camera and 3x3 K')
    context=np.asarray(c['context_indices'])
    if not np.array_equal(context,np.linspace(0,n-1,16,dtype=int)):
        raise ValueError('Native export uses 16 uniform, view-major contexts')
    if not np.allclose(K,K[0],rtol=0,atol=1e-10):raise ValueError('Use one consistent crop/K per segment')
    return E,K,names


def verify(out,c,fingerprint):
    manifest=json.loads((out/'input_manifest.json').read_text())
    if manifest['config_fingerprint']!=fingerprint:raise ValueError('Configuration changed; select a new output directory')
    for section in ['source_hashes','snapshot_hashes','code_hashes']:
        for p,h in manifest[section].items():
            if sha(p)!=h:raise ValueError('Immutable input/code changed: '+p)
    return manifest


def code_files():
    """Own GS code plus the shared road math and actual training implementation."""
    review = Path(__file__).resolve().parents[1]
    paths = list((review/'gs').glob('*.py'))
    paths += [review/'pipeline/gs.py', review/'io_utils.py', review/'runtime.py',
              review/'road/sampling.py', review/'road/surface.py', review/'road/fitting.py',
              GS_ROOT/'arguments/__init__.py', GS_ROOT/'scene/__init__.py',
              GS_ROOT/'scene/colmap_loader.py', GS_ROOT/'scene/gaussian_model.py',
              GS_ROOT/'gaussian_renderer/__init__.py', GS_ROOT/'utils/general_utils.py',
              GS_ROOT/'utils/loss_utils.py', GS_ROOT/'utils/sky_mask_utils.py']
    return sorted(path for path in paths if not path.name.startswith('test_'))
