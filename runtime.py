"""Project-relative inputs and explicit local installations of external backends."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
config_file = Path(os.environ.get('GLRMAP_GS_PATHS', ROOT/'paths.local.json')).expanduser()
PATHS = json.loads(config_file.read_text()) if config_file.exists() else {}


def path(name, default):
    return (ROOT/Path(PATHS.get(name, default)).expanduser()).resolve()


GS_ROOT = path('gs_backend', ROOT.parent/'gaussian-splatting')
ROMA_ASSETS = path('roma_assets', ROOT)
ROMAV2_SOURCE = path('romav2_source', ROOT/'third_party/RoMaV2')
DINOV3_SOURCE = path('dinov3_source', ROOT/'third_party/dinov3')
SPARSE_WEIGHTS = path('sparse_weights', Path.home()/'.cache/torch/hub/checkpoints')
RESPLAT = path('resplat', ROOT.parent/'resplat')
GPU_LOCK = path('gpu_lock', ROOT/'cache/gpu.lock')


def python_for(name):
    return str(path('python_'+name, Path.home()/'miniconda3/envs'/name/'bin/python'))
