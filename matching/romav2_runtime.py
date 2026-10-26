"""Local, offline RoMa v2 loading; no changes to upstream source or old caches."""
from contextlib import contextmanager
import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys
from ..io_utils import sha

# Reuse the existing offline source/weight installation; never import old pipelines.
from ..runtime import ROOT, ROMA_ASSETS as BASE, ROMAV2_SOURCE, DINOV3_SOURCE
ROMAV2_REVISION = '95c9968145c8906b7b59383258e9f73b02853d89'
DINOV3_REVISION = 'adc254450203739c8149213a7a69d8d905b4fcfa'
WEIGHT_URL = 'https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt'
def local_path(value):
    path = (BASE / value).resolve()
    if BASE not in path.parents:
        raise ValueError(f'Checkpoint must remain under configured roma_assets: {value}')
    return path


def import_local_romav2():
    source = (ROMAV2_SOURCE / 'src').resolve()
    sys.path.insert(0, str(source))
    install_local_imports()
    module = importlib.import_module('romav2')
    if source not in Path(module.__file__).resolve().parents:
        raise RuntimeError('RoMa v2 was imported from another project')
    return module


def install_local_imports():
    """Keep inherited editable-install finders from redirecting this package."""
    if any(getattr(finder, '_superdim_local_romav2', False) for finder in sys.meta_path):
        for finder in sys.meta_path:
            if getattr(finder, '_superdim_local_romav2', False):
                sys.meta_path.remove(finder)
                sys.meta_path.insert(0, finder)
                return

    class LocalRoMaFinder(importlib.abc.MetaPathFinder):
        _superdim_local_romav2 = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'romav2':
                search = [str(ROMAV2_SOURCE / 'src')]
            elif fullname.startswith('romav2.'):
                search = path
            else:
                return None
            return importlib.machinery.PathFinder.find_spec(fullname, search)

    sys.meta_path.insert(0, LocalRoMaFinder())


@contextmanager
def offline_hub(checkpoint):
    import torch
    original_weights = torch.hub.load_state_dict_from_url
    original_hub_load = torch.hub.load
    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(ROOT / 'cache/torch/hub'))
    local_dino = DINOV3_SOURCE
    if not (local_dino / 'hubconf.py').is_file():
        raise FileNotFoundError('Local pinned DINOv3 source is missing')

    def weights(url, *args, **kwargs):
        if url != WEIGHT_URL:
            raise RuntimeError(f'Unexpected weight download request: {url}')
        return torch.load(checkpoint, map_location=kwargs.get('map_location', 'cpu'), weights_only=True)

    def hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != f'facebookresearch/dinov3:{DINOV3_REVISION}' or model != 'dinov3_vitl16':
            raise RuntimeError(f'Unexpected hub source: {repo_or_dir}/{model}')
        kwargs.pop('skip_validation', None)
        return original_hub_load(str(local_dino), model, *args, source='local', **kwargs)

    torch.hub.load_state_dict_from_url = weights
    torch.hub.load = hub_load
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original_weights
        torch.hub.load = original_hub_load
        torch.hub.set_dir(original_hub_dir)


def load_model(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this process; use a GPU-enabled execution session')
    checkpoint = local_path(config['checkpoint'])
    if sha(checkpoint) != config['checkpoint_sha256']:
        raise RuntimeError('RoMa v2 checkpoint SHA256 mismatch')
    module = import_local_romav2()
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(8)
    with offline_hub(checkpoint):
        model = module.RoMaV2(module.RoMaV2.Cfg(setting=config['setting'], compile=False))
    model.bidirectional = config['bidirectional']
    model.threshold = config['threshold']
    if model.threshold is not None:
        raise ValueError('This pipeline preserves continuous overlap; threshold must be null')
    if config['local_correlation'] != 'native_torch':
        raise ValueError('Only the verified native torch correlation backend is configured')
    importlib.import_module('romav2.local_correlation').local_corr = None
    chunk_size = config['kde_chunk_size']
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('kde_chunk_size must be a positive integer')

    def bounded_kde(x, std=0.1, half=True):
        if half:
            x = x.half()
        return torch.cat([
            (-(torch.cdist(x[start:start+chunk_size], x)**2)/(2*std**2)).exp().sum(-1)
            for start in range(0, len(x), chunk_size)
        ])

    # Same KDE expression as upstream; bound intermediate memory only.
    importlib.import_module('romav2.romav2').kde = bounded_kde
    return model.eval()


def query_field(field, xy, image_hw):
    """Query a BHWC dense field in original COLMAP edge-origin pixels."""
    import torch
    from torch.nn import functional as F
    h, w = image_hw
    xy = torch.as_tensor(xy, device=field.device, dtype=field.dtype)
    grid = xy / xy.new_tensor([w, h]) * 2 - 1
    return F.grid_sample(field.permute(0, 3, 1, 2), grid.reshape(1, 1, -1, 2),
                         mode='bilinear', align_corners=False, padding_mode='zeros')[0, :, 0].T
