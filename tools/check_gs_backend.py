"""Check the pinned Gaussian backend and optionally exercise its CUDA training API."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from ..io_utils import sha
from ..runtime import ROOT, GS_ROOT, GPU_LOCK


def check_sources(backend):
    manifest = json.loads((ROOT/'patches/gaussian-backend.json').read_text())
    revisions = {'.': manifest['backend']['revision'],
                 'submodules/diff-gaussian-rasterization': manifest['rasterizer']['revision'],
                 **manifest['submodules']}
    for name, expected in revisions.items():
        actual = subprocess.check_output(['git', '-C', str(backend/name), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError('Backend revision mismatch: '+name)
    for section, folder in [('backend', backend),
                            ('rasterizer', backend/'submodules/diff-gaussian-rasterization')]:
        spec = manifest[section]
        patch = ROOT/'patches'/spec['patch']
        if sha(patch) != spec['patch_sha256']:
            raise ValueError('Patch checksum mismatch: '+str(patch))
        for name, expected in spec['patched_files_sha256'].items():
            if sha(folder/name) != expected:
                raise ValueError('Backend source differs from the distributed patch: '+str(folder/name))
    return manifest


def cuda_check(backend):
    import numpy as np
    import torch
    from PIL import Image
    from plyfile import PlyData, PlyElement
    if not torch.cuda.is_available():
        raise RuntimeError('--cuda requires a CUDA GPU and a CUDA-enabled PyTorch')
    sys.path.insert(0, str(backend))
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import masked_l1_loss, masked_ssim
    from utils.sky_mask_utils import ValidMaskSupervisor
    if Path(inspect.getfile(Scene)).resolve() != backend/'scene/__init__.py':
        raise ValueError('A different scene package was imported')
    if 'init_ply' not in inspect.signature(Scene).parameters:
        raise ValueError('Scene lacks init_ply')
    # Verify gradients are zero at ignored pixels, including SSIM windows crossing
    # the mask boundary. This must hold before involving the renderer.
    prediction = torch.full((3, 16, 16), .25, device='cuda', requires_grad=True)
    truth = torch.zeros_like(prediction)
    valid = torch.ones((1, 16, 16), device='cuda')
    valid[:, :, :8] = 0
    loss = masked_l1_loss(prediction, truth, valid)+1-masked_ssim(prediction, truth, valid)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :8]) == 0
    assert torch.count_nonzero(prediction.grad[:, :, 8:]) > 0
    with tempfile.TemporaryDirectory(prefix='glrmap_gs_backend_') as directory:
        root = Path(directory)
        source, output = root/'source', root/'trained'
        for folder in [source/'images', source/'sparse/0', root/'masks', output]:
            folder.mkdir(parents=True)
        rows = []
        for index in range(2):
            name = f'{index:06d}.jpg'
            Image.new('RGB', (32, 32), (80, 100, 120)).save(source/'images'/name)
            Image.fromarray(np.full((32, 32), 255, np.uint8)).save(root/'masks'/f'{index:06d}.png')
            rows.append(f'{index+1} 1 0 0 0 {-index*.05} 0 0 1 {name}\n\n')
        (source/'sparse/0/cameras.txt').write_text('1 PINHOLE 32 32 32 32 16 16\n')
        (source/'sparse/0/images.txt').write_text(''.join(rows))
        (source/'sparse/0/points3D.txt').write_text('# Camera-only test scene\n')
        fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
        fields += [f'f_rest_{i}' for i in range(45)]
        fields += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        points = np.zeros(3, dtype=[(name, 'f4') for name in fields])
        points['x'] = [-.1, 0., .1]
        points['z'] = 2.
        points['opacity'] = np.log(.7/.3)
        for index in range(3):
            points[f'scale_{index}'] = np.log(.05)
        points['rot_0'] = 1.
        ply = root/'init.ply'
        PlyData([PlyElement.describe(points, 'vertex')], text=False).write(ply)
        parser = argparse.ArgumentParser()
        lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
        args = parser.parse_args(['-s', str(source), '-m', str(output), '--resolution', '1',
                                  '--iterations', '1', '--position_lr_max_steps', '1'])
        model = GaussianModel(3, 'default')
        scene = Scene(lp.extract(args), model, init_ply=str(ply), shuffle=False)
        np.testing.assert_array_equal(model._xyz.detach().cpu().numpy(),
                                      np.column_stack([points[key] for key in 'xyz']))
        assert model._features_rest.shape == (3, 15, 3)
        assert model._exposure.shape == (2, 3, 4)
        model.training_setup(op.extract(args))
        camera = scene.getTrainCameras()[0]
        valid = ValidMaskSupervisor(root/'masks').target_for(camera)
        assert valid.shape == (1, 32, 32) and valid.min() == 1
        package = render(camera, model, pp.extract(args), torch.zeros(3, device='cuda'),
                         use_trained_exp=False, separate_sh=False)
        required = ['render', 'depth', 'mode_depth', 'mode_ids', 'mode_weights', 'accumulated_alpha']
        assert all(name in package for name in required)
        assert package['render'].shape == (3, 32, 32)
        assert torch.isfinite(package['depth']).all() and package['depth'].max() > 0
        loss = masked_l1_loss(package['render'], camera.original_image.cuda(), valid)
        loss.backward()
        assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all()
        assert model._features_dc.grad is not None and torch.isfinite(model._features_dc.grad).all()
        model.optimizer.step()
        assert torch.isfinite(model._xyz).all()
    torch.cuda.synchronize()
    rasterizer = importlib.import_module('diff_gaussian_rasterization')
    rasterizer_cuda = importlib.import_module('diff_gaussian_rasterization._C')
    knn_cuda = importlib.import_module('simple_knn._C')
    return dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                torch_cuda=torch.version.cuda, init_ply=True, masked_gradients=True,
                CUDA_render_backward_optimizer_step=True,
                rasterizer_python=rasterizer.__file__, rasterizer_cuda=rasterizer_cuda.__file__,
                simple_knn_cuda=knn_cuda.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, default=GS_ROOT)
    parser.add_argument('--cuda', action='store_true', help='Run tiny PLY/mask/render/backward checks on the GPU')
    parser.add_argument('--output', type=Path, help='Save the check report as JSON')
    args = parser.parse_args()
    backend = args.backend.resolve()
    manifest = check_sources(backend)
    result = dict(source_checks_passed=True, backend=str(backend),
                  backend_revision=manifest['backend']['revision'],
                  rasterizer_revision=manifest['rasterizer']['revision'])
    if args.cuda:
        from ..gs.state import gpu
        with gpu(dict(gpu_lock=str(GPU_LOCK))):
            result['cuda'] = cuda_check(backend)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
