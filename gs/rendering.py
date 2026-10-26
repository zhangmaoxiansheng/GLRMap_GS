"""Common gsplat renderer used to audit initialization coordinate conversion.

Video/HTML publishing remains in visualization; this module only renders audit images.
"""
import numpy as np
import torch
from .conversion import coefficients


def sh3_gpu_model(vertices):
    coeffs,_,degree = coefficients(vertices)
    result = dict(
        means=torch.from_numpy(np.column_stack([vertices[a] for a in 'xyz']).copy()).cuda(),
        quats=torch.from_numpy(np.column_stack([vertices[f'rot_{i}'] for i in range(4)]).copy()).cuda(),
        scales=torch.from_numpy(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).copy()).cuda().exp(),
        opacities=torch.from_numpy(vertices['opacity'].copy()).cuda().sigmoid(),
        colors=torch.from_numpy(coeffs.copy()).cuda(),sh_degree=degree)
    return result


def gpu_model(vertices):
    if any(n.startswith('f_rest_') for n in vertices.dtype.names):
        return sh3_gpu_model(vertices)
    return dict(means=torch.from_numpy(np.column_stack([vertices[n] for n in 'xyz']).copy()).cuda(),
        quats=torch.from_numpy(np.column_stack([vertices[f'rot_{i}'] for i in range(4)]).copy()).cuda(),
        scales=torch.from_numpy(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).copy()).cuda().exp(),
        opacities=torch.from_numpy(vertices['opacity'].copy()).cuda().sigmoid(),
        colors=torch.from_numpy(np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)]).copy()).cuda()[:, None],
        sh_degree=0)


def camera_render(model, E, K, width=666, height=370, marker=None):
    from gsplat.rendering import rasterization
    fields = model.copy()
    if marker is not None:
        fields['colors'] = marker
        fields['sh_degree'] = None
    with torch.no_grad():
        rgbd, alpha, _ = rasterization(**fields,
            viewmats=torch.tensor(np.array(E, dtype=np.float32), device='cuda')[None],
            Ks=torch.tensor(np.array(K, dtype=np.float32), device='cuda')[None],
            width=width, height=height, near_plane=.01, far_plane=200.,
            eps2d=.1, rasterize_mode='antialiased', packed=True,
            absgrad=False, sparse_grad=False, render_mode='RGB+ED')
    rgb = rgbd[0, ..., :3].cpu().numpy().clip(0, 1)
    return rgb, alpha[0, ..., 0].cpu().numpy()
