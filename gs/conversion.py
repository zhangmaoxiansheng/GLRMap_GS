"""Native 16x128x240 ReSplat XYZ, source-camera covariance and SH3 to fixed SfM world."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def coefficients(vertices):
    names = sorted([n for n in vertices.dtype.names if n.startswith('f_rest_')], key=lambda n:int(n.rsplit('_',1)[1]))
    count = 1 + len(names) // 3
    degree = int(np.sqrt(count)) - 1
    assert len(names) % 3 == 0 and (degree + 1)**2 == count
    dc = np.stack([vertices[f'f_dc_{i}'] for i in range(3)], axis=-1)
    rest = np.stack([vertices[n] for n in names], axis=-1)
    return np.concatenate([dc[:,None,:],rest.reshape(-1,3,count-1).transpose(0,2,1)],axis=1), names, degree


def conversion(vertices, scene, contexts):
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast
    pivot = scene['c2w'][contexts[8]].astype(np.float64)
    poses = torch.from_numpy(scene['c2w'][contexts])
    relative = (torch.linalg.inv(poses[8])[None] @ poses).numpy()
    export_rotation = np.linalg.inv(relative[8, :3, :3])
    undo = np.linalg.inv(export_rotation)
    xyz = np.column_stack([vertices[n] for n in 'xyz'])
    centered_positions = xyz @ undo.T
    world = vertices.copy()
    world_positions = centered_positions @ pivot[:3, :3].T + pivot[:3, 3]
    for j, axis in enumerate('xyz'):
        world[axis] = world_positions[:, j]
    per_view = len(world)//16
    assert per_view == 128*240 and len(world) == 16*per_view
    local_quats = np.column_stack([vertices[n] for n in ('rot_1', 'rot_2', 'rot_3', 'rot_0')])
    centered = vertices.copy()
    for j, axis in enumerate('xyz'):
        centered[axis] = centered_positions[:, j]
    for view in range(16):
        block = slice(view*per_view, (view+1)*per_view)
        for target, matrix in ((centered, relative[view, :3, :3] @ undo),
                               (world, pivot[:3, :3] @ relative[view, :3, :3] @ undo)):
            q = (Rotation.from_matrix(matrix) * Rotation.from_quat(local_quats[block])).as_quat()
            for field, column in (('rot_0', 3), ('rot_1', 0), ('rot_2', 1), ('rot_3', 2)):
                target[field][block] = q[:, column]
    coeffs, rest_names, degree = coefficients(vertices)
    assert degree == 3
    directions = np.random.default_rng(19).normal(size=(512, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rotation = Rotation.from_matrix(pivot[:3, :3]).as_matrix()
    world_basis = _eval_sh_bases_fast(16, torch.from_numpy(directions)).numpy()
    local_basis = _eval_sh_bases_fast(16, torch.from_numpy(directions @ rotation)).numpy()
    transform = np.zeros((16, 16))
    for band in range(4):
        block = slice(band**2, (band+1)**2)
        transform[block, block] = np.linalg.lstsq(world_basis[:, block], local_basis[:, block], rcond=None)[0]
    error = float(np.max(abs(world_basis @ transform-local_basis)))
    assert error < 1e-6
    rotated = np.einsum('ij,njc->nic', transform.astype(np.float32), coeffs, optimize=True)
    for j in range(3):
        world[f'f_dc_{j}'] = rotated[:, 0, j]
    rest = rotated[:, 1:].transpose(0, 2, 1).reshape(len(world), -1)
    for j, name in enumerate(rest_names):
        world[name] = rest[:, j]
    for name in ('opacity', 'scale_0', 'scale_1', 'scale_2'):
        assert np.array_equal(world[name], vertices[name])
    return world, centered, pivot, dict(SH3_basis_max_error=error, source_view_rotation_applied=True,
        gaussians_per_view=per_view, normalization='Original float32 pivot-relative poses; rigid transform only, no scale fitting')
