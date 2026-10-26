"""Current native-camera lateral diagnostics; no legacy extended-trajectory inference."""
import numpy as np


def lateral_weights(frames):
    if frames < 20:
        raise ValueError("At least 20 frames are needed")
    times=np.linspace(0.,1.,frames)
    knots=np.array([0.,.05,.25,.30,.50,.70,.75,.95,1.])
    values=np.array([0.,0.,-1.,-1.,0.,1.,1.,0.,0.])
    segment=np.clip(np.searchsorted(knots,times,side="right")-1,0,len(knots)-2)
    progress=(times-knots[segment])/(knots[segment+1]-knots[segment])
    eased=.5-.5*np.cos(np.pi*progress)
    return values[segment]+eased*(values[segment+1]-values[segment])


def lateral(extrinsics, intrinsics, amplitude, anchor=None, frames=96):
    """anchor=None advances along native cameras; an index freezes one camera first."""
    extrinsics, intrinsics = np.asarray(extrinsics), np.asarray(intrinsics)
    if extrinsics.shape != (len(intrinsics), 4, 4) or intrinsics.shape[1:] != (3, 3):
        raise ValueError('Expected native world-to-camera poses and pixel K')
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Positive amplitude in SfM world units required')
    if anchor is None:
        source = np.linalg.inv(extrinsics)
        target_K = intrinsics.copy()
    else:
        if not 0 <= anchor < len(extrinsics):
            raise ValueError('Lateral anchor outside segment')
        source = np.repeat(np.linalg.inv(extrinsics[anchor])[None], frames, axis=0)
        target_K = np.repeat(intrinsics[anchor][None], frames, axis=0)
    shifted = source.copy()
    right = source[:, :3, 0]/np.linalg.norm(source[:, :3, 0], axis=1)[:, None]
    offsets = amplitude*lateral_weights(len(source))
    shifted[:, :3, 3] += offsets[:, None]*right
    # Express displacement in each original camera frame, not arbitrary world axes.
    local = np.einsum('nij,nj->ni', np.linalg.inv(source[:, :3, :3]), shifted[:, :3, 3]-source[:, :3, 3])
    audit = dict(max_local_vertical=float(abs(local[:, 1]).max()),
                 max_local_forward=float(abs(local[:, 2]).max()),
                 max_rotation_change=float(abs(shifted[:, :3, :3]-source[:, :3, :3]).max()),
                 amplitude_world=amplitude, world_units_are_not_meters=True,
                 no_novel_view_ground_truth=True, sequence='center_left_center_right_center')
    if max(audit[key] for key in ['max_local_vertical', 'max_local_forward', 'max_rotation_change']) > 1e-8:
        raise ValueError(f'Non-lateral camera motion: {audit}')
    return dict(extrinsics=np.linalg.inv(shifted), intrinsics=target_K, lateral_offsets_world=offsets), audit
