"""Optional direct DA3 error screening; no XYZ edits, occlusion or leave-out pruning."""
from dataclasses import dataclass
import numpy as np

@dataclass
class FilterConfig:
    relative_depth_threshold: float = .2
    min_conflict_views: int = 2
    conflict_fraction: float = .5


def geometry_protected_mask(points,min_views=3,min_angle=3.,max_reprojection=1.):
    """Protect supported, well-conditioned points using existing statistics.

    No new leave-out tests are computed. This is an operational protection
    rule, not a claim of independently measured correctness.
    """
    angle=np.asarray(points['angle']);error=np.asarray(points['max_reprojection'])
    return ((np.asarray(points['track_length'])>=min_views)
        &np.isfinite(angle)&(angle>=min_angle)
        &np.isfinite(error)&(error<=max_reprojection))


def select_supported_depth_conflicts(camera_z, aligned_depth, usable, config=None):
    """Symmetric depth error at the actual tracked pixels.

    usable denotes high-confidence static depth samples, independent of XYZ.
    No occlusion or cross-view depth-surface consistency checks are performed.
    Both points ahead of and behind the prediction may vote for rejection.
    """
    config = config or FilterConfig()
    z=np.asarray(camera_z);d=np.asarray(aligned_depth)
    good=np.asarray(usable,bool)&np.isfinite(z)&np.isfinite(d)&(z>0)&(d>0)
    relative=(z-d)/np.maximum(d,1e-12)
    conflict=good&(np.abs(relative)>config.relative_depth_threshold)
    negative=conflict.sum(1);evidence=good.sum(1)
    reject=(negative>=config.min_conflict_views)&(negative>config.conflict_fraction*evidence)
    return dict(keep=~reject,reject=reject,conflict_views=negative,evaluable_views=evidence,
        agreement_views=(good&~conflict).sum(1),observation_conflicts=conflict,
        observation_agrees=good&~conflict)
