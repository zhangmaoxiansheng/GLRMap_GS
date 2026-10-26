"""Local anchor evidence and rank-band gating; no legacy comparisons or point edits."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from ..matching.sparse import valid_mask
from ..sfm.geometry import angle_degrees
from .sampling import bilinear

NOT_SELECTED, RELIABLE, UNKNOWN, UNRELIABLE = range(4)
STATUS_NAMES = ['not_selected', 'reliable', 'unknown', 'unreliable']
REASONS = {0:'not_selected', 1:'insufficient_anchors', 2:'anchor_agreement',
           3:'anchor_conflict', 4:'mixed_or_borderline_residuals',
           5:'pose_or_scale_untrusted', 6:'depth_or_mask_boundary'}

@dataclass
class AnchorConfig:
    min_views: int = 3
    max_reprojection_px: float = 2.
    min_angle_deg: float = 3.
    patch_size: int = 32  # processed Pi3X pixels; limits propagation distance
    min_local_anchors: int = 3
    surface_log_jump: float = .05  # adjacent depth pixels, not a flat-plane test
    good_log_error: float = float(np.log(1.10))
    bad_log_error: float = float(np.log(1.20))
    max_log_mad: float = .05
    unknown_strength: float = .1
    max_pose_angle_deg: float = 3.
    max_center_error_fraction: float = .02


def surface_components(depth, valid, config):
    """4-connected depth-continuous components confined to each image grid cell."""
    h, w = depth.shape
    yy, xx = np.indices((h, w)); ids = np.arange(h*w).reshape(h, w)
    cell = (yy//config.patch_size)*((w+config.patch_size-1)//config.patch_size)+xx//config.patch_size
    logd = np.log(np.maximum(np.nan_to_num(depth), 1e-12))
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:] & valid[:-1]
    jump_h = np.abs(logd[:, 1:]-logd[:, :-1]) > config.surface_log_jump
    jump_v = np.abs(logd[1:]-logd[:-1]) > config.surface_log_jump
    boundary = np.zeros_like(valid)
    bh = (valid[:, 1:] != valid[:, :-1]) | (horizontal & jump_h)
    bv = (valid[1:] != valid[:-1]) | (vertical & jump_v)
    boundary[:, 1:] |= bh; boundary[:, :-1] |= bh
    boundary[1:] |= bv; boundary[:-1] |= bv
    horizontal &= ~jump_h & (cell[:, 1:] == cell[:, :-1])
    vertical &= ~jump_v & (cell[1:] == cell[:-1])
    a = np.r_[ids[:, :-1][horizontal], ids[:-1][vertical]]
    b = np.r_[ids[:, 1:][horizontal], ids[1:][vertical]]
    graph = coo_matrix((np.ones(len(a), np.uint8), (a, b)), shape=(h*w, h*w)).tocsr()
    _, labels = connected_components(graph, directed=False)
    return labels.reshape(h, w), boundary


def sample_anchors(depth, valid, labels, boundary, xy, sfm_z):
    """Use real track pixels, with no interpolation across surface/mask edges."""
    d, inside = bilinear(depth, xy)
    h, w = depth.shape
    q = np.nan_to_num(xy, nan=0., posinf=0., neginf=0.)
    x = np.clip(np.floor(q[:, 0]).astype(int), 0, w-1)
    y = np.clip(np.floor(q[:, 1]).astype(int), 0, h-1)
    x1 = np.minimum(x+1, w-1); y1 = np.minimum(y+1, h-1)
    component = labels[y, x]
    good = inside & np.isfinite(d) & (d>0) & np.isfinite(sfm_z) & (sfm_z>0)
    for yy, xx in [(y,x), (y,x1), (y1,x), (y1,x1)]:
        good &= valid[yy,xx] & ~boundary[yy,xx] & (labels[yy,xx] == component)
    residual = np.full(len(xy), np.nan)
    residual[good] = np.log(d[good]/sfm_z[good])
    return residual, component, good


def local_status(depth, valid, top30, anchor_xy, anchor_z, anchor_ids, config, trusted=True):
    labels, boundary = surface_components(depth, valid, config)
    residual, component, usable = sample_anchors(depth, valid, labels, boundary, anchor_xy, anchor_z)
    # A SfM point is either a gate anchor or an assessment point in ALL views.
    heldout = anchor_ids % 5 == 0
    size = int(labels.max())+1
    count = np.zeros(size, np.int32)
    med_abs = np.full(size, np.nan, np.float32); signed = med_abs.copy(); mad = med_abs.copy()
    selected = np.flatnonzero(usable & ~heldout)
    order = selected[np.argsort(component[selected], kind='stable')]
    cuts = np.r_[0, np.flatnonzero(np.diff(component[order]))+1, len(order)]
    for start, end in zip(cuts[:-1], cuts[1:]):
        ids = order[start:end]
        if not len(ids): continue
        group = component[ids[0]]
        _, first = np.unique(anchor_ids[ids], return_index=True)
        values = residual[ids[first]]
        count[group] = len(values)
        signed[group] = np.median(values); med_abs[group] = np.median(np.abs(values))
        mad[group] = 1.4826*np.median(np.abs(values-signed[group]))
    status = np.full(depth.shape, NOT_SELECTED, np.uint8); status[top30] = UNKNOWN
    reason = np.zeros(depth.shape, np.uint8); reason[top30] = 1
    enough = top30 & (count[labels] >= config.min_local_anchors) & ~boundary
    reason[enough] = 4
    good = enough & (med_abs[labels] <= config.good_log_error) & (mad[labels] <= config.max_log_mad)
    bad = enough & (med_abs[labels] >= config.bad_log_error)
    status[good] = RELIABLE; reason[good] = 2
    status[bad] = UNRELIABLE; reason[bad] = 3
    reason[top30 & boundary] = 6
    if not trusted:
        status[top30] = UNKNOWN; reason[top30] = 5
    weight = np.zeros(depth.shape, np.float32)
    weight[status == RELIABLE] = 1.
    weight[(status == UNKNOWN) & (reason != 5) & ~boundary] = config.unknown_strength
    return dict(top30_mask=top30, prior_status=status, depth_prior_weight=weight,
        depth_prior_weight_unknown0=(status == RELIABLE).astype(np.float32),
        anchor_error_map=med_abs[labels], anchor_signed_log_error_map=signed[labels],
        anchor_log_mad_map=mad[labels], anchor_count=count[labels], reason=reason,
        surface_component=labels, surface_boundary=boundary,
        anchor_residual=residual, anchor_component=component, anchor_usable=usable,
        anchor_heldout=heldout)


def geometry_anchors(model, images, poses, params, masks, config):
    """Select original SfM points; recompute errors at current native resolution."""
    by_id = {im.image_id:i for i,im in enumerate(images)}
    centers = np.stack([im.projection_center() for im in images])
    records = [[] for _ in images]; accepted = 0
    rejection = dict(few_static_views=0, nonpositive=0, reprojection=0, angle=0)
    for pid, point in model.points3D.items():
        track = [(by_id[e.image_id], images[by_id[e.image_id]].points2D[e.point2D_idx].xy)
                 for e in point.track.elements if e.image_id in by_id]
        track = [(i,xy) for i,xy in track if valid_mask(np.asarray(xy)[None], masks[i])[0]]
        if len(track)<config.min_views:
            rejection['few_static_views']+=1;continue
        idx=np.array([i for i,_ in track]);xy=np.array([xy for _,xy in track])
        pp=poses[idx];cam=np.einsum('nij,j->ni',pp[:,:,:3],point.xyz)+pp[:,:,3]
        if not np.isfinite(cam).all() or (cam[:,2]<=0).any():
            rejection['nonpositive']+=1;continue
        focal=np.stack([params[idx,0],params[idx,4] if params.shape[-1]==5 else params[idx,0]],-1)
        projected=cam[:,:2]/cam[:,2,None]*focal+params[idx,1:3]
        error=np.linalg.norm(projected-xy,axis=1)
        if error.max()>config.max_reprojection_px:
            rejection['reprojection']+=1;continue
        angle=angle_degrees(point.xyz,centers[idx])
        if angle<config.min_angle_deg:
            rejection['angle']+=1;continue
        accepted+=1
        for j,i in enumerate(idx):records[i].append((int(pid),*xy[j],cam[j,2],error[j],angle))
    return [np.array(r,dtype=float).reshape(-1,6) for r in records],dict(
        sparse_points=len(model.points3D),qualified_points=accepted,rejected=rejection,
        reprojection_rule='Maximum error across retained static true observations, native image pixels')


def residual_stats(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return dict(count=len(x),median_log=float(np.median(x)) if len(x) else None,
        absolute_log_p50_p90=np.percentile(np.abs(x),[50,90]).tolist() if len(x) else None)


def local_error_p90(maps):
    labels=maps['surface_component'];component=maps['anchor_component']
    ids=np.flatnonzero(maps['anchor_usable'] & ~maps['anchor_heldout'])
    order=ids[np.argsort(component[ids],kind='stable')]
    values=np.full(int(labels.max())+1,np.nan,np.float32)
    cuts=np.r_[0,np.flatnonzero(np.diff(component[order]))+1,len(order)]
    for start,end in zip(cuts[:-1],cuts[1:]):
        selected=order[start:end]
        if len(selected):values[component[selected[0]]]=np.percentile(np.abs(maps['anchor_residual'][selected]),90)
    return values[labels]


def select_band(top30,top60,status,p90,max_p90=np.log(1.10)):
    assert np.all(~top30 | top60), 'The relaxed ranking must contain the original top30'
    band=top60 & ~top30
    accepted=band & (status==RELIABLE) & np.isfinite(p90) & (p90<=max_p90)
    return top30 | accepted,accepted
