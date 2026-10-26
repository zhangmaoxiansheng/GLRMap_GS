"""Fixed per-image seeds and bidirectional queries; no Track dependency."""
import numpy as np
from .romav2_runtime import query_field
from .snap import precision_to_original
from .sparse import valid_mask

SEEDS = 2500

def prepared_seeds(path, count, image_id):
    """Read explicit source identities without regenerating or moving their pixels."""
    with np.load(path, allow_pickle=False) as arrays:
        seed = {key: arrays[key] for key in ['xy', 'kind', 'key', 'image_id']}
    if (seed['xy'].shape != (count, 2) or seed['kind'].shape != (count,)
            or seed['key'].shape != (count,) or int(seed['image_id']) != image_id
            or not np.isfinite(seed['xy']).all()):
        raise ValueError(f'Prepared source seeds disagree with image identity or budget: {path}')
    identities = np.column_stack([seed['kind'], seed['key']])
    if not np.isin(seed['kind'], [1, 2, 3]).all() or len(np.unique(identities, axis=0)) != count:
        raise ValueError(f'Prepared source identities must be unique: {path}')
    return seed


def image_seeds(mask, seed, count=SEEDS):
    """Unique valid pixel centers, stratified then filled; deterministic."""
    y, x = np.nonzero(mask > 0)
    if len(x) < count:
        raise ValueError('Too few valid source pixels')
    rng = np.random.default_rng(seed)
    cell = (y * 32 // mask.shape[0]) * 48 + x * 48 // mask.shape[1]
    order = rng.permutation(len(x))
    _, first = np.unique(cell[order], return_index=True)
    chosen = order[first]
    if len(chosen) > count:
        chosen = rng.choice(chosen, count, replace=False)
    else:
        used = np.zeros(len(x), bool); used[chosen] = True
        extra = order[~used[order]][:count-len(chosen)]
        chosen = np.r_[chosen, extra]
    return np.column_stack((x[chosen]+.5, y[chosen]+.5)).astype(np.float32)


def query_matches(preds, source, target, seed_xy, cameras, masks, matcher_wh=(512,512), min_overlap=.5, max_cycle_px=2.):
    import torch
    a, b = ('AB', 'BA') if source == 0 else ('BA', 'AB')
    ca, cb = cameras[source], cameras[target]
    q = torch.as_tensor(seed_xy, device=preds[f'warp_{a}'].device)
    dst = (query_field(preds[f'warp_{a}'], q, (ca.height, ca.width))+1)*q.new_tensor([cb.width, cb.height])/2
    back = (query_field(preds[f'warp_{b}'], dst, (cb.height, cb.width))+1)*q.new_tensor([ca.width, ca.height])/2
    overlap = torch.minimum(query_field(preds[f'overlap_{a}'], q, (ca.height, ca.width))[:, 0],
                            query_field(preds[f'overlap_{b}'], dst, (cb.height, cb.width))[:, 0])
    cycle = torch.linalg.norm(back-q, dim=1)
    # Upstream precision keys are checked against the actual returned prediction.
    forward_precision = preds[f'precision_{a}']
    reverse_precision = preds[f'precision_{b}']
    # Forward precision describes TARGET coordinates; reverse describes SOURCE.
    pb = query_field(forward_precision.reshape(*forward_precision.shape[:3], -1), q, (ca.height, ca.width)).reshape(-1, 2, 2).cpu().numpy()
    pa = query_field(reverse_precision.reshape(*reverse_precision.shape[:3], -1), dst, (cb.height, cb.width)).reshape(-1, 2, 2).cpu().numpy()
    pa = precision_to_original(pa, matcher_wh, [ca.width, ca.height])
    pb = precision_to_original(pb, matcher_wh, [cb.width, cb.height])
    eigen = np.stack([np.linalg.eigvalsh(pa), np.linalg.eigvalsh(pb)], 1)
    sigma = np.max(1/np.sqrt(np.maximum(eigen[:, :, 0], 1e-15)), 1)
    dst = dst.cpu().numpy(); ov = overlap.cpu().numpy(); cy = cycle.cpu().numpy()
    eligible = valid_mask(seed_xy, masks[source]) & valid_mask(dst, masks[target]) & (ov >= min_overlap) & (cy <= max_cycle_px)
    return dict(src=seed_xy.copy(), dst=dst, overlap=ov, cycle=cy, sigma=sigma, eligible=eligible)
