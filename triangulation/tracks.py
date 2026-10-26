"""Direct-reference stars keyed by (reference image index, shared seed index)."""
from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np
from ..sfm.geometry import model_data
from ..matching.filters import shared_filter_settings

def build_shared(root, pairs, anchor_names=None, minimum_seed_index=0):
    start = time.monotonic(); manifest = json.loads((root/'manifest.json').read_text())
    pair_cap=int(manifest.get('per_pair_cap',5000))
    filters=shared_filter_settings(manifest)
    model, images, _ = model_data(Path(manifest['source'])/'sparse/0')
    index = {im.name: i for i, im in enumerate(images)}
    stars = defaultdict(list); reports = []
    for pair_number, (a, b) in enumerate(pairs):
        if anchor_names is not None and a not in anchor_names and b not in anchor_names:continue
        with np.load(root/'shared/direct_pairs'/f'{Path(a).stem}_{Path(b).stem}.npz') as z:
            # Anchor descriptors/evidence stay in the pair cache; this stage uses
            # only the selected coordinates, identities and acceptance fields.
            d = {k: z[k] for k in ['raw0', 'raw1', 'final0', 'final1', 'raw_eligible',
                 'raw_epi', 'final_epi', 'source_side', 'seed_index', 'existing_id',
                 'snap_status', 'raw_overlap', 'raw_cycle', 'sigma_max_original_px',
                 'raw_cached_joint_pass'] if k in z.files}
        assert len(d['raw0'])==pair_cap,'Cached query count differs from experiment budget'
        keep = d['raw_eligible'] & (d.get('final_epi',d['raw_epi']) <= filters['match_max_epipolar_px'])
        keep &= d['seed_index'] >= minimum_seed_index
        if anchor_names is not None:
            keep &= np.where(d['source_side']==0,a in anchor_names,b in anchor_names)
        existing_ids=d.get('existing_id',np.zeros(len(d['raw0']),np.int64));snap_status=d.get('snap_status',np.zeros(len(d['raw0']),np.uint8))
        p0,p1=d.get('final0',d['raw0']),d.get('final1',d['raw1'])
        for r in np.flatnonzero(keep):
            side = int(d['source_side'][r]); anchor = index[a if side == 0 else b]; target = index[b if side == 0 else a]
            src, dst = (p0[r],p1[r]) if side == 0 else (p1[r],p0[r])
            original = d['raw1'][r] if side==0 else d['raw0'][r]
            old=int(existing_ids[r]);snapped=bool(snap_status[r]==1)
            stars[(anchor, int(d['seed_index'][r]))].append((target, src, dst, d['raw_overlap'][r], d['raw_cycle'][r], d['sigma_max_original_px'][r], d['raw_cached_joint_pass'][r], pair_number, int(r),original,old,snapped))
        reports.append(dict(pair=[a,b], raw=len(d['raw0']), raw_eligible=int(d['raw_eligible'].sum()), output_edges=int(keep.sum())))
    image=[]; xy=[]; conf=[]; cycle=[]; sigma=[]; joint=[]; tracks=[]; track_keys=[]; evidence=[];alternatives={};existing_excluded=0
    for key, observations in sorted(stars.items()):
        anchor, seed = key; source_xy = observations[0][1]
        assert all(np.array_equal(source_xy, r[1]) for r in observations), 'Shared source moved between pairs'
        assert len({r[0] for r in observations}) == len(observations)
        if any(r[10]>0 for r in observations):existing_excluded+=1;continue
        nodes=[]
        for i, position, ov, cy, sg, jp, pair, row in [(anchor, source_xy, 1., 0., 0., True, -1, -1)] + [
                (r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in observations]:
            nodes.append(len(image)); image.append(i); xy.append(position); conf.append(ov); cycle.append(cy); sigma.append(sg); joint.append(jp)
            evidence.append((pair, row))
        tid=len(tracks)
        for obs,r in enumerate(observations,1):
            if r[11] and not np.array_equal(r[2],r[9]):
                alternatives[(tid,obs)]=[dict(xy=r[9].copy(),source='original_roma_of_unique_snapped_pair',edge_id=int(r[7])*pair_cap+int(r[8]),track_id=tid,identity_verified=True,independent_match=True)]
        tracks.append(nodes); track_keys.append(key)
    node_ids = np.array([n for t in tracks for n in t], np.int32)
    report = dict(mode='shared', input_pairs=len(pairs), minimum_seed_index=minimum_seed_index,
                  raw_sampled=sum(r['raw'] for r in reports),
                  output_edges=sum(r['output_edges'] for r in reports), tracks=len(tracks),
                  three_view_tracks=sum(len(t)>=3 for t in tracks), four_view_tracks=sum(len(t)>=4 for t in tracks),
                  existing_tracks_excluded=existing_excluded,
                  seconds=time.monotonic()-start, pairs=reports,filter_settings=filters,
                  identity='(reference image ID, fixed source seed ID); all observations are direct RoMa matches from that original source. Different reference stars are kept separate; no nearest-pixel or 3D merging.')
    return dict(model=model, images=images, tracks=tracks, nodes=node_ids, offsets=np.r_[0,np.cumsum(np.array([len(t) for t in tracks],np.int64))],
                node_image=np.array(image,np.int16), node_xy=np.array(xy,float).reshape(-1,2), node_conf=np.array(conf),
                node_cycle=np.array(cycle), node_sigma=np.array(sigma), node_joint=np.array(joint),
                track_keys=np.array(track_keys,np.int32).reshape(-1,2), observation_evidence=np.array(evidence,np.int32).reshape(-1,2), alternatives=alternatives,filter_settings=filters,report=report)
