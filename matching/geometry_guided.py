"""RoMaV2 direct queries, paired SIFT/SP evidence and identity-preserving Snap."""
from pathlib import Path
import json
import sqlite3
import time
import numpy as np
from PIL import Image
from ..io_utils import save_npz, write_json
from ..sfm.geometry import model_data, epipolar_error
from .sparse import valid_mask, offline_sparse_weights, sift_data, PAIR_BASE, pair_seed
from .romav2_runtime import load_model, query_field
from ..runtime import ROOT
from .seeds import image_seeds, query_matches, prepared_seeds
from .filters import shared_filter_settings
from .snap import identity_snap

def match_all(root):
    import torch
    from lightglue import SuperPoint,LightGlue
    from lightglue.utils import load_image
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());source=Path(manifest['source'])
    seed_count=int(manifest.get('seeds_per_image',2500))
    if seed_count<1500:raise ValueError('seeds_per_image must be at least 1500')
    pair_cap=2*seed_count
    filters=shared_filter_settings(manifest)
    model,images,by_name=model_data(source/'sparse/0');kp,old=sift_data(model,images,source/'database.db')
    pairs=sorted({tuple(p) for ps in manifest['policies'].values() for p in ps})
    active_names={name for pair in pairs for name in pair}
    images=[im for im in images if im.name in active_names]
    masks={im.name:np.array(Image.open(root/'shared/sky_masks/foreground_valid_hard'/f'{Path(im.name).stem}.png')) for im in images}
    started=time.monotonic()
    with offline_sparse_weights():
        extractor=SuperPoint(max_num_keypoints=2048).eval().cuda();matcher=LightGlue(features='superpoint',filter_threshold=.2).eval().cuda()
    features={};seeds={}
    with torch.inference_mode():
        for im in images:
            path=root/'shared/superpoint'/f'{Path(im.name).stem}.pt'
            if path.exists():f=torch.load(path,map_location='cpu',weights_only=True)
            else:
                f={k:v.cpu() for k,v in extractor.extract(load_image(source/'images'/im.name).cuda(),resize=1024).items()};torch.save(f,path)
            features[im.name]=f;sp=f['keypoints'][0].numpy()+.5
            if manifest.get('prepared_source_seeds',False):
                seed=prepared_seeds(root/'shared/seeds'/f'{Path(im.name).stem}.npz',seed_count,im.image_id)
                if not valid_mask(seed['xy'],masks[im.name]).all():
                    raise ValueError(f'Prepared source seed outside valid pixels: {im.name}')
                seeds[im.name]=seed
                continue
            sid=np.flatnonzero(valid_mask(kp[im.name],masks[im.name])&(old[im.name]==0))
            pid=np.flatnonzero(valid_mask(sp,masks[im.name]))
            # Deterministic spatial strata within each feature pool, then free fill.
            rng=np.random.default_rng(pair_seed(im.name,'sparse_source',142))
            def choose(ids,xy,count):
                order=rng.permutation(ids);cells=(xy[order,1]//34).astype(int)*64+(xy[order,0]//30).astype(int)
                _,first=np.unique(cells,return_index=True);first=order[first];remaining=order[~np.isin(order,first)]
                return np.r_[first,remaining][:count]
            sid=choose(sid,kp[im.name],500);pid=choose(pid,sp,1000)
            count=seed_count-len(sid)-len(pid);free=image_seeds(masks[im.name],pair_seed(im.name,'shared_free',142),count=count)
            seed=dict(xy=np.concatenate([kp[im.name][sid],sp[pid],free]).astype(np.float32),kind=np.r_[np.ones(len(sid),np.uint8),np.full(len(pid),2,np.uint8),np.full(count,3,np.uint8)],
                key=np.r_[sid.astype(np.int64),pid.astype(np.int64),np.arange(count,dtype=np.int64)],image_id=np.array(im.image_id))
            assert len(seed['xy'])==seed_count
            seed_path=root/'shared/seeds'/f'{Path(im.name).stem}.npz'
            if manifest.get('validate_reused_seeds',False):
                with np.load(seed_path) as prior:
                    assert all(np.array_equal(value,prior[key]) for key,value in seed.items()),'Cached matches require exactly the same source seeds'
            save_npz(root/'shared/seeds'/f'{Path(im.name).stem}.npz',**seed);seeds[im.name]=seed
    torch.cuda.synchronize();sparse_seconds=time.monotonic()-started;del extractor
    config=json.loads((ROOT/'config/romav2.json').read_text())
    config.update(manifest.get('matcher_overrides',{}))
    start=time.monotonic();roma=load_model(config);torch.cuda.synchronize();load_seconds=time.monotonic()-start
    matcher_wh=[roma.W_hr or roma.W_lr,roma.H_hr or roma.H_lr]
    reports=[];inferred=0;reused=0
    db=sqlite3.connect(f'file:{source/"database.db"}?mode=ro',uri=True)
    with torch.inference_mode():roma.match(source/'images'/pairs[0][0],source/'images'/pairs[0][1])
    torch.cuda.synchronize();full_begin=time.monotonic()
    with torch.inference_mode():
        for number,(a,b) in enumerate(pairs):
            target=root/'shared/direct_pairs'/f'{Path(a).stem}_{Path(b).stem}.npz';timing=target.with_suffix('.json')
            if target.exists() and timing.exists():reports.append(json.loads(timing.read_text()));reused+=1;continue
            inferred+=1
            ims=[by_name[a],by_name[b]];cs=[model.cameras[im.camera_id] for im in ims];ms=[masks[a],masks[b]]
            torch.cuda.synchronize();begin=time.monotonic();preds=roma.match(source/'images'/a,source/'images'/b);torch.cuda.synchronize();match_seconds=time.monotonic()-begin
            tick=time.monotonic();q=[query_matches(preds,i,1-i,seeds[[a,b][i]]['xy'],cs,ms,matcher_wh=matcher_wh,min_overlap=filters['match_min_overlap'],max_cycle_px=filters['match_max_cycle_px']) for i in range(2)];torch.cuda.synchronize();query_seconds=time.monotonic()-tick
            raw0=np.concatenate([q[0]['src'],q[1]['dst']]);raw1=np.concatenate([q[0]['dst'],q[1]['src']]);overlap=np.concatenate([s['overlap'] for s in q]);cycle=np.concatenate([s['cycle'] for s in q]);sigma=np.concatenate([s['sigma'] for s in q]);eligible=np.concatenate([s['eligible'] for s in q])
            tick=time.monotonic();out=matcher({'image0':{k:v.cuda() for k,v in features[a].items()},'image1':{k:v.cuda() for k,v in features[b].items()}})
            spm=out['matches'][0].cpu().numpy();scores=out['scores'][0].cpu().numpy();good=scores>=.2;spm=spm[good];scores=scores[good]
            sp0=features[a]['keypoints'][0].numpy()+.5;sp1=features[b]['keypoints'][0].numpy()+.5
            id0,id1=ims[0].image_id,ims[1].image_id;row=db.execute('SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?',(min(id0,id1)*PAIR_BASE+max(id0,id1),)).fetchone()
            sift=np.frombuffer(row[2],np.uint32).reshape(row[0],row[1]).copy() if row and row[0] else np.empty((0,2),np.uint32)
            if id0>id1:sift=sift[:,::-1]
            p0={int(p):k for k,p in enumerate(old[a]) if p>0};p1={int(p):k for k,p in enumerate(old[b]) if p>0}
            extra=np.array([(p0[p],p1[p]) for p in sorted(p0.keys()&p1.keys())],np.uint32).reshape(-1,2);sift=np.unique(np.concatenate([sift,extra]),axis=0)
            anchors=dict(p0=np.concatenate([kp[a][sift[:,0]],sp0[spm[:,0]]]),p1=np.concatenate([kp[b][sift[:,1]],sp1[spm[:,1]]]),
                kind=np.r_[np.ones(len(sift),np.uint8),np.full(len(spm),2,np.uint8)],key0=np.r_[sift[:,0].astype(np.int64),spm[:,0]],key1=np.r_[sift[:,1].astype(np.int64),spm[:,1]],
                old0=np.r_[old[a][sift[:,0]],np.zeros(len(spm),np.int64)],old1=np.r_[old[b][sift[:,1]],np.zeros(len(spm),np.int64)])
            x0=torch.as_tensor(anchors['p0'],device='cuda');x1=torch.as_tensor(anchors['p1'],device='cuda')
            fw=(query_field(preds['warp_AB'],x0,(cs[0].height,cs[0].width))+1)*x0.new_tensor([cs[1].width,cs[1].height])/2
            bw=(query_field(preds['warp_BA'],x1,(cs[1].height,cs[1].width))+1)*x0.new_tensor([cs[0].width,cs[0].height])/2
            anchor_cycle=torch.maximum(torch.linalg.norm(fw-x1,dim=1),torch.linalg.norm(bw-x0,dim=1)).cpu().numpy()
            ov=torch.minimum(query_field(preds['overlap_AB'],x0,(cs[0].height,cs[0].width))[:,0],query_field(preds['overlap_BA'],x1,(cs[1].height,cs[1].width))[:,0]).cpu().numpy()
            epi=epipolar_error(model,ims[0],ims[1],anchors['p0'],anchors['p1'])
            anchor_valid=valid_mask(anchors['p0'],ms[0])&valid_mask(anchors['p1'],ms[1])&(anchor_cycle<=2)&(ov>=.5)&(epi<=2)
            # Existing SP identity needs a unique complete two-end SIFT witness.
            known=np.flatnonzero((anchors['old0']>0)&(anchors['old0']==anchors['old1'])&anchor_valid)
            if len(known):
                from scipy.spatial import cKDTree
                tree=cKDTree(np.column_stack([anchors['p0'][known],anchors['p1'][known]]))
                for anchor in np.flatnonzero((anchors['kind']==2)&anchor_valid):
                    near=tree.query_ball_point(np.r_[anchors['p0'][anchor],anchors['p1'][anchor]],np.sqrt(2)*.5)
                    near=known[np.array(near,int)];near=near[(np.linalg.norm(anchors['p0'][near]-anchors['p0'][anchor],axis=1)<=.5)&(np.linalg.norm(anchors['p1'][near]-anchors['p1'][anchor],axis=1)<=.5)]
                    values=np.unique(anchors['old0'][near])
                    if len(values)==1:anchors['old0'][anchor]=anchors['old1'][anchor]=values[0]
            valid_ids=np.flatnonzero(anchor_valid);av={k:v[valid_ids] for k,v in anchors.items()}
            final0=raw0.copy();final1=raw1.copy();status=np.zeros(pair_cap,np.uint8);chosen=np.full(pair_cap,-1,np.int32);existing=np.zeros(pair_cap,np.int64)
            for side,name in enumerate([a,b]):
                local=av if side==0 else dict(p0=av['p1'],p1=av['p0'],kind=av['kind'],key0=av['key1'],key1=av['key0'],old0=av['old1'],old1=av['old0'])
                src,dst=(raw0[:seed_count],raw1[:seed_count]) if side==0 else (raw1[seed_count:],raw0[seed_count:])
                final,s,c=identity_snap(src,dst,seeds[name]['kind'],seeds[name]['key'],local)
                at=slice(side*seed_count,(side+1)*seed_count);status[at]=s
                positive=c>=0;c[positive]=valid_ids[c[positive]];chosen[at]=c
                if side==0:final1[at]=final
                else:final0[at]=final
                rows=np.flatnonzero(positive)+side*seed_count
                existing[rows]=np.maximum(anchors['old0'][c[positive]],anchors['old1'][c[positive]])
            assert np.array_equal(final0[:seed_count],raw0[:seed_count]) and np.array_equal(final1[seed_count:],raw1[seed_count:])
            raw_epi=epipolar_error(model,ims[0],ims[1],raw0,raw1);final_epi=epipolar_error(model,ims[0],ims[1],final0,final1)
            joint=eligible&(final_epi<=filters['two_view_max_epipolar_px'])
            arrays=dict(raw0=raw0,raw1=raw1,final0=final0,final1=final1,raw_overlap=overlap,raw_cycle=cycle,raw_epi=raw_epi,final_epi=final_epi,
                raw_eligible=eligible,raw_cached_joint_pass=joint,sigma_max_original_px=sigma,source_side=np.repeat([0,1],seed_count),seed_index=np.tile(np.arange(seed_count),2),
                snap_status=status,snap_anchor_index=chosen,existing_id=existing,anchor_valid=anchor_valid,anchor_lg_scores=np.r_[np.full(len(sift),np.nan),scores])
            arrays.update({f'anchor_{k}':v for k,v in anchors.items()});save_npz(target,**arrays)
            torch.cuda.synchronize();snap_seconds=time.monotonic()-tick
            record=dict(pair=[a,b],match_seconds=match_seconds,query_seconds=query_seconds,sparse_match_and_snap_seconds=snap_seconds,
                        total_pair_seconds=time.monotonic()-begin,sampled=pair_cap,eligible=int(eligible.sum()),snapped=int(((status==1)&eligible).sum()),
                        sift_anchors=len(sift),sp_anchors=len(spm),existing_evidence=int((existing>0).sum()),final_epipolar_pass=int((eligible&(final_epi<=filters['match_max_epipolar_px'])).sum()))
            write_json(timing,record);reports.append(record)
            if number%30==0 or number==len(pairs)-1:print(f'RoMa+Snap {number+1}/{len(pairs)}, snap={record["snapped"]}',flush=True)
    db.close();write_json(root/'shared/matching_report.json',dict(records=reports,superpoint_and_seeds_seconds=sparse_seconds,model_load_seconds=load_seconds,
                union_pairs=len(pairs),actual_union_inference_pairs=inferred,cached_union_pairs=reused,model_warmup_inference_pairs=1,
                matching_union_wall_seconds=time.monotonic()-full_begin,per_pair_cap=pair_cap,matcher_setting=config['setting'],matcher_wh=matcher_wh,filter_settings=filters,
                seed_recipe=(manifest.get('seed_recipe','Explicit prepared source identities') if manifest.get('prepared_source_seeds',False)
                             else f'up to 500 unregistered SIFT + 1000 SP + valid stratified free fill to {seed_count} per image; same seeds across ALL policies'),
                snap='complete same-source-feature pair; source coordinates immutable; failed Snap exact Raw fallback; real pre-Snap candidate repair'))
