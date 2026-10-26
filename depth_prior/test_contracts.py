"""Existing depth/anchor numerical contracts retargeted to organized implementations."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pycolmap
from PIL import Image
from ..io_utils import save_npz
from ..triangulation.geometry import project_batch
from .mapping import camera_similarity, check_mapping
from .anchors import AnchorConfig, local_status, RELIABLE, UNKNOWN, UNRELIABLE, geometry_anchors, select_band, local_error_p90
from .refinement import refine
from .sampling import bilinear, sample
from .cloud import uncovered_prior_points
from .filter import FilterConfig, select_supported_depth_conflicts, geometry_protected_mask
from . import mapping
from .cache import native_intrinsics, fields_from_pi3x, read_arrays
from .cloud import unproject_clouds, write_cloud, export_prior
from .track_pipeline import refine_tracks, filter_da3
from . import anchor_maps
from .band_pipeline import relax_band
from .da3_constraints import constraints, depth_fit


class CameraAlignmentTest(unittest.TestCase):
    def test_orientation_recovers_similarity_for_straight_camera_path(self):
        angle=.7;c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        scale=.25;t=np.array([3.,-2.,5.])
        predicted=np.tile(np.eye(4),(5,1,1));predicted[:,2,3]=np.arange(5)*3
        target=predicted.copy();target[:,:3,:3]=R
        target[:,:3,3]=scale*predicted[:,:3,3]@R.T+t
        recovered,Q,offset,meta=camera_similarity(predicted,target)
        self.assertAlmostEqual(recovered,scale)
        np.testing.assert_allclose(Q,R,atol=1e-12)
        np.testing.assert_allclose(offset,t,atol=1e-12)
        self.assertLess(meta['camera_center_rmse'],1e-12)
        point=np.array([1.,2.,10.])
        np.testing.assert_allclose(recovered*Q@point+offset,scale*R@point+t,atol=1e-12)


class AnchorConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.depth=np.ones((16,16))*10
        self.valid=np.ones((16,16),bool)
        self.xy=np.array([[2.,2.],[4.,3.],[5.,6.]])
        self.ids=np.array([1,2,3]);self.z=np.full(3,10.)
        self.config=AnchorConfig(patch_size=16)

    def run_gate(self,**kw):
        args=dict(depth=self.depth,valid=self.valid,top30=self.valid,anchor_xy=self.xy,
                  anchor_z=self.z,anchor_ids=self.ids,config=self.config)
        args.update(kw);return local_status(**args)

    def test_agreement_and_conflict(self):
        a=self.run_gate();self.assertTrue((a['prior_status']==RELIABLE).all())
        b=self.run_gate(anchor_z=self.z/2)
        self.assertTrue((b['prior_status']==UNRELIABLE).all())
        self.assertTrue((b['depth_prior_weight']==0).all())

    def test_no_anchors_is_unknown_not_wrong(self):
        a=self.run_gate(anchor_xy=np.empty((0,2)),anchor_z=np.empty(0),anchor_ids=np.empty(0,dtype=int))
        self.assertTrue((a['prior_status']==UNKNOWN).all())
        np.testing.assert_allclose(a['depth_prior_weight'],.1)
        self.assertFalse(a['depth_prior_weight_unknown0'].any())

    def test_holdout_never_votes(self):
        a=self.run_gate(anchor_ids=np.array([5,10,15]),anchor_z=self.z/2)
        self.assertTrue((a['prior_status']==UNKNOWN).all())
        self.assertFalse(a['anchor_count'].any())
        self.assertTrue(np.isfinite(a['anchor_residual']).all())

    def test_duplicate_anchors_do_not_inflate_support(self):
        a=self.run_gate(anchor_ids=np.array([1,1,1]))
        self.assertTrue((a['prior_status']==UNKNOWN).all())

    def test_bridge_road_depth_layers_not_propagated(self):
        d=self.depth.copy();d[:,8:]=30
        a=self.run_gate(depth=d)
        self.assertTrue((a['prior_status'][:,:7]==RELIABLE).all())
        self.assertTrue((a['prior_status'][:,9:]==UNKNOWN).all())
        self.assertFalse(a['anchor_count'][:,9:].any())

    def test_dynamic_barrier_and_not_selected(self):
        valid=self.valid.copy();valid[:,8]=False
        a=self.run_gate(valid=valid,top30=valid)
        self.assertFalse(a['prior_status'][:,8].any())
        self.assertFalse(a['depth_prior_weight'][:,8].any())
        self.assertTrue((a['prior_status'][:,9:]==UNKNOWN).all())

    def test_grid_limits_propagation(self):
        a=self.run_gate(config=AnchorConfig(patch_size=8))
        self.assertTrue((a['prior_status'][8:,8:]==UNKNOWN).all())
        self.assertFalse(a['anchor_count'][8:,8:].any())

    def test_pose_untrusted_disables_even_weak_prior(self):
        a=self.run_gate(trusted=False)
        self.assertTrue((a['prior_status']==UNKNOWN).all())
        self.assertFalse(a['depth_prior_weight'].any())

    def test_mixed_surface_is_unknown(self):
        a=self.run_gate(anchor_z=np.array([10.,10./np.exp(.12),10./np.exp(-.12)]))
        self.assertTrue((a['prior_status']==UNKNOWN).all())

    def test_coordinate_resize_and_pose_validation(self):
        names=['a'];poses=np.eye(4)[None];k=np.array([[[10.,0,8],[0,10,8],[0,0,1]]])
        data=dict(image_names=np.array(names),world_to_camera=poses.copy(),poses=poses.copy(),native_intrinsics=k.copy(),
                  intrinsics=k*np.array([.5,.5,1])[None,:,None],image=np.zeros((1,8,8,3)))
        pred=dict(image_names=np.array(names),conf_logits=np.zeros((1,8,8)),local_points=np.zeros((1,8,8,3)))
        check_mapping(data,pred,names,poses[:,:3],k,(16,16))
        with self.assertRaises(AssertionError):check_mapping(data,pred,names,poses[:,:3],k,(32,16))
        data['world_to_camera'][0,0,3]=1
        with self.assertRaises(AssertionError):check_mapping(data,pred,names,poses[:,:3],k,(16,16))

    def test_single_view_sfm_point_cannot_be_anchor(self):
        im=SimpleNamespace(image_id=1,projection_center=lambda:np.zeros(3),points2D=[SimpleNamespace(xy=np.array([3.,3.]))])
        point=SimpleNamespace(xyz=np.array([0.,0.,10.]),track=SimpleNamespace(elements=[SimpleNamespace(image_id=1,point2D_idx=0)]))
        model=SimpleNamespace(points3D={1:point})
        records,stats=geometry_anchors(model,[im],np.eye(4)[None,:3],np.array([[10.,8.,8.,0.]]),[self.valid],self.config)
        self.assertEqual(stats['qualified_points'],0);self.assertEqual(len(records[0]),0)


class ConfidenceBandTest(unittest.TestCase):
    def test_original_preserved_and_only_supported_band_enters(self):
        old=np.array([True,False,False,False,False,False])
        candidate=np.array([True,True,True,True,True,False])
        adopted,band=select_band(old,candidate,np.array([3,1,2,1,1,1]),
                                np.array([np.nan,.01,.01,.2,np.nan,.01]))
        np.testing.assert_array_equal(adopted,[True,True,False,False,False,False])
        np.testing.assert_array_equal(band,[False,True,False,False,False,False])

    def test_invalid_original_subset_fails(self):
        with self.assertRaises(AssertionError):
            select_band(np.array([True]),np.array([False]),np.array([1]),np.array([0.]))

    def test_assessment_and_unusable_anchors_do_not_gate(self):
        maps=dict(surface_component=np.array([[0,1,2]]),anchor_component=np.array([0,0,0,1]),
                  anchor_usable=np.array([True,True,False,True]),anchor_heldout=np.array([False,True,False,False]),
                  anchor_residual=np.array([.02,10.,10.,-.03]))
        np.testing.assert_allclose(local_error_p90(maps),[[.02,.03,np.nan]])


class Pi3XRefinementTest(unittest.TestCase):
    def test_zero_weight_invalid_neighbors_do_not_erase_exact_depth(self):
        field = np.array([[10., np.nan], [np.inf, np.nan]])
        values, inside = bilinear(field, np.array([[0., 0.], [.5, 0.], [0., .5], [-1., 0.]]))
        self.assertEqual(values[0], 10.)
        self.assertFalse(np.isfinite(values[1:3]).any())
        np.testing.assert_array_equal(inside, [True, True, True, False])

    def test_bilinear_sampling_keeps_fractional_values_and_border_clamp(self):
        field = np.array([[10., 20.], [30., 40.]])
        values, inside = bilinear(field, np.array([[.25, .5], [1., 1.], [1.75, 1.75], [np.nan, 0.]]))
        np.testing.assert_array_equal(values[:3], [22.5, 40., 40.])
        np.testing.assert_array_equal(inside, [True, True, True, False])

    def test_supplements_use_3d_coverage_and_per_point_footprint(self):
        existing=np.array([[0.,0.,10.],[2.,0.,10.]])
        prior=np.array([[0.,0.,10.],[.1,0.,10.],[0.,0.,11.],[2.3,0.,10.]])
        keep,distance=uncovered_prior_points(existing,prior,np.array([.2,.2,.2,.4]))
        np.testing.assert_array_equal(keep,[False,False,True,False])
        np.testing.assert_allclose(distance,[0.,.1,1.,.3])

    def setUp(self):
        self.poses=np.tile(np.eye(4)[:3],(1,3,1,1))
        self.poses[0,:,0,3]=[0.,-.1,-.2]
        self.params=np.tile([500.,450.,250.,0.],(1,3,1))
        self.initial=np.array([[0.,0.,12.]])
        self.xy=project_batch(self.initial,self.poses,self.params)[0]

    def test_small_baseline_depth_prior_corrects_biased_triangulation(self):
        depth=np.full((1,3),10.);conf=np.array([[.9,.8,.7]])
        soft,replace,anchor,active=refine(self.initial,self.poses,self.params,self.xy,depth,conf,
            np.ones((1,3),bool),.02,8)
        self.assertTrue(active[0]);self.assertEqual(anchor[0],0)
        np.testing.assert_allclose(replace,[[0.,0.,10.]])
        self.assertLess(abs(soft[0,2]-10),.05)
        self.assertGreater(soft[0,2],10.)  # Retains influence of the original observations.

    def test_disabled_depth_keeps_original_and_best_eligible_anchor_wins(self):
        depth=np.full((1,3),10.);conf=np.array([[.9,.8,.99]])
        soft,replace,best,active=refine(self.initial,self.poses,self.params,self.xy,depth,conf,
            np.zeros((1,3),bool),.02,8)
        np.testing.assert_array_equal(soft,self.initial);np.testing.assert_array_equal(replace,self.initial)
        self.assertFalse(active[0])
        _,replace,best,_=refine(self.initial,self.poses,self.params,self.xy,depth,conf,
            np.array([[False,True,False]]),.02,8)
        self.assertEqual(best[0],1)
        pixel,z,_=project_batch(replace,self.poses,self.params)
        np.testing.assert_allclose(pixel[:,1],self.xy[:,1]);self.assertAlmostEqual(z[0,1],10.)

    def test_true_pixel_samples_resized_depth_not_current_xyz_projection(self):
        yy,xx=np.indices((6,8));field=dict(depth=xx+2.*yy,conf=np.ones((6,8)),
            spread=np.zeros((6,8)),mask=np.ones((12,16),bool),threshold=.8,
            K=np.array([[2.,0,4],[0,2.,3],[0,0,1.]]))
        # Native pixel (10,8), native K f=4,c=(8,6) => processed pixel (5,4).
        d,_,_,good=sample([field],np.array([[0]]),np.array([[[.5,.5,1.]]]),
            np.array([[[10.,8.]]]),0)
        self.assertTrue(good[0,0]);self.assertAlmostEqual(d[0,0],13.)


class DirectDepthFilterTest(unittest.TestCase):
    def test_protection_requires_all_three_geometry_conditions(self):
        points=dict(track_length=np.array([3,2,3,3,3,3]),
            angle=np.array([3.,10.,2.9,3.,np.nan,3.]),
            max_reprojection=np.array([1.,.1,.1,1.01,.1,np.nan]))
        np.testing.assert_array_equal(geometry_protected_mask(points),[True,False,False,False,False,False])

    def test_twenty_percent_tolerance_accepts_moderate_depth_disagreement(self):
        z=np.array([[8.5,8.5],[7.9,7.9],[12.1,12.1]])
        result=select_supported_depth_conflicts(z,np.full_like(z,10.),np.ones_like(z,bool),FilterConfig(.2))
        np.testing.assert_array_equal(result['reject'],[False,True,True])

    def test_rejects_in_front_and_behind_without_occlusion_exemption(self):
        z=np.array([[8.,8.],[12.,12.],[10.,10.]])
        result=select_supported_depth_conflicts(z,np.full_like(z,10.),np.ones_like(z,bool),FilterConfig(.1))
        np.testing.assert_array_equal(result['reject'],[True,True,False])

    def test_one_view_conflict_and_exact_half_do_not_remove_point(self):
        z=np.array([[8.,10.],[8.,8.],[8.,8.]])
        good=np.array([[1,1],[1,0],[1,1]],bool)
        result=select_supported_depth_conflicts(z,np.full_like(z,10.),good,FilterConfig(.1))
        np.testing.assert_array_equal(result['reject'],[False,False,True])

    def test_unknown_and_nan_have_no_vote(self):
        z=np.array([[8.,8.,8.],[8.,8.,8.]])
        depth=np.array([[10.,np.nan,10.],[10.,10.,10.]])
        good=np.array([[1,1,0],[0,0,0]],bool)
        result=select_supported_depth_conflicts(z,depth,good,FilterConfig(.1))
        np.testing.assert_array_equal(result['reject'],[False,False])
        np.testing.assert_array_equal(result['evaluable_views'],[1,0])


class DepthPipelineTests(unittest.TestCase):
    def scene(self, root):
        source, prior, reference = root/'source', root/'prior', root/'reference'
        for folder in [source/'images', source/'static_valid_masks', source/'sparse/0',
                       prior/'pi3x', reference/'B1_shared']:
            folder.mkdir(parents=True)
        for name in ['images.bin', 'cameras.bin']:
            (source/'sparse/0'/name).write_bytes(b'fixed source model')
        names = np.array([f'{i}.png' for i in range(3)])
        poses = np.tile(np.eye(4)[:3], (3, 1, 1))
        poses[:, 0, 3] = [0., -.1, -.2]
        camera = pycolmap.Camera(model='PINHOLE', width=28, height=28, params=[20., 30., 14., 14.])
        params = np.tile([20., 14., 14., 0., 30.], (3, 1))
        images = [SimpleNamespace(name=str(name), camera_id=1, image_id=i+1,
                  cam_from_world=lambda pose=pose: SimpleNamespace(matrix=lambda: pose),
                  projection_center=lambda pose=pose: -pose[:, :3].T@pose[:, 3])
                  for i, (name, pose) in enumerate(zip(names, poses))]
        model = SimpleNamespace(cameras={1: camera}, points3D={})
        world_to_camera = np.tile(np.eye(4), (3, 1, 1))
        world_to_camera[:, :3] = poses
        native = native_intrinsics(params)
        intrinsics = native.copy()
        intrinsics[:, :2] *= .5
        rgb = np.full((3, 14, 14, 3), 125, np.uint8)
        for name in names:
            Image.fromarray(np.full((28, 28, 3), 125, np.uint8)).save(source/'images'/str(name))
            Image.fromarray(np.full((28, 28), 255, np.uint8)).save(source/'static_valid_masks'/str(name))
        data = dict(image=rgb, poses=np.linalg.inv(world_to_camera), intrinsics=intrinsics, image_names=names,
                    native_intrinsics=native, world_to_camera=world_to_camera)
        save_npz(prior/'input_conditions.npz', **data)
        yy, xx = np.indices((14, 14))
        rays = np.stack([xx, yy, np.ones_like(xx)], -1)@np.linalg.inv(intrinsics[0]).T
        predicted = dict(local_points=np.tile((rays*10).astype(np.float32), (3, 1, 1, 1)),
                         conf_logits=np.tile(np.arange(196, dtype=np.float32).reshape(14, 14), (3, 1, 1)),
                         camera_poses=data['poses'].copy(), image_names=names)
        save_npz(prior/'pi3x/prediction.npz', **predicted)
        (prior/'pi3x/inference.json').write_text(json.dumps(dict(pose_conditioned=True, intrinsics_conditioned=True, depth_conditioned=False)))
        scale, rotation, translation, alignment = camera_similarity(predicted['camera_poses'], data['poses'])
        (prior/'camera_alignment.json').write_text(json.dumps(alignment))
        fields = fields_from_pi3x(data, predicted, source, images, poses, params, scale)
        exported = unproject_clouds(data, predicted, fields, params, scale, rotation, translation)
        write_cloud(prior, 'PI3X_FIXED', exported['fixed'], exported['rgb'], exported['source_image_index'],
                    exported['source_pixel'], exported['confidence'],
                    dict(per_frame_confidence_threshold=[field['threshold'] for field in fields]))
        xyz = np.array([[0., 3., 13.], [0., -3., 13.]], np.float32)
        pp, qq = np.tile(poses, (2, 1, 1, 1)), np.tile(params, (2, 1, 1))
        xy = project_batch(xyz, pp, qq)[0]
        points = dict(xyz=xyz, rgb=np.full((2, 3), 100, np.uint8), track_ids=np.arange(2), track_length=np.full(2, 3),
                      quality=np.ones(2, np.uint8), reprojection=np.zeros(2), max_reprojection=np.zeros(2),
                      angle=np.array([1., 5.]), leave_one_out_px=np.ones(2), relative_xyz_stability=np.ones(2)*.01)
        save_npz(reference/'B1_shared/new_points.npz', **points)
        save_npz(reference/'B1_shared/point_observations.npz', offsets=np.array([0, 3, 6]),
                 image_index=np.tile(np.arange(3), 2), xy=xy.reshape(-1, 2), errors=np.zeros(6))
        (reference/'manifest.json').write_text(json.dumps(dict(status='complete', source=str(source), pair_budget=3)))
        return source, prior, reference, model, images, poses, params, data

    def test_cached_conditions_and_refinement_preserve_real_tracks_and_label_fill(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, prior, reference, model, images, poses, params, data = self.scene(root)
            cached = root/'cached.npz'
            save_npz(cached, image=data['image'], intrinsics=data['intrinsics'], image_names=data['image_names'], extrinsics=poses)
            before = (reference/'B1_shared/new_points.npz').read_bytes()
            with patch.object(mapping, 'model_data', return_value=(model, images, {})):
                mapping.prepare_conditions(source, cached, root/'new_conditions.npz')
                export_prior(prior, source, root/'exported')
                for key, value in read_arrays(root/'exported/PI3X_FIXED/new_points.npz').items():
                    np.testing.assert_equal(value, read_arrays(prior/'PI3X_FIXED/new_points.npz')[key])
                outputs, mixed = refine_tracks(reference, prior, root/'refined')
                with patch.object(anchor_maps, 'model_data', return_value=(model, images, {})):
                    gated = relax_band(reference, prior, root/'refined', root/'band')
                # Insufficient anchors must not change any existing top30 fused point.
                for key in mixed:
                    np.testing.assert_equal(gated[key], mixed[key])
            for key, value in read_arrays(root/'new_conditions.npz').items():
                np.testing.assert_equal(value, data[key])
            original = read_arrays(reference/'B1_shared/new_points.npz')
            self.assertLess(outputs[0]['xyz'][0, 2], 13.)
            self.assertAlmostEqual(outputs[1]['xyz'][0, 2], 10.)
            for cloud in outputs:
                np.testing.assert_array_equal(cloud['xyz'][1], original['xyz'][1])
                self.assertEqual(cloud['quality'][0], 0)
                self.assertEqual(cloud['quality'][1], 1)
            self.assertEqual((reference/'B1_shared/new_points.npz').read_bytes(), before)
            np.testing.assert_array_equal(read_arrays(root/'refined/REPLACED/point_observations.npz')['xy'],
                                          read_arrays(reference/'B1_shared/point_observations.npz')['xy'])
            self.assertGreater(len(mixed['xyz']), 2)
            self.assertTrue((mixed['track_ids'][2:] == -1).all())
            self.assertTrue((mixed['track_length'][2:] == 0).all())
            self.assertTrue((mixed['point_origin'][2:] == 1).all())

    def test_da3_screen_keeps_geometry_protected_points_and_never_moves_xyz(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, prior, reference, model, images, poses, params, data = self.scene(root)
            da3 = root/'da3.npz'
            save_npz(da3, depth=np.full((3, 14, 14), 10.), conf=np.tile(np.arange(196).reshape(14, 14), (3, 1, 1)),
                     intrinsics=data['intrinsics'], extrinsics=poses, image_names=data['image_names'])
            with patch.object(mapping, 'model_data', return_value=(model, images, {})):
                rejected = filter_da3(reference, da3, root/'filtered')
            np.testing.assert_array_equal(rejected, [True, False])
            np.testing.assert_array_equal(read_arrays(root/'filtered/KEPT/new_points.npz')['xyz'],
                                          read_arrays(reference/'B1_shared/new_points.npz')['xyz'][[1]])

    def test_sparse_anchor_projection_supports_distinct_fy(self):
        with TemporaryDirectory() as directory:
            source, prior, reference, model, images, poses, params, data = self.scene(Path(directory))
            xyz = np.array([0., 3., 13.])
            pixel = project_batch(xyz[None], poses[None], params[None])[0][0]
            for image, xy in zip(images, pixel):
                image.points2D = [SimpleNamespace(xy=xy)]
            model.points3D = {1: SimpleNamespace(xyz=xyz, track=SimpleNamespace(elements=[
                SimpleNamespace(image_id=i+1, point2D_idx=0) for i in range(3)]))}
            records, stats = geometry_anchors(model, images, poses, params, [np.ones((28, 28), bool)]*3,
                                              AnchorConfig(min_angle_deg=.1, max_reprojection_px=1e-8))
            self.assertEqual(stats['qualified_points'], 1)
            self.assertTrue(all(len(record) == 1 for record in records))


class LowAngleConstraintTests(unittest.TestCase):
    def scene(self, depths=(5., 5., 5.)):
        fields = [dict(depth=np.full((160, 256), depth), conf=np.ones((160, 256)),
                       K=np.array([[100., 0, 128.], [0, 100., 80.], [0, 0, 1.]]),
                       mask=np.ones((160, 256), bool), spread=np.zeros((160, 256)), threshold=.5)
                  for depth in depths]
        poses = np.tile(np.c_[np.eye(3), np.zeros(3)], (1, 3, 1, 1))
        poses[0, :, 0, 3] = -np.arange(3)*.02
        params = np.tile([100., 128., 80., 0.], (1, 3, 1))
        xy = project_batch(np.array([[0., 0., 5.]]), poses, params)[0]
        return fields, poses, params, xy, np.array([[0, 1, 2]])

    def test_valid_small_angle_depth_improves_xyz_without_moving_observations(self):
        fields, poses, params, xy, im = self.scene()
        original = xy.copy()
        prior = constraints(fields, im, xy, poses, params)
        result, weight, active = depth_fit(np.array([[0., 0., 7.]]), poses, params, xy, prior, np.array([.6]))
        self.assertTrue(active[0])
        self.assertTrue(weight.any())
        self.assertLess(abs(result[0, 2]-5), .2)
        np.testing.assert_array_equal(xy, original)

    def test_low_confidence_foreground_blocks_constraint(self):
        fields, poses, params, xy, im = self.scene((10., 5., 10.))
        fields[1]['conf'][:] = 0
        prior = constraints(fields, im, xy, poses, params)
        self.assertTrue((prior['occluded'][0, [0, 2]] > 0).all())
        self.assertFalse(prior['valid'].any())
        initial = np.array([[0., 0., 7.]])
        result, weight, active = depth_fit(initial, poses, params, xy, prior, np.array([.6]))
        np.testing.assert_array_equal(result, initial)
        self.assertFalse(weight.any())
        self.assertFalse(active.any())

    def test_cross_view_projection_uses_distinct_target_fy(self):
        fields, poses, params, _, im = self.scene()
        params = np.concatenate([params, np.full((1, 3, 1), 160.)], axis=2)
        yy, _ = np.indices(fields[0]['depth'].shape)
        for field in fields:
            field['K'][1, 1] = 160.
            field['depth'] = 5.+.02*(yy-80)
        # A sloped surface makes an incorrect target y-coordinate change the sampled depth.
        z = (5+np.sqrt(25+6.4))/2
        xy = project_batch(np.array([[0., .5, z]]), poses, params)[0]
        prior = constraints(fields, im, xy, poses, params, tolerance=.005)
        self.assertTrue(prior['valid'].all())
