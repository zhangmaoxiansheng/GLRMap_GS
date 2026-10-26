"""Focused checks for the SfM→query→Snap→Track→XYZ boundary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap
from PIL import Image

from ..io_utils import save_npz
from ..sfm.geometry import fixed_camera_parameters
from ..sfm.reconstruction import static_masks, oriented_similarity, run_logged
from ..matching.filters import shared_filter_settings
from ..matching.pairs import sparse_pairs
from ..matching.seeds import image_seeds, query_matches, prepared_seeds
from ..matching.snap import identity_snap
from . import tracks
from .geometry import project_batch, dlt_batch, fit_xyz, robust_initial, repair_track
from .reconstruct import reconstruct
from .batch import merge_chunks


class ReconstructionContracts(unittest.TestCase):
    def test_sfm_masks_preserve_inputs_and_reject_changed_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ['sfm/block_00', 'masks/vehicle_valid', 'sky_all/foreground_valid_hard']:
                (root/folder).mkdir(parents=True)
            dynamic = np.full((370, 666), 255, np.uint8)
            dynamic[10, 20] = 0
            sky = np.full_like(dynamic, 255)
            sky[:20] = 0
            Image.fromarray(dynamic).save(root/'masks/vehicle_valid/000000.png')
            Image.fromarray(sky).save(root/'sky_all/foreground_valid_hard/000000.png')
            static_masks(root, 0, (0, 1, 1))
            np.testing.assert_array_equal(np.array(Image.open(root/'masks/sfm_static/000000.jpg.png')),
                                          np.minimum(dynamic, sky))
            np.testing.assert_array_equal(np.array(Image.open(root/'masks/vehicle_valid/000000.png')), dynamic)
            sky[30, 30] = 0
            Image.fromarray(sky).save(root/'sky_all/foreground_valid_hard/000000.png')
            with self.assertRaisesRegex(ValueError, 'Changed static mask'):
                static_masks(root, 0, (0, 1, 1))

    def test_sfm_similarity_uses_rotations_to_resolve_straight_path(self):
        moving = np.column_stack([np.arange(4), np.zeros((4, 2))])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([3., 4., 5.])
        reference = 2*moving@rotation.T+translation
        a = np.tile(np.eye(3), (4, 1, 1))
        scale, found, shift = oriented_similarity(moving, reference, a, rotation@a)
        np.testing.assert_allclose(scale, 2)
        np.testing.assert_allclose(found, rotation)
        np.testing.assert_allclose(shift, translation)

    def test_failed_sfm_command_has_no_done_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('GLRMap_GS.sfm.reconstruction.subprocess.run',
                       return_value=SimpleNamespace(returncode=2)):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    run_logged(['colmap', 'feature_extractor'], root, 'features', root)
            self.assertFalse((root/'features.done.json').exists())

    def test_filters_keep_two_view_gate_stricter(self):
        with self.assertRaises(ValueError):
            shared_filter_settings({'filter_settings': {'two_view_min_angle_deg': .5}})
        with self.assertRaises(ValueError):
            shared_filter_settings({'filter_settings': {'two_view_max_reprojection_px': 3.}})

    def test_covisibility_counts_points_not_boolean_connectivity(self):
        points = {i: SimpleNamespace(error=error, track=SimpleNamespace(length=lambda n=n: n))
                  for i, error, n in [(1, .5, 3), (2, 1., 4), (3, 1.1, 4), (4, .2, 2)]}
        images = [SimpleNamespace(points2D=[SimpleNamespace(point3D_id=i, has_point3D=lambda: True)
                                            for i in ids]) for ids in [[1, 1, 2, 3, 4], [1, 2, 2], []]]
        self.assertEqual(sparse_pairs(SimpleNamespace(points3D=points), images),
                         {(0, 1): 2, (0, 2): 0, (1, 2): 0})
        self.assertEqual(sparse_pairs(SimpleNamespace(points3D={}), images),
                         {(0, 1): 0, (0, 2): 0, (1, 2): 0})

    def test_shared_seed_coordinates_are_deterministic_valid_pixel_centers(self):
        mask = np.ones((80, 100), np.uint8)
        mask[:20] = 0
        a = image_seeds(mask, 17, count=100)
        np.testing.assert_array_equal(a, image_seeds(mask, 17, count=100))
        self.assertEqual(len(np.unique(a, axis=0)), 100)
        self.assertTrue((a % 1 == .5).all())
        x, y = np.floor(a).astype(int).T
        self.assertTrue((mask[y, x] > 0).all())

    def test_query_keeps_source_and_converts_target_precision_to_original_pixels(self):
        import torch
        axis = torch.tensor([-.5, .5])
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        warp = torch.stack([x, y], -1)[None]
        predictions = {f'warp_{d}': warp for d in ['AB', 'BA']}
        predictions.update({f'overlap_{d}': torch.ones(1, 2, 2, 1) for d in ['AB', 'BA']})
        predictions['precision_AB'] = torch.eye(2).expand(1, 2, 2, 2, 2)*4
        predictions['precision_BA'] = torch.eye(2).expand(1, 2, 2, 2, 2)*9
        cameras = [SimpleNamespace(width=8, height=8), SimpleNamespace(width=20, height=10)]
        masks = [np.ones((8, 8), np.uint8), np.ones((10, 20), np.uint8)]
        source = np.array([[2., 2.], [6., 6.]], np.float32)
        result = query_matches(predictions, 0, 1, source, cameras, masks)
        np.testing.assert_array_equal(result['src'], source)
        np.testing.assert_allclose(result['dst'], [[5., 2.5], [15., 7.5]])
        np.testing.assert_allclose(result['cycle'], 0)
        np.testing.assert_allclose(result['sigma'], 20/512/2)

    def test_prepared_seeds_keep_pixels_and_reject_ambiguous_source_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'seeds.npz'
            values = dict(xy=np.array([[2.25, 3.75], [7.5, 8.5]], np.float32),
                          kind=np.array([1, 3], np.uint8), key=np.array([10, 11]), image_id=np.array(42))
            save_npz(path, **values)
            loaded = prepared_seeds(path, 2, 42)
            for key in values:
                np.testing.assert_array_equal(loaded[key], values[key])
            with self.assertRaises(ValueError):
                prepared_seeds(path, 2, 43)
            values['kind'][1] = 1
            values['key'][1] = 10
            save_npz(path, **values)
            with self.assertRaisesRegex(ValueError, 'unique'):
                prepared_seeds(path, 2, 42)

    def test_snap_requires_same_source_identity_and_preserves_source_coordinates(self):
        source = np.array([[100., 200.]], np.float32)
        target = np.array([[150.8, 200.]], np.float32)
        anchors = dict(p0=source.copy(), p1=np.array([[150., 200.]], np.float32),
                       kind=np.array([2]), key0=np.array([17]), key1=np.array([23]),
                       old0=np.array([0]), old1=np.array([0]))
        final, status, _ = identity_snap(source, target, [2], [17], anchors)
        np.testing.assert_array_equal(source, anchors['p0'])
        np.testing.assert_array_equal(final, anchors['p1'])
        self.assertEqual(status[0], 1)
        final, _, chosen = identity_snap(source, target, [2], [18], anchors)
        np.testing.assert_array_equal(final, target)
        self.assertEqual(chosen[0], -1)

    def test_reference_seed_forms_three_view_track_and_cannot_move(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'shared/direct_pairs').mkdir(parents=True)
            (root/'manifest.json').write_text(json.dumps({'source': str(root/'source'), 'per_pair_cap': 1}))
            images = [SimpleNamespace(name=f'{c}.jpg') for c in ['A', 'B', 'C']]
            pairs = [('A.jpg', 'B.jpg'), ('A.jpg', 'C.jpg')]
            arrays = dict(raw0=np.array([[100., 200.]], np.float32), raw1=np.array([[20., 30.]], np.float32),
                          raw_eligible=np.array([True]), raw_epi=np.array([.1]), source_side=np.array([0]),
                          seed_index=np.array([5]), raw_overlap=np.array([.9]), raw_cycle=np.array([.1]),
                          sigma_max_original_px=np.array([1.]), raw_cached_joint_pass=np.array([True]))
            for _, name in pairs:
                save_npz(root/f'shared/direct_pairs/A_{Path(name).stem}.npz', **arrays)
            with patch.object(tracks, 'model_data', return_value=(object(), images, {})):
                built = tracks.build_shared(root, pairs)
                np.testing.assert_array_equal(built['track_keys'], [[0, 5]])
                np.testing.assert_array_equal(built['node_image'], [0, 1, 2])
                selected = tracks.build_shared(root, pairs, minimum_seed_index=5)
                np.testing.assert_array_equal(selected['track_keys'], built['track_keys'])
                empty = tracks.build_shared(root, pairs, minimum_seed_index=6)
                self.assertEqual(len(empty['tracks']), 0)
                arrays['raw0'] += 1
                save_npz(root/'shared/direct_pairs/A_C.npz', **arrays)
                with self.assertRaisesRegex(AssertionError, 'Shared source moved'):
                    tracks.build_shared(root, pairs)

    def test_pinhole_distinct_focals_projection_jacobian_and_xyz(self):
        camera = pycolmap.Camera(model='PINHOLE', width=1920, height=1080, params=[900, 1100, 960, 540])
        xyz = np.array([[.4, .8, 5.]])
        poses = np.broadcast_to(np.eye(4)[:3], (1, 3, 3, 4)).copy()
        poses[0, :, 0, 3] = [0, -1, -2]
        params = np.broadcast_to(fixed_camera_parameters(camera), (1, 3, 5))
        xy, _, jac = project_batch(xyz, poses, params)
        rays = np.concatenate([camera.cam_from_img(xy.reshape(-1, 2)).reshape(1, 3, 2),
                               np.ones((1, 3, 1))], axis=2)
        for i in range(3):
            np.testing.assert_allclose(xy[0, i], camera.img_from_cam(poses[0, i, :, :3]@xyz[0]+poses[0, i, :, 3]))
        for axis in range(3):
            delta = np.zeros_like(xyz)
            delta[:, axis] = 1e-6
            numeric = (project_batch(xyz+delta, poses, params)[0]-project_batch(xyz-delta, poses, params)[0])/2e-6
            np.testing.assert_allclose(jac[..., axis], numeric, atol=1e-6)
        np.testing.assert_allclose(dlt_batch(poses, rays), xyz, atol=1e-10)
        np.testing.assert_allclose(fit_xyz(xyz+.01, poses, params, xy, np.ones((1, 3), bool)), xyz, atol=1e-6)

    def test_outlier_repair_requires_real_same_track_evidence(self):
        poses = np.array([np.column_stack([np.eye(3), [-x, 0, 0]]) for x in [0., .5, 1., 1.5]])
        params = np.tile([900., 960., 540., 0.], (4, 1))
        centers = -poses[:, :, 3]
        xyz = np.array([.3, .1, 8.])
        xy = project_batch(xyz[None], poses[None], params[None])[0][0]
        truth = xy.copy()
        xy[3, 1] += 9
        rays = np.column_stack([(xy-params[:, 1:3])/params[:, :1], np.ones(4)])
        initial = robust_initial(poses[None], params[None], xy[None], rays[None], centers[None])[0][0]
        fake = {3: [dict(xy=truth[3], source='projected_xyz', identity_verified=True, track_id=12)]}
        self.assertEqual(repair_track(initial, poses, params, xy, rays, centers, fake, 12)['status'], 'outliers_removed')
        real = {3: [dict(xy=truth[3], source='original_roma_of_unique_snapped_pair', identity_verified=True, track_id=12)]}
        result = repair_track(initial, poses, params, xy, rays, centers, real, 12)
        self.assertEqual(result['status'], 'repaired')
        self.assertTrue(result['keep'].all())

    def test_empty_tracks_export_integer_offsets_and_ids(self):
        camera = pycolmap.Camera(model='PINHOLE', width=64, height=48, params=[50, 50, 32, 24])
        image = SimpleNamespace(name='A.jpg', camera_id=1,
                                cam_from_world=lambda: SimpleNamespace(matrix=lambda: np.eye(4)[:3]),
                                projection_center=lambda: np.zeros(3))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'shared/direct_pairs').mkdir(parents=True)
            (root/'manifest.json').write_text(json.dumps({'source': str(root/'source')}))
            with patch.object(tracks, 'model_data', return_value=(SimpleNamespace(cameras={1: camera}), [image], {})):
                built = tracks.build_shared(root, [])
            built['compute_holdouts'] = False
            metric = reconstruct(root, 'EMPTY', built)
            self.assertEqual(metric['accepted_new_points'], 0)
            with np.load(root/'EMPTY_shared/new_points.npz') as points:
                self.assertEqual(points['xyz'].shape, (0, 3))
                self.assertEqual(points['track_ids'].dtype.kind, 'i')
            with np.load(root/'EMPTY_shared/point_observations.npz') as obs:
                np.testing.assert_array_equal(obs['offsets'], [0])

    def test_chunk_track_ids_shift_by_proposed_count(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [root/'chunk0', root/'chunk1']
            for part, proposed, tid in zip(parts, [10, 4], [8, 3]):
                part.mkdir()
                save_npz(part/'new_points.npz', xyz=np.array([[0., 0., 5.]]), rgb=np.zeros((1, 3), np.uint8),
                         track_ids=np.array([tid]), track_length=np.array([2]))
                save_npz(part/'tracks.npz', offsets=np.arange(proposed+1)*2)
                save_npz(part/'point_observations.npz', offsets=np.array([0, 2]), image_index=np.array([0, 1]),
                         xy=np.ones((2, 2)), errors=np.zeros(2))
            merge_chunks(root/'merged', parts, 2, {})
            with np.load(root/'merged/new_points.npz') as points:
                np.testing.assert_array_equal(points['track_ids'], [8, 13])
            with np.load(root/'merged/point_observations.npz') as obs:
                np.testing.assert_array_equal(obs['offsets'], [0, 2, 4])

    def test_missing_chunk_does_not_reserve_final_destination(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                merge_chunks(root/'merged', [root/'missing'], 2, {})
            self.assertFalse((root/'merged').exists())

    def test_triangulated_rgb_uses_each_tracks_source_pixel_floor(self):
        camera = pycolmap.Camera(model='PINHOLE', width=64, height=48, params=[50, 50, 32, 24])
        poses = np.tile(np.eye(4)[:3], (2, 1, 1))
        poses[1, 0, 3] = -1.
        images = [SimpleNamespace(name=f'{i}.png', camera_id=1,
                  cam_from_world=lambda p=p: SimpleNamespace(matrix=lambda: p),
                  projection_center=lambda p=p: -p[:, :3].T@p[:, 3]) for i, p in enumerate(poses)]
        xyz = np.array([[.13, .17, 5.], [.37, .23, 6.]])
        xy = project_batch(xyz, np.tile(poses, (2, 1, 1, 1)),
                           np.tile([50., 32., 24., 0.], (2, 2, 1)))[0]
        # Opposite reference images, deliberately different source colors.
        node_image = np.array([0, 1, 1, 0])
        node_xy = np.concatenate([xy[0], xy[1, ::-1]])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'images').mkdir()
            (root/'manifest.json').write_text(json.dumps(dict(source=str(root))))
            expected = []
            for i in range(2):
                yy, xx = np.indices((48, 64))
                photo = np.stack([xx, yy, np.full_like(xx, 100+i)], -1).astype(np.uint8)
                Image.fromarray(photo).save(root/'images'/images[i].name)
                x, y = np.floor(node_xy[i*2]).astype(int)
                expected.append(photo[y, x])
            built = dict(images=images, model=SimpleNamespace(cameras={1: camera}), tracks=[[0, 1], [2, 3]],
                         node_image=node_image, node_xy=node_xy, node_conf=np.ones(4), node_cycle=np.zeros(4),
                         node_sigma=np.ones(4), node_joint=np.ones(4, bool), offsets=np.array([0, 2, 4]),
                         nodes=np.arange(4), report=dict(input_pairs=1, seconds=0.), compute_holdouts=False)
            metric = reconstruct(root, 'RGB', built)
            self.assertEqual(metric['accepted_new_points'], 2)
            with np.load(root/'RGB_shared/new_points.npz') as points:
                np.testing.assert_array_equal(points['rgb'], expected)
                np.testing.assert_allclose(points['xyz'], xyz, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
