"""Road lineage, fit acceptance and supported ray sampling checks."""
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
from . import selection
from .fitting import fit_patches, load_patches
from .sampling import query_surface, select_cells, sample_road
from .surface import camera_basis
from . import reconstruct
from .lane_protection import marking_votes, clear_surface


def plane(depth=5., first=0, stop=100, support=10000.):
    return dict(first=first, stop=stop, origin=np.zeros(3), basis=np.eye(3),
                surface=dict(coefficients=np.array([depth, 0., 0., 0., 0.]), offset=np.zeros(2),
                             scale=np.ones(2), support_min=np.full(2, -support), support_max=np.full(2, support)))


class RoadContracts(unittest.TestCase):
    def test_lane_protection_removes_wrong_height_surface_not_real_points(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.array([[0, 255]+[0]*10], np.uint8)).save(root/'0.png')
            images = [SimpleNamespace(name='0.jpg')]
            xyz = np.array([[10., 0., 2.]], np.float32)
            original = xyz.copy()
            dense = dict(xyz=np.array([[10.01, 0., 100.], [30., 0., 2.], [40., 0., 2.]]),
                         source_image_index=np.zeros(3, int),
                         source_xy=np.array([[10.5, .5], [10.5, .5], [1.5, .5]]))
            keep, report = clear_surface(dense, xyz, images, root, np.zeros(3), np.eye(3), .02)
            np.testing.assert_array_equal(keep, [False, True, False])
            np.testing.assert_array_equal(xyz, original)
            self.assertEqual(report['fitted_removed'], 2)

    def test_marking_requires_two_retained_votes_and_half_the_track(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.array([[255, 0, 0, 0]], np.uint8)).save(root/'0.png')
            obs = dict(offsets=np.array([0, 2, 5, 10]), image_index=np.zeros(10, int),
                       xy=np.array([[x+.5, .5] for x in [0, 0, 0, 3, 3, 0, 0, 3, 3, 3]]))
            selected, votes = marking_votes(obs, [SimpleNamespace(name='0.jpg')], root)
            np.testing.assert_array_equal(votes, [2, 1, 2])
            np.testing.assert_array_equal(selected, [True, False, False])

    def test_road_selection_uses_retained_static_observations_and_preserves_xyz(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            branch = root/'B1_shared'
            branch.mkdir()
            for folder in ['road_masks', 'static_valid_masks']:
                (source/folder).mkdir(parents=True)
            images = [SimpleNamespace(name=f'{i}.jpg') for i in range(4)]
            road_mask = np.array([[255, 0, 255]], np.uint8)
            static = np.array([[255, 255, 0]], np.uint8)
            for image in images:
                Image.fromarray(road_mask).save(source/'road_masks'/f'{Path(image.name).stem}.png')
                Image.fromarray(static).save(source/'static_valid_masks'/f'{Path(image.name).stem}.png')
            points = dict(xyz=np.arange(15, dtype=np.float32).reshape(5, 3),
                          rgb=np.arange(15, dtype=np.uint8).reshape(5, 3), track_length=np.array([2, 3, 4, 1, 2]))
            save_npz(branch/'new_points.npz', **points)
            save_npz(branch/'point_observations.npz', offsets=np.array([0, 2, 5, 9, 10, 12]),
                     image_index=np.array([0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 0, 1]),
                     xy=np.array([[x+.5, .5] for x in [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 2]]))
            (branch/'metrics.json').write_text(json.dumps({'input_pairs': 3}))
            (root/'manifest.json').write_text(json.dumps({'source': str(source)}))
            with patch.object(selection, 'model_data', return_value=(None, images, {})):
                result, rows, frames = selection.extract_road(root, root/'ROAD_shared')
            np.testing.assert_array_equal(rows, [0, 2])
            np.testing.assert_array_equal(frames, [0, 0])
            for field in points:
                np.testing.assert_array_equal(result[field], points[field][[0, 2]])

    def fitting_data(self):
        rng = np.random.default_rng(54)
        poses = np.broadcast_to(np.eye(4)[:3], (100, 3, 4)).copy()
        poses[:, 0, 3] = -np.linspace(0, 4, 100)
        origin, forward, lateral, up, _ = camera_basis(poses)
        uv = rng.uniform([0, -1], [4, 1], (4000, 2))
        height = -1+.001*uv[:, 0]**2+.005*uv[:, 1]+.0005*uv[:, 0]*uv[:, 1]
        local = np.column_stack([uv, height])
        xyz = origin+local@np.stack([forward, lateral, up])
        road = dict(xyz=xyz, track_length=np.full(4000, 4), angle=np.full(4000, 5.),
                    reprojection=np.full(4000, .1), max_reprojection=np.full(4000, .2))
        return road, np.arange(4000), rng.integers(0, 100, 4000), poses

    def test_fit_artifact_records_window_and_reloads_same_geometry(self):
        road, rows, frames, poses = self.fitting_data()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            patches = fit_patches(road, rows, frames, poses, root)
            loaded = load_patches(root/'surfaces')
            self.assertEqual((loaded[0]['first'], loaded[0]['stop']), (0, 100))
            for key in ['origin', 'basis']:
                np.testing.assert_array_equal(patches[0][key], loaded[0][key])
            for key in patches[0]['surface']:
                np.testing.assert_array_equal(patches[0]['surface'][key], loaded[0]['surface'][key])
            self.assertTrue(json.loads((root/'fit_report.json').read_text())['patches'][0]['passed'])

    def test_insufficient_holdout_fails_instead_of_silently_skipping_patch(self):
        road, rows, frames, poses = self.fitting_data()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'failed road fit'):
                fit_patches(road, rows*5+1, frames, poses, root)
            self.assertFalse(json.loads((root/'fit_report.json').read_text())['patches'][0]['passed'])
            self.assertFalse((root/'surfaces/patch_00.npz').exists())

    def test_query_keeps_overlap_weighting_and_source_ray(self):
        patches = [plane(4, 0, 100), plane(6, 80, 220)]
        pose, intrinsic = np.eye(4)[:3], np.eye(3)
        xy = np.array([[0., 0.], [1., 2.]])
        result = query_surface(patches, 90, xy, pose, intrinsic)
        expected = (4*.5+6*.55)/1.05
        np.testing.assert_allclose(result['depth'], expected)
        np.testing.assert_allclose(result['xyz'][:, :2]/result['xyz'][:, 2:], xy)
        self.assertTrue(result['valid'].all())

    def test_unsupported_negative_and_grazing_rays_have_no_depth(self):
        pose, intrinsic = np.eye(4)[:3], np.eye(3)
        tests = [(plane(support=1.), 10, [[5., 0.]]),
                 (plane(-5.), 10, [[0., 0.]]),
                 (plane(), 10, [[100., 0.]]),
                 (plane(), 200, [[0., 0.]])]
        for surface, index, xy in tests:
            with self.subTest(index=index, xy=xy):
                result = query_surface([surface], index, np.array(xy), pose, intrinsic)
                self.assertFalse(result['valid'].any())
                self.assertTrue(np.isnan(result['depth']).all())
                self.assertTrue(np.isnan(result['xyz']).all())

    def test_same_query_handles_both_camera_resolutions_without_pixel_shift(self):
        low = np.array([[545., 0, 333.], [0, 545., 185.], [0, 0, 1.]])
        scale = 900/666
        high = np.diag([scale, scale, 1.])@low
        xy = np.array([[200., 300.], [400., 250.]])
        a = query_surface([plane()], 10, xy, np.eye(4)[:3], low)
        b = query_surface([plane()], 10, xy*scale, np.eye(4), high)
        np.testing.assert_allclose(a['xyz'], b['xyz'], atol=1e-12)
        np.testing.assert_array_equal(a['valid'], b['valid'])

    def test_cell_selection_keeps_actual_best_point_not_voxel_center(self):
        xyz = np.array([[.001, .002, 5.], [.003, .002, 5.], [.02, .002, 5.]])
        rows = select_cells(xyz, np.array([3., 1., 2.]), np.zeros(3), .01)
        np.testing.assert_array_equal(rows, [1, 2])
        np.testing.assert_array_equal(xyz[rows], xyz[[1, 2]])

    def test_sampling_rgb_and_pixel_lineage_are_exact(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            for folder in ['images', 'road_masks', 'static_valid_masks']:
                (source/folder).mkdir()
            rgb = np.arange(4*6*3, dtype=np.uint8).reshape(4, 6, 3)
            Image.fromarray(rgb).save(source/'images/0.png')
            for folder in ['road_masks', 'static_valid_masks']:
                Image.fromarray(np.full((4, 6), 255, np.uint8)).save(source/folder/'0.png')
            camera = pycolmap.Camera(model='PINHOLE', width=6, height=4, params=[3, 3, 3, 2])
            image = SimpleNamespace(name='0.png', camera_id=1,
                                    cam_from_world=lambda: SimpleNamespace(matrix=lambda: np.eye(4)[:3]),
                                    projection_center=lambda: np.zeros(3))
            dense, _, candidates = sample_road(source, SimpleNamespace(cameras={1: camera}), [image], [plane()])
            self.assertEqual(candidates, 6)
            self.assertTrue((dense['source_xy'] % 2 == 0).all())
            x, y = dense['source_xy'].T
            np.testing.assert_array_equal(dense['rgb'], rgb[y, x])
            np.testing.assert_allclose(camera.img_from_cam(dense['xyz'].astype(float)), dense['source_xy'], atol=1e-6)

    def test_fusion_preserves_nonroad_and_input_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)/'input'
            root.mkdir()
            source = root/'source'
            (source/'sparse/0').mkdir(parents=True)
            for name in ['images.bin', 'cameras.bin']:
                (source/'sparse/0'/name).write_bytes(b'camera input')
            branch = root/'B1_shared'
            branch.mkdir()
            xyz = np.array([[1., 2., 3.], [4., 5., 6.]], np.float32)
            rgb = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)
            save_npz(branch/'new_points.npz', xyz=xyz, rgb=rgb)
            save_npz(branch/'point_observations.npz', offsets=np.array([0, 2, 4]))
            (branch/'metrics.json').write_text(json.dumps({'input_pairs': 3}))
            (root/'manifest.json').write_text(json.dumps({'source': str(source)}))
            original = (branch/'new_points.npz').read_bytes()
            dense = dict(xyz=np.array([[7., 8., 9.]], np.float32), rgb=np.array([[70, 80, 90]], np.uint8),
                         source_image_index=np.array([0]), source_xy=np.array([[2, 2]]))
            image = SimpleNamespace(name='000000.jpg', cam_from_world=lambda: SimpleNamespace(matrix=lambda: np.eye(4)[:3]))
            camera = SimpleNamespace(model_name='PINHOLE')
            output = root.parent/'output'
            with patch.object(reconstruct, 'model_data', return_value=(SimpleNamespace(cameras={1: camera}), [image], {})), \
                 patch.object(reconstruct, 'extract_road', return_value=({'xyz': xyz[:1]}, np.array([0]), np.array([0]))), \
                 patch.object(reconstruct, 'fit_patches', return_value=[plane()]), \
                 patch.object(reconstruct, 'sample_road', return_value=(dense, [], 1)):
                reconstruct.run(root, output)
            with np.load(output/'FUSED_AFTER/new_points.npz') as fused:
                np.testing.assert_array_equal(fused['xyz'], np.concatenate([xyz[1:], dense['xyz']]))
                np.testing.assert_array_equal(fused['rgb'], np.concatenate([rgb[1:], dense['rgb']]))
            self.assertEqual((branch/'new_points.npz').read_bytes(), original)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError):
                reconstruct.run(root, output)


if __name__ == '__main__':
    unittest.main()
