"""Road covariance/identity, native source cells and restart input contracts; CPU only."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from ..io_utils import save_npz, sha, write_json
from .initialization import road_targets, source_masks
from .road import road_gaussians
from .state import FIELDS, cameras, gpu, verify, road_frozen_fields
from ..pipeline.gs import plan


class GSContracts(unittest.TestCase):
    def test_appearance_training_keeps_geometry_frozen_and_is_opt_in(self):
        self.assertEqual(road_frozen_fields({}), FIELDS)
        self.assertEqual(road_frozen_fields({'road_train_appearance': False}), FIELDS)
        self.assertEqual(road_frozen_fields({'road_train_appearance': True}),
                         ('_xyz', '_scaling', '_rotation'))
        for invalid in ['false', 1, None]:
            with self.assertRaisesRegex(ValueError, 'must be a boolean'):
                road_frozen_fields({'road_train_appearance': invalid})

    def patches(self):
        return [dict(first=0, stop=400, origin=np.zeros(3), basis=np.eye(3),
                     surface=dict(coefficients=np.array([5., 0, 0, 0, 0]), offset=np.zeros(2),
                                  scale=np.ones(2), support_min=np.full(2, -10000.),
                                  support_max=np.full(2, 10000.))) for _ in range(3)]

    def test_same_source_road_rows_have_identical_all_fields_across_windows(self):
        xyz = np.array([[1, 2, 5], [2, 3, 5], [3, 4, 5]], np.float32)
        rgb = np.array([[255, 100, 0], [50, 40, 30], [0, 128, 255]], np.uint8)
        frames = np.array([100, 150, 200])
        whole, normals = road_gaussians(xyz, rgb, frames, self.patches(), .002)
        subset, _ = road_gaussians(xyz[1:], rgb[1:], frames[1:], self.patches(), .002)
        self.assertEqual(whole[1:].tobytes(), subset.tobytes())
        np.testing.assert_array_equal(np.column_stack([whole[k] for k in 'xyz']), xyz)
        np.testing.assert_allclose(np.column_stack([whole[f'f_dc_{i}'] for i in range(3)])*.28209479177387814+.5,
                                   rgb/255., atol=1e-7)
        rotations = Rotation.from_quat(np.column_stack([whole[f'rot_{i}'] for i in [1, 2, 3, 0]])).as_matrix()
        np.testing.assert_allclose(rotations[:, :, 2], normals, atol=1e-7)
        np.testing.assert_allclose(np.exp(whole['scale_0']), .002*.7, rtol=1e-6)
        np.testing.assert_allclose(np.exp(whole['scale_2']), .002*.12, rtol=1e-6)

    def test_road_without_source_patch_fails_instead_of_fabricating_normals(self):
        with self.assertRaisesRegex(ValueError, 'no corresponding fitted patch'):
            road_gaussians(np.ones((1, 3), np.float32), np.ones((1, 3), np.uint8),
                           np.array([500]), self.patches(), .002)

    def test_any_dynamic_source_pixel_removes_native_cell_without_dilation(self):
        valid = np.full((256, 480), 255, np.uint8)
        covered = np.zeros_like(valid)
        valid[2, 2] = 0
        covered[6, 8] = 255
        dynamic, road, _ = source_masks([valid, np.full_like(valid, 255)],
                                       [covered, np.zeros_like(covered)], ['A.jpg', 'B.jpg'])
        np.testing.assert_array_equal(np.flatnonzero(dynamic), [241])
        np.testing.assert_array_equal(np.flatnonzero(road), [724])
        keep = ~(dynamic | road)
        self.assertTrue(keep[240])
        self.assertTrue(keep[30720:].all())
        # Uncovered semantic road remains, rather than being removed as road globally.
        self.assertEqual(int(keep.sum()), 2*30720-2)

    def targets(self, directory):
        root = Path(directory)
        depth = np.array([[[0., 4.], [0., 0.]]], np.float32)
        mask = depth > 0
        save_npz(root/'depth.npz', depth=depth, correction_mask=mask, image_names=['A.jpg'])
        cv2.imwrite(str(root/'A.png'), mask[0].astype(np.uint8)*255)
        return root, depth, mask

    def test_coverage_targets_are_reordered_by_image_name(self):
        with TemporaryDirectory() as directory:
            root, depth, mask = self.targets(directory)
            cv2.imwrite(str(root/'B.png'), mask[0].astype(np.uint8)*255)
            save_npz(root/'depth.npz', depth=np.concatenate([depth, depth*2]),
                     correction_mask=np.concatenate([mask, mask]), image_names=['A.jpg', 'B.jpg'])
            actual, supported = road_targets(root/'depth.npz', ['B.jpg', 'A.jpg'], (2, 2), root)
            np.testing.assert_array_equal(actual, np.concatenate([depth*2, depth]))
            np.testing.assert_array_equal(supported, np.concatenate([mask, mask]))

    def test_semantic_road_cannot_replace_uncovered_native_road(self):
        with TemporaryDirectory() as directory:
            root, _, _ = self.targets(directory)
            cv2.imwrite(str(root/'A.png'), np.full((2, 2), 255, np.uint8))
            with self.assertRaisesRegex(ValueError, 'actual curve coverage'):
                road_targets(root/'depth.npz', ['A.jpg'], (2, 2), root)

    def test_nonpositive_supported_depth_is_rejected(self):
        with TemporaryDirectory() as directory:
            root, depth, mask = self.targets(directory)
            depth[mask] = -1
            save_npz(root/'depth.npz', depth=depth, correction_mask=mask, image_names=['A.jpg'])
            with self.assertRaisesRegex(ValueError, 'positive, finite'):
                road_targets(root/'depth.npz', ['A.jpg'], (2, 2), root)

    def camera_config(self, root):
        names = np.array([f'{i:06d}.jpg' for i in range(100)])
        poses = np.tile(np.eye(4), (100, 1, 1))
        intrinsics = np.tile(np.array([[500., 0, 333], [0, 500, 185], [0, 0, 1]]), (100, 1, 1))
        save_npz(root/'cameras.npz', image_names=names, extrinsics=poses, intrinsics=intrinsics)
        return dict(cameras=str(root/'cameras.npz'), first_frame=0, stop_frame=100,
                    context_indices=np.linspace(0, 99, 16, dtype=int).tolist(),
                    steps_per_view_per_phase=10, output=str(root/'output'))

    def test_native_context_camera_contract_and_per_view_budget(self):
        with TemporaryDirectory() as directory:
            config = self.camera_config(Path(directory))
            result = plan(config, 'test')
            self.assertEqual(result['total_steps'], 2000)
            self.assertEqual(result['phase_steps'], 1000)
            config['steps_per_view_per_phase'] = 20
            self.assertEqual(plan(config, 'test')['total_steps'], 4000)
            config['context_indices'][-1] = 98
            with self.assertRaisesRegex(ValueError, '16 uniform'):
                cameras(config)

    def test_resume_rejects_changed_input_snapshot_and_code(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/name for name in ['source', 'snapshot', 'code']]
            for path in paths:
                path.write_text('original')
            record = dict(config_fingerprint='test', **{key: {str(path): sha(path)} for key, path in
                          zip(['source_hashes', 'snapshot_hashes', 'code_hashes'], paths)})
            write_json(root/'input_manifest.json', record)
            verify(root, {}, 'test')
            for path in paths:
                path.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Immutable input/code changed'):
                    verify(root, {}, 'test')
                path.write_text('original')
            with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                verify(root, {}, 'other')

    def test_twenty_view_window_preserves_camera_selection_and_budget(self):
        with TemporaryDirectory() as directory:
            config = self.camera_config(Path(directory))
            config.update(first_frame=40, stop_frame=60,
                          context_indices=np.linspace(0, 19, 16, dtype=int).tolist())
            poses, intrinsics, names = cameras(config)
            self.assertEqual(poses.shape, (20, 4, 4))
            self.assertEqual(intrinsics.shape, (20, 3, 3))
            self.assertEqual(names.tolist(), [f'{i:06d}.jpg' for i in range(40, 60)])
            self.assertEqual(plan(config, 'short')['total_steps'], 400)
            config['stop_frame'] = 59
            with self.assertRaises(ValueError):
                cameras(config)

    def test_explicit_single_patch_tangent_preserves_road_geometry(self):
        xyz = np.array([[1, 2, 5]], np.float32)
        rgb = np.array([[255, 100, 0]], np.uint8)
        frames = np.array([10])
        baseline, _ = road_gaussians(xyz, rgb, frames, self.patches(), .002)
        short, _ = road_gaussians(xyz, rgb, frames, self.patches()[:1], .002, tangent_patch_index=0)
        self.assertEqual(short.tobytes(), baseline.tobytes())

    def test_gpu_lock_released_on_failure(self):
        with TemporaryDirectory() as directory:
            config = dict(gpu_lock=str(Path(directory)/'gpu.lock'))
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with gpu(config):
                    raise RuntimeError('test failure')
            with gpu(config):
                pass


if __name__ == '__main__':
    unittest.main()
