"""Offline payloads, full projection, unchanged geometry and timestamped lateral previews."""
import base64
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from ..io_utils import save_npz, sha, write_json
from .clouds import viewer
from .html import media_report
from .projection import raster, coverage
from .report import run
from .trajectory import lateral, lateral_weights
from .video import frame_schedule


class VisualizationContracts(unittest.TestCase):
    def test_projection_uses_nearest_positive_z_and_original_rgb(self):
        points = dict(xyz=np.array([[0, 0, 2], [0, 0, 4], [0, 0, -1], [100, 0, 1], [np.nan, 0, 1]]),
                      rgb=np.arange(15, dtype=np.uint8).reshape(5, 3), track_length=np.array([2, 3, 3, 3, 3]))
        K = np.array([[2., 0, 2], [0, 2, 2], [0, 0, 1]])
        result = raster(points, np.eye(4), K, 4, 4)
        self.assertEqual(result['projected_points'], 2)
        self.assertEqual(result['occupied_pixels'], 1)
        self.assertEqual(result['depth'][2, 2], 2)
        np.testing.assert_array_equal(result['rgb'][2, 2], points['rgb'][0])
        multi = raster(points, np.eye(4)[:3], K, 4, 4, multi=True)
        self.assertEqual(multi['depth'][2, 2], 4)

    def test_coverage_weights_partial_edge_tiles_by_actual_roi_pixels(self):
        roi = np.ones((13, 13), bool)
        hit = np.zeros_like(roi)
        hit[0, [0, 1, 2]] = True
        report = coverage(hit, roi)
        self.assertAlmostEqual(report['pixel_fraction'], 3/169)
        self.assertAlmostEqual(report['supported_tile_area_fraction'], 144/169)
        self.assertEqual(coverage(hit, np.zeros_like(roi))['pixel_fraction'], 0)

    def cameras(self):
        K = np.array([[5., 0, 6], [0, 5, 4], [0, 0, 1]])
        camera = SimpleNamespace(width=12, height=8, model_name='PINHOLE', calibration_matrix=lambda: K,
                                 mean_focal_length=lambda: 5.)
        images = []
        for index in range(3):
            pose = np.eye(4)[:3]
            pose[0, 3] = -index*.1
            transform = SimpleNamespace(matrix=lambda pose=pose: pose, rotation=SimpleNamespace(matrix=lambda: np.eye(3)))
            images.append(SimpleNamespace(name=f'{index:06d}.jpg', camera_id=1,
                           projection_center=lambda pose=pose: -pose[:, 3], cam_from_world=lambda transform=transform: transform))
        return SimpleNamespace(cameras={1: camera}), images

    def fixture(self, root):
        source = root/'source'
        for folder in ['images', 'road_masks', 'static_valid_masks', 'sparse/0']:
            (source/folder).mkdir(parents=True)
        for name in ['cameras.bin', 'images.bin']:
            (source/'sparse/0'/name).write_bytes(b'camera-loader mocked in this report integration check')
        write_json(source/'manifest.json', dict(signature=dict(start_seconds=10, fps=10)))
        for index in range(3):
            Image.fromarray(np.full((8, 12, 3), 100, np.uint8)).save(source/'images'/f'{index:06d}.jpg')
            Image.fromarray(np.zeros((8, 12), np.uint8)).save(source/'static_valid_masks'/f'{index:06d}.png')
            Image.fromarray(np.full((8, 12), 255, np.uint8)).save(source/'road_masks'/f'{index:06d}.png')
        write_json(root/'manifest.json', dict(source=str(source), viewer_branches=['TRACK', 'CURVE'],
                   viewer_labels={'TRACK': 'Track', 'CURVE': 'Curve'}, viewer_title='Small report'))
        xyz = np.array([[0., 0, 5], [0, 0, 10], [2, 0, 5], [100, 0, 5]], np.float32)
        rgb = np.arange(12, dtype=np.uint8).reshape(4, 3)
        for name in ['TRACK', 'CURVE']:
            (root/name).mkdir()
            arrays = dict(xyz=xyz, rgb=rgb)
            if name == 'TRACK':
                arrays['track_length'] = np.array([2, 3, 4, 2])
            save_npz(root/name/'new_points.npz', **arrays)
            write_json(root/name/'metrics.json', dict(accepted_new_points=4, input_pairs=2,
                       has_track_observations=name == 'TRACK'))
        return self.cameras()

    def test_single_file_viewer_preserves_sampling_identity_and_counts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, images = self.fixture(root)
            output = root/'cloud.html'
            viewer(root, output, model, images, dict(start_seconds=10, fps=10), max_points=2)
            html = output.read_text()
            config = json.loads(re.search(r'<script id="settings" type="application/json">(.*?)</script>', html, re.S)[1])
            self.assertEqual([(r['count'], r['originalCount']) for r in config['clouds']], [(2, 4), (2, 4)])
            self.assertFalse(config['clouds'][1]['hasTrackObservations'])
            payload = re.search(r'data-cloud="0">(.*?)</script>', html, re.S)[1]
            packed = np.frombuffer(base64.b64decode(payload), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,)), ('quality', 'u1')])
            chosen = np.sort(np.random.default_rng(20261015).choice(4, 2, replace=False))
            with np.load(root/'TRACK/new_points.npz') as arrays:
                np.testing.assert_array_equal(packed['xyz'], (arrays['xyz'][chosen].astype(float)-config['origin']).astype(np.float32))
                np.testing.assert_array_equal(packed['rgb'], arrays['rgb'][chosen])
            self.assertNotRegex(html, r'<script[^>]+src=|fetch\(')
            self.assertNotIn('__SETTINGS_JSON__', html)

    def test_report_full_projection_ignores_zero_static_masks_and_bev_uses_all_points(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, images = self.fixture(root)
            hashes = {str(path): sha(path) for path in root.rglob('*') if path.is_file()}
            with patch('GLRMap_GS.visualization.report.model_data',
                       return_value=(model, images, {image.name: image for image in images})):
                run(root, max_points=2, image_name='000000.jpg')
            out = root/'comparison'
            projection = json.loads((out/'projection_metrics.json').read_text())
            self.assertFalse(projection['full_cloud_mask_clipping'])
            self.assertEqual(projection['clouds'][0]['input_points'], 4)
            self.assertEqual(projection['clouds'][0]['occupied_pixels'], 2)
            self.assertEqual(json.loads((out/'road_bev_side.json').read_text())['clouds'][0]['input_count'], 4)
            for name in ['bev_projection.html', 'camera_projection.html']:
                html = (out/name).read_text()
                self.assertIn('data:image/png;base64,', html)
                self.assertNotRegex(html, r'src="(?!data:)')
            self.assertEqual(hashes, {path: sha(path) for path in hashes})

    def test_no_index_or_external_media_dependencies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (2, 2)).save(root/'image.png')
            media_report(root/'report.html', '<title>', 'note', [('photo', root/'image.png')])
            self.assertIn('&lt;title&gt;', (root/'report.html').read_text())
            self.assertIn('data:image/png;base64,', (root/'report.html').read_text())
            with self.assertRaisesRegex(ValueError, 'index'):
                media_report(root/'index.html', 'title', 'note', [])

    def test_lateral_motion_preserves_rotation_intrinsics_and_has_no_vertical_shift(self):
        E = np.tile(np.eye(4), (100, 1, 1))
        E[:, 2, 3] = -np.arange(100)*.01
        K = np.tile(np.eye(3), (100, 1, 1))
        for anchor in [None, 50]:
            trajectory, audit = lateral(E, K, .06891966864338281, anchor=anchor)
            count = 100 if anchor is None else 96
            np.testing.assert_array_equal(trajectory['intrinsics'], K[:count])
            pose = np.linalg.inv(trajectory['extrinsics'])
            np.testing.assert_allclose(pose[:, 0, 3], .06891966864338281*lateral_weights(count), atol=1e-16)
            self.assertLess(audit['max_local_vertical'], 1e-8)
            self.assertLess(audit['max_local_forward'], 1e-8)
            self.assertEqual(audit['max_rotation_change'], 0)
        np.testing.assert_array_equal(E[:, 0, 3], 0)

    def test_overlap_blends_identical_global_times_without_duplicates(self):
        segments = [dict(first_frame=start, stop_frame=start+100) for start in range(3200, 4101, 75)]
        frames, transitions = frame_schedule(segments)
        self.assertEqual([r['global_frame'] for r in frames], list(range(3200, 4200)))
        self.assertEqual(len(transitions), 12)
        blended = [r for r in frames if 'newer_weight' in r]
        self.assertEqual(len(blended), 120)
        for item in blended:
            self.assertGreater(item['newer_weight'], 0)
            self.assertLess(item['newer_weight'], 1)
            self.assertEqual(item['newer'], item['older']+1)

    def test_nonoverlapping_or_empty_video_intervals_fail(self):
        for segments in [[dict(first_frame=0, stop_frame=0)],
                         [dict(first_frame=0, stop_frame=100), dict(first_frame=100, stop_frame=200)]]:
            with self.assertRaises(ValueError):
                frame_schedule(segments)


if __name__ == '__main__':
    unittest.main()
