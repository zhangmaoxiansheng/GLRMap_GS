"""Batch boundaries, required caches, true overlap and failed-child queue semantics."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from ..io_utils import save_npz, sha, write_json
from .cloud import plan as cloud_plan, matching_cache_sizes
from . import gs_queue
from . import window
from .demo import demo_images, check_road_rendering
from .. import runtime


class PipelineContracts(unittest.TestCase):
    def test_demo_requires_exactly_100_consecutive_native_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for frame in range(3400, 3500):
                (root/f'{frame:06d}.jpg').touch()
            with patch('PIL.Image.open', return_value=SimpleNamespace(size=(666, 370))):
                self.assertEqual(len(demo_images(root)), 100)
                (root/'003450.jpg').rename(root/'003500.jpg')
                with self.assertRaisesRegex(ValueError, 'consecutive'):
                    demo_images(root)
                (root/'003500.jpg').unlink()
                with self.assertRaisesRegex(ValueError, 'exactly 100'):
                    demo_images(root)

    def test_demo_rejects_near_clipped_road_without_rescaling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'road_assets/road_depth_targets.npz'
            path.parent.mkdir()
            save_npz(path, depth=np.array([0., .3, .6]), correction_mask=[False, True, True])
            check_road_rendering(root)
            save_npz(path, depth=np.array([0., .1, .6]), correction_mask=[False, True, True])
            before = sha(path)
            with self.assertRaisesRegex(ValueError, 'incompatible with the pinned renderer'):
                check_road_rendering(root)
            self.assertEqual(sha(path), before)
            audit = json.loads((root/'road_assets/renderer_audit.json').read_text())
            self.assertFalse(audit['passed'])
            self.assertEqual(audit['clipped_pixels'], 1)

    def test_python_path_preserves_virtual_environment_symlink(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root/'base/python'
            base.parent.mkdir()
            base.touch()
            executable = root/'venv/bin/python'
            executable.parent.mkdir(parents=True)
            executable.symlink_to(base)
            with patch.dict(runtime.PATHS, python_matching=str(executable)):
                self.assertEqual(runtime.python_for('matching'), str(executable))
                self.assertNotEqual(runtime.python_for('matching'), str(base))

    def test_window_pairs_keep_direct_neighbors_but_not_unrelated_pairs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = [[f'{i:06d}.jpg', f'{i+1:06d}.jpg'] for i in range(100)]
            pairs.append(['000000.jpg', '000050.jpg'])
            write_json(root/'manifest.json', dict(policies={'B1': pairs}))
            window.select_window_pairs(root, 40, 60)
            result = json.loads((root/'manifest.json').read_text())
            self.assertEqual(result['full_batch_pair_budget'], 101)
            self.assertEqual(result['pair_budget'], 22)
            self.assertIn(['000000.jpg', '000050.jpg'], result['policies']['B1'])
            self.assertNotIn(['000000.jpg', '000001.jpg'], result['policies']['B1'])
            self.assertEqual(result['reference_image_names'], [f'{i:06d}.jpg' for i in range(40, 60)])

    def test_window_plan_limits_work_and_scales_training_budget(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['video.mp4', 'anchor.npz', 'matching_settings.json', 'frames_manifest.json']:
                (root/name).touch()
            write_json(root/'plan.json', dict(fps=10, blocks=[[3200, 4200, 3300]],
                       source_video=str(root/'video.mp4'), first_anchor=str(root/'anchor.npz')))
            args = SimpleNamespace(input=root, output=root/'new', block=0, first_frame=3840,
                                   stop_frame=3860, reuse_sfm=False, settings=None)
            result = window.plan(args)
            self.assertEqual(result['views'], 20)
            self.assertEqual(result['total_training_steps'], 400)
            self.assertFalse((root/'new').exists())
            args.stop_frame = 4200
            with self.assertRaisesRegex(ValueError, '20–100'):
                window.plan(args)

    def test_window_failed_child_stops_before_matching_or_training(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)/'new'
            args = SimpleNamespace(reuse_sfm=False, python_sfm='sfm-python')
            planned = dict(output=str(out), views=20, first_frame=3840, stop_frame=3860)
            with patch.object(window, 'prepare_inputs'), \
                 patch.object(window.subprocess, 'run', return_value=SimpleNamespace(returncode=9)) as child:
                with self.assertRaisesRegex(RuntimeError, 'sfm failed:9'):
                    window.run(args, planned)
            self.assertEqual(child.call_count, 1)
            status = json.loads((out/'progress.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['stage'], 'sfm')
            self.assertFalse((out/'gs').exists())

    def test_cloud_blocks_have_1000_views_and_100_anchor_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(fps=10, blocks=[[n, n+1000, n+100] for n in [3200, 4100, 5000]],
                          source_video='video.mp4', first_anchor='anchor.npz')
            write_json(root/'plan.json', config)
            plan = cloud_plan(root)
            self.assertEqual(plan['unique_views'], 2800)
            self.assertEqual(plan['stages'][-2:], ['road', 'report'])
            self.assertFalse(plan['GS_training'])
            config['blocks'][1][0] += 1
            write_json(root/'plan.json', config)
            with self.assertRaises(ValueError):
                cloud_plan(root)

    def test_report_without_real_pair_or_seed_caches_cannot_skip_matching(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'shared').mkdir()
            write_json(root/'manifest.json', dict(policies={'B1': [['A.jpg', 'B.jpg']]}))
            write_json(root/'shared/matching_report.json', dict(complete=True))
            with self.assertRaisesRegex(FileNotFoundError, 'explicitly rebuild'):
                matching_cache_sizes(root)
            for folder in ['direct_pairs', 'seeds', 'superpoint']:
                (root/'shared'/folder).mkdir()
            for filename in ['direct_pairs/A_B.npz', 'direct_pairs/A_B.json', 'seeds/A.npz',
                             'seeds/B.npz', 'superpoint/A.pt', 'superpoint/B.pt']:
                (root/'shared'/filename).write_bytes(b'cache inventory fixture')
            self.assertEqual(len(matching_cache_sizes(root)), 7)
            (root/'shared/seeds/A.npz').unlink()
            with self.assertRaises(FileNotFoundError):
                matching_cache_sizes(root)

    def test_gs_windows_cover_1000_unique_views_with_25_view_overlap(self):
        windows = gs_queue.windows(3200, 4200)
        self.assertEqual(len(windows), 13)
        count = np.zeros(1000, int)
        for first, stop in windows:
            count[first-3200:stop-3200] += 1
        self.assertEqual((int(count.min()), int(count.max())), (1, 2))
        self.assertTrue(all(a[1]-b[0] == 25 for a, b in zip(windows, windows[1:])))
        with self.assertRaises(ValueError):
            gs_queue.windows(3200, 4201)

    def config(self, root, first):
        root.mkdir()
        names = np.array([f'{n:06d}.jpg' for n in range(first, first+100)])
        E = np.tile(np.eye(4), (100, 1, 1))
        E[:, 0, 3] = -np.arange(first, first+100)*.01
        K = np.tile(np.eye(3), (100, 1, 1))
        save_npz(root/'cameras.npz', image_names=names, extrinsics=E, intrinsics=K)
        return dict(output=str(root/'gs'), first_frame=first, stop_frame=first+100,
                    cameras=str(root/'cameras.npz'), reference_cameras=str(root/'cameras.npz'),
                    road_ply=str(root/'road.ply'), context_indices=np.linspace(0, 99, 16, dtype=int).tolist(),
                    steps_per_view_per_phase=10, python_resplat='python')

    def test_overlap_requires_same_cameras_and_every_shared_road_attribute(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older, newer = self.config(root/'A', 0), self.config(root/'B', 75)
            save_npz(root/'A/source_rows.npz', dense_road_source_rows=[0, 1, 2])
            save_npz(root/'B/source_rows.npz', dense_road_source_rows=[1, 2, 3])
            a = np.array([(0., .5), (1., .6), (2., .7)], dtype=[('x', 'f4'), ('opacity', 'f4')])
            b = np.array([(1., .6), (2., .7), (3., .8)], dtype=a.dtype)
            clouds = {older['road_ply']: a, newer['road_ply']: b}
            reader = SimpleNamespace(read=lambda path, **kw: {'vertex': SimpleNamespace(data=clouds[str(path)])})
            with patch.dict('sys.modules', {'plyfile': SimpleNamespace(PlyData=reader)}):
                result = gs_queue.overlap(older, newer)
                self.assertEqual(result['common_road_rows'], 2)
                b['opacity'][0] = .9
                with self.assertRaisesRegex(ValueError, 'Gaussian attributes'):
                    gs_queue.overlap(older, newer)
                b['opacity'][0] = .6
                with np.load(root/'B/cameras.npz') as arrays:
                    values = {key: arrays[key] for key in arrays.files}
                values['extrinsics'][0, 1, 3] += .01
                save_npz(root/'B/cameras.npz', **values)
                with self.assertRaisesRegex(ValueError, 'identical camera'):
                    gs_queue.overlap(older, newer)

    def test_verified_group_checks_budget_and_render_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root/'group', 0)
            out = Path(config['output'])
            out.mkdir()
            ply = out/'final.ply'
            ply.write_bytes(b'final Gaussian fields checked by the GS training stage')
            html = out/'render_comparison.html'
            html.write_text('standalone HTML')
            counts = {f'{n:06d}.jpg': 10 for n in range(100)}
            report = dict(iterations=2000, road_fields_bit_exact=True, phase_counts=[counts, counts],
                          final_PLY=str(ply), final_sha256=sha(ply), road_count=100, Gaussian_count=120, elapsed_seconds=1.)
            write_json(out/'training_report.json', report)
            write_json(out/'visualization_complete.json', dict(config_fingerprint='test', artifact_hashes={str(html): sha(html)}))
            with patch.object(gs_queue, 'verify'):
                self.assertEqual(gs_queue.verify_group(config, 'test')['background_count'], 20)
                report['phase_counts'][0]['000000.jpg'] = 9
                write_json(out/'training_report.json', report)
                with self.assertRaisesRegex(ValueError, 'verification'):
                    gs_queue.verify_group(config, 'test')

    def test_failed_gs_child_does_not_publish_or_report_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root/'group', 0)
            path = root/'queue.json'
            settings = root/'group.json'
            write_json(settings, config)
            queue = dict(output=str(root/'queue'), configs=[str(settings)])
            write_json(path, queue)
            child = SimpleNamespace(wait=lambda: 5)
            with patch.object(gs_queue.subprocess, 'Popen', return_value=child), \
                 patch.object(gs_queue.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*2**30)), \
                 patch.object(gs_queue, 'join_forward') as publish:
                self.assertEqual(gs_queue.run(queue, [(config, 'test')], path, False), 5)
                publish.assert_not_called()
            out = Path(queue['output'])
            self.assertFalse((out/'summary.json').exists())
            self.assertEqual(json.loads((out/'progress.json').read_text())['stage'], 'failed')

    def test_new_queue_cannot_adopt_or_overwrite_historical_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root/'group', 0)
            settings = root/'group.json'
            write_json(settings, config)
            out = root/'historical'
            out.mkdir()
            write_json(out/'plan.json', dict(original_experiment=True))
            original_hash = sha(out/'plan.json')
            queue = dict(output=str(out), configs=[str(settings)])
            path = root/'queue.json'
            write_json(path, queue)
            with patch.object(gs_queue.subprocess, 'Popen') as child:
                with self.assertRaisesRegex(ValueError, 'fresh queue output'):
                    gs_queue.run(queue, [(config, 'test')], path, False)
                child.assert_not_called()
            self.assertEqual(sha(out/'plan.json'), original_hash)
            self.assertFalse((out/'queue_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
