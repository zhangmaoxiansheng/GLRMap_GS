"""Point-cloud WebGL, full-cloud projection and BEV from explicit current branches."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..io_utils import sha, write_json
from ..sfm.geometry import model_data
from .clouds import branch_inputs, load_cloud, viewer
from .projection import raster, coverage
from .bev import compare
from .html import media_report


def projections(specs, source, model, image, output, png_only=False):
    camera = model.cameras[image.camera_id]
    if camera.model_name not in ['PINHOLE', 'SIMPLE_PINHOLE']:
        raise ValueError('Full projection requires the current pinhole reconstruction')
    pose, intrinsics = image.cam_from_world().matrix(), camera.calibration_matrix()
    width, height = camera.width, camera.height
    photo_path = source/'images'/image.name
    with Image.open(photo_path) as photo:
        original = photo.convert('RGB')
    if original.size != (width, height):
        raise ValueError('Source photo and camera dimensions differ')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    strip = Image.new('RGB', ((len(specs)+1)*width, height+60), '#10151d')
    strip.paste(original, (0, 60))
    draw = ImageDraw.Draw(strip)
    draw.text((8, 8), f'Source: {image.name}', fill='white', font=font)
    stem = Path(image.name).stem+'.png'
    road_path = source/'road_masks'/stem
    static_path = source/'static_valid_masks'/stem
    roi = None
    if road_path.exists():
        with Image.open(road_path) as mask:
            roi = np.array(mask) > 127
        with Image.open(static_path) as mask:
            roi &= np.array(mask) > 127
        if roi.shape != (height, width):
            raise ValueError('Road ROI has wrong crop/resolution')
    records = []
    for index, spec in enumerate(specs, 1):
        points = load_cloud(spec['path'])
        result = raster(points, pose, intrinsics, width, height)
        # Full projection never intersects a road, dynamic or sky mask.
        strip.paste(Image.fromarray(result['rgb']), (index*width, 60))
        draw.text((index*width+8, 8), spec['label'], font=font, fill='white')
        draw.text((index*width+8, 32), f'{result["occupied_pixels"]:,} pixels / {len(points["xyz"]):,} points', font=font, fill='white')
        record = dict(name=spec['name'], input_points=result['input_points'], projected_points=result['projected_points'],
                      occupied_pixels=result['occupied_pixels'], source_sha256=sha(spec['path']))
        if roi is not None:
            record['target_road_roi_coverage'] = coverage(np.isfinite(result['depth']), roi)
        records.append(record)
    strip.save(output/'camera_projection.png')
    report = dict(camera_image=image.name, width=width, height=height, point_size_pixels=1,
                  cam_from_world=pose.tolist(), intrinsics=intrinsics.tolist(), source_image_sha256=sha(photo_path),
                  clouds=records, full_cloud_mask_clipping=False, unchanged_xyz_rgb=True,
                  limitation='Pixel coverage measures projected density, not independent 3D accuracy; nearest-point z-buffer is not complete scene occlusion')
    write_json(output/'projection_metrics.json', report)
    if not png_only:
        media_report(output/'camera_projection.html', '同相机完整点云投影',
                 '同Pose/K、同画幅、1像素点、原始RGB。完整投影不按mask裁剪；点密度不代表独立几何精度。',
                 [('原图与完整点云投影', output/'camera_projection.png')])
    return report


def run(root, output=None, branches=None, image_name=None, max_points=250000, png_only=False):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root/'comparison'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    source = Path(manifest['source'])
    model, images, lookup = model_data(source/'sparse/0')
    specs = branch_inputs(root, branches)
    image = lookup[image_name] if image_name else images[(len(images)-1)//2]
    inputs = [root/'manifest.json', source/'manifest.json', source/'sparse/0/cameras.bin', source/'sparse/0/images.bin']
    inputs += [path for spec in specs for path in [spec['path'], spec['metrics_path']]]
    inputs.append(source/'images'/image.name)
    if (source/'road_masks'/f'{Path(image.name).stem}.png').exists():
        inputs += [source/folder/f'{Path(image.name).stem}.png' for folder in ['road_masks', 'static_valid_masks']]
    guards = {str(path): sha(path) for path in inputs}
    projections(specs, source, model, image, output, png_only=png_only)
    pose_names = manifest.get('reference_image_names', [im.name for im in images])
    poses = np.stack([lookup[name].cam_from_world().matrix() for name in pose_names])
    compare(specs, poses, output/'road_bev_side.png')
    if not png_only:
        media_report(output/'bev_projection.html', manifest.get('viewer_title', root.name),
                 '投影与BEV使用全量点云；BEV仅统一坐标显示，不拟合或压平道路。',
                 [('完整点云投影', output/'camera_projection.png'), ('同尺度BEV与侧视', output/'road_bev_side.png')])
        signature = json.loads((source/'manifest.json').read_text())['signature']
        viewer(root, output/'raw_3d_interactive.html', model, images, signature, branches, max_points)
    if any(sha(path) != digest for path, digest in guards.items()):
        raise ValueError('Reconstruction inputs changed while generating the report')
    write_json(output/'report.json', dict(complete=True, source_sha256=guards, source_inputs_unchanged=True,
                                        projection_uses_full_cloud=True, bev_uses_full_cloud=True,
                                        standalone_html=[] if png_only else ['raw_3d_interactive.html', 'bev_projection.html', 'camera_projection.html']))
    return output/'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--branches', nargs='+')
    parser.add_argument('--image-name')
    parser.add_argument('--max-points', type=int, default=250000, help='Presentation cap only; projection/BEV use all points')
    parser.add_argument('--png-only', action='store_true', help='Write PNG and statistics without HTML')
    args = parser.parse_args()
    print(run(args.root, args.output, args.branches, args.image_name, args.max_points, args.png_only))


if __name__ == '__main__':
    main()
