"""RGB video encoding, PTS checks and same-timestamp overlap previews; no GS fusion."""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from ..io_utils import sha, write_json


def encode(path, width, height, fps):
    return subprocess.Popen(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '8',
        '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,duration', '-of', 'json', str(path)]))['streams'][0]


def verify_video(path, count, fps):
    info = probe(path)
    if int(info['nb_read_frames']) != count or info['r_frame_rate'] != f'{fps}/1':
        raise ValueError(f'Video count/FPS mismatch: {path}')
    frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]))['frames']
    if len(frames) != count or any(abs(float(item['best_effort_timestamp_time'])-i/fps)>1e-6 for i, item in enumerate(frames)):
        raise ValueError(f'Video timestamps are not i/fps: {path}')
    return dict(probe=info, sha256=sha(path), all_pts_equal_i_divided_by_fps=True)


def frame_schedule(segments, blend_frames=10):
    if not segments or blend_frames <= 0:
        raise ValueError('Ordered overlapping segments and positive blend length required')
    if any(segment['stop_frame'] <= segment['first_frame'] for segment in segments):
        raise ValueError('Empty/reversed video frame interval')
    transitions = []
    for index, (older, newer) in enumerate(zip(segments, segments[1:])):
        a, b, c, d = older['first_frame'], newer['first_frame'], older['stop_frame'], newer['stop_frame']
        if not a < b < c < d:
            raise ValueError('Expected strictly ordered, overlapping frame intervals')
        length = min(blend_frames, c-b)
        start = (b+c)//2-length//2
        transitions.append(dict(older=index, newer=index+1, overlap=[b, c], blend=[start, start+length]))
    frames, current = [], 0
    for global_frame in range(segments[0]['first_frame'], segments[-1]['stop_frame']):
        transition = transitions[current] if current < len(transitions) else None
        if transition and global_frame >= transition['blend'][1]:
            current += 1
            transition = transitions[current] if current < len(transitions) else None
        if transition and transition['blend'][0] <= global_frame < transition['blend'][1]:
            weight = (global_frame-transition['blend'][0]+.5)/(transition['blend'][1]-transition['blend'][0])
            frames.append(dict(global_frame=global_frame, older=current, newer=current+1, newer_weight=weight))
        else:
            frames.append(dict(global_frame=global_frame, group=current))
    for item in frames:
        for index in ([item['group']] if 'group' in item else [item['older'], item['newer']]):
            if not segments[index]['first_frame'] <= item['global_frame'] < segments[index]['stop_frame']:
                raise ValueError('Blend requests a frame outside its trained interval')
    return frames, transitions


def join_forward(segments, output, blend_frames=10):
    """segments contain first_frame, stop_frame and video; frames share global timestamps."""
    output = Path(output)
    frames, transitions = frame_schedule(segments, blend_frames)
    hashes = {str(segment['video']): sha(segment['video']) for segment in segments}
    captures = [cv2.VideoCapture(str(segment['video'])) for segment in segments]
    writer = None
    try:
        width, height = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        for cap, segment in zip(captures, segments):
            if (not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != segment['stop_frame']-segment['first_frame']
                    or abs(cap.get(cv2.CAP_PROP_FPS)-10)>1e-8
                    or (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height)):
                raise ValueError(f'Inconsistent source video: {segment["video"]}')
        def frame(index, global_frame):
            local = global_frame-segments[index]['first_frame']
            cap = captures[index]
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != local:
                cap.set(cv2.CAP_PROP_POS_FRAMES, local)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode group {index}, global frame {global_frame}')
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        writer = encode(output, width, height, 10)
        for item in frames:
            global_frame = item['global_frame']
            if 'group' in item:
                image = frame(item['group'], global_frame)
            else:
                a, b = frame(item['older'], global_frame), frame(item['newer'], global_frame)
                weight = item['newer_weight']
                image = np.round((1-weight)*a+weight*b).astype(np.uint8)
            writer.stdin.write(image.tobytes())
    finally:
        if writer is not None:
            writer.stdin.close()
            code = writer.wait()
        for cap in captures:
            cap.release()
    if code:
        raise RuntimeError(f'ffmpeg failed: {output}')
    info = verify_video(output, len(frames), 10)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Source videos changed during preview encoding')
    report = dict(complete=True, first=segments[0]['first_frame'], stop=segments[-1]['stop_frame'],
                  count=len(frames), gaps=0, duplicate_global_frames=0, declared_intervals_overlap=True,
                  transitions=transitions, frames=frames, video=info, source_sha256=hashes, model_fusion=False)
    write_json(output.with_suffix('.json'), report)
    return report
