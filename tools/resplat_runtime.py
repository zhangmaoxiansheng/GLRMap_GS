"""Native ReSplat paths, GPU ownership and10FPS output verification."""
from contextlib import contextmanager
import fcntl
import json
import subprocess
import time
from ..runtime import ROOT, GS_ROOT, RESPLAT, GPU_LOCK, path
from ..io_utils import sha, write_json as save

CHECKPOINT = path('resplat_checkpoint', RESPLAT/'pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth')


def probe(video, count):
    data = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,nb_read_frames,avg_frame_rate:format=duration','-of','json',str(video)],text=True))
    stream = data['streams'][0]
    assert int(stream['nb_read_frames']) == count and stream['avg_frame_rate'] == '10/1'
    assert abs(float(data['format']['duration'])-count/10) < .01
    return data


@contextmanager
def gpu_slot():
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open('a') as lock:
        while True:
            try:fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Waiting for GPU lock', GPU_LOCK, flush=True);time.sleep(15);continue
            free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free >= 18000:break
            fcntl.flock(lock, fcntl.LOCK_UN)
            print('Waiting for GPU memory',free,'MiB',flush=True);time.sleep(15)
        try:yield
        finally:fcntl.flock(lock, fcntl.LOCK_UN)
