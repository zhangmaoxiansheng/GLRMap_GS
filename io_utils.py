"""Atomic experiment artifacts and streaming SHA256; no reconstruction policy."""
import hashlib
from pathlib import Path
import json
import numpy as np

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    path=Path(path);temporary=path.with_suffix('.part')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def export_ply(path,xyz,rgb):
    xyz=np.asarray(xyz,dtype=np.float32).reshape(-1,3);rgb=np.asarray(rgb,dtype=np.uint8).reshape(-1,3)
    vertex=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k,name in enumerate('xyz'):vertex[name]=xyz[:,k]
    for k,name in enumerate(['red','green','blue']):vertex[name]=rgb[:,k]
    with Path(path).open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        stream.write(vertex.tobytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()
