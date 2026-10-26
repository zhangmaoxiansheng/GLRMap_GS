"""Fixed SfM cameras, COLMAP edge-origin pixels and world-coordinate rays."""
import numpy as np
import pycolmap

def model_data(path):
    model = pycolmap.Reconstruction(str(path))
    images = sorted(model.images.values(), key=lambda im: im.name)
    return model, images, {im.name: im for im in images}


def rays(camera, xy):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if not len(xy): return np.empty((0, 3))
    return np.column_stack((camera.cam_from_img(xy), np.ones(len(xy))))


def fixed_camera_parameters(camera):
    """XYZ fitter parameters: fx,cx,cy,k, with optional distinct fy."""
    if camera.model_name=='SIMPLE_RADIAL':return np.asarray(camera.params)
    if camera.model_name=='SIMPLE_PINHOLE':return np.r_[camera.params,0.]
    if camera.model_name=='PINHOLE':
        fx,fy,cx,cy=camera.params
        if np.isclose(fx,fy,rtol=1e-10,atol=1e-10):return np.array([fx,cx,cy,0.])
        return np.array([fx,cx,cy,0.,fy])
    raise ValueError(f'Unsupported fixed XYZ fitter camera: {camera.model_name}')


def essential(im0, im1):
    p0, p1 = im0.cam_from_world().matrix(), im1.cam_from_world().matrix()
    rotation = p1[:, :3] @ p0[:, :3].T
    translation = p1[:, 3]-rotation@p0[:, 3]
    x, y, z = translation
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return cross@rotation


def epipolar_error(model, im0, im1, xy0, xy1):
    c0, c1 = model.cameras[im0.camera_id], model.cameras[im1.camera_id]
    a, b = rays(c0, xy0), rays(c1, xy1)
    e = essential(im0, im1)
    l1, l0 = a@e.T, b@e
    residual = np.abs((b*l1).sum(1))
    f0, f1 = float(c0.mean_focal_length()), float(c1.mean_focal_length())
    return np.maximum(residual/np.maximum(np.linalg.norm(l0[:, :2], axis=1), 1e-12)*f0,
                      residual/np.maximum(np.linalg.norm(l1[:, :2], axis=1), 1e-12)*f1)


def angle_degrees(xyz, centers):
    unit = xyz-centers
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    dot = np.clip(unit@unit.T, -1, 1)
    return float(np.degrees(np.arccos(dot)).max())
