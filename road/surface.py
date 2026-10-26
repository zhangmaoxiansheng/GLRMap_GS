"""Current five-coefficient road surface; coordinates and GRID are in SfM units."""
import numpy as np

GRID = .00420966


def camera_basis(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    centers = -np.einsum("vji,vj->vi", rotations, translations)
    camera_down = np.einsum("vji,j->vi", rotations, np.array([0.0, 1.0, 0.0]))
    up = -camera_down.mean(axis=0)
    up /= np.linalg.norm(up)
    forward = centers[-1] - centers[0]
    forward -= up * np.dot(forward, up)
    forward /= np.linalg.norm(forward)
    lateral = np.cross(up, forward)
    lateral /= np.linalg.norm(lateral)
    origin = centers[0]
    return origin, forward, lateral, up, centers


def design(uv):
    u,v=uv.T
    return np.column_stack((np.ones(len(uv)),u,v,u*u,u*v))


def fit_surface(local,grid,seed=54):
    """RANSAC initialization then robust, density-balanced smooth road fit."""
    offset=np.median(local[:,:2],axis=0)
    scale=np.maximum(np.ptp(local[:,:2],axis=0)/2,grid*10)
    uv=(local[:,:2]-offset)/scale
    A=design(uv)
    rng=np.random.default_rng(seed)
    subset=rng.choice(len(local),min(30000,len(local)),replace=False)
    plane=A[:,:3]
    best_count=-1; initial=None
    for _ in range(400):
        chosen=rng.choice(subset,3,replace=False)
        if abs(np.linalg.det(plane[chosen]))<1e-4:
            continue
        candidate=np.linalg.solve(plane[chosen],local[chosen,2])
        residual=abs(plane[subset]@candidate-local[subset,2])
        count=int((residual<grid*3).sum())
        if count>best_count:
            best_count=count; initial=candidate
    if initial is None or best_count<len(subset)*.4:
        raise ValueError('RoMa road lacks a dominant consistent surface')
    weight=(abs(plane@initial-local[:,2])<grid*6).astype(float)
    cells=np.floor(local[:,:2]/(grid*8)).astype(np.int32)
    _,idx,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
    balance=1./np.sqrt(counts[idx])
    coef=np.r_[initial,0.,0.]
    for _ in range(12):
        w=weight*balance
        coef=np.linalg.lstsq(A*np.sqrt(w)[:,None],local[:,2]*np.sqrt(w),rcond=None)[0]
        residual=local[:,2]-A@coef
        sigma=max(1.4826*np.median(abs(residual-np.median(residual))),grid*.15)
        cutoff=min(grid*6,max(grid*2,sigma*4))
        r=residual/cutoff
        weight=np.maximum(0.,1.-r*r)**2
    return dict(coefficients=coef,offset=offset,scale=scale,
        support_min=np.percentile(local[:,:2],.25,axis=0),
        support_max=np.percentile(local[:,:2],99.75,axis=0)),residual,weight>.01


def evaluate(surface,uv):
    q=(uv-surface['offset'])/surface['scale']
    c=surface['coefficients']; u,v=q.T
    height=design(q)@c
    slope=np.column_stack(((c[1]+2*c[3]*u+c[4]*v)/surface['scale'][0],
                           (c[2]+c[4]*u)/surface['scale'][1]))
    n=np.column_stack((-slope,np.ones(len(uv))))
    n/=np.linalg.norm(n,axis=1)[:,None]
    return height,n


def intersect(surface,center,rays,origin,basis):
    """Exact camera-Z intersection with height polynomial; positive near root."""
    p=(center-origin)@basis.T
    d=rays@basis.T
    u0,v0=(p[:2]-surface['offset'])/surface['scale']
    du,dv=(d[:,:2]/surface['scale']).T
    c=surface['coefficients']
    A=c[3]*du*du+c[4]*du*dv
    B=c[1]*du+c[2]*dv+2*c[3]*u0*du+c[4]*(u0*dv+v0*du)-d[:,2]
    C=c[0]+c[1]*u0+c[2]*v0+c[3]*u0*u0+c[4]*u0*v0-p[2]
    disc=B*B-4*A*C
    # Stable quadratic roots; use the exact linear root in the planar limit.
    q=-.5*(B+np.copysign(np.sqrt(np.maximum(disc,0)),B))
    with np.errstate(divide='ignore',invalid='ignore'):
        z0=q/A; z1=C/q; linear=-C/B
    z0=np.where((z0>0)&(disc>=0),z0,np.inf)
    z1=np.where((z1>0)&(disc>=0),z1,np.inf)
    z=np.minimum(z0,z1)
    z=np.where(abs(A)<1e-12,linear,z)
    z=np.where(np.isfinite(z)&(z>0),z,np.nan)
    xyz=center+z[:,None]*rays
    local=(xyz-origin)@basis.T
    supported=np.isfinite(xyz).all(1)&np.all(local[:,:2]>=surface['support_min']-.05,axis=1)&np.all(local[:,:2]<=surface['support_max']+.05,axis=1)
    return xyz,z,supported
