"""Camera-space parallax. Pose matrices are camera-to-world, optical x right/y down/z forward."""
import numpy as np


def parallax(depth, intrinsics, world_from_current, world_from_reference, valid, reference_depth=None, tolerance=0.02, reference_valid=None):
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).copy()
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('depth and valid must be matching HxW arrays')
    h, w = depth.shape
    k = np.asarray(intrinsics, dtype=np.float64)
    tc, tr = np.asarray(world_from_current), np.asarray(world_from_reference)
    if k.shape != (3, 3) or tc.shape != (4, 4) or tr.shape != (4, 4):
        raise ValueError('expected K 3x3 and camera-to-world poses 4x4')
    if not all(np.isfinite(x).all() for x in (k, tc, tr)):
        raise ValueError('camera calibration contains non-finite values')
    y, x = np.indices((h, w), dtype=np.float64)
    pixels = np.stack([x.ravel(), y.ravel(), np.ones(h*w)])
    xyz = np.linalg.solve(k, pixels) * depth.ravel()
    ref = (np.linalg.inv(tr) @ tc) @ np.vstack([xyz, np.ones(h*w)])
    projected = k @ ref[:3]
    z = ref[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = projected[0]/projected[2], projected[1]/projected[2]
    m = mask.ravel() & np.isfinite(depth.ravel()) & (depth.ravel()>0) & np.isfinite(u) & np.isfinite(v) & (z>0) & (u>=0) & (u<w) & (v>=0) & (v<h)
    if reference_depth is None:
        raise ValueError('reference depth required for visibility validation')
    if reference_depth is not None:
        rd = np.asarray(reference_depth)
        if rd.shape != depth.shape: raise ValueError('reference depth must match image shape')
        rv = None if reference_valid is None else np.asarray(reference_valid, dtype=bool)
        if rv is not None and rv.shape != depth.shape: raise ValueError('reference_valid must match image shape')
        idx = np.flatnonzero(m)
        ui = np.clip(np.rint(u[idx]).astype(int), 0, w-1)
        vi = np.clip(np.rint(v[idx]).astype(int), 0, h-1)
        observed = rd[vi, ui]
        m[idx] &= np.isfinite(observed) & (observed>0) & (np.abs(z[idx]-observed)<=tolerance)
        if rv is not None: m[idx] &= rv[vi, ui]
    flow = np.zeros((h*w, 2), dtype=np.float32)
    flow[m, 0] = ((u[m]-pixels[0,m])/w).astype(np.float32)
    flow[m, 1] = ((v[m]-pixels[1,m])/h).astype(np.float32)
    return flow.reshape(h,w,2), m.reshape(h,w)


def masked_average_pool(flow, valid, size=4):
    flow, valid = np.asarray(flow), np.asarray(valid, bool)
    if flow.ndim != 3 or flow.shape[-1] != 2 or flow.shape[:2] != valid.shape:
        raise ValueError('expected HxWx2 flow and HxW mask')
    h,w = valid.shape
    if h < size or w < size or size < 1: raise ValueError('invalid pooling size')
    out = np.zeros((size,size,2), dtype=np.float32)
    mask = np.zeros((size,size,2), dtype=bool)
    for i in range(size):
        for j in range(size):
            # Adaptive pooling follows floor(start), ceil(end).
            y0,y1 = i*h//size, ((i+1)*h+size-1)//size
            x0,x1 = j*w//size, ((j+1)*w+size-1)//size
            m = valid[y0:y1,x0:x1]
            if m.any():
                vals = flow[y0:y1,x0:x1][m]
                if not np.isfinite(vals).all(): raise ValueError('valid flow contains non-finite values')
                out[i,j] = vals.mean(axis=0); mask[i,j] = True
    return out.reshape(-1), mask.reshape(-1)
