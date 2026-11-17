"""Dual-teacher geometry. Poses use world-from-frame homogeneous matrices."""
import numpy as np
from scipy.spatial.transform import Rotation

def ee_difference(current_world_from_root, current_world_from_ee, anchor_world_from_root, anchor_world_from_ee, *, position_scale_m, rotation_scale_rad):
    if position_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ValueError('Positive position and rotation scales are required')
    matrices=[np.asarray(x,dtype=np.float64) for x in (current_world_from_root,current_world_from_ee,anchor_world_from_root,anchor_world_from_ee)]
    if any(x.shape!=(4,4) or not np.isfinite(x).all() for x in matrices):raise ValueError('Expected finite 4x4 world-from-frame transforms')
    for x in matrices:
        if not np.allclose(x[3],[0,0,0,1],atol=1e-6) or not np.allclose(x[:3,:3].T@x[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(x[:3,:3]),1.,atol=1e-5):raise ValueError('Invalid SE(3) transform')
    c=np.linalg.inv(matrices[0])@matrices[1]; a=np.linalg.inv(matrices[2])@matrices[3]
    # Current-to-Anchor correction, expressed in corresponding arm-root axes.
    # Left multiplication: R_delta @ R_current_root == R_anchor_root.
    pos=a[:3,3]-c[:3,3];rot=Rotation.from_matrix(a[:3,:3]@c[:3,:3].T).as_rotvec()
    return np.concatenate([pos/position_scale_m,rot/rotation_scale_rad]).astype(np.float32)

def pool_match(flow_px, current_background, anchor_background, backward_px, confidence, *, confidence_threshold=.5, fb_threshold_px=1.5, min_valid_fraction=.1):
    import cv2
    f=np.asarray(flow_px,np.float32); b=None if backward_px is None else np.asarray(backward_px,np.float32);c=np.asarray(confidence,np.float32)
    h,w=f.shape[:2]
    if f.shape!=(h,w,2) or (b is not None and b.shape!=f.shape) or c.shape!=(h,w):raise ValueError('Mismatched RGB flow shape')
    mc=np.asarray(current_background,bool);ma=np.asarray(anchor_background,bool)
    if mc.shape!=(h,w) or ma.shape!=(h,w):raise ValueError('Mask dimensions differ from RGB')
    if h<4 or w<4 or not 0<=min_valid_fraction<=1:raise ValueError('Invalid pooling geometry')
    yy,xx=np.mgrid[:h,:w].astype(np.float32);x=xx+f[...,0];y=yy+f[...,1]
    inb=np.isfinite(f).all(axis=-1)&np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<w-1)&(y>=0)&(y<h-1)
    sx=np.where(inb,x,-1).astype(np.float32);sy=np.where(inb,y,-1).astype(np.float32)
    bg=cv2.remap(ma.astype(np.uint8),sx,sy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    valid=mc&bg&inb&np.isfinite(c)
    if confidence_threshold is not None:
        valid &= c>=confidence_threshold
    if fb_threshold_px is not None:
        if b is None:raise ValueError('Backward flow required for consistency filtering')
        bx=cv2.remap(b[...,0],sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
        by=cv2.remap(b[...,1],sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
        fb=np.hypot(f[...,0]+bx,f[...,1]+by)
        valid &= np.isfinite(fb)&(fb<=fb_threshold_px)
    out=np.zeros((4,4,2),np.float32);mask=np.zeros_like(out,dtype=bool);coverage=np.zeros((4,4),np.float32)
    for i in range(4):
        for j in range(4):
            y0=i*h//4;y1=((i+1)*h+3)//4;x0=j*w//4;x1=((j+1)*w+3)//4
            m=valid[y0:y1,x0:x1];coverage[i,j]=m.mean()
            if m.any() and coverage[i,j]>=min_valid_fraction:
                v=f[y0:y1,x0:x1][m].mean(axis=0);out[i,j]=[v[0]/w,v[1]/h];mask[i,j]=True
    return out.reshape(32),mask.reshape(32),coverage,valid
