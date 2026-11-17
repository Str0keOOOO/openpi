"""First-frame measurements. All angles in degrees, distances in metres."""
import gzip
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
import robocasa
import robosuite

NAMES = ['X', 'Y', 'Yaw', 'Height'] + [f'J{i}' for i in range(1, 8)]
SCALE = np.array([.1, .1, 10., .1] + [10.] * 7)
TOL = np.array([.001, .001, .1, .001] + [.1] * 7)


def model(xml):
    root = ET.fromstring(xml)
    for el in root.find('asset'):
        f = el.get('file')
        if not f: continue
        parts = f.split('/')
        for name, base in [('robosuite', Path(robosuite.__file__).parent), ('robocasa', Path(robocasa.__file__).parent)]:
            if name in parts:
                i = len(parts) - 1 - parts[::-1].index(name)
                el.set('file', str(base.joinpath(*parts[i+1:])))
                break
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def pose(m, state):
    d = mujoco.MjData(m)
    if len(state) != 1 + m.nq + m.nv: raise ValueError('Unexpected flat MuJoCo state length')
    d.time = state[0]; d.qpos[:] = state[1:1+m.nq]; d.qvel[:] = state[1+m.nq:]
    mujoco.mj_forward(m, d)
    bid = m.body('mobilebase0_base').id
    w,x,y,z = d.xquat[bid]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    names = ['mobilebase0_joint_torso_height'] + [f'robot0_joint{i}' for i in range(1,8)]
    ids = [m.joint(n).id for n in names]
    q = d.qpos[m.jnt_qposadr[ids]].copy()
    return np.r_[d.xpos[bid], yaw, q], ids


def delta(p, a):
    xy = p[:2]-a[:2]; c,s = math.cos(a[3]),math.sin(a[3])
    return np.r_[c*xy[0]+s*xy[1], -s*xy[0]+c*xy[1],
                 math.degrees((p[3]-a[3]+math.pi)%(2*math.pi)-math.pi),
                 p[4]-a[4], np.degrees(p[5:]-a[5:])]


def measure_pair(job):
    source, task, human, anchor = job
    ds = Path(source)/task/'lerobot'
    hx = ds/'extras'/f'episode_{human:06d}'; ax = ds/'extras'/f'episode_{anchor:06d}'
    xml = gzip.open(hx/'model.xml.gz', 'rt').read()
    if xml != gzip.open(ax/'model.xml.gz','rt').read(): raise ValueError(f'Scene XML mismatch: {task}/{human}')
    m = model(xml)
    with np.load(hx/'states.npz') as z: hp,_ = pose(m,z['states'][0])
    with np.load(ax/'states.npz') as z: ap,ids = pose(m,z['states'][0])
    for idx,p in [(human,hp),(anchor,ap)]:
        f = next((ds/'data').glob(f'*/episode_{idx:06d}.parquet'))
        obs = np.asarray(pd.read_parquet(f,columns=['observation.state']).iloc[0,0])
        if obs.shape != (16,) or np.linalg.norm(obs[:2]-p[:2])>.001: raise ValueError('LeRobot/MuJoCo first-frame mismatch')
    lo,hi = -SCALE.copy(),SCALE.copy();lo[3]=0
    for i,j in enumerate(ids):
        if not m.jnt_limited[j]: continue
        q=ap[4+i];low,high=m.jnt_range[j]
        if q<low-1e-5 or q>high+1e-5: raise ValueError('Anchor starts outside joint limit')
        scale=1. if i==0 else 180/math.pi
        k=3+i;lo[k]=max(lo[k],(low-q)*scale);hi[k]=min(hi[k],(high-q)*scale)
    return dict(task=task,human=human,anchor=anchor,anchor_pose=ap.tolist(),
                human_delta=delta(hp,ap).tolist(),lo=lo.tolist(),hi=hi.tolist(),scene_sha256=hashlib.sha256(xml.encode()).hexdigest())
