"""CPU-only exact first-state contact screening; full rollout checks still apply."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import gzip
import math
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from state import model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa_anchor'))
from anchor_pose import set_base_pose


@lru_cache(maxsize=3)
def load_scene(source, task, human, anchor):
    root = Path(source) / task / 'lerobot/extras'
    h = root / f'episode_{human:06d}'
    a = root / f'episode_{anchor:06d}'
    xml = gzip.open(h / 'model.xml.gz', 'rt').read()
    if xml != gzip.open(a / 'model.xml.gz', 'rt').read():
        raise ValueError('Human/Anchor XML mismatch')
    m = model(xml)
    robot = ET.fromstring(xml).find(".//body[@name='robot0_base']")
    if robot is None:
        raise ValueError('Robot root missing from scene XML')
    # Same robot subtree as env.robots[0].robot_model.root. The full simulator
    # independently applies detect_robot_collision before accepting any sample.
    geoms = {m.geom(g.attrib['name']).id for g in robot.iter('geom')}
    joints = [m.joint(j.attrib['name']).id for j in robot.iter('joint')]
    if any(m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL) for j in joints):
        raise ValueError('Unexpected non-scalar robot joint')
    with np.load(h / 'states.npz') as z:
        hs = z['states'][0].copy()
    with np.load(a / 'states.npz') as z:
        anchor_state = z['states'][0].copy()
    if len(hs) != 1 + m.nq + m.nv or len(anchor_state) != len(hs):
        raise ValueError('Unexpected simulation state shape')
    return m, hs, anchor_state, geoms, joints


def screen_candidates(source, ref, vectors):
    """Evaluate candidate first states; never step physics, render or use a GPU."""
    m, hs, anchor_state, robot_geoms, robot_joints = load_scene(
        source, ref['task'], ref['human'], ref['anchor'])
    results = []
    for v in vectors:
        d = mujoco.MjData(m)
        d.time = hs[0]
        d.qpos[:] = hs[1:1 + m.nq]
        d.qvel[:] = hs[1 + m.nq:]
        for jid in robot_joints:
            d.qpos[m.jnt_qposadr[jid]] = anchor_state[1 + m.jnt_qposadr[jid]]
            d.qvel[m.jnt_dofadr[jid]] = 0
        valid = True
        for name, change in zip(
                ['mobilebase0_joint_torso_height'] + [f'robot0_joint{i}' for i in range(1, 8)],
                [v[3], *np.radians(v[4:])]):
            jid = m.joint(name).id
            d.qpos[m.jnt_qposadr[jid]] += change
            if m.jnt_limited[jid] and not m.jnt_range[jid, 0] <= d.qpos[m.jnt_qposadr[jid]] <= m.jnt_range[jid, 1]:
                valid = False
        if not valid:
            results.append(dict(valid=False, reason='joint_limit'))
            continue
        adapter = SimpleNamespace(
            model=SimpleNamespace(body_name2id=lambda n: m.body(n).id,
                                  get_joint_qpos_addr=lambda n: m.jnt_qposadr[m.joint(n).id]),
            data=SimpleNamespace(qpos=d.qpos, body_xpos=d.xpos, body_xquat=d.xquat),
            forward=lambda: mujoco.mj_forward(m, d))
        p = ref['anchor_pose']
        c, s = math.cos(p[3]), math.sin(p[3])
        xy = [p[0] + c * v[0] - s * v[1], p[1] + s * v[0] + c * v[1]]
        set_base_pose(SimpleNamespace(sim=adapter), xy, p[3] + math.radians(v[2]), None)
        contacts = []
        for contact in d.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in robot_geoms) != (g2 in robot_geoms):
                contacts.append([m.geom(g1).name, m.geom(g2).name, float(contact.dist)])
        results.append(dict(valid=not contacts, reason='collision' if contacts else 'clear',
                            contacts=contacts[:8], robot_geom_count=len(robot_geoms)))
    return results
