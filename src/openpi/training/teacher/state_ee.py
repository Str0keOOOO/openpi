"""EE teachers from recorded LeRobot State; no MuJoCo or renderer dependency.

State: base world XYZ/XYZW [0:7], base-relative TCP XYZ [7:10],
base-relative hand-body XYZW [10:14], gripper [14:16]. A rigid hand-to-TCP
right rotation cancels in R_anchor @ R_current.T. Robot XML is read only to
check the reference frames, never to execute simulation or solve IK.
"""
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from openpi.training.teacher.provenance import file_digest

EE_VERSION = 'lerobot_state_ee_v1'
EE_FILE = 'ee_from_state.npz'
_xml_proofs = {}


def _local_pose(node):
    if any(k in node.attrib for k in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
        raise ValueError('Unsupported XML rotation on EE reference path: '+str(node.attrib))
    t = np.eye(4)
    t[:3, 3] = np.fromstring(node.get('pos', '0 0 0'), sep=' ')
    q = np.fromstring(node.get('quat', '1 0 0 0'), sep=' ')
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return t


def robot_frame_proof(path):
    raw = Path(path).read_bytes()
    key = hashlib.sha256(raw).hexdigest()
    if key in _xml_proofs:
        return _xml_proofs[key]
    text = gzip.decompress(raw)
    tree = ET.fromstring(text)
    parent = {child: p for p in tree.iter() for child in p}
    def find(tag, name):
        nodes = [n for n in tree.iter(tag) if n.get('name') == name]
        if len(nodes) != 1:
            raise ValueError(f'Unknown or ambiguous EE reference {tag} {name}')
        return nodes[0]
    def chain(node):
        out = [node]
        while node in parent:
            node = parent[node]
            out.append(node)
        return out[::-1]
    def relative(a, b):
        ca, cb = chain(a), chain(b)
        shared = 0
        while shared < min(len(ca), len(cb)) and ca[shared] is cb[shared]:
            shared += 1
        def tail(nodes):
            t = np.eye(4)
            for node in nodes[shared:]:
                if node.tag not in ('body', 'site'):
                    raise ValueError('Non-body/site in robot coordinate path')
                if node.tag == 'body' and any(c.tag in ('joint', 'freejoint') for c in node):
                    raise ValueError('State base-to-root/hand-to-TCP is not rigid; additional saved joint data required')
                t = t @ _local_pose(node)
            return t
        return np.linalg.inv(tail(ca)) @ tail(cb)
    base_root = relative(find('site', 'mobilebase0_center'), find('site', 'robot0_right_center'))
    hand_tcp = relative(find('body', 'robot0_right_hand'), find('site', 'gripper0_right_grip_site'))
    proof = dict(xml_sha256=key, xml_text_sha256=hashlib.sha256(text).hexdigest(), base_site='mobilebase0_center', root_site='robot0_right_center', State_orientation_body='robot0_right_hand', tcp_site='gripper0_right_grip_site', base_from_root=base_root.tolist(), hand_from_tcp=hand_tcp.tolist(), conversion='fixed root rotation; position offsets and rigid right TCP rotation cancel in the relative difference')
    _xml_proofs[key] = proof
    return proof


@lru_cache(maxsize=32)
def _state_table(path, size, mtime):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['observation.state', 'frame_index'])
    states = np.stack(table['observation.state'].to_pylist()).astype(np.float64)
    frames = np.asarray(table['frame_index'].to_pylist()).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 16 or not np.isfinite(states).all() or not np.array_equal(frames, np.arange(len(states))):
        raise ValueError('Invalid State dimensions, finite values or frame indexing: '+path)
    norms = np.linalg.norm(states[:, 10:14], axis=-1)
    if np.any(np.abs(norms-1.) > 1e-3):
        raise ValueError('Invalid recorded EE quaternion: '+path)
    return states


def state_path(dataset, task, episode):
    return Path(dataset)/task/'lerobot'/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'


def read_state_table(path):
    path = Path(path)
    st = path.stat()
    return _state_table(str(path.resolve()), st.st_size, st.st_mtime_ns)


def state_ee_difference(current, anchor, base_from_root, *, position_scale_m=.1, rotation_scale_rad=1.):
    if position_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ValueError('Positive EE normalization scales required')
    c = np.asarray(current, np.float64)
    a = np.asarray(anchor, np.float64)
    if c.shape[-1] != 16 or a.shape != (16,) or not np.isfinite(c).all() or not np.isfinite(a).all():
        raise ValueError('Expected finite recorded State16 and one Anchor State16')
    r_root_base = np.asarray(base_from_root, np.float64)[:3, :3].T
    pos = (a[7:10]-c[..., 7:10]) @ r_root_base.T
    rc = Rotation.from_quat(c[..., 10:14]).as_matrix()
    ra = Rotation.from_quat(a[10:14]).as_matrix()
    relative = r_root_base @ (ra @ np.swapaxes(rc, -1, -2)) @ r_root_base.T
    rot = Rotation.from_matrix(relative).as_rotvec()
    return np.concatenate([pos/position_scale_m, rot/rotation_scale_rad], axis=-1).astype(np.float32)


def episode_teacher(dataset, task, episode, anchor, *, position_scale_m=.1, rotation_scale_rad=1.):
    dataset = Path(dataset)
    current_path = state_path(dataset, task, episode)
    anchor_path = state_path(dataset, task, anchor)
    ds = dataset/task/'lerobot'
    current_xml = ds/'extras'/f'episode_{episode:06d}'/'model.xml.gz'
    anchor_xml = ds/'extras'/f'episode_{anchor:06d}'/'model.xml.gz'
    proof = robot_frame_proof(current_xml)
    if proof['xml_text_sha256'] != robot_frame_proof(anchor_xml)['xml_text_sha256']:
        raise ValueError('Current and Anchor robot/scene XML differ')
    states = read_state_table(current_path)
    a = read_state_table(anchor_path)[0]
    values = state_ee_difference(states, a, proof['base_from_root'], position_scale_m=position_scale_m, rotation_scale_rad=rotation_scale_rad)
    sources = [current_path, anchor_path, current_xml, anchor_xml, dataset/'human_anchor_pairs.csv', ds/'extras'/f'episode_{episode:06d}'/'provenance.json', ds/'extras'/f'episode_{anchor:06d}'/'provenance.json']
    meta = dict(cache_version=EE_VERSION, task=task, episode=episode, anchor_episode=anchor, reference_frame=0, source='LeRobot observation.state', state_position_slice=[7, 10], state_quaternion_xyzw_slice=[10, 14], ee_delta='anchor minus current; rotation log(R_anchor_root @ R_current_root.T)', position_scale_m=position_scale_m, rotation_scale_rad=rotation_scale_rad, frames=len(values), robot_frame_proof=proof, source_sha256={str(p.resolve()): file_digest(p) for p in sources})
    return values, np.ones(values.shape, bool), meta


@lru_cache(maxsize=32)
def _load_teacher(path, size, mtime):
    with np.load(path, allow_pickle=False) as z:
        values, valid = z['ee_latent'].copy(), z['ee_valid'].copy()
        indices, meta = z['frame_index'].copy(), json.loads(str(z['metadata']))
    if meta.get('cache_version') != EE_VERSION or values.shape != (meta['frames'], 6) or valid.shape != values.shape or not np.array_equal(indices, np.arange(len(values))) or not np.isfinite(values[valid]).all() or not valid.all():
        raise ValueError('Invalid State EE teacher: '+path)
    for source, expected in meta['source_sha256'].items():
        if file_digest(source) != expected:
            raise ValueError('State EE source changed: '+source)
    return values, valid, meta


def load_teacher(folder):
    path = Path(folder)/EE_FILE
    st = path.stat()
    return _load_teacher(str(path.resolve()), st.st_size, st.st_mtime_ns)
