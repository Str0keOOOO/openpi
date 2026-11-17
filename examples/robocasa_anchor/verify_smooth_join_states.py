"""Offline check of the actual connector and operation, without a rendering context."""
import gzip,json
from pathlib import Path
import mujoco,numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen
from soft_joint_limits import check_soft_joint_limits

def verify(folder):
    r=json.loads((folder/'summary.json').read_text());rank=next(x['rank'] for x in r['attempts'] if x['status']=='accepted');trial=folder/f'candidate_{rank}'
    extra=Path('/data4/xuchenfei/robocasa365/data/household_augmented')/r['task']/f"lerobot/extras/episode_{r['episode']:06d}"
    # Use the official asset path remapper. Camera overrides are irrelevant to
    # joint/contact checks and are deliberately left out of this offline model.
    remapper=Kitchen.__new__(Kitchen);remapper._cam_configs={};remapper.generative_textures=None
    model=mujoco.MjModel.from_xml_string(remapper.edit_model_xml(gzip.open(extra/'model.xml.gz','rt').read()));data=mujoco.MjData(model)
    states=[]
    for name in ['transition_actual.npz','operation_actual.npz']:
        with np.load(trial/name) as z:
            states.extend(z['sim_state_before_action']);states.append(z['final_sim_state'])
    states=np.asarray(states);joints=[model.joint(i).name for i in range(model.njnt) if (model.joint(i).name or '').startswith(('robot0_','mobilebase0_'))]
    limits=check_soft_joint_limits(model,joints,states)
    robots={i for i in range(model.ngeom) if (model.body(int(model.geom_bodyid[i])).name or '').startswith(('robot0_','mobilebase0_','gripper0_'))}
    self_contacts=[]
    for frame,s in enumerate(states):
        data.time=s[0];data.qpos[:]=s[1:1+model.nq];data.qvel[:]=s[1+model.nq:];mujoco.mj_forward(model,data)
        for c in data.contact:
            g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in robots or g2 not in robots or float(c.dist)>=-.005:continue
            b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
            if b1==b2 or model.body_parentid[b1]==b2 or model.body_parentid[b2]==b1 or model.body_parentid[b1]==model.body_parentid[b2]:continue
            self_contacts.append(dict(frame=frame,distance_m=float(c.dist),bodies=[model.body(b1).name,model.body(b2).name]))
    result=dict(states_checked=len(states),joint_limits=limits,nonadjacent_self_penetrations=self_contacts,passed=limits['accepted'] and not self_contacts)
    (folder/'actual_states_validation.json').write_text(json.dumps(result,indent=2));print(folder.name,result['passed'],len(states),flush=True)
    return result

if __name__=='__main__':
    for folder in sorted(Path('/data5/xuchenfei/handoff_comparison/revised').glob('*_ep*')):
        if (folder/'summary.json').exists():verify(folder)
