"""Shared fixed-start benchmark. Planning imports neither simulation nor policies."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'robocasa_small'))
# When run as a script, import the old module under an explicit unique name.
import importlib.util
spec=importlib.util.spec_from_file_location('robocasa_legacy_benchmark',HERE.parent/'robocasa_small/main.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def slots(config):
    for task_idx,task in enumerate(config['tasks']):
        for i in range(config['baseline_per_task']):
            yield dict(task=task,id=f'baseline_{i:02d}',family='baseline',axis=None,level=None,repeat=i,seed=config['seed']+i)
        for axis,name in enumerate(config['names']):
            for level_idx,level in enumerate(['small','medium','large']):
                for i in range(config['single_per_dimension_level']):
                    positive=(i+level_idx+axis+task_idx)%2==0
                    yield dict(task=task,id=f'single_{name}_{level}_{i}',family='single',axis=name,level=level,
                               axis_index=axis,sign=1 if axis==3 or positive else -1,repeat=i,seed=config['seed']+i)
        for group_idx,(lo,hi) in enumerate(config['multi_dimension_groups']):
            for level in config['multi_amplitudes']:
                for i in range(config['multi_per_group_level']):
                    yield dict(task=task,id=f'multi_{lo}_{hi}_{level}_{i}',family='multi',axis=f'{lo}-{hi}',level=level,
                               dimension_range=[lo,hi],repeat=i,seed=config['seed']+i)


def propose(config,case,attempt):
    seed=int(legacy.digest([config['seed'],case['task'],case['id'],attempt])[:16],16)
    rng=np.random.default_rng(seed);scale=np.asarray(config['training_limits']);tol=np.asarray(config['tolerances']);v=np.zeros(11)
    if case['family']=='single':
        i=case['axis_index'];lo,hi=config['level_multipliers'][['small','medium','large'].index(case['level'])]
        low=max(lo*scale[i],np.nextafter(tol[i],np.inf));v[i]=case['sign']*rng.uniform(low,hi*scale[i])
    elif case['family']=='multi':
        lo,hi=case['dimension_range'];n=int(rng.integers(lo,hi+1));ids=rng.choice(11,n,replace=False)
        _,upper=config['multi_amplitudes'][case['level']]
        for i in ids:v[i]=(1 if i==3 else rng.choice([-1,1]))*rng.uniform(np.nextafter(tol[i],np.inf),upper*scale[i])
        if case['level']=='strong':
            i=int(rng.choice(ids));v[i]=(1 if i==3 or v[i]>0 else -1)*rng.uniform(np.nextafter(scale[i],np.inf),1.5*scale[i])
    return v


def signature(config):
    return legacy.digest({k:v for k,v in config.items() if k not in ['models','inference']})


def in_stratum(config,case,values):
    values=np.asarray(values);active=np.flatnonzero(np.abs(values)>np.asarray(config['tolerances']))
    if not np.isfinite(values).all() or values[3]<0:return False
    if case['family']=='baseline':return len(active)==0
    normalized=np.abs(values)/np.asarray(config['training_limits'])
    if case['family']=='single':
        i=case['axis_index'];lo,hi=config['level_multipliers'][['small','medium','large'].index(case['level'])]
        return len(active)==1 and active[0]==i and np.sign(values[i])==case['sign'] and lo<normalized[i]<=hi
    lo,hi=case['dimension_range'];lower,upper=config['multi_amplitudes'][case['level']]
    return lo<=len(active)<=hi and lower<normalized.max()<=upper


def load_anchors(config):
    if not config.get('anchors_manifest'):raise RuntimeError('No held-out actual-Anchor manifest selected. Refusing to use training Anchors as test scenes.')
    records=json.loads(Path(config['anchors_manifest']).read_text())
    train=set()
    for p in Path(config['training_dataset']).glob('*/lerobot/extras/episode_*/ep_meta.json'):
        d=json.loads(p.read_text());train.add((int(d['layout_id']),int(d['style_id'])))
    for row in records:
        root=Path(row['dataset'])/row['task']/'lerobot/extras'/f"episode_{row['anchor_episode']:06d}"
        meta=json.loads((root/'ep_meta.json').read_text())
        if (int(meta['layout_id']),int(meta['style_id'])) in train:raise ValueError('Training/test layout-style overlap: '+str(root))
        if not (root/'states.npz').exists() or not (root/'model.xml.gz').exists():raise FileNotFoundError(root)
        row['extras']=str(root);row['ep_meta']=meta;row['xml_sha256']=hashlib.sha256((root/'model.xml.gz').read_bytes()).hexdigest()
    for task in config['tasks']:
        if not any(r['task']==task for r in records):raise ValueError('No held-out Anchor for '+task)
    return records


def restore(wrapper,row,flat):
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    root=Path(row['extras']);env=wrapper.unwrapped.env
    if hashlib.sha256((root/'model.xml.gz').read_bytes()).hexdigest()!=row['xml_sha256']:raise RuntimeError('Reference Anchor model changed after test-set selection')
    reset_to(env,dict(model=gzip.open(root/'model.xml.gz','rt').read(),ep_meta=json.dumps(row['ep_meta']),states=flat))
    before=env.sim.data.qpos.copy()
    env.cur_time=0;env.timestep=0;env.done=False;env._obs_cache={};env._reset_observables()
    for robot in env.robots:
        robot.base_pos=env.sim.data.get_body_xpos(robot.robot_model.root_body).copy()
        robot.base_ori=env.sim.data.get_body_xmat(robot.robot_model.root_body).copy()
        cc=robot.composite_controller;cc.update_state()
        for ctrl in cc.part_controllers.values():ctrl.update(force=True)
        cc.reset()
    if not np.array_equal(before,env.sim.data.qpos):raise RuntimeError('Controller initialization altered physical state')


def apply(wrapper,requested):
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(HERE.parent/'robocasa_anchor'))
    from anchor_pose import set_base_pose
    env=wrapper.unwrapped.env;m=env.sim.model;data=env.sim.data
    bid=m.body_name2id('mobilebase0_base');old=data.qpos.copy();base=data.body_xpos[bid].copy()
    yaw=Rotation.from_quat(np.asarray(data.body_xquat[bid])[[1,2,3,0]]).as_euler('xyz')[2]
    basis=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    names=['mobilebase0_joint_torso_height']+[f'robot0_joint{i}' for i in range(1,8)]
    addr=[m.get_joint_qpos_addr(n) for n in names]
    if np.any(requested[:3]):set_base_pose(env,base[:2]+basis@requested[:2],yaw+math.radians(requested[2]),None)
    data.qpos[addr]+=np.r_[requested[3],np.radians(requested[4:])];env.sim.forward()
    limits=[]
    for name,address in zip(names,addr):
        jid=m.joint_name2id(name);q=float(data.qpos[address])
        if m.jnt_limited[jid] and not m.jnt_range[jid,0]<=q<=m.jnt_range[jid,1]:limits.append(name)
    actual_yaw=Rotation.from_quat(np.asarray(data.body_xquat[bid])[[1,2,3,0]]).as_euler('xyz')[2]
    actual=np.r_[basis.T@(data.body_xpos[bid][:2]-base[:2]),math.degrees((actual_yaw-yaw+math.pi)%(2*math.pi)-math.pi),data.qpos[addr[0]]-old[addr[0]],np.degrees(data.qpos[addr[1:]]-old[addr[1:]])]
    collision=legacy.official_collision(env)
    return actual,dict(status='invalid_joint_limit' if limits else 'invalid_official_collision' if collision else 'ready',joint_limit_violations=limits,official_collision=collision,contacts=legacy.contacts(env))


def sample(config,cases,output):
    anchors=load_anchors(config)
    for case in cases:
        dest=output/'starts'/case['task']/case['id'];marker=dest/'configuration.json'
        if marker.exists():continue
        options=[r for r in anchors if r['task']==case['task']]
        rotation=int(legacy.digest([case['task'],case['id']])[:8],16)
        row=options[rotation%len(options)]
        scene=dict(layout_id=row['ep_meta']['layout_id'],style_id=row['ep_meta']['style_id'],seed=row.get('scene_seed',0),object_split='pretrain')
        wrapper=legacy.make_env(case['task'],scene);dest.mkdir(parents=True,exist_ok=True)
        try:
            with np.load(Path(row['extras'])/'states.npz') as z:reference=z['states'][0].copy()
            accepted=False
            for attempt in range(1 if case['family']=='baseline' else config['max_sampling_attempts_per_slot']):
                restore(wrapper,row,reference);requested=propose(config,case,attempt);actual,outcome=apply(wrapper,requested)
                if not np.allclose(actual,requested,atol=1e-7,rtol=0):outcome['status']='invalid_actual_delta'
                elif not in_stratum(config,case,actual):outcome['status']='invalid_actual_stratum'
                event=dict(attempt=attempt,requested=requested.tolist(),actual=actual.tolist(),**outcome)
                with (dest/'attempts.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                if outcome['status']!='ready':continue
                env=wrapper.unwrapped.env;np.savez_compressed(dest/'state.npz',state=env.sim.get_state().flatten(),qpos=env.sim.data.qpos.copy(),qvel=env.sim.data.qvel.copy())
                legacy.save_images(dest,legacy.observation(wrapper));accepted=True;break
            legacy.dump(marker,dict(case=case,accepted=accepted,anchor=row,actual_delta=actual.tolist(),test_signature=signature(config),outcome=outcome))
        finally:wrapper.close()
    rows=[]
    for path in (output/'starts').glob('*/*/configuration.json'):
        d=json.loads(path.read_text());rows.append(dict(**d['case'],accepted=d['accepted'],sampling_status=d['outcome']['status'],actual_delta=d['actual_delta']))
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(output/'fixed_starts.csv',index=False)
        legacy.dump(output/'sampling_summary.json',dict(planned=len(list(slots(config))),processed=len(rows),valid=sum(r['accepted'] for r in rows),invalid=sum(not r['accepted'] for r in rows),reasons=dict(collections.Counter(r['sampling_status'] for r in rows))))


def evaluate(config,cases,output,model,port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    import imageio.v2 as imageio
    model_info=config['models'][model]
    if not model_info['checkpoint']:raise RuntimeError('Choose a frozen checkpoint for '+model)
    if model in ['M2','M3']:
        paths=[config['models'][label].get('training_manifest') for label in ['M2','M3']]
        if not all(paths):raise RuntimeError('Provide actual M2 and M3 training-data manifests for identity verification')
        if legacy.digest(json.loads(Path(paths[0]).read_text()))!=legacy.digest(json.loads(Path(paths[1]).read_text())):raise RuntimeError('M2/M3 training datasets are different')
    client=WebsocketClientPolicy('127.0.0.1',port);meta=client.get_server_metadata()
    if not meta.get('explicit_inference_seed'):raise RuntimeError('Use this benchmark seeded_server.py for per-chunk reproducible inference')
    if str(Path(meta['checkpoint']).resolve())!=str(Path(model_info['checkpoint']).resolve()):raise RuntimeError('Service checkpoint mismatch')
    if meta.get('policy_config')!=model_info['policy_config']:raise RuntimeError('Service model architecture/config mismatch')
    for case in cases:
        source=output/'starts'/case['task']/case['id'];c=json.loads((source/'configuration.json').read_text())
        dest=output/'results'/model/case['task']/case['id'];f=dest/'result.json'
        if f.exists() and json.loads(f.read_text()).get('status')=='completed':continue
        if not c['accepted']:continue
        if c['test_signature']!=signature(config):raise RuntimeError('Fixed-test configuration changed')
        row=c['anchor'];scene=dict(layout_id=row['ep_meta']['layout_id'],style_id=row['ep_meta']['style_id'],seed=row.get('scene_seed',0),object_split='pretrain')
        wrapper=legacy.make_env(case['task'],scene);dest.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
        try:
            with np.load(source/'state.npz') as z:flat=z['state'].copy();q=z['qpos'].copy()
            restore(wrapper,row,flat);env=wrapper.unwrapped.env
            if not np.array_equal(env.sim.data.qpos,q):raise RuntimeError('Fixed initial state mismatch')
            np.random.seed(case['seed']);env.rng.bit_generator.state=np.random.default_rng(case['seed']).bit_generator.state
            obs=legacy.observation(wrapper);plan=collections.deque();actions=[];success=False;chunk_id=0
            with imageio.get_writer(dest/'review.mp4',fps=20,codec='libx264',pixelformat='yuv420p') as writer:
                for step in range(get_task_horizon(case['task'])):
                    if not plan:
                        inputs=legacy.policy_input(obs,config['inference']['resize_size']);inputs['__inference_seed']=int(legacy.digest([case['seed'],case['id'],chunk_id])[:8],16)
                        chunk=np.asarray(client.infer(inputs)['actions']);chunk_id+=1
                        if chunk.ndim!=2 or chunk.shape[1]!=12 or not np.isfinite(chunk).all():raise ValueError('Invalid action chunk')
                        np.save(dest/f'chunk_{chunk_id:05d}.npy',chunk);plan.extend(chunk[:config['inference']['replan_steps']])
                    action=plan.popleft();actions.append(action.copy());obs,_,_,_,info=wrapper.step(convert_action(action));success=bool(info['success'])
                    if step%2==0 or success:writer.append_data(wrapper.render())
                    if success:break
            np.savez_compressed(dest/'actions.npz',actions=actions)
            legacy.dump(f,dict(**case,status='completed',success=success,steps=len(actions),termination='success' if success else 'timeout',total_seconds=time.perf_counter()-started,actual_delta=c['actual_delta'],checkpoint=model_info['checkpoint']))
        except Exception as exc:legacy.dump(f,dict(**case,status='error',error=repr(exc)))
        finally:wrapper.close()


def report(config,output):
    import pandas as pd
    for model in config['models']:
        rows=[json.loads(p.read_text()) for p in (output/'results'/model).glob('*/*/result.json')]
        if not rows:continue
        d=pd.DataFrame(rows)
        if 'success' not in d:d['success']=False
        d.to_csv(output/f'{model}_results.csv',index=False)
        completed=d[d.status=='completed'];groups=completed.groupby(['task','family','axis','level'],dropna=False).success.agg(['sum','count','mean']);groups.to_csv(output/f'{model}_breakdown.csv')
        summary={'overall':dict(completed=len(completed),successes=int(completed.success.sum()),success_rate=float(completed.success.mean()) if len(completed) else None),'planned':len(list(slots(config))),'errors':int((d.status=='error').sum())}
        legacy.dump(output/f'{model}_summary.json',summary)
        completed.groupby('task').success.agg(['sum','count','mean']).to_csv(output/f'{model}_tasks.csv')
        completed.groupby('family').success.agg(['sum','count','mean']).to_csv(output/f'{model}_families.csv')
        completed.groupby(['family','axis','level'],dropna=False).success.agg(['sum','count','mean']).to_csv(output/f'{model}_axes.csv')
        lines=[f'# {model}',f'Completed {len(completed)}/{len(list(slots(config)))}; errors {int((d.status=="error").sum())}', '| Task / perturbation / axis / level | Success |','|---|---:|']
        for key,row in groups.iterrows():lines.append(f'| {key} | {int(row["sum"])}/{int(row["count"])} ({100*row["mean"]:.1f}%) |')
        (output/f'{model}_summary.md').write_text('\n'.join(lines))


def main():
    # Descriptive IDs support independent Anchor/Augmented baselines. The legacy
    # M2/M3 same-training-data check remains specific to its original ablation pair.
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=HERE/'benchmark.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['plan','sample','evaluate','report'],default='plan');p.add_argument('--model',choices=['M1','M2','M3','ANCHOR','AUGMENTED']);p.add_argument('--port',type=int);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--gpus',type=int,nargs='+');p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
    if c.get('protocol')=='grouped_same_level_v2':
        from grouped import plan as grouped_plan,digest as grouped_digest,dump as grouped_dump
        if a.mode=='plan':
            grouped_dump(a.output/'plan.json',grouped_plan(c));grouped_dump(a.output/'plan_summary.json',dict(per_model=1460,scenes_per_task=5,baseline=20,body=480,arm=480,joint=480,actual_valid_starts='not verified until sample'));return
        if a.mode=='sample':
            if not a.gpus:p.error('Grouped sampling requires --gpus')
            import subprocess
            subprocess.run([sys.executable,str(HERE/'build_grouped.py'),'--config',str(a.config),'--output',str(a.output),'--gpus',*map(str,a.gpus)],check=True);return
        import grouped_evaluation
        if a.mode=='report':grouped_evaluation.report(c,a.output);return
        if not a.model or not a.port:p.error('evaluate requires --model and --port')
        grouped_evaluation.evaluate(c,a.output,a.model,a.port,gpu=a.gpu,shard_index=a.shard_index,shard_count=a.shard_count);grouped_evaluation.report(c,a.output);return
    cases=list(slots(c))
    marker=a.output/'test_signature.txt';sig=signature(c)
    if marker.exists() and marker.read_text()!=sig:raise RuntimeError('Changed test set; choose a new output directory')
    marker.write_text(sig);legacy.dump(a.output/'plan.json',cases)
    legacy.dump(a.output/'config_used.json',c)
    counts=dict(collections.Counter(x['family'] for x in cases));legacy.dump(a.output/'plan_summary.json',dict(per_model=len(cases),three_models=3*len(cases),families=counts,actual_valid_starts='not_verified_until_sample',ood_included=False))
    selected=[x for i,x in enumerate(cases) if i%a.shard_count==a.shard_index]
    if a.mode=='plan':print(counts,'per model',len(cases),'No simulation or inference started.');return
    if a.mode=='sample':sample(c,selected,a.output)
    elif a.mode=='report':report(c,a.output)
    else:
        if not a.model or not a.port:p.error('evaluate requires --model and --port')
        evaluate(c,selected,a.output,a.model,a.port);report(c,a.output)


if __name__=='__main__':main()
