"""Build and freeze the 1460-case grouped start set; never load a policy."""
import argparse,fcntl,concurrent.futures,csv,hashlib,json,os,subprocess,sys,time,collections,shutil
from pathlib import Path
from grouped import *
HERE=Path(__file__).resolve().parent

def summarize(config,output,*,final=False):
 cases=plan(config);rows=[];counts=collections.Counter();scene_rows=[]
 for task in config['tasks']:
  for i in range(config['scenes_per_task']):
   p=output/'scenes'/task/f'scene_{i:02d}'/'summary.json'
   if p.exists():scene_rows.append(json.loads(p.read_text()))
 for case in cases:
  p=output/'scenes'/case['task']/f"scene_{case['scene_index']:02d}"/'starts'/case['id']/'configuration.json'
  if p.exists():
   r=json.loads(p.read_text());rows.append(r);counts['valid' if r['accepted'] else 'invalid']+=1
  elif final:rows.append(dict(case=case,accepted=False,reason='scene_build_error_or_missing_configuration'));counts['missing']+=1
 summary=dict(planned=len(cases),processed=counts['valid']+counts['invalid'],valid=counts['valid'],invalid=counts['invalid'],missing=len(cases)-counts['valid']-counts['invalid'],scenes_completed=sum(r.get('status')=='completed' for r in scene_rows),scene_errors=sum(r.get('status')=='error' for r in scene_rows),total_scenes=20,policy_actions_executed=0,status='complete' if final and counts['valid']==len(cases) else 'complete_with_infeasible_slots' if final else 'generating',by_task={task:dict(valid=sum(r['accepted'] for r in rows if r['case']['task']==task),invalid=sum(not r['accepted'] for r in rows if r['case']['task']==task)) for task in config['tasks']})
 dump(output/'sampling_summary.json',summary)
 if final:
  fields=['task','scene_index','id','family','level','sample_index','accepted','actual_delta','state_file','anchor','reason','attempts']
  with (output/'fixed_starts.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for r in rows:w.writerow({k:json.dumps(r[k]) if isinstance(r.get(k),list) else r.get(k,r['case'].get(k)) for k in fields})
  dump(output/'invalid_starts.json',[r for r in rows if not r['accepted']]);dump(output/'scene_summary.json',scene_rows)
  manifest=[];balance=[]
  import imageio.v2 as imageio
  for r in rows:
   c=r['case'];entry=dict(case=c,accepted=r['accepted'],reason=r.get('reason'))
   if r['accepted']:
    file=Path(r['state_file'])
    with np.load(file) as z:
     if not np.isfinite(z['state']).all():raise RuntimeError('Non-finite frozen state')
    for camera,path in r['images'].items():
     img=imageio.imread(path)
     if img.shape!=(256,256,3) or img.dtype!=np.uint8:raise RuntimeError('Invalid real RGB')
    entry.update(configuration_file=str(output/'scenes'/c['task']/f"scene_{c['scene_index']:02d}"/'starts'/c['id']/'configuration.json'),state_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),actual_delta=r['actual_delta'])
    for axis in c['active_dimensions']:balance.append(dict(task=c['task'],family=c['family'],level=c['level'],axis=config['names'][axis],bin=c['magnitude_subbins'][axis],sign=c['signs'][axis],value=r['actual_delta'][axis]))
   manifest.append(entry)
  dump(output/'frozen_manifest.json',dict(version=2,protocol='grouped_same_level_v2',config_sha256=digest({k:v for k,v in config.items() if k!='models'}),planned=len(cases),valid=counts['valid'],cases=manifest,policy_actions_executed=0))
  with (output/'actual_dimension_samples.csv').open('w') as f:
   columns=['task','family','level','axis','bin','sign','value'];w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(balance)
  try:
   from PIL import Image,ImageDraw
   canvas=Image.new('RGB',(5*256,4*284),'white');draw=ImageDraw.Draw(canvas)
   for t,task in enumerate(config['tasks']):
    for i in range(5):
     anchor=output/'scenes'/task/f'scene_{i:02d}'/'anchor';image=anchor/'robot0_agentview_left.png';x=i*256;y=t*284;draw.text((x+4,y+2),f'{task} scene{i}',fill='black')
     if image.exists():canvas.paste(Image.open(image).convert('RGB'),(x,y+28))
   canvas.save(output/'all_20_anchor_views.png')
  except Exception as ex:dump(output/'contact_sheet_error.json',dict(error=str(ex)))
  lines=['固定测试集：4任务×5场景×(1原点+3类×3档×8配置)=1460个计划起点。','这里只生成物理起点，没有加载π0.5或运行任务成功率评测。','body同时扰动X/Y/Yaw/Height；arm同时扰动J1-J7；joint同时扰动11维。','三档分别为增强范围的0-0.5、0.5-1、1-1.5倍，即0-5、5-10、10-15cm或度。','XY位于Anchor底盘朝向的局部前后/左右坐标，Yaw绕世界竖直轴；Height是躯干升降关节量；关节角相对Anchor第一帧。','每个场景每类每档8个请求中，各双向维度4正4负；每任务每类每档40个请求，每维每个幅度小区间8个。','同一样本全部参与维度在同一档；独立打散各维符号和幅度区间，未变维度为零。','相同槽位碰撞或越界时，在同一方向、同一档和同一幅度子区间内重新采样，最多50次；不能生成的保留原槽位和失败原因，不缩小扰动或用任务成功筛选。','训练场景隔离采用418条Human中不存在的layout/style组合，物品仍采用官方pretrain split。布局、风格单独可能见过，这是组合隔离，不是整个布局或风格都未见。','Anchor通过官方任务reset获得场景、目标和摆放，机器人采用对应官方Anchor+默认关节姿态；原点因初始碰撞不合法时只重新生成场景种子，不执行任务、不据成功率选场景。','每个Anchor都保存实际第一帧、XML、完整MuJoCo state、ctrl、运行时模型/相机参数、任务metadata及三路RGB；每个配置另外保存其实际state和三路首帧。','未来测评应使用grouped.restore恢复，先载XML/任务metadata，再恢复运行时模型及state，重置控制器且不改变物理qpos，不额外仿真稳定。','本目录frozen_manifest.json列出固定起点及hash，模型评测只能读取，不重新采样。','',json.dumps(summary,ensure_ascii=False)]
  if config.get('replacement_sampling')=='whole_tier_legal_points':
   lines[5]='原始请求目标是每场景各双向维度4正4负，每任务每类每档每维每幅度小区间8个；这些是重采样的软目标，实际可行分布见realized_balance.csv。'
   lines[7]='碰撞或越界请求在同一场景、同一类别、同一档位重新选取其他合法点，允许改变符号和幅度子区间；不缩小档位、不使用任务成功筛选。'
  (output/'说明.txt').write_text('\n'.join(lines)+'\n')
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=HERE/'grouped_config.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--workers-per-gpu',type=int,default=2);a=p.parse_args();config=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
 lock=(a.output/'builder.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (a.output/'FROZEN').exists():print('Already frozen; fixed starts are unchanged',flush=True);return
 signature=digest({k:v for k,v in config.items() if k!='models'});marker=a.output/'test_signature.txt'
 if marker.exists() and marker.read_text()!=signature:raise RuntimeError('Existing test set has a different specification')
 marker.write_text(signature);dump(a.output/'config.json',config);dump(a.output/'plan.json',plan(config));audit=train_scene_audit(config['training_dataset']);dump(a.output/'training_scene_audit.json',audit)
 gpus={}
 for g in a.gpus:
  result=subprocess.run(['nvidia-smi','-i',str(g),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
  if not result.returncode:
   uuid,free=result.stdout.strip().split(',')
   if int(free)>4000:gpus[g]=uuid.strip()
 if not gpus:raise RuntimeError('No GPU with sufficient free memory for independent rendering')
 import queue,threading
 q=queue.Queue();lock=threading.Lock();started=time.time()
 # Same scene index first across four tasks, then the remaining indices.
 for i in range(5):
  for task in config['tasks']:q.put((task,i))
 def worker(g):
  while True:
   try:task,i=q.get_nowait()
   except queue.Empty:return
   folder=a.output/'scenes'/task/f'scene_{i:02d}';folder.mkdir(parents=True,exist_ok=True);done=folder/'summary.json'
   if done.exists() and json.loads(done.read_text()).get('status')=='completed':continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
   cmd=[sys.executable,'-u',str(HERE/'sample_grouped_scene.py'),'--config',str(a.config),'--output',str(a.output),'--task',task,'--scene',str(i),'--gpu',str(g)]
   dump(folder/'command.json',dict(command=cmd,gpu_uuid=gpus[g]))
   with (folder/'run.log').open('a') as log:
    try:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    except subprocess.TimeoutExpired:dump(done,dict(task=task,scene_index=i,status='error',error='Scene generation exceeded 30 minute timeout'))
   with lock:
    s=summarize(config,a.output);s['elapsed_seconds']=time.time()-started;dump(a.output/'progress.json',s);print(json.dumps(s),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)*a.workers_per_gpu) as pool:list(pool.map(worker,[g for g in gpus for _ in range(a.workers_per_gpu)]))
 final=summarize(config,a.output,final=True);final['elapsed_seconds']=time.time()-started;dump(a.output/'finished.json',final);print(json.dumps(final),flush=True)
if __name__=='__main__':main()
