"""Independent real-execution comparison of first-frame and pre-contact joins."""
import concurrent.futures,json,os,shutil,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,imageio.v2 as imageio
CODE=Path(__file__).parent
ROOT=Path('/data4/xuchenfei/robocasa365/data/household_augmented')
HUMAN=Path('/data4/xuchenfei/robocasa365/data/household_augmented')
OUTPUT=Path(os.environ.get('HANDOFF_COMPARISON_OUTPUT','/data5/xuchenfei/handoff_comparison/revised'))
CASES=[('OpenDrawer',467,0),('OpenStandMixerHead',441,81),('PickPlaceCounterToCabinet',331,86),('PickPlaceSinkToCounter',265,11)]
if os.environ.get('HANDOFF_ONLY_TASK'):CASES=[c for c in CASES if c[0]==os.environ['HANDOFF_ONLY_TASK']]
CAM='robot0_agentview_left'

def merge(paths,out):
 writer=imageio.get_writer(str(out),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast'])
 try:
  for path in paths:
   cap=cv2.VideoCapture(str(path))
   if not cap.isOpened():raise ValueError('Unreadable video '+str(path))
   try:
    while True:
     ok,bgr=cap.read()
     if not ok:break
     writer.append_data(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   finally:cap.release()
 finally:writer.close()

def side_by_side(old,new,out,old_transition,new_transition,join):
 caps=[cv2.VideoCapture(str(old)),cv2.VideoCapture(str(new))];last=[None,None];done=[False,False];i=0
 writer=imageio.get_writer(str(out),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast'])
 try:
  while True:
   fresh=False
   for j,c in enumerate(caps):
    if done[j]:continue
    ok,frame=c.read()
    if ok:last[j]=frame;fresh=True
    else:done[j]=True
   if not fresh:break
   images=[]
   for j,frame in enumerate(last):
    if frame is None:raise ValueError('Empty comparison video')
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);canvas=np.zeros((300,256,3),np.uint8);canvas[44:]=rgb
    title='OLD: join Human frame 0' if j==0 else f'NEW: join Human frame {join}'
    phase='transition' if i<(old_transition if j==0 else new_transition) else 'operation'
    cv2.putText(canvas,title,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    cv2.putText(canvas,f't={i/20:.2f}s  {phase}'+(' [ended]' if done[j] else ''),(5,35),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1);images.append(canvas)
   writer.append_data(np.concatenate(images,axis=1));i+=1
 finally:
  writer.close()
  for c in caps:c.release()

def path_length(x):return float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()) if len(x)>1 else 0.

def run(case):
 task,ep,human=case;start=time.time();folder=OUTPUT/f'{task}_ep{ep:06d}';folder.mkdir(parents=True,exist_ok=True);extra=ROOT/task/'lerobot/extras'/f'episode_{ep:06d}';oldprov=json.loads((extra/'provenance.json').read_text())
 old=ROOT/task/f'lerobot/videos/chunk-000/observation.images.{CAM}'/f'episode_{ep:06d}.mp4';shutil.copy2(old,folder/'old.mp4')
 source=HUMAN/task/f'lerobot/videos/chunk-000/observation.images.{CAM}'/f'episode_{human:06d}.mp4';shutil.copy2(source,folder/'human_reference.mp4')
 gpu=0;uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
 attempts=[];accepted=None
 for rank in range(3):
  trial=folder/f'candidate_{rank}';trial.mkdir(exist_ok=True)
  cmd=[sys.executable,'-u',str(CODE/'preflight_smooth_join.py'),'--task',task,'--episode',str(human),'--gpu',str(gpu),'--output',str(trial),'--execute-transition','--execute-operation','--reference-anchor-extras',str(extra),'--restore-initial-extras',str(extra),'--delta11',*(['0']*11),'--handoff-rank',str(rank),'--handoff-duration-scale',os.environ.get('HANDOFF_DURATION_SCALE','1.0')]
  with (trial/'run.log').open('w') as log:
   try:q=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
   except subprocess.TimeoutExpired:attempts.append(dict(rank=rank,status='timeout'));continue
  r=json.loads((trial/'result.json').read_text());reason=r.get('error') or r.get('failure_reason') or r.get('transition',{}).get('failure_reason')
  if r.get('status')=='operation_success':
   tr=r['transition'];op=r['operation']
   if tr.get('collision_review_required') or op.get('collision_review_required'):reason='Contact review flag';r['status']='contact_rejected'
   elif tr['final_position_error_m']>=.003 or tr['final_rotation_error_rad']>=.04:reason='Final EE tracking error';r['status']='tracking_rejected'
   else:
    with np.load(extra/'states.npz') as z:initial=z['states'][0]
    with np.load(trial/'transition_actual.npz') as z:
     if not np.allclose(z['sim_state_before_action'][0],initial,atol=1e-9,rtol=0):raise ValueError('Initial state changed')
    accepted=(trial,r);attempts.append(dict(rank=rank,status='accepted'));break
  attempts.append(dict(rank=rank,status=r.get('status'),reason=reason))
 if accepted is None:
  result=dict(task=task,episode=ep,human=human,status='no_accepted_candidate',attempts=attempts,seconds=time.time()-start)
 else:
  trial,r=accepted;tr=r['transition'];plan=r['handoff_plan'];merge([trial/f'transition_{CAM}.mp4',trial/f'{CAM}.mp4'],folder/'new.mp4');side_by_side(folder/'old.mp4',folder/'new.mp4',folder/'comparison.mp4',oldprov['transition_frames'],tr['steps'],plan['join_frame'])
  with np.load(extra/'ee_world.npz') as z:old_ee=z['ee_xyz_quat_xyzw'][:,:3]
  with np.load(trial/'transition_actual.npz') as z:new_tr=z['ee_world_xyz_quat_xyzw'][:,:3]
  with np.load(trial/'operation_actual.npz') as z:new_op=z['ee_world_xyz_quat_xyzw'][:,:3]
  old_proxy=path_length(old_ee[:oldprov['transition_frames']])+plan['human_initial_approach_path_m'];new_proxy=path_length(new_tr)+plan['remaining_precontact_path_m']
  result=dict(task=task,episode=ep,human=human,status='task_success',source_join_frame=plan['join_frame'],first_protected_event_frame=plan['first_protected_event_frame'],old_transition_s=oldprov['transition_frames']/20,new_transition_s=tr['steps']/20,old_approach_length_proxy_m=old_proxy,new_approach_length_proxy_m=new_proxy,approach_proxy_reduction_fraction=1-new_proxy/max(old_proxy,1e-12),old_total_actual_path_m=path_length(old_ee),new_total_actual_path_m=path_length(np.concatenate([new_tr,new_op])),final_position_error_m=tr['final_position_error_m'],final_rotation_error_rad=tr['final_rotation_error_rad'],comparison=str(folder/'comparison.mp4'),new_video=str(folder/'new.mp4'),source_frame_index_offset=plan['join_frame'],attempts=attempts,seconds=time.time()-start,note='Both videos at original 20Hz; shorter video holds its final frame in side-by-side view. Approach length is a proxy: measured connector plus retained source precontact path, not a claim of globally shortest executed task path.')
 (folder/'summary.json').write_text(json.dumps(result,indent=2));return result

def main():
 OUTPUT.mkdir(parents=True,exist_ok=True);results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for r in pool.map(run,CASES):
   results.append(r);(OUTPUT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
 (OUTPUT/'finished.json').write_text(json.dumps(dict(processed=len(results),task_success=sum(r['status']=='task_success' for r in results),dataset_modified=False),indent=2))
if __name__=='__main__':main()
