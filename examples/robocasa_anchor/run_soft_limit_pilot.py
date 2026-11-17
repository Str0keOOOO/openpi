import concurrent.futures,json,os,subprocess,sys,time,traceback
from pathlib import Path
import pandas as pd
from retime_payload import build,write_json
R=Path('/data4/xuchenfei/robocasa365/data/household_augmented');W=Path('/data4/xuchenfei/robocasa365/output/soft_limits_pilot_gpu46');OLD=Path('/data4/xuchenfei/robocasa365/output/transition_speed_gpu456');CODE=Path(__file__).parent

def main():
 W.mkdir(parents=True,exist_ok=True)
 old=pd.read_csv(OLD/'results.csv');bad=old[(old.status=='kept_original') & old.reason.fillna('').str.contains('joint-limit violation')].head(4)
 cases=[dict(task=x.task,episode=int(x.episode),human=int(x.human),role='previous_joint_limit_rejection') for x in bad.itertuples()]
 cases.extend([dict(task='OpenStandMixerHead',episode=191,human=98,role='strict_success_control'),dict(task='PickPlaceSinkToCounter',episode=382,human=26,role='strict_success_control')])
 write_json(W/'config.json',dict(cases=cases,angular_cap_deg=.6,max_consecutive_frames=6,linear_tolerance_m=.001,raw_human_audit='/data4/xuchenfei/robocasa365/output/joint_soft_limit_audit/summary.json',original_batch='continues unchanged',dataset_write=False))
 uuids={g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in [4,6]};results=[]
 def run(case,g):
  t=time.time();out=W/'trials'/f"{case['task']}_ep{case['episode']:06d}";out.mkdir(parents=True,exist_ok=True);extra=R/case['task']/'lerobot/extras'/f"episode_{case['episode']:06d}"
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  cmd=[sys.executable,'-u',str(CODE/'preflight_soft_limits.py'),'--task',case['task'],'--episode',str(case['human']),'--gpu',str(g),'--output',str(out),'--execute-transition','--execute-operation','--reference-anchor-extras',str(extra),'--restore-initial-extras',str(extra),'--delta11',*(['0']*11)]
  row=dict(case,gpu=g)
  try:
   with (out/'run.log').open('a') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
   data=json.loads((out/'result.json').read_text());row.update(status=data['status'],reason=data.get('error') or data.get('failure_reason'))
   soft=data.get('soft_joint_limit_report',{});row['soft_limit_accepted']=soft.get('accepted')
   row['max_arm_excess_deg']=max([r['max_excess'] for r in soft.get('joints',[]) if r['unit']=='deg'],default=0)
   row['longest_excess_frames']=max([r['longest_consecutive_frames'] for r in soft.get('joints',[])],default=0)
   if data['status']=='operation_success':
    payload=W/'payloads'/out.name;ready=build(R,case['task'],case['episode'],out,payload)
    provenance=json.loads((payload/'provenance.json').read_text());provenance['soft_joint_limit_report']=soft;write_json(payload/'provenance.json',provenance)
    row.update(status='verified_payload',video=str(payload/'robot0_agentview_left.mp4'),old_transition_s=ready['old_transition_frames']/20,new_transition_s=ready['new_transition_frames']/20,final_position_error_m=ready['final_position_error_m'],final_rotation_error_rad=ready['final_rotation_error_rad'])
  except Exception as e:row.update(status='rejected_or_error',reason=str(e))
  row['seconds']=time.time()-t;return row
 # Two real workers at a time, assigned one per GPU; the original 12 keep running.
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for start in range(0,len(cases),2):
   futures=[pool.submit(run,c,g) for c,g in zip(cases[start:start+2],[4,6])]
   for future in concurrent.futures.as_completed(futures):
    row=future.result();results.append(row);print(json.dumps(row),flush=True);pd.DataFrame(results).to_csv(W/'results.csv',index=False);write_json(W/'progress.json',dict(total=len(cases),processed=len(results),verified=sum(r['status']=='verified_payload' for r in results),running=len(results)<len(cases)))
 write_json(W/'finished.json',dict(total=len(cases),verified=sum(r['status']=='verified_payload' for r in results),not_accepted=sum(r['status']!='verified_payload' for r in results),original_batch_unchanged=True))
 (W/'report_zh.md').write_text('# 轻微软限位独立试跑\n\n规则：角度超出不超过0.6度、超过旧容差的连续段不超过6帧；直线关节仍为原1mm。完整任务成功、最终EE误差、接触检查、实际第一帧一致性以及同步数据导出均需通过。官方数据中更大的少数超限不被此规则自动接纳。没有修改现有数据集或旧批处理。\n\n'+'```text\n'+pd.DataFrame(results).to_string(index=False)+'\n```')
if __name__=='__main__':main()
