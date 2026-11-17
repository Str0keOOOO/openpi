"""Read-only screening of measured handoff reversals, detours and slow joins."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np

def measured(p,n):
    before=p[n-1]-p[max(0,n-6)];after=p[min(len(p)-1,n+4)]-p[n]
    a,b=float(np.linalg.norm(before)),float(np.linalg.norm(after))
    cosine=float(before@after/(a*b)) if a>=.003 and b>=.003 else None
    end=min(len(p),n+60);prefix=p[:end];d=np.linalg.norm(np.diff(prefix,axis=0),axis=1);length=float(d.sum());net=float(np.linalg.norm(prefix[-1]-prefix[0]));eff=net/max(length,1e-12)
    tr=float(np.linalg.norm(np.diff(p[:n],axis=0),axis=1).sum());tr_speed=tr/max((n-1)/20,.05)
    op=np.linalg.norm(np.diff(p[n:],axis=0),axis=1)*20;moving=op[op>.005];speed=float(np.quantile(moving,.75)) if len(moving) else 0.
    reversal=cosine is not None and cosine<-.25 and a+b>=.015
    detour=length-net>=.08 and eff<=.70 and length>=.15
    slow=n>=30 and n/len(p)>=.25 and tr>=.02 and speed>.03 and tr_speed<.6*speed
    return dict(handoff_cosine=cosine,handoff_before_displacement_m=a,handoff_after_displacement_m=b,approach_path_m=length,approach_net_m=net,approach_extra_m=length-net,approach_efficiency=eff,transition_path_m=tr,transition_mean_speed_m_s=tr_speed,human_operation_moving_speed_p75_m_s=speed,join_reversal=reversal,approach_detour=detour,slow_transition=slow,selected=reversal or detour or slow)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];errors=[];counts={};started=time.time();all_eps=0
    for f in sorted(a.dataset.glob('*/lerobot/extras/episode_*/provenance.json')):
        all_eps+=1;r=json.loads(f.read_text());n=int(r.get('transition_frames',0))
        if n<=0:continue
        try:
            with np.load(f.parent/'ee_world.npz') as z:pts=z['ee_xyz_quat_xyzw'][:,:3]
            if n>=len(pts) or not np.isfinite(pts).all():raise ValueError('Invalid EE path/transition split')
            task=f.parents[3].name;ep=int(f.parent.name.split('_')[-1]);row=dict(task=task,episode=ep,human=int(r['source_human_episode_index']),old_steps=n,old_total_frames=len(pts),data_type=r.get('data_type'),**measured(pts,n));rows.append(row)
            ct=counts.setdefault(task,dict(with_transition=0,selected=0,join_reversal=0,approach_detour=0,slow_transition=0));ct['with_transition']+=1
            for k in list(ct)[1:]:ct[k]+=int(row[k])
        except Exception as ex:errors.append(dict(path=str(f),error=str(ex)))
    for name,rr in [('all_transitions.csv',rows),('selected.csv',[r for r in rows if r['selected']])]:
        if rr:
            with (a.output/name).open('w') as out:w=csv.DictWriter(out,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    report=dict(total_episodes=all_eps,with_transition=len(rows),selected=sum(r['selected'] for r in rows),by_task=counts,errors=errors,seconds=time.time()-started,criteria=dict(reversal='0.25s average displacements each >=3mm, cosine<-0.25, combined displacement>=15mm',detour='first transition+3s: path minus direct distance>=8cm, net/path<=0.70, path>=15cm',slow='transition>=1.5s and >=25% total, path>=2cm, mean speed<60% of operation moving-speed P75'),interpretation='Screening candidates, not proof of defective data. Replace only after real execution succeeds and improves measured path/time.')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
