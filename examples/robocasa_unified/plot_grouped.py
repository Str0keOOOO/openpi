"""Visualize the actual fixed-start coverage; reads data only."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.output;c=json.loads((r/'config.json').read_text());m=json.loads((r/'frozen_manifest.json').read_text());original=json.loads((r/'plan_initial.json').read_text()) if (r/'plan_initial.json').exists() else [];original_cases={(x['task'],x['scene_index'],x['id']):x for x in original};families=['body','arm','joint'];levels=['small','medium','large'];fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
 for ax,task in zip(axes.flat,c['tasks']):
  values=np.array([[sum(x['accepted'] and x['case']['task']==task and x['case']['family']==f and x['case']['level']==l for x in m['cases']) for f in families] for l in levels]);ax.imshow(values,vmin=0,vmax=40,cmap='Blues');ax.set_xticks(range(3),['Body (4D)','Arm (7D)','Joint (11D)']);ax.set_yticks(range(3),levels);ax.set_title(task)
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{values[i,j]}/40',ha='center',va='center',color='white' if values[i,j]>25 else 'black',fontsize=14)
 fig.suptitle('Legal fixed starts / planned starts (initial collision and joint-limit checks)',fontsize=14);fig.savefig(r/'coverage_by_task_family_level.png',dpi=220);fig.savefig(r/'coverage_by_task_family_level.svg');plt.close(fig)
 fig,axes=plt.subplots(4,3,figsize=(14,14),constrained_layout=True);bins=np.arange(-1.5,1.500001,.1)
 for axis,ax in enumerate(axes.flat):
  if axis>=11:ax.axis('off');continue
  actual=[];planned=[]
  for x in m['cases']:
   case=x['case']
   if axis not in case['active_dimensions']:continue
   requested=original_cases.get((case['task'],case['scene_index'],case['id']),case);tier=levels.index(requested['level']);lo,hi=c['level_multipliers'][tier];mid=lo+(requested['magnitude_subbins'][axis]+.5)*(hi-lo)/5;planned.append(mid*requested['signs'][axis])
   if x['accepted']:actual.append(x['actual_delta'][axis]/c['training_limits'][axis])
  ph,_=np.histogram(planned,bins);ah,_=np.histogram(actual,bins);ax.bar((bins[:-1]+bins[1:])/2,ah,width=.085,label='Legal starts',alpha=.8);ax.stairs(ph,bins,color='black',linewidth=1.2,label='Requested quota');ax.set_title(c['names'][axis]);ax.set_xlabel('Delta / augmentation bound');ax.set_ylabel('Active samples');ax.axvline(0,color='gray',linewidth=.6)
  if axis==0:ax.legend(fontsize=8)
 fig.suptitle('Requested marginal balance versus the physically legal subset',fontsize=14);fig.savefig(r/'actual_vs_requested_dimension_coverage.png',dpi=200);fig.savefig(r/'actual_vs_requested_dimension_coverage.svg');plt.close(fig)
 audit=json.loads((r/'physical_sampling_audit.json').read_text())
 with (r/'scenes.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(audit['scenes'][0]));w.writeheader();w.writerows(audit['scenes'])
 print('Saved coverage plots and20-scene CSV')
if __name__=='__main__':main()
