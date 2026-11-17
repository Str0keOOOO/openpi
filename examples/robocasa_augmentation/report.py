"""Plot measured first-frame perturbations; does not load MuJoCo or a policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    d=pd.read_csv(a.work/'actual_distribution.csv');attempts=pd.read_csv(a.work/'attempts.csv')
    names=['X','Y','Yaw','Height']+[f'J{i}' for i in range(1,8)]
    scales=np.array([.1,.1,10,.1]+[10]*7);tol=np.array([.001,.001,.1,.001]+[.1]*7)
    figures=a.work/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,4,figsize=(18,11))
    labels=['human','perturb_single','perturb_multi'];colors=['#6b7280','#2563eb','#dc2626']
    rows=[]
    for i,name in enumerate(names):
        ax=axes.flat[i];edges=np.linspace(0,1,4) if i==3 else np.linspace(-1,1,7)
        arrays=[]
        for label in labels:
            x=d.loc[d.data_type==label,name].to_numpy();x=x[np.abs(x)>tol[i]]/scales[i];arrays.append(x)
        ax.hist(arrays,bins=edges,stacked=True,label=labels,color=colors)
        ax.set(title=name,xlabel='Normalized actual perturbation (inactive excluded)',ylabel='Samples')
        ax.axvline(0,color='black',lw=.5);ax.legend(fontsize=7)
        for task,part in d.groupby('task'):
            active=part[np.abs(part[name])>tol[i]]
            rows.append(dict(task=task,dimension=name,active=len(active),minimum=part[name].min(),maximum=part[name].max(),mean=part[name].mean(),std=part[name].std()))
    axes.flat[-1].axis('off');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/f'unified_11d_distribution.{ext}',dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(a.work/'dimension_statistics.csv',index=False)
    counts=d.groupby(['task','data_type','derived_type'],dropna=False).size().rename('count');counts.to_csv(a.work/'type_counts.csv')
    failures=attempts[~attempts.status.str.startswith('exported_',na=False)]
    failures.groupby(['status','reason'],dropna=False).size().rename('count').to_csv(a.work/'failure_statistics.csv')
    report=['# Measured augmentation report','',
            'Original Human and Anchor data are preserved. All new trajectories come from Human retargeting; the actual paired Anchor first frame defines the reference.',
            'Single/Multi are classified with 1 mm translation/height and 0.1 degree angular tolerances. Zero perturbations and out-of-range original Human samples are retained and identifiable.',
            'No policy was trained. Exported trials passed official initial collision, IK, executed robot joint limits, contact-review rejection and task-success checks.',
            'Teacher labels are not used. Normalized histograms show nonzero perturbations; original type labels are unchanged. Sparse bins do not imply unreachable bins.',
            '', '## Counts', counts.to_string(),'', '## Attempt outcomes',attempts.status.value_counts().to_string(),'',
            'A successful trial reaches the task condition at least once during the recorded execution. This does not claim global geometric optimality or formal continuous-time collision freedom.',
            '', 'See actual_distribution.csv for physical-unit deltas, attempts.csv for successful and failed attempts, and figures/ for plots.']
    (a.work/'report.md').write_text('\n\n'.join(report))


if __name__=='__main__':main()
