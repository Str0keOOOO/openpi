"""Compute equal-sample-exposure schedule from explicitly supplied actual sampler counts."""
import argparse, json, math
p=argparse.ArgumentParser()
p.add_argument('--reference-samples',type=int,required=True,help='actual number of eligible reference window starts')
p.add_argument('--new-samples',type=int,required=True,help='actual number of eligible new training window starts')
p.add_argument('--reference-batch',type=int,required=True)
p.add_argument('--new-batch',type=int,required=True)
p.add_argument('--reference-steps',type=int,default=75000)
a=p.parse_args()
if min(a.reference_samples,a.new_samples,a.reference_batch,a.new_batch,a.reference_steps)<=0: p.error('all counts must be positive')
epochs=a.reference_steps*a.reference_batch/a.reference_samples
steps=math.ceil(epochs*a.new_samples/a.new_batch)
print(json.dumps({'reference_global_batch':a.reference_batch,'reference_samples':a.reference_samples,'reference_steps':a.reference_steps,'reference_epochs_equivalent':epochs,'new_samples':a.new_samples,'new_global_batch':a.new_batch,'recommended_steps':steps,'new_epochs_equivalent':steps*a.new_batch/a.new_samples,'assumption':'uniform sample exposure; weighted mixture sampling requires task-specific accounting'},indent=2))
