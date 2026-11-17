"""Temporarily reserve GPU memory; release cooperatively before training."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--fraction',type=float,default=.9)
    p.add_argument('--status',type=Path,required=True);p.add_argument('--release-file',type=Path,required=True)
    a=p.parse_args()
    if not 0<a.fraction<1:raise ValueError('fraction must be between zero and one')
    uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(uuids)!=torch.cuda.device_count():raise RuntimeError('Visible GPU mapping mismatch')
    allocations=[];rows=[];stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def write(status,error=None):
        a.status.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),fraction=a.fraction,gpus=rows,error=error,time=time.time()),indent=2));tmp.replace(a.status)
    def usage(uuid):
        raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
        return tuple(map(int,raw.strip().split(',')))
    try:
        for index,uuid in enumerate(uuids):
            if stopping or a.release_file.exists():break
            torch.cuda.set_device(index)
            torch.empty(1,device=f'cuda:{index}')
            total,used,free=usage(uuid);target=int(total*a.fraction)
            amount=target-used-8
            if amount<0 or free<amount+128:raise RuntimeError(f'{uuid}: insufficient free memory for requested fraction')
            remaining=amount*2**20
            while remaining>0:
                size=min(512*2**20,remaining)
                allocations.append(torch.empty(size,dtype=torch.uint8,device=f'cuda:{index}'))
                remaining-=size
            torch.cuda.synchronize(index)
            actual=usage(uuid)
            rows.append(dict(uuid=uuid,visible_index=index,total_mib=total,target_used_mib=target,actual_used_mib=actual[1],free_mib=actual[2],tensor_mib=torch.cuda.memory_allocated(index)/2**20))
            write('reserving');print(json.dumps(rows[-1]),flush=True)
        write('held')
        while not stopping and not a.release_file.exists():time.sleep(1)
    except Exception as exc:
        write('failed',repr(exc));raise
    finally:
        allocations.clear()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):torch.cuda.empty_cache()
        if len(rows)==len(uuids):write('released')
        print('reservation released',flush=True)

if __name__=='__main__':main()
