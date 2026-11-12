"""Detached matched full-layer run; robust NVML/CUDA queue and accepted-update checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())

def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(dict(time=time.strftime('%F %T'),**value),ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def gpu():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,timeout=15)
        if p.returncode:return None,(p.stderr or p.stdout).strip()
        return int(p.stdout.strip()),None
    except (OSError,ValueError,subprocess.TimeoutExpired) as e:return None,str(e)
def cuda():
    try:
        p=subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(); torch.cuda.mem_get_info(); print("CUDA_OK")'],capture_output=True,text=True,timeout=60)
        return p.returncode==0,(p.stderr or p.stdout)[-1500:]
    except (OSError,subprocess.TimeoutExpired) as e:return False,str(e)
def paused():return (ROOT/'controls/pause_requested.json').exists()
def launch(mode):
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/('launch_'+mode+'.json')).exists(),'Existing queue must be inspected before relaunch'
        (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as f:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run',mode],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        v=dict(pid=child.pid,detached=True,host='server-a',model_pair='Gemma3-4B -> Full-Sync DST+Rank64 -> Qwen3-4B',memory_tokens=0,adapters=0,state_width=256,receiver_layers=36,updates_per_stage=1536,checks_only=mode=='checks',static_memory=False)
        save('launch_'+mode+'.json',dict(mode=mode,**v));print(json.dumps(v),flush=True)
def run(mode):
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);action=None
        try:
            env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
            for action in [['assets.py'],['prepare.py'],['checks.py','cpu']]:
                save('runner_status.json',dict(status='running_cpu_checks',pid=os.getpid(),action=action))
                child=subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,env=env);child.check_returncode()
            for action in ([['checks.py','stage1']]+([['train.py','stage1'],['checks.py','stage2'],['train.py','stage2'],['evaluate.py']] if mode=='train' else [])):
                while True:
                    if paused():save('runner_status.json',dict(status='paused',action=action));return
                    free,error=gpu()
                    if error:
                        save('runner_status.json',dict(status='waiting_gpu_unavailable',pid=os.getpid(),action=action,gpu_error=error));time.sleep(30);continue
                    if free<CFG['minimum_free_gpu_mib']:
                        save('runner_status.json',dict(status='waiting_gpu',pid=os.getpid(),action=action,free_mib=free,required_mib=CFG['minimum_free_gpu_mib']));time.sleep(30);continue
                    ok,probe=cuda()
                    if not ok:
                        save('runner_status.json',dict(status='waiting_gpu_unavailable',pid=os.getpid(),action=action,gpu_error=probe));time.sleep(30);continue
                    save('runner_status.json',dict(status='running',pid=os.getpid(),action=action,free_mib=free));print(time.strftime('%F %T'),'START',action,flush=True)
                    child=subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT)
                    if child.returncode==75:save('runner_status.json',dict(status='paused',action=action));return
                    if child.returncode:
                        _,error=gpu();ok,probe=cuda() if error is None else (False,error)
                        if not ok:
                            save('runner_status.json',dict(status='waiting_gpu_unavailable',pid=os.getpid(),action=action,gpu_error=probe));time.sleep(30);continue
                        child.check_returncode()
                    break
        except Exception as error:
            v=dict(status='failed',action=action,message=str(error));save('failure.json',v);save('runner_status.json',v);raise
        save('runner_status.json',dict(status='ready_for_training' if mode=='checks' else 'completed'))
if __name__=='__main__':launch(sys.argv[2]) if sys.argv[1]=='launch' else run(sys.argv[2])
