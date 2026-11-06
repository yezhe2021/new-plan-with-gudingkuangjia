"""Detached shared-GPU queue; source-only preprocessing then Qwen-only M64 training."""
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
def free_memory():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
def paused():return (ROOT/'controls/pause_requested.json').exists()
def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(),'Existing queue: inspect before relaunch'
        (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        v=dict(pid=p.pid,detached=True,host='server-a',memory_tokens=64,adapter_count=4,adapter_rank=16,adapter_after_layers=[6,12,18,24],only_arm='M64+A4',execution='source-only GPU cache / Qwen-only GPU training + host saved-tensor offload')
        save('launch.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        action=None
        try:
            env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
            for action in [['prepare.py'],['cpu_checks.py']]:
                save('runner_status.json',dict(status='running_cpu_checks',pid=os.getpid(),action=action))
                subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,env=env,check=True)
            actions=[(['precompute.py','probe'],CFG['minimum_free_source_mib']),(['smoke.py'],CFG['minimum_free_training_mib']),
                (['precompute.py'],CFG['minimum_free_source_mib']),(['train_new.py'],CFG['minimum_free_training_mib']),(['evaluate_new.py'],CFG['minimum_free_training_mib'])]
            for action,minimum in actions:
                required=minimum
                while True:
                    if paused():save('runner_status.json',dict(status='paused',action=action));return
                    free=free_memory()
                    if free<required:
                        save('runner_status.json',dict(status='waiting_gpu',pid=os.getpid(),action=action,free_mib=free,required_mib=required))
                        time.sleep(30);continue
                    save('runner_status.json',dict(status='running',pid=os.getpid(),action=action,free_mib=free,required_mib=required))
                    print(time.strftime('%F %T'),'START',action,'free_mib',free,flush=True)
                    child=subprocess.run([sys.executable,'-u',str(ROOT/'gpu_entry.py'),*action],cwd=ROOT)
                    if child.returncode==75:save('runner_status.json',dict(status='paused',action=action));return
                    if child.returncode==76:
                        # Full context exits before retry. Never shrink/drop the experiment.
                        required=min(32256,max(required+2048,free+1024))
                        save('runner_status.json',dict(status='waiting_gpu_after_oom',pid=os.getpid(),action=action,required_mib=required))
                        print(time.strftime('%F %T'),'OOM retained exact run; wait for',required,'MiB',flush=True);time.sleep(30);continue
                    child.check_returncode();break
        except Exception as error:
            v=dict(status='failed',action=action,message=str(error));save('failure.json',v);save('runner_status.json',v);raise
        save('runner_status.json',dict(status='completed'))
if __name__=='__main__':launch() if sys.argv[1]=='launch' else run()
