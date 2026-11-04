"""Detached single-GPU prepare/check/train/evaluate pipeline."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent

def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(),'Existing launch requires inspection'
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
        assert free>=24576,('GPU busy',free)
        (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True,training_updates=1536,evaluation_samples=512)
        save('launch.json',value);print(json.dumps(value),flush=True)

def run():
    actions=['prepare.py','cpu_checks.py','smoke.py','train_new.py','evaluate_new.py']
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for action in actions:
                if (ROOT/'controls/pause_requested.json').exists():save('runner_status.json',dict(status='paused'));return
                save('runner_status.json',dict(status='running',action=action,pid=os.getpid(),time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START',action,flush=True)
                result=subprocess.run([sys.executable,'-u',str(ROOT/action)],cwd=ROOT)
                if result.returncode==75:save('runner_status.json',dict(status='paused'));return
                result.check_returncode()
        except Exception as error:
            value=dict(status='failed',action=action,message=str(error),time=time.strftime('%F %T'))
            save('failure.json',value);save('runner_status.json',value);save('status.json',value);raise
        save('runner_status.json',dict(status='completed',time=time.strftime('%F %T')))

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
