"""Detached frozen A/C/D native KV substitution and masking,64/task, no GSM8K."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent


def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(),'Existing launch requires inspection'
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
        assert free>=24*1024,('GPU busy',free)
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True,training=False,automatic_resume_other_experiments=False)
        save('launch.json',value);print(json.dumps(value))


def run():
    actions=[['experiment.py', 'prepare'], ['experiment.py', 'smoke'], ['experiment.py', 'evaluate', 'arc'], ['experiment.py', 'evaluate', 'hellaswag'], ['experiment.py', 'evaluate', 'hotpotqa'], ['experiment.py', 'summarize']]
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for action in actions:
                save('runner_status.json',dict(status='running',action=action,pid=os.getpid(),time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START',action,flush=True)
                subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,check=True)
        except Exception as exc:
            value=dict(status='failed',type=type(exc).__name__,message=str(exc),action=action,time=time.strftime('%F %T'))
            save('failure.json',value);save('runner_status.json',value);save('status.json',value);raise
        save('runner_status.json',dict(status='completed',finished_at=time.strftime('%F %T')))


if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
