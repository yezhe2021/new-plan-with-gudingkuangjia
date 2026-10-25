"""Detached, locked, fail-stop Reader-only training and evaluation."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())


def save(name,value):
    p=ROOT/name
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(p)


def ready():
    assert not (ROOT/'failure.json').exists(), 'Failure requires inspection'
    assert not (ROOT/'completed.json').exists(), 'Already completed'
    assert (ROOT/'prepared/completed.json').exists(), 'Not prepared'
    assert shutil.disk_usage(ROOT).free/2**30 > CFG['disk_reserve_gib']+4
    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True)
    assert float(out.strip())/1024 >= CFG['queue_min_free_gpu_gib'], 'GPU busy'


def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():
            raise RuntimeError('Existing launch; inspect before resuming')
        ready()
        subprocess.run([sys.executable,str(ROOT/'experiment.py'),'verify'],cwd=ROOT,check=True)
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,started_at=time.strftime('%F %T'),interpreter=sys.executable,detached=True)
        save('launch.json',value)
        print(json.dumps(value))


def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready()
        save('runner_status.json',dict(status='running',pid=os.getpid(),started_at=time.strftime('%F %T')))
        actions=[['smoke'],['train']]+[['evaluate',t] for t in CFG['evaluation_tasks']]+[['summarize']]
        try:
            for action in actions:
                print(time.strftime('%F %T'),'START',action,flush=True)
                subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),*action],cwd=ROOT,check=True)
        except Exception as exc:
            save('runner_status.json',dict(status='failed',pid=os.getpid(),error=str(exc)))
            if not (ROOT/'failure.json').exists():
                save('failure.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
        save('runner_status.json',dict(status='completed',pid=os.getpid(),finished_at=time.strftime('%F %T')))


if __name__=='__main__':
    if sys.argv[1]=='--launch': launch()
    elif sys.argv[1]=='--run': run()
    else: raise ValueError(sys.argv[1])
