"""Detached dependency queue: smoke → diagnostic → Reader training → evaluation."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())


def save(name,value):
    p=ROOT/name
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(p)


def readiness():
    parent=Path(CFG['wait_for_experiment'])
    if (parent/'failure.json').exists():
        raise RuntimeError('Prior experiment failed; inspect it before using GPU')
    completed=parent/'completed.json'
    if not completed.exists():
        return dict(ready=False,reason='prior_experiment_running',dependency=str(parent))
    d=json.loads(completed.read_text())
    assert d.get('status')=='completed'
    if 'manifest_sha256' in d:
        assert d['manifest_sha256']==hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest()
    if (parent/'runner_status.json').exists():
        runner=json.loads((parent/'runner_status.json').read_text())
        if runner['status']!='completed':
            return dict(ready=False,reason='prior_runner_exiting')
    free_disk=shutil.disk_usage(ROOT).free/2**30
    if free_disk < CFG['disk_reserve_gib']+4:
        return dict(ready=False,reason='disk_reserve',free_disk_gib=free_disk)
    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True)
    free_gpu=float(out.strip())/1024
    if free_gpu < CFG['queue_min_free_gpu_gib']:
        return dict(ready=False,reason='gpu_busy',free_gpu_gib=free_gpu)
    return dict(ready=True,reason='ready',free_gpu_gib=free_gpu)


def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'failure.json').exists(), 'Failure requires review'
        assert not (ROOT/'completed.json').exists(), 'Already completed'
        assert not (ROOT/'launch.json').exists(), 'Existing launch; inspect before restarting'
        assert (ROOT/'prepared/completed.json').exists()
        subprocess.run([sys.executable,str(ROOT/'experiment.py'),'verify'],cwd=ROOT,check=True)
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=process.pid,started_at=time.strftime('%F %T'),detached=True,interpreter=sys.executable)
        save('launch.json',record)
        print(json.dumps(record))


def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            previous_reason=None
            while True:
                state=readiness()
                if state['ready']:break
                rec=dict(status='queued',time=time.strftime('%F %T'),pid=os.getpid(),**state)
                save('runner_status.json',rec)
                save('status.json',dict(phase='waiting_for_gpu',**rec))
                if state['reason']!=previous_reason:
                    print(time.strftime('%F %T'),'QUEUED',state,flush=True)
                    previous_reason=state['reason']
                time.sleep(CFG['queue_retry_seconds'])
            save('runner_status.json',dict(status='running',pid=os.getpid(),started_at=time.strftime('%F %T')))
            actions=[['experiment.py','smoke'],['diagnostic.py'],['experiment.py','train']]
            actions += [['experiment.py','evaluate',task] for task in CFG['evaluation_tasks']]
            actions += [['experiment.py','summarize']]
            for action in actions:
                print(time.strftime('%F %T'),'START',action,flush=True)
                save('runner_status.json',dict(status='running',pid=os.getpid(),phase=action,time=time.strftime('%F %T')))
                subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,check=True)
        except Exception as exc:
            save('runner_status.json',dict(status='failed',pid=os.getpid(),error=str(exc)))
            if not (ROOT/'failure.json').exists():
                save('failure.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
        save('runner_status.json',dict(status='completed',pid=os.getpid(),finished_at=time.strftime('%F %T')))


if __name__=='__main__':
    if sys.argv[1]=='--launch':launch()
    elif sys.argv[1]=='--run':run()
    else:raise ValueError(sys.argv[1])
