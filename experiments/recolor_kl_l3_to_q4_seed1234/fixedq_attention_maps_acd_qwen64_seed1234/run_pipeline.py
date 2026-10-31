"""Durable fixed-Native-Q attention map queue after Phase1."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())


def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(), 'Existing launch requires inspection'
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True,training=False,
            dependency=CFG['wait_for_experiment'],automatic_resume_other_experiments=False)
        save('launch.json',value);print(json.dumps(value,ensure_ascii=False))


def waiting():
    import hashlib
    dependency=Path(CFG['wait_for_experiment']);previous=None
    while True:
        assert dependency.exists(), 'Configured dependency disappeared'
        if (dependency/'manifest.json').exists():
            assert hashlib.sha256((dependency/'manifest.json').read_bytes()).hexdigest()==CFG['wait_for_manifest_sha256']
        if (dependency/'failure.json').exists():
            raise RuntimeError('Preceding mechanism experiment failed; do not silently start new training.')
        done=dependency/'completed.json'
        if done.exists():
            record=json.loads(done.read_text())
            assert record['status']=='completed' and record['manifest_sha256']==CFG['wait_for_manifest_sha256']
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
            state='waiting_for_gpu_release' if free<CFG['min_free_gpu_mib'] else 'dependency_completed_gpu_ready'
            detail=dict(gpu_free_mib=free)
        else:
            pid=json.loads((dependency/'launch.json').read_text())['pid']
            try: os.kill(pid,0)
            except ProcessLookupError: raise RuntimeError('Preceding experiment stopped without successful completion.')
            state='waiting_for_previous_experiment';detail=dict(dependency_pid=pid)
        value=dict(status='queued' if state!='dependency_completed_gpu_ready' else 'ready',phase=state,
            dependency=str(dependency),time=time.strftime('%F %T'),pid=os.getpid(),**detail)
        save('status.json',value);save('queue_status.json',value)
        if state!=previous: print(time.strftime('%F %T'),state,detail,flush=True);previous=state
        if state=='dependency_completed_gpu_ready': return
        time.sleep(30)


def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        action=['wait_for_previous_experiment']
        try:
            save('runner_status.json',dict(status='queued',action=action,pid=os.getpid(),time=time.strftime('%F %T')))
            waiting()
            actions=[['experiment.py','prepare'],['experiment.py','smoke'],['experiment.py','evaluate','arc'],['experiment.py','evaluate','hotpotqa'],['experiment.py','summarize']]
            for action in actions:
                if (ROOT/'controls/pause_requested.json').exists():
                    save('status.json',dict(status='paused',time=time.strftime('%F %T')));return
                save('runner_status.json',dict(status='running',action=action,pid=os.getpid(),time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START',action,flush=True)
                result=subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,check=False)
                if result.returncode==75:
                    save('runner_status.json',dict(status='paused',time=time.strftime('%F %T')));return
                result.check_returncode()
        except Exception as exc:
            value=dict(status='failed',type=type(exc).__name__,message=str(exc),action=action,time=time.strftime('%F %T'))
            save('failure.json',value);save('runner_status.json',value);save('status.json',value);raise
        save('runner_status.json',dict(status='completed',finished_at=time.strftime('%F %T')))


if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
