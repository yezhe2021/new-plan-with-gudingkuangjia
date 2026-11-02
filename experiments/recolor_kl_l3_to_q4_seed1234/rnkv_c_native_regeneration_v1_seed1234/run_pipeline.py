"""Server-side dependency queue; waits without holding GPU or client SSH."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())

def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text())
def stopped():return (ROOT/'controls/pause_requested.json').exists()

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(),'Existing queue requires inspection'
        assert (ROOT/'manifest.json').exists() and read(ROOT/'cpu_checks.json')['status']=='completed'
        (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True,dependency=CFG['dependency'],automatic_start_after_success=True)
        save('launch.json',value);print(json.dumps(value),flush=True)

def wait_for_dependency():
    dep=Path(CFG['dependency'])
    while True:
        if stopped():return 'paused'
        if (dep/'failure.json').exists():
            save('dependency_blocked.json',dict(status='dependency_failed',source=str(dep/'failure.json'),details=read(dep/'failure.json')))
            return 'dependency_failed'
        state=read(dep/'runner_status.json') if (dep/'runner_status.json').exists() else {}
        if state.get('status') in ['failed','paused']:
            save('dependency_blocked.json',dict(status='dependency_'+state['status'],details=state));return 'dependency_'+state['status']
        complete=dep/'completed.json'
        if complete.exists() and state.get('status')=='completed':
            assert read(complete)['manifest_sha256']==CFG['dependency_manifest_sha256']
            assert read(complete)['status']=='completed'
            # A completed marker is written before the runner process exits.
            # Wait for actual release, and avoid confusing PID reuse with this runner.
            pid=read(dep/'launch.json')['pid'];proc=Path('/proc')/str(pid)/'cmdline'
            try:alive=str(dep).encode() in proc.read_bytes()
            except FileNotFoundError:alive=False
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
            if not alive and free>=24*1024:return 'ready'
            phase='waiting_gpu_release'
        else:phase='waiting_dependency'
        value=dict(status=phase,dependency=str(dep),dependency_action=state.get('action'),pid=os.getpid(),time=time.strftime('%F %T'))
        save('runner_status.json',value);save('status.json',value)
        time.sleep(30)

def run():
    actions=[['smoke.py'],['train.py'],['evaluate.py','arc'],['evaluate.py','hotpotqa'],['evaluate.py','summarize']]
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        action=['wait_for_dependency']
        try:
            result=wait_for_dependency()
            if result!='ready':save('runner_status.json',dict(status=result,time=time.strftime('%F %T')));return
            save('dependency_released.json',dict(status='completed',dependency=CFG['dependency'],time=time.strftime('%F %T')))
            for action in actions:
                if stopped():save('runner_status.json',dict(status='paused',time=time.strftime('%F %T')));return
                value=dict(status='running',action=action,pid=os.getpid(),time=time.strftime('%F %T'))
                save('runner_status.json',value);print(time.strftime('%F %T'),'START',action,flush=True)
                result=subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,check=False)
                if result.returncode==75:
                    save('runner_status.json',dict(status='paused',time=time.strftime('%F %T')));return
                result.check_returncode()
        except Exception as exc:
            value=dict(status='failed',action=action,type=type(exc).__name__,message=str(exc),time=time.strftime('%F %T'))
            save('failure.json',value);save('runner_status.json',value);save('status.json',value);raise
        save('runner_status.json',dict(status='completed',finished_at=time.strftime('%F %T')))

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
