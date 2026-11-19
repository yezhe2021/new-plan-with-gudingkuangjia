"""Adopt the running first training without interrupting it; cache later phases."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;TOP=HERE.parent

def save(name,value):
    path=TOP/name;path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def launch():
    assert not (HERE/'cached_launch.json').exists()
    with (TOP/'logs/cached_pipeline.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u',str(HERE/'cached_queue.py'),'run'],cwd=TOP,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save('controls/cached_launch.json',dict(pid=p.pid,time=time.strftime('%F %T'),detached=True));print(p.pid,flush=True)

def run():
    adoption=json.loads((HERE/'adoption.json').read_text());pid=adoption.get('training_child_pid')
    with (TOP/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save('runner_status.json',dict(status='running',arm='restricted_19_30',action='adopt_existing_training',pid=os.getpid(),training_child_pid=pid,time=time.strftime('%F %T')))
            while pid and live(pid):time.sleep(10)
            if (TOP/'controls/pause_requested.json').exists():save('runner_status.json',dict(status='paused'));return
            completed=TOP/'arms/restricted_19_30/runs/completed.json'
            actions=[]
            if adoption.get('resume_first_training'):
                actions.append(('restricted_19_30','train_gemma.py',[]))
            else:assert completed.exists() and json.loads(completed.read_text())['status']=='completed','Adopted training exited without completing'
            actions.append(('restricted_19_30','evaluate_gemma.py',[]))
            actions += [('all_36',n,[]) for n in ['cpu_checks.py','smoke.py','train_gemma.py','evaluate_gemma.py']]
            actions += [(None,'evaluate_gemma.py',['summarize'])]
            for arm,script,args in actions:
                if (TOP/'controls/pause_requested.json').exists():save('runner_status.json',dict(status='paused'));return
                env=dict(os.environ)
                if arm:env['RNKV_GEMMA_ARM']=arm
                else:env.pop('RNKV_GEMMA_ARM',None)
                save('runner_status.json',dict(status='running',arm=arm,action=script,pid=os.getpid(),cache_enabled=True,time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START CACHED',arm,script,flush=True)
                r=subprocess.run([sys.executable,'-u',str(HERE/'cache_entry.py'),script,*args],cwd=TOP,env=env)
                if r.returncode==75:save('runner_status.json',dict(status='paused',arm=arm));return
                r.check_returncode()
        except Exception as exc:
            value=dict(status='failed',message=str(exc),time=time.strftime('%F %T'),cache_queue=True)
            save('failure.json',value);save('runner_status.json',value);raise
        save('runner_status.json',dict(status='completed',time=time.strftime('%F %T'),cache_enabled=True))

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
