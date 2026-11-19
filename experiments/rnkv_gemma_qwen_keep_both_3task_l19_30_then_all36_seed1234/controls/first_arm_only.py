"""User-requested hold: finish the running first arm, evaluate, then stop."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;TOP=HERE.parent

def save(name,value):
    p=TOP/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def launch():
    assert (HERE/'hold_second_arm.json').exists()
    assert not (HERE/'first_only_launch.json').exists()
    with (TOP/'logs/first_arm_only.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-u',str(HERE/'first_arm_only.py'),'run'],cwd=TOP,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save('controls/first_only_launch.json',dict(pid=p.pid,time=time.strftime('%F %T'),detached=True,all36_enabled=False));print(p.pid,flush=True)

def run():
    adoption=json.loads((HERE/'first_only_adoption.json').read_text());pid=adoption['training_child_pid']
    with (TOP/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save('runner_status.json',dict(status='running',arm='restricted_19_30',action='adopt_existing_training',training_child_pid=pid,pid=os.getpid(),all36_held=True,time=time.strftime('%F %T')))
            while live(pid):time.sleep(10)
            if (HERE/'pause_requested.json').exists():save('runner_status.json',dict(status='paused',all36_held=True));return
            done=TOP/'arms/restricted_19_30/runs/completed.json'
            assert done.exists() and json.loads(done.read_text())['status']=='completed','First training exited without completing'
            save('runner_status.json',dict(status='running',arm='restricted_19_30',action='evaluate_gemma.py',pid=os.getpid(),all36_held=True,time=time.strftime('%F %T')))
            env=dict(os.environ,RNKV_GEMMA_ARM='restricted_19_30')
            r=subprocess.run([sys.executable,'-u',str(HERE/'cache_entry.py'),'evaluate_gemma.py'],cwd=TOP,env=env)
            if r.returncode==75:save('runner_status.json',dict(status='paused',all36_held=True));return
            r.check_returncode()
        except Exception as error:
            v=dict(status='failed',message=str(error),all36_held=True,time=time.strftime('%F %T'))
            save('failure.json',v);save('runner_status.json',v);raise
        v=dict(status='held_after_first_arm',first_arm='completed',all36_started=False,reason='User requested mechanism discussion before second training',time=time.strftime('%F %T'))
        save('runner_status.json',v);save('status.json',v);save('controls/first_only_completed.json',v)

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
