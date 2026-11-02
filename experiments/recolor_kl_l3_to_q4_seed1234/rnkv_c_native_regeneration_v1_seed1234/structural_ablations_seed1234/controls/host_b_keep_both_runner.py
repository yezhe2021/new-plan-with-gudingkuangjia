"""Resume the second ablation on B after cancelling A's waiting queue."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent.parent

def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def launch():
    assert not (ROOT/'controls/server_b_keep_both_launch.json').exists()
    assert (ROOT/'arms/memory_only/evaluation/completed.json').exists()
    assert (ROOT/'controls/server_a_cancelled_ack.json').exists()
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
    assert free>=24*1024,('GPU busy',free)
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/server_b_keep_both.log').open('a') as stream:
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=p.pid,host='server-b',arm='keep_both',detached=True,started_at=time.strftime('%F %T'),updates=1024)
    save('controls/server_b_keep_both_launch.json',value);print(json.dumps(value),flush=True)

def run():
    with (ROOT/'controls/server_b_keep_both.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for action in [['ablation_smoke.py','keep_both'],['ablation_train.py','keep_both'],['ablation_evaluate.py','keep_both'],['ablation_evaluate.py','summarize']]:
            if (ROOT/'controls/pause_requested.json').exists():
                save('controls/server_b_keep_both_status.json',dict(status='paused',time=time.strftime('%F %T')));return
            value=dict(status='running',host='server-b',arm='keep_both',action=action,pid=os.getpid(),time=time.strftime('%F %T'))
            save('controls/server_b_keep_both_status.json',value);save('runner_status.json',value)
            print(time.strftime('%F %T'),'START',action,flush=True)
            result=subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,check=False)
            if result.returncode==75:
                save('controls/server_b_keep_both_status.json',dict(status='paused',time=time.strftime('%F %T')));return
            result.check_returncode()
        value=dict(status='completed',host='server-b',arm='keep_both',all_arms_completed=True,finished_at=time.strftime('%F %T'))
        save('controls/server_b_keep_both_completed.json',value);save('controls/server_b_keep_both_status.json',value);save('runner_status.json',value)

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:
        try:run()
        except Exception as exc:
            value=dict(status='failed',host='server-b',arm='keep_both',type=type(exc).__name__,message=str(exc),time=time.strftime('%F %T'))
            save('controls/server_b_keep_both_failure.json',value);save('controls/server_b_keep_both_status.json',value);save('runner_status.json',value);raise
