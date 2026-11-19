"""Detached serial queue: partial-layer train/eval, fresh all-layer train/eval."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
TOP=Path(__file__).resolve().parent
CFG=json.loads((TOP/'config.json').read_text())

def save(name,value):
    p=TOP/name;p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)

def launch():
    with (TOP/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (TOP/'launch.json').exists(),'Inspect prior launch before restarting'
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
        assert free>=24576,('GPU busy',free)
        (TOP/'logs').mkdir(exist_ok=True);(TOP/'controls').mkdir(exist_ok=True)
        with (TOP/'logs/pipeline.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(TOP/'run_pipeline.py'),'run'],cwd=TOP,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        v=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True,arm_order=CFG['arm_order'],updates_per_arm=1536)
        save('launch.json',v);print(json.dumps(v),flush=True)

def run():
    actions=[(None,'prepare.py',[])]
    for arm in CFG['arm_order']:
        actions += [(arm,n,[]) for n in ['cpu_checks.py','smoke.py','train_gemma.py','evaluate_gemma.py']]
    actions.append((None,'evaluate_gemma.py',['summarize']))
    with (TOP/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for arm,script,args in actions:
                if (TOP/'controls/pause_requested.json').exists():save('runner_status.json',dict(status='paused',arm=arm));return
                env=dict(os.environ)
                if arm:env['RNKV_GEMMA_ARM']=arm
                else:env.pop('RNKV_GEMMA_ARM',None)
                save('runner_status.json',dict(status='running',arm=arm,action=script,time=time.strftime('%F %T'),pid=os.getpid()))
                print(time.strftime('%F %T'),'START',arm,script,flush=True)
                result=subprocess.run([sys.executable,'-u',str(TOP/script),*args],cwd=TOP,env=env)
                if result.returncode==75:save('runner_status.json',dict(status='paused',arm=arm,action=script));return
                result.check_returncode()
        except Exception as error:
            v=dict(status='failed',arm=arm,action=script,message=str(error),time=time.strftime('%F %T'))
            save('failure.json',v);save('runner_status.json',v);save('status.json',v);raise
        save('runner_status.json',dict(status='completed',time=time.strftime('%F %T')))

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
