"""One-GPU structural-ablation queue: checks, matched training, seven-task evaluation."""
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
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(),'Existing launch requires inspection'
        assert (ROOT/'manifest.json').exists() and json.loads((ROOT/'cpu_checks.json').read_text())['status']=='completed'
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
        assert free>=24*1024,('GPU busy',free)
        (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True,
            trained_arms=[a for a,c in CFG['arms'].items() if c['train']],updates_per_trained_arm=1024,evaluation_rows_per_arm=448)
        save('launch.json',value);print(json.dumps(value),flush=True)

def run():
    actions=[['ablation_smoke.py',a] for a in CFG['arms']]
    # Inference-only keep-both, if selected by the user, can finish before long training.
    for a,c in CFG['arms'].items():
        if not c['train']:actions.append(['ablation_evaluate.py',a])
    for a,c in CFG['arms'].items():
        if c['train']:actions += [['ablation_train.py',a],['ablation_evaluate.py',a]]
    actions.append(['ablation_evaluate.py','summarize'])
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for action in actions:
                if (ROOT/'controls/pause_requested.json').exists():
                    save('runner_status.json',dict(status='paused',time=time.strftime('%F %T')));return
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
