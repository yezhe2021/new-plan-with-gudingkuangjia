"""Prepare/check on CPU, wait for first Gemma evaluation, then frozen Llama diagnostics."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())

def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def launch():
    assert not (ROOT/'launch.json').exists(),'Inspect existing launch before restarting'
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
    with (ROOT/'logs/pipeline.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=p.pid,time=time.strftime('%F %T'),detached=True,training=False,wait_for=CFG['wait_for']);save('launch.json',value);print(json.dumps(value),flush=True)

def action(args,cpu=False):
    if (ROOT/'controls/pause_requested.json').exists():raise SystemExit(75)
    save('runner_status.json',dict(status='running',action=args,pid=os.getpid(),time=time.strftime('%F %T')))
    env=dict(os.environ)
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    print(time.strftime('%F %T'),'START',args,flush=True)
    r=subprocess.run([sys.executable,'-u',str(ROOT/args[0]),*args[1:]],cwd=ROOT,env=env)
    if r.returncode==75:raise SystemExit(75)
    r.check_returncode()

def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            action(['prepare.py'],cpu=True);action(['cpu_checks.py'],cpu=True)
            wait=Path(CFG['wait_for']);marker=wait/'controls/first_only_completed.json'
            save('runner_status.json',dict(status='waiting_for_gemma_first_arm_evaluation',pid=os.getpid(),time=time.strftime('%F %T')))
            while True:
                if (ROOT/'controls/pause_requested.json').exists():raise SystemExit(75)
                if marker.exists():
                    v=json.loads(marker.read_text());assert v['first_arm']=='completed' and v['all36_started']==False
                    complete=wait/'arms/restricted_19_30/completed.json';assert json.loads(complete.read_text())['status']=='completed'
                    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True))
                    if free>=24576:break
                f=wait/'failure.json'
                if f.exists() and json.loads(f.read_text()).get('status')=='failed':raise RuntimeError('Gemma first arm failed; do not bypass its completion gate')
                time.sleep(30)
            for a in [['experiment.py','precompute'],['experiment.py','smoke'],['experiment.py','run'],['experiment.py','counterfactual'],['summarize.py']]:action(a)
        except SystemExit as exc:
            if exc.code==75:save('runner_status.json',dict(status='paused'));return
            raise
        except Exception as exc:
            value=dict(status='failed',message=str(exc),time=time.strftime('%F %T'));save('failure.json',value);save('runner_status.json',value);raise
        save('runner_status.json',dict(status='completed',time=time.strftime('%F %T')))

if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    else:run()
