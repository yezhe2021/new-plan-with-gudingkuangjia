"""Detached dependency gate: finish Llama diagnostics first; never start old Gemma all36."""
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
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(dict(time=time.strftime('%F %T'),**value),ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'launch.json').exists(),'Launch already exists; inspect before resuming'
        (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=child.pid,detached=True,wait_for=CFG['wait_for'],scope='ONLY Gemma19-30; new Recolor+C; three-task fresh Keep-both',
            c_updates=800,memory_updates=1536,all36_started=False)
        save('launch.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)

def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        action=None
        try:
            env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
            for action in [['prepare.py'],['cpu_checks.py']]:
                subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT,env=env,check=True)
            save('runner_status.json',dict(status='waiting_for_llama_mechanism_completion',pid=os.getpid(),wait_for=CFG['wait_for'],all36_started=False))
            print(time.strftime('%F %T'),'WAIT Llama mechanism experiment1/2/3 completed.json',flush=True)
            dep=Path(CFG['wait_for'])
            while True:
                if (ROOT/'controls/pause_requested.json').exists():save('runner_status.json',dict(status='paused'));return
                if (dep/'failure.json').exists():raise RuntimeError('Llama mechanism dependency failed; Gemma GPU stages not started')
                if (dep/'completed.json').exists():
                    value=json.loads((dep/'completed.json').read_text());assert value['status']=='completed'
                    import hashlib
                    assert value['manifest_sha256']==hashlib.sha256((dep/'manifest.json').read_bytes()).hexdigest()
                    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
                    if free>=24576:
                        save('dependency_released.json',dict(status='completed',dependency=value,gpu_free_mib=free));break
                time.sleep(30)
            actions=[['fit_recolor.py','check'],['fit_recolor.py'],['smoke_c.py'],['calibrate_c.py'],['smoke.py'],['train_new.py'],['evaluate_new.py','parity'],['evaluate_new.py']]
            for action in actions:
                if (ROOT/'controls/pause_requested.json').exists():save('runner_status.json',dict(status='paused'));return
                save('runner_status.json',dict(status='running',pid=os.getpid(),action=action));print(time.strftime('%F %T'),'START',action,flush=True)
                child=subprocess.run([sys.executable,'-u',str(ROOT/action[0]),*action[1:]],cwd=ROOT)
                if child.returncode==75:save('runner_status.json',dict(status='paused',action=action));return
                child.check_returncode()
        except Exception as error:
            value=dict(status='failed',action=action,message=str(error));save('failure.json',value);save('runner_status.json',value);raise
        save('runner_status.json',dict(status='completed'))

if __name__=='__main__':launch() if sys.argv[1]=='launch' else run()
