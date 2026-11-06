"""Operational-only recovery: wait through NVML/CUDA outages, reuse completed cache and checks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text())
EXPECTED_MANIFEST='a3dbf1dcf63ba6b734a47f74376c80428d31c1866e484b597e5e69ce81785295'

def read(path):return json.loads(Path(path).read_text())
def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(dict(time=time.strftime('%F %T'),**value),ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def query_gpu():
    try:
        v=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,timeout=15)
    except (OSError,subprocess.TimeoutExpired) as e:return None,str(e)
    if v.returncode:return None,(v.stderr or v.stdout or f'nvidia-smi exit{v.returncode}').strip()[:1500]
    try:return int(v.stdout.strip()),None
    except ValueError:return None,'Unexpected nvidia-smi response: '+v.stdout[:500]

def cuda_probe():
    code="import json,torch; assert torch.cuda.is_available(), 'No CUDA GPUs are available'; free,total=torch.cuda.mem_get_info(); print(json.dumps(dict(free_mib=free//2**20,total_mib=total//2**20)))"
    try:v=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
    except (OSError,subprocess.TimeoutExpired) as e:return False,str(e)
    if v.returncode:return False,(v.stderr or v.stdout)[-1800:]
    return True,v.stdout.strip()

def execution():
    path=ROOT/'controls/use_gpu_resident.json'
    if not path.exists():return ROOT/'gpu_entry.py',CFG['minimum_free_training_mib'],None
    record=read(path);entry=ROOT/'controls/gpu_resident_entry.py'
    assert record['manifest_sha256']==EXPECTED_MANIFEST and record['entry_sha256']==hashlib.sha256(entry.read_bytes()).hexdigest()
    return entry,max(CFG['minimum_free_training_mib'],record['minimum_free_mib']),record

def paused():return (ROOT/'controls/pause_requested.json').exists()
def set_status(**v):
    value=dict(pid=os.getpid(),recovery_controller='controls/recover_queue.py',**v)
    save('runner_status.json',value);save('status.json',value)

def finished(name):
    p=ROOT/name
    if not p.exists():return False
    v=read(p);assert v['manifest_sha256']==EXPECTED_MANIFEST and v['status']=='completed';return True

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'controls/recovery_launch.json').exists(),'Recovery already launched; inspect existing process'
        # Prove the original queue/another recovery is not still running.
        with (ROOT/'pipeline.lock').open('a') as pipeline:
            fcntl.flock(pipeline,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        v=dict(pid=child.pid,detached=True,original_manifest_sha256=EXPECTED_MANIFEST,reason='Resume same experiment after GPU recovery or user-requested execution change; preserve scientific configuration and checkpoints',execution_mode='gpu_resident' if (ROOT/'controls/use_gpu_resident.json').exists() else 'cpu_offload')
        save('controls/recovery_launch.json',v);print(json.dumps(v),flush=True)

def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sys.path.insert(0,str(ROOT));import experiment_common as E
        action=None
        try:
            E.verify();assert E.identity()==EXPECTED_MANIFEST
            assert finished('base_cache/completed.json') and finished('smoke/completed.json') and finished('evaluation_parity.json')
            archive=ROOT/'attempts/20261105_nvml_queue_exit';archive.mkdir(parents=True,exist_ok=True)
            for name in ['failure.json','runner_status.json','status.json']:
                p=ROOT/name
                if p.exists() and not (archive/name).exists():p.replace(archive/name)
            save('controls/recovery_provenance.json',dict(status='deployed',original_manifest_sha256=EXPECTED_MANIFEST,
                original_code_and_config_unchanged=True,reuse_completed_cache=True,reuse_passed_cpu_gpu_checks=True,
                changes='Operational controller waits through GPU faults; optional explicit GPU-resident runtime override follows user request and passed update-parity/memory checks',execution_override=execution()[2],
                original_failure=read(archive/'failure.json') if (archive/'failure.json').exists() else None))
            print(time.strftime('%F %T'),'RECOVERY controller active; retain all cache, scientific code/config unchanged',flush=True)
            for action,marker in [('train_new.py','runs/completed.json'),('evaluate_new.py','completed.json')]:
                if finished(marker):continue
                entry,required,execution_record=execution();last_error=None
                while True:
                    if paused():set_status(status='paused',action=action);return
                    free,error=query_gpu()
                    if error:
                        set_status(status='waiting_gpu_unavailable',phase='gpu_access_recovery',action=action,gpu_error=error,training_started=(ROOT/'runs/last.pt').exists())
                        if error!=last_error:print(time.strftime('%F %T'),'WAIT GPU ACCESS',error,flush=True);last_error=error
                        time.sleep(30);continue
                    if free<required:
                        set_status(status='waiting_gpu',action=action,free_mib=free,required_mib=required);time.sleep(30);continue
                    ok,cuda=cuda_probe()
                    if not ok:
                        set_status(status='waiting_gpu_unavailable',phase='cuda_access_recovery',action=action,gpu_error=cuda,free_mib=free)
                        if cuda!=last_error:print(time.strftime('%F %T'),'WAIT CUDA ACCESS',cuda,flush=True);last_error=cuda
                        time.sleep(30);continue
                    set_status(status='running',action=action,free_mib=free,required_mib=required,cuda_probe=cuda,execution_mode='gpu_resident' if execution_record else 'cpu_offload')
                    print(time.strftime('%F %T'),'RECOVERY START',action,'free_mib',free,flush=True)
                    child=subprocess.run([sys.executable,'-u',str(entry),action],cwd=ROOT)
                    if child.returncode==75:set_status(status='paused',action=action);return
                    if child.returncode==76:
                        if execution_record and free>=30720:raise RuntimeError('GPU-resident execution exceeded memory despite free GPU; checkpoint retained, CPU offload will not be silently re-enabled')
                        required=min(32256,max(required+2048,free+1024));set_status(status='waiting_gpu_after_oom',action=action,required_mib=required);time.sleep(30);continue
                    if child.returncode:
                        _,error=query_gpu()
                        if error:
                            set_status(status='waiting_gpu_unavailable',action=action,child_exit=child.returncode,gpu_error=error);time.sleep(30);continue
                        ok,cuda=cuda_probe()
                        if not ok:
                            set_status(status='waiting_gpu_unavailable',action=action,child_exit=child.returncode,gpu_error=cuda);time.sleep(30);continue
                        child.check_returncode()
                    break
            set_status(status='completed')
        except Exception as error:
            v=dict(status='failed',action=action,message=str(error));save('failure.json',v);set_status(**v);raise

def self_test():
    from unittest.mock import patch
    from types import SimpleNamespace
    with patch('subprocess.run',return_value=SimpleNamespace(returncode=255,stderr='Failed to initialize NVML: Unknown Error',stdout='')):
        free,error=query_gpu();assert free is None and 'NVML' in error
    with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stderr='',stdout='19739\n')):
        assert query_gpu()==(19739,None)
    with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stderr='',stdout='bad')):
        assert query_gpu()[0] is None
    with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('nvidia-smi',15)):
        assert query_gpu()[0] is None
    print('RECOVERY GPU-QUERY FAILURE TESTS PASSED')
if __name__=='__main__':
    {'launch':launch,'run':run,'self-test':self_test}[sys.argv[1]]()
