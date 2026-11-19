"""User-requested pause controller; frozen experiment sources stay unchanged."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tarfile
import time
import traceback

ROOT=Path('/hy-tmp/yezhe/异构模型/nativechat_evidence_kv_baselines_v1_seed1234')
CONTROL=Path(__file__).resolve().parent
PARENT=675276
TRAINER=680806


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def process(pid):
    p=Path(f'/proc/{pid}')
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_ticks':fields[19],
                'cmd':(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode().strip()}
    except (FileNotFoundError,ProcessLookupError):
        return None


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def summary(status,request,completed=None,last=None,error=None):
    result={'experiment':ROOT.name,'timestamp_utc':now(),'server':'B','status':status,
            'full_experiment_complete':False,'configuration':read(ROOT/'config.json'),
            'checkpoint_selection':'held-out Stage A loss only, including epoch zero',
            'pause_request':request,'llama_stage_a':completed,'last_epoch_archive':last,
            'error':error,'evaluation':{},
            'paths':{'experiment':str(ROOT),'logs':str(ROOT/'logs/pipeline.log'),
                     'pause_controller_log':str(CONTROL/'pause.log')}}
    for p in sorted((ROOT/'evaluation').rglob('summary.json')):
        result['evaluation'][str(p.relative_to(ROOT))]=read(p)
    save(ROOT/'summary.json',result)
    shared=ROOT.parent/'EXPERIMENT_RESULTS.md'
    with shared.open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write('\n\n## '+ROOT.name+' — user-requested pause\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        f.flush();os.fsync(f.fileno())
    save(ROOT/'reports/shared_summary_updated.json',{'path':str(shared),'status':status,'timestamp_utc':now()})


def main():
    lock=(CONTROL/'pause.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent,trainer=process(PARENT),process(TRAINER)
    assert parent and parent['cmd'].endswith(str(ROOT/'run_pipeline.py')),parent
    assert trainer and trainer['ppid']==PARENT and trainer['cmd'].endswith(str(ROOT/'train.py')+' llama our_a'),trainer
    request={'requested_utc':now(),'instruction':'Finish current Llama Stage A, preserve results, then pause all later stages.',
             'dispatcher':parent,'trainer':trainer,'controller_pid':os.getpid(),
             'frozen_sources_changed':False,'phase':'arming'}
    save(ROOT/'reports/pause_request.json',request)
    # Terminate only the waiting dispatcher. Python subprocesses do not inherit
    # a parent-death signal here; the actual Stage A trainer keeps running.
    assert process(PARENT)['start_ticks']==parent['start_ticks']
    assert process(TRAINER)['start_ticks']==trainer['start_ticks']
    os.kill(PARENT,signal.SIGTERM)
    for _ in range(30):
        p=process(PARENT)
        if p is None or p['state']=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Dispatcher did not terminate; no further process signalled')
    child=process(TRAINER)
    assert child and child['start_ticks']==trainer['start_ticks'] and child['state']!='Z','Stage A must remain alive'
    request.update(phase='waiting_for_stage_a_completion',dispatcher_stopped=True,stage_a_training_continues=True)
    save(ROOT/'reports/pause_request.json',request)
    print(now(),'ARMED: dispatcher stopped; Stage A continues',flush=True)
    deadline=time.monotonic()+36*3600
    completed_path=ROOT/'training/llama/our_a/completed.json'
    while time.monotonic()<deadline:
        child=process(TRAINER)
        alive=child and child['start_ticks']==trainer['start_ticks'] and child['state']!='Z'
        if completed_path.exists() and not alive:break
        if not alive:raise RuntimeError('Stage A exited without its completed.json; preserve logs and diagnose')
        time.sleep(20)
    else:raise TimeoutError('Pause controller deadline reached; dispatcher already disabled')
    completed=read(completed_path)
    selected=Path(completed['checkpoint']['path'])
    assert selected.is_file() and sha(selected)==completed['checkpoint']['sha256']
    assert completed['completed'] and completed['updates']==768
    sys.path.insert(0,str(ROOT))
    from common import verify
    verify()
    # Preserve the final update as well as the validation-selected checkpoint.
    # Optimizer state stays in the original RAM-disk resume file, untouched.
    import torch
    cfg=read(ROOT/'config.json')
    resume=Path(cfg['scratch_directory'])/'resume_latest.pt'
    state=torch.load(resume,map_location='cpu',weights_only=True)
    assert state['identity']['family']=='llama' and state['identity']['stage']=='our_a'
    assert state['step']==768
    destination=Path(cfg['durable_stage_a_directory'])/'our_llama_stage_a_last.pt'
    assert not destination.exists(),'Do not overwrite an existing last-update archive'
    size=sum(v.numel()*v.element_size() for v in state['state'].values())
    assert shutil.disk_usage(destination.parent).free>size+(512<<20),'Insufficient persistent space for last-update archive'
    temporary=destination.with_suffix('.pt.tmp')
    torch.save({k:state[k] for k in ('identity','state','epoch','batch','step','history')},temporary)
    temporary.replace(destination)
    last={'path':str(destination),'sha256':sha(destination),'persistent':True,'selected_for_stage_b':False,
          'purpose':'Archive final Stage A update; validation-selected checkpoint remains authoritative'}
    request.update(phase='paused_after_stage_a',completed_utc=now(),stage_a_training_continues=False,
                   selected_checkpoint_verified=True,last_update_checkpoint=last)
    save(ROOT/'reports/pause_request.json',request)
    save(ROOT/'reports/paused_after_stage_a.json',{'paused':True,'timestamp_utc':now(),'completed':completed,'last_update_archive':last,
         'stage_b_started':False,'original_resume_file_preserved':str(resume),'frozen_manifest_verified':True})
    save(ROOT/'status.json',{'phase':'paused_by_user_after_llama_stage_a','timestamp_utc':now(),'full_experiment_complete':False,
                           'completed_stage':'llama/our_a','next_stage_not_started':'llama/our_b','updates':768})
    summary('paused_by_user_after_llama_stage_a',request,completed,last)
    archive=ROOT/'paused_stage_a_scripts_and_results.tgz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py','.json','.jsonl','.md','.log'):continue
            if 'models' in p.relative_to(ROOT).parts:continue
            t.add(p,arcname=p.relative_to(ROOT))
        t.add(Path(__file__),arcname='pause_control/pause_after_stage_a.py')
    print(now(),'PAUSED: selected and last-update weights durable; all lightweight results archived',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        report={'phase':'pause_controller_failed','timestamp_utc':now(),'error':repr(error),'traceback':traceback.format_exc()}
        save(ROOT/'reports/pause_controller_failure.json',report)
        print(json.dumps(report),flush=True)
        request=read(ROOT/'reports/pause_request.json') if (ROOT/'reports/pause_request.json').exists() else None
        if request and request.get('dispatcher_stopped'):
            summary('paused_controller_failed_preserve_existing_artifacts',request,error=repr(error))
        raise
