"""Finish only GSM8K, then pause; do not modify the frozen evaluator."""
from pathlib import Path
import datetime
import fcntl
import importlib.util
import json
import os
import signal
import sys
import tarfile
import time
import traceback

BASE=Path('/hy-tmp/yezhe/异构模型')
HERE=BASE/'heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825'
CONTROL=Path(__file__).resolve().parent
PARENT=726780


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def read(path):return json.loads(path.read_text())


def process(pid):
    p=Path(f'/proc/{pid}')
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_ticks':fields[19],
                'cmd':(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode().strip()}
    except (FileNotFoundError,ProcessLookupError):return None


def log(message):print(now(),message,flush=True)


def main():
    controller_lock=(CONTROL/'pause.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=process(PARENT)
    request_path=HERE/'reports/pause_after_gsm8k_request.json'
    if parent is None:
        request=read(request_path)
        assert request.get('dispatcher_stopped') and request['dispatcher']['pid']==PARENT
        assert request['dispatcher']['cmd'].endswith(str(HERE/'run_evaluation.py'))
        children=request.get('existing_children',[])
        for child in children:
            live=process(child['pid'])
            if live and live['state']!='Z':
                assert live['start_ticks']==child['start_ticks'] and '--benchmark gsm8k ' in live['cmd'],live
        request.update(controller_pid=os.getpid(),recovery_utc=now(),phase='finishing_gsm8k_only')
        save(request_path,request)
        save(HERE/'reports/pause_after_gsm8k_recovery.json',{'timestamp_utc':now(),'previous_failure_preserved':True,
             'repair':'Wait for current child to exit before acquiring original pipeline lock','controller_pid':os.getpid()})
        log('RECOVERED: dispatcher remains stopped; existing GSM8K shard retained')
    else:
        assert parent['cmd'].endswith(str(HERE/'run_evaluation.py')),parent
        request={'requested_utc':now(),'controller_pid':os.getpid(),'dispatcher':parent,
             'instruction':'Finish all three 128-item GSM8K conditions, then pause; do not start Qasper extension.',
             'frozen_sources_changed':False,'phase':'arming'}
        arm_dispatcher(parent,request)
        request=read(request_path)
        children=request.get('existing_children',[])
    finish_gsm8k(request,children)


def arm_dispatcher(parent,request):
    save(HERE/'reports/pause_after_gsm8k_request.json',request)
    # Stop scheduling before examining the live child list. Existing evaluation
    # children keep running; no process group or unrelated job is signalled.
    assert process(PARENT)['start_ticks']==parent['start_ticks']
    os.kill(PARENT,signal.SIGSTOP)
    try:
        for _ in range(30):
            p=process(PARENT)
            if p and p['state'] in ('T','t'):break
            time.sleep(.1)
        else:raise RuntimeError('Dispatcher did not stop')
        children=[]
        for pid in Path(f'/proc/{PARENT}/task/{PARENT}/children').read_text().split():
            child=process(int(pid))
            if child is None or child['state']=='Z':continue
            assert '-m heterofold evaluate' in child['cmd'] and '--benchmark gsm8k ' in child['cmd'] and str(HERE/'results') in child['cmd'],child
            children.append(child)
        assert len(children)<=1,children
    except BaseException:
        os.kill(PARENT,signal.SIGCONT)
        raise
    os.kill(PARENT,signal.SIGKILL)
    request.update(phase='finishing_gsm8k_only',dispatcher_stopped=True,existing_children=children)
    save(HERE/'reports/pause_after_gsm8k_request.json',request)
    log('ARMED: old dispatcher stopped; current GSM8K shard preserved; Qasper scheduling disabled')


def finish_gsm8k(request,children):
    save(HERE/'status.json',{'phase':'finishing_gsm8k_before_user_requested_pause','timestamp_utc':now(),
                           'qasper_scheduling_disabled':True,'controller_pid':os.getpid()})
    deadline=time.monotonic()+6*3600
    while children:
        remaining=[]
        for child in children:
            live=process(child['pid'])
            if live and live['start_ticks']==child['start_ticks'] and live['state']!='Z':remaining.append(child)
        if not remaining:break
        if time.monotonic()>deadline:raise TimeoutError('Current GSM8K shard exceeded controller wait deadline')
        time.sleep(10)
        children=remaining
    # GPU process teardown can delay descriptor cleanup. Acquire after the old
    # child exits, with a bounded wait rather than failing at the handoff.
    pipeline_lock=(HERE/'pipeline.lock').open('a')
    deadline=time.monotonic()+60
    while True:
        try:
            fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic()>deadline:raise
            time.sleep(1)
    spec=importlib.util.spec_from_file_location('frozen_eval_128',HERE/'run_evaluation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for key,value in module.environment().items():os.environ[key]=value
    manifest=read(HERE/'manifest.json');module.verify_manifest(manifest)
    qasper_before={c:module.collect(HERE/f'results/{c}/qasper')[0] for c in module.CONDITIONS}
    for condition in module.CONDITIONS:
        rows,_,_=module.collect(HERE/f'results/{condition}/gsm8k')
        for start,end in module.pending_ranges(rows,module.CFG['shard_size']['gsm8k']):
            log(f'CONTINUE GSM8K ONLY {condition} {start}:{end}')
            module.evaluate(condition,'gsm8k',start,end)
            module.summarize('finishing_gsm8k_before_user_requested_pause')
    for condition in module.CONDITIONS:
        assert sorted(module.collect(HERE/f'results/{condition}/gsm8k')[0])==list(range(128))
        assert module.collect(HERE/f'results/{condition}/qasper')[0]==qasper_before[condition],'Qasper outputs changed unexpectedly'
    module.verify_manifest(manifest)
    result=module.summarize('paused_by_user_after_gsm8k',shared=True)
    counts={task:{c:len(module.collect(HERE/f'results/{c}/{task}')[0]) for c in module.CONDITIONS} for task in module.TASKS}
    report={'paused':True,'timestamp_utc':now(),'completed_tasks':['arc','gsm8k'],'counts':counts,
            'qasper_extension_started':False,'frozen_manifest_verified':True,'original_weights_and_outputs_preserved':True,
            'full_experiment_complete':result['completed']}
    save(HERE/'reports/paused_after_gsm8k.json',report)
    save(HERE/'status.json',{'phase':'paused_by_user_after_gsm8k','timestamp_utc':now(),
                           'completed_conditions_128':6,'qasper_extension_started':False,'full_experiment_complete':False})
    request.update(phase='paused_after_gsm8k',completed_utc=now(),counts=counts)
    save(HERE/'reports/pause_after_gsm8k_request.json',request)
    with tarfile.open(HERE/'paused_after_gsm8k_scripts_and_results.tgz','w:gz') as tar:
        for p in sorted(HERE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl','.md','.log'):
                tar.add(p,arcname=p.relative_to(HERE))
        tar.add(Path(__file__),arcname='pause_control/pause_after_gsm8k.py')
    log('PAUSED: GSM8K 128/128 for all three conditions; Qasper unchanged; summary appended and lightweight artifacts archived')


if __name__=='__main__':
    try:main()
    except Exception as error:
        record={'phase':'pause_controller_failed','timestamp_utc':now(),'error':repr(error),'traceback':traceback.format_exc()}
        save(HERE/'reports/pause_after_gsm8k_failure.json',record)
        request=read(HERE/'reports/pause_after_gsm8k_request.json') if (HERE/'reports/pause_after_gsm8k_request.json').exists() else {}
        if request.get('dispatcher_stopped'):
            save(HERE/'status.json',record)
            try:
                spec=importlib.util.spec_from_file_location('frozen_eval_128_failure',HERE/'run_evaluation.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.summarize('incomplete_during_user_requested_gsm8k_pause',repr(error),shared=True)
            except Exception:traceback.print_exc()
        log(json.dumps(record));raise
