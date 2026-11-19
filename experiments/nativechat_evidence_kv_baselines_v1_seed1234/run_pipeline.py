"""Locked native correctness -> fresh baselines -> validation gate -> frozen fusion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from common import ROOT, CFG, save, read, rows, signature, digest, log, freeze, verify, scratch


def source_signature():
    files=list(ROOT.glob('*.py'))+list((ROOT/'vendor').rglob('*.py'))+[ROOT/'config.json']
    return signature({str(p.relative_to(ROOT)):digest(p) for p in sorted(files)})


def run(script,*args):
    log('PIPELINE '+script+' '+' '.join(args))
    save(ROOT/'status.json',{'phase':'pipeline_step','script':script,'arguments':list(args),
                           'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,check=True)


def shared_summary(status, fusion=False, error=None):
    result={'experiment':CFG['name'],'status':status,'server':'B','configuration':CFG,
            'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'fusion_completed':fusion,'error':error,'gold_evidence_mechanism_pilot':True,
            'not_official_full_HotpotQA':True,'evaluation':{},'validation_gates':{},
            'checkpoint_selection':'held-out losses only; initializations eligible',
            'paths':{'logs':str(ROOT/'logs/pipeline.log'),'raw_results':str(ROOT/'evaluation')},
            'checkpoint_storage':{'temporary':CFG['scratch_directory'],'temporary_is_volatile':True,
                                  'stage_a_final':CFG['durable_stage_a_directory'],'other_final':str(ROOT/'models')}}
    for path in sorted((ROOT/'evaluation').rglob('summary.json')):
        result['evaluation'][str(path.relative_to(ROOT))]=read(path)
    for path in sorted((ROOT/'evaluation/validation').rglob('gate.json')):
        result['validation_gates'][str(path.relative_to(ROOT))]=read(path)
    save(ROOT/'summary.json',result)
    shared=ROOT.parent/'EXPERIMENT_RESULTS.md'
    with shared.open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        f.flush()
    save(ROOT/'reports/shared_summary_updated.json',{'path':str(shared),'status':status})


def startup():
    current=source_signature()
    path=ROOT/'reports/startup.json'
    if path.exists() and read(path).get('source_signature')==current and read(path).get('passed'):
        log('Startup report matches current source; not rerunning')
        return
    if (ROOT/'manifest.json').exists():
        raise RuntimeError('Frozen source changed; preserve old experiment and use a new version')
    for directory,required in [(scratch(),32),(Path(CFG['durable_stage_a_directory']),4.5),(ROOT,2.5)]:
        directory.mkdir(parents=True,exist_ok=True)
        free=shutil.disk_usage(directory).free/2**30
        if free<required:raise RuntimeError(f'Insufficient storage at {directory}: {free:.2f} GiB < {required}; existing files untouched')
    run('prepare.py')
    run('startup.py')
    run('smoke.py','recolor')
    for family in ('llama','gemma'):
        run('smoke.py',family)
    assert source_signature()==current,'Source changed during preflight'
    save(path,{'passed':True,'source_signature':current,'native':read(ROOT/'reports/native_startup.json'),
        'algorithm_smokes':{f:read(ROOT/f'reports/smoke_{f}.json') for f in ('recolor','llama','gemma')},
        'smoke_parameters_never_used_for_formal_training':True,'volatile_optimizer_statistics_storage':True})


def main(startup_only):
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'pipeline.pid').write_text(str(os.getpid())+'\n')
    try:
        startup()
        if not (ROOT/'manifest.json').exists():freeze()
        verify()
        if startup_only:
            save(ROOT/'status.json',{'phase':'startup_checks_passed'});log('STARTUP CHECKS PASSED');return
        if not (ROOT/'evaluation/validation/native/summary.json').exists():
            run('evaluate.py','validation','--native')
        if not read(ROOT/'evaluation/validation/native/gate.json')['passed']:
            if not (ROOT/'evaluation/pilot/native/summary.json').exists():run('evaluate.py','pilot','--native')
            shared_summary('native_validation_gate_not_passed; translator training not started')
            save(ROOT/'status.json',{'phase':'stopped_at_native_gate','run_finished':True,'full_experiment_complete':False})
            log('NATIVE GATE NOT PASSED: preserved native results; no translator tuning or fusion launched');return
        for family in ('llama','gemma'):
            for stage in ('our_a','our_b','hf_fit','hf_calib'):
                marker=ROOT/f'training/{family}/{stage}/completed.json'
                # Recolor statistics/maps are volatile. Always let hf_fit check
                # their identity until its folded final map is durable.
                needs_recolor=(stage=='hf_fit' and not (ROOT/f'training/{family}/hf_calib/completed.json').exists())
                if not marker.exists() or needs_recolor:run('train.py',family,stage)
            if not (ROOT/f'evaluation/validation/{family}/summary.json').exists():
                run('evaluate.py','validation','--family',family)
        allowed=all(read(ROOT/f'evaluation/validation/{f}/gate.json')['passed'] for f in ('llama','gemma'))
        if not (ROOT/'evaluation/pilot/all/summary.json').exists():
            arguments=['pilot']+(['--fusion'] if allowed else [])
            run('evaluate.py',*arguments)
        shared_summary('completed' if allowed else 'single_source_completed; fusion_skipped_by_validation_gate',allowed)
        save(ROOT/'status.json',{'phase':'completed' if allowed else 'completed_single_source_fusion_skipped',
                                'fusion_completed':allowed,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        log('BASELINE EXPERIMENT FINISHED; no receiver training or automatic follow-up candidates')
    except Exception as error:
        failure={'phase':'failed','error':repr(error),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        save(ROOT/'status.json',failure)
        shared_summary('failed_or_incomplete',error=repr(error))
        traceback.print_exc();raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--startup-only',action='store_true');args=ap.parse_args();main(args.startup_only)
