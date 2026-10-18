"""One locked, resumable continuation followed by unchanged official evaluations."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from followup_common import HERE,save,log


def run(script,*args):
    log('PIPELINE '+script+' '+' '.join(args))
    save(HERE/'status.json',{'phase':'pipeline_step','script':script,'arguments':list(args),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    subprocess.run([sys.executable,'-u',str(HERE/script),*args],cwd=HERE,check=True)


def main(startup):
    lock=(HERE/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'pipeline.pid').write_text(str(os.getpid())+'\n')
    try:
        if startup:
            run('tests.py')
            run('prepare_followup.py')
            run('followup_train.py','smoke')
            save(HERE/'status.json',{'phase':'startup_checks_passed'})
            log('STARTUP CHECKS PASSED');return
        assert json.loads((HERE/'reports/smoke.json').read_text())['passed']
        if not (HERE/'manifest.json').exists():run('followup_train.py','freeze')
        if not (HERE/'stage_b/completed.json').exists():run('followup_train.py','train')
        if not (HERE/'diagnostics/summary.json').exists():run('diagnose.py')
        for family in ('mcq','mmlu'):
            if not (HERE/f'evaluation/{family}/new_stage_b.json').exists():run('evaluate.py',family)
        if not (HERE/'evaluation/mmlu_pro/new_stage_b.json').exists():
            for action in ('prepare','smoke','run'):run('evaluate_mmlu_pro.py',action)
        run('evaluate.py','summarize')
        save(HERE/'status.json',{'phase':'completed','updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        log('STUDENT-PREFIX FOLLOW-UP COMPLETED')
    except Exception as error:
        save(HERE/'status.json',{'phase':'failed','error':repr(error),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        traceback.print_exc();raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--startup-only',action='store_true');args=ap.parse_args();main(args.startup_only)
