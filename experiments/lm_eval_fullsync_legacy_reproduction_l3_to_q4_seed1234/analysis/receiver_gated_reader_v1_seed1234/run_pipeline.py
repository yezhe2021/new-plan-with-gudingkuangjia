"""Exclusive, resumable startup -> one reader training -> original official tests."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from experiment import HERE, PARENT, CONFIG, save, log, digest


def run(script, *args):
    log('PIPELINE ' + script + ' ' + ' '.join(args))
    save(HERE / 'status.json', {'phase': 'pipeline_step', 'script': script, 'arguments': list(args),
                              'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    subprocess.run([sys.executable, '-u', str(HERE / script), *args], cwd=HERE, check=True)


def summary():
    from experiment import verify_manifest, PREVIOUS
    verify_manifest()
    report = {'experiment': HERE.name, 'completed': True, 'server': 'B', 'configuration': CONFIG,
              'pilot_subset_not_full_benchmark': True, 'selection': json.loads((HERE / 'reader/selection.json').read_text()),
              'checkpoint_sha256': digest(HERE / 'reader/best.pt'), 'families': {},
              'diagnostics': json.loads((HERE / 'diagnostics/summary.json').read_text())}
    for family in ('mcq', 'mmlu', 'mmlu_pro'):
        new = json.loads((HERE / f'evaluation/{family}/new_reader.json').read_text())
        old = json.loads((PREVIOUS / f'evaluation/{family}/new_stage_b.json').read_text())
        report['families'][family] = {'reader': new['results'], 'frozen_translator_baseline': old['results']}
    save(HERE / 'summary.json', report)
    text = '\n## ' + HERE.name + '\n\n' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    text += '\n\nCompleted. Server B; seed 1234. Frozen latest translator; reader only. '
    text += 'Same predecessor training/validation splits: 256 references, 24 held-out references, 32 held-out fixed prefixes; 128/128/128/128 benchmark pilots. '
    text += 'Checkpoint selected on held-out mixed loss with epoch zero eligible.\n\n'
    text += '```json\n' + json.dumps(report, ensure_ascii=False, indent=2) + '\n```\n'
    text += '\nRaw results: ' + str(HERE / 'evaluation') + '; log: ' + str(HERE / 'logs/pipeline.log') + '\n'
    shared = PARENT.parents[0] / 'EXPERIMENT_RESULTS.md'
    shared.parent.mkdir(exist_ok=True, parents=True)
    with shared.open('a', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if not (HERE / 'reports/shared_summary_appended.json').exists():
            f.write(text)
            f.flush()
            save(HERE / 'reports/shared_summary_appended.json', {'path': str(shared), 'completed': True})


def main(startup_only):
    lock = (HERE / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (HERE / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    try:
        if not (HERE / 'manifest.json').exists():
            run('train.py', 'prepare')
            run('tests.py')
            run('train.py', 'smoke')
            run('train.py', 'freeze')
        if startup_only:
            save(HERE / 'status.json', {'phase': 'startup_checks_passed'})
            log('STARTUP CHECKS PASSED')
            return
        if not (HERE / 'reader/completed.json').exists():
            run('train.py', 'train')
        for family in ('mcq', 'mmlu'):
            if not (HERE / f'evaluation/{family}/new_reader.json').exists():
                run('evaluate.py', family)
        if not (HERE / 'evaluation/mmlu_pro/new_reader.json').exists():
            for action in ('prepare', 'smoke', 'run'):
                run('evaluate_mmlu_pro.py', action)
        if not (HERE / 'diagnostics/summary.json').exists():
            run('diagnose.py')
        summary()
        save(HERE / 'status.json', {'phase': 'completed', 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        log('EXTERNAL READER EXPERIMENT COMPLETED')
    except Exception as error:
        status = {'phase': 'failed', 'error': repr(error), 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        save(HERE / 'status.json', status)
        with (PARENT.parents[0] / 'EXPERIMENT_RESULTS.md').open('a', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write('\n\n' + HERE.name + ': FAILED/INCOMPLETE ' + json.dumps(status) + '; log=' + str(HERE / 'logs/pipeline.log') + '\n')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--startup-only', action='store_true')
    main(ap.parse_args().startup_only)
