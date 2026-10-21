"""Single ordered job; no GPU work while the preceding UFD job is alive."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import sys
import json
import time
import fcntl
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
PARENT = Path(CFG['parent'])


def run(*args):
    print(f'[{time.strftime("%F %T")}] START {args}', flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), *args], cwd=ROOT, check=True)


def parent_alive():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        except (OSError, PermissionError):
            continue
        if str(PARENT / 'experiment.py') in command or str(PARENT / 'run_pipeline.py') in command:
            return True
    return False


def wait_for_gpu():
    last_reason = None
    while True:
        alive = parent_alive()
        ready = (PARENT / 'completed.json').exists() and not alive
        output = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        free = float(output.stdout.splitlines()[0]) / 1024 if output.returncode == 0 else 0
        reason = 'previous_ufd_pipeline_not_finished' if not ready else 'insufficient_free_gpu'
        if ready and free >= CFG['queue_min_free_gpu_gib']:
            import torch
            if torch.cuda.is_available():
                print(f'[{time.strftime("%F %T")}] GPU READY {free:.2f} GiB free; prior UFD pipeline completed', flush=True)
                return
            reason = 'cuda_unavailable'
        state = dict(time=time.strftime('%F %T'), status='waiting', phase='ordered_gpu_queue', reason=reason,
                     parent_alive=alive, parent_completed=(PARENT / 'completed.json').exists(), free_gpu_gib=free,
                     retry_seconds=CFG['queue_retry_seconds'], no_gpu_models_loaded=True)
        (ROOT / 'queue_status.json').write_text(json.dumps(state, indent=2) + '\n')
        if reason != last_reason:
            print(f'[{time.strftime("%F %T")}] QUEUED {state}', flush=True)
            last_reason = reason
        if not alive and (PARENT / 'failure.json').exists() and not ready:
            raise RuntimeError('Previous UFD experiment failed; do not preempt or blindly restart it. See parent failure.json.')
        time.sleep(CFG['queue_retry_seconds'])


with (ROOT / 'pipeline.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run('prepare')
        wait_for_gpu()
        run('smoke')
        run('train')
        for task in ['arc_challenge', 'openbookqa', 'hellaswag', 'mmlu_pro', 'sciq', 'gsm8k']:
            run('evaluate', task)
        run('summarize')
    except Exception as exc:
        # Stage subprocesses record their own failures. Also retain queue/launch
        # errors instead of silently leaving a stale waiting status.
        if not (ROOT / 'failure.json').exists():
            failure = dict(time=time.strftime('%F %T'), status='failed_or_incomplete',
                           type=type(exc).__name__, message=str(exc))
            (ROOT / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
            (ROOT / 'status.json').write_text(json.dumps(failure, indent=2) + '\n')
            record = dict(experiment=CFG['name'], config=CFG, checkpoint_selection=CFG['selection'],
                          raw_results=str(ROOT / 'evaluation/per_sample.jsonl'),
                          log=str(ROOT / 'logs/pipeline.log'), **failure)
            with (ROOT.parent / 'EXPERIMENT_RESULTS.md').open('a', encoding='utf-8') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write('\n\n## ' + CFG['name'] + '\n\n```json\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n```\n')
        raise
