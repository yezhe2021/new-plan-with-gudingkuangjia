"""One locked, resumable pipeline; no unattended duplicate relaunch."""
import os
import sys
import json
import time
import fcntl
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())


def run(action):
    print(f'[{time.strftime("%F %T")}] START {action}', flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), action], cwd=ROOT, check=True)


def wait_gpu():
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    while True:
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free >= CFG['queue_min_free_gpu_gib']:
            print(f'GPU READY {free:.2f}GiB free; no other job is stopped', flush=True)
            return
        state = dict(status='waiting', phase='gpu_queue', free_gpu_gib=free, required_gib=CFG['queue_min_free_gpu_gib'])
        (ROOT / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
        print(state, flush=True)
        time.sleep(CFG['queue_retry_seconds'])


with (ROOT / 'pipeline.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'completed.json').exists():
        print('Pilot already completed; no relaunch', flush=True)
        sys.exit(0)
    (ROOT / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    try:
        run('prepare')
        wait_gpu()
        for action in ['capture', 'smoke', 'train', 'evaluate', 'summarize']:
            run(action)
    except Exception as exc:
        # Stage failures append their own machine record. Also record launcher errors.
        if not (ROOT / 'failure.json').exists():
            spec = __import__('importlib.util', fromlist=['util'])
            module_spec = spec.spec_from_file_location('pilot_failure', ROOT / 'experiment.py')
            module = spec.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            error = dict(status='failed_or_incomplete', type=type(exc).__name__, message=str(exc))
            module.C.save(ROOT / 'failure.json', error)
            module.status(phase='pipeline_failure', **error)
            module.shared('failed_or_incomplete', error)
        raise
