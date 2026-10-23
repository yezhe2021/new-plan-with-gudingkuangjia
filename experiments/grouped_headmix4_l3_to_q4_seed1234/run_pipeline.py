"""Idempotent launcher. It starts only after the prior experiment fully completes."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def dependency_ready(directory):
    parent = Path(directory)
    if (parent / 'failure.json').exists():
        return False, 'previous_experiment_failed'
    if not (parent / 'completed.json').exists():
        return False, 'previous_experiment_not_completed'
    record = json.loads((parent / 'completed.json').read_text())
    digest = hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest()
    if record.get('manifest_sha256') != digest or record.get('status') != 'completed':
        return False, 'previous_completion_identity_mismatch'
    with (parent / 'pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, 'previous_pipeline_still_exiting'
    return True, 'previous_experiment_completed'


def readiness(root=ROOT, config=CFG):
    if (root / 'completed.json').exists():
        return dict(ready=False, reason='already_completed')
    if (root / 'failure.json').exists():
        return dict(ready=False, reason='failed_requires_review')
    if not (root / 'prepared/completed.json').exists():
        return dict(ready=False, reason='not_prepared')
    ready, reason = dependency_ready(config['wait_for_experiment'])
    if not ready:
        return dict(ready=False, reason=reason)
    free_disk = shutil.disk_usage(root).free / 2**30
    if free_disk < config['disk_reserve_gib'] + 8:
        return dict(ready=False, reason='insufficient_disk', free_disk_gib=free_disk)
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'],
                            check=True, capture_output=True, text=True)
    free = float(result.stdout.strip()) / 1024
    if free < config['queue_min_free_gpu_gib']:
        return dict(ready=False, reason='insufficient_gpu_memory', free_gpu_gib=free)
    return dict(ready=True, reason='ready', free_gpu_gib=free)


def launch():
    # Covers the small gap between spawning the runner and the runner taking its lock.
    with (ROOT / 'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'launch.json').exists():
            record = json.loads((ROOT / 'launch.json').read_text())
            proc = Path('/proc') / str(record['pid'])
            if proc.exists() and str(ROOT).encode() in (proc / 'cmdline').read_bytes():
                print(json.dumps(dict(launched=False, reason='already_running', pid=record['pid'])))
                return
            if not (ROOT / 'completed.json').exists():
                print(json.dumps(dict(launched=False, reason='previous_launch_exited_requires_review')))
                return
        state = readiness()
        if not state['ready']:
            save(ROOT / 'queue_status.json', dict(time=time.strftime('%F %T'), **state))
            print(json.dumps(dict(launched=False, **state)))
            return
        # Readiness never silently bypasses the immutable code/data validation.
        subprocess.run([sys.executable, str(ROOT / 'experiment.py'), 'verify'], check=True, cwd=ROOT)
        (ROOT / 'logs').mkdir(exist_ok=True)
        with (ROOT / 'logs/pipeline.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_pipeline.py'), '--run'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(ROOT / 'launch.json', dict(pid=process.pid, time=time.strftime('%F %T'), interpreter=sys.executable))
        print(json.dumps(dict(launched=True, pid=process.pid)))


def run():
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = readiness()
        if not state['ready']:
            raise RuntimeError('Launch preconditions changed: ' + str(state))
        save(ROOT / 'runner_status.json', dict(status='running', pid=os.getpid(), started_at=time.strftime('%F %T')))
        actions = [['smoke', 'a'], ['train', 'a'], ['smoke', 'b'], ['train', 'b']]
        actions += [['evaluate', task] for task in CFG['evaluation_tasks']]
        actions += [['summarize']]
        try:
            for args in actions:
                print(time.strftime('%F %T'), 'START', args, flush=True)
                subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), *args], cwd=ROOT, check=True)
        except Exception as exc:
            save(ROOT / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
            save(ROOT / 'runner_status.json', dict(status='failed', pid=os.getpid()))
            raise
        save(ROOT / 'runner_status.json', dict(status='completed', pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-if-ready', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.launch_if_ready:
        launch()
    elif args.run:
        run()
    else:
        print(json.dumps(readiness()))
