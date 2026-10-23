"""Server-only conditional launch; independent of SSH and the desktop app."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WAIT_REASONS = {'previous_experiment_not_completed', 'previous_pipeline_still_exiting',
                'insufficient_gpu_memory', 'insufficient_disk'}


def save(value):
    path = HERE / 'status.json'
    temp = HERE / 'status.tmp'
    temp.write_text(json.dumps(dict(time=time.strftime('%F %T'), pid=os.getpid(), **value),
                               ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def classify(record):
    if record.get('launched') or record.get('reason') == 'already_running':
        return 'started'
    if record.get('reason') == 'already_completed':
        return 'completed'
    if record.get('reason') in WAIT_REASONS:
        return 'waiting'
    return 'blocked'


def confirm_started(record):
    pid = record.get('pid')
    if pid is None:
        raise RuntimeError('Launcher did not report a runner PID')
    # Bounded startup handshake only; never monitor subsequent training progress.
    for _ in range(15):
        failure = ROOT / 'failure.json'
        if failure.exists():
            raise RuntimeError('Training failed during startup: ' + failure.read_text()[:1000])
        proc = Path('/proc') / str(pid) / 'cmdline'
        if not proc.exists() or str(ROOT).encode() not in proc.read_bytes():
            raise RuntimeError('Launched pipeline exited before startup confirmation')
        path = ROOT / 'runner_status.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('pid') == pid and state.get('status') == 'running':
                return
        time.sleep(1)
    raise RuntimeError('Runner startup confirmation timed out')


def run():
    with (HERE / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        try:
            while True:
                result = subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py'), '--launch-if-ready'],
                                        cwd=ROOT, check=True, capture_output=True, text=True)
                lines = [line for line in result.stdout.splitlines() if line.strip()]
                if not lines:
                    raise RuntimeError('Launcher returned no status')
                record = json.loads(lines[-1])
                state = classify(record)
                if state == 'started':
                    confirm_started(record)
                save(dict(status=state, launcher=record))
                marker = (state, record.get('reason'))
                if marker != previous:
                    print(time.strftime('%F %T'), json.dumps(dict(status=state, launcher=record)), flush=True)
                    previous = marker
                if state != 'waiting':
                    return
                time.sleep(60)
        except Exception as exc:
            save(dict(status='failed', error=dict(type=type(exc).__name__, message=str(exc))))
            raise


if __name__ == '__main__':
    run()
