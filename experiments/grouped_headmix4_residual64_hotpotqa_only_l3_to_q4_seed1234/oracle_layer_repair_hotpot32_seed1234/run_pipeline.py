"""Detached Oracle layer-repair evaluation; stop on failure."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def launch():
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / 'launch.json').exists(), 'Existing launch; inspect before restart'
        free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'], text=True).strip()) / 1024
        assert free >= 24, ('GPU busy', free)
        (ROOT / 'logs').mkdir(exist_ok=True)
        with (ROOT / 'logs/pipeline.log').open('a') as stream:
            p = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_pipeline.py'), 'run'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=p.pid, started_at=time.strftime('%F %T'), detached=True, training=False, interpreter=sys.executable)
        save('launch.json', record)
        print(json.dumps(record))


def run():
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for action in [['prepare'], ['evaluate']]:
                phase = ' '.join(action)
                save('runner_status.json', dict(status='running', phase=phase, pid=os.getpid(), time=time.strftime('%F %T')))
                print(time.strftime('%F %T'), 'START', phase, flush=True)
                subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), *action], cwd=ROOT, check=True)
        except Exception as exc:
            record = dict(status='failed', type=type(exc).__name__, message=str(exc), time=time.strftime('%F %T'))
            save('failure.json', record)
            save('runner_status.json', record)
            save('status.json', record)
            raise
        save('runner_status.json', dict(status='completed', pid=os.getpid(), finished_at=time.strftime('%F %T')))


if __name__ == '__main__':
    if sys.argv[1] == 'launch':
        launch()
    elif sys.argv[1] == 'run':
        run()
    else:
        raise ValueError(sys.argv[1])
