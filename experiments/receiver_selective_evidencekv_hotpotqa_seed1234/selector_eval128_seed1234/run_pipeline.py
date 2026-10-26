"""Immediate, detached, single-run launcher; never waits for or changes other experiments."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():
            old=json.loads((ROOT/'launch.json').read_text());p=Path('/proc')/str(old['pid'])
            active=p.exists() and str(ROOT).encode() in (p/'cmdline').read_bytes()
            print(json.dumps(dict(launched=False,reason='already_running' if active else 'already_launched_requires_review',pid=old['pid'])))
            return
        assert not (ROOT/'failure.json').exists(),'Existing failure requires review'
        assert (ROOT/'prepared/completed.json').exists(),'Prepare data first'
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(ROOT/'launch.json',dict(pid=p.pid,time=time.strftime('%F %T'),interpreter=sys.executable))
        print(json.dumps(dict(launched=True,pid=p.pid)))


def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(ROOT/'runner_status.json',dict(status='running',pid=os.getpid(),started_at=time.strftime('%F %T')))
        try:
            for action in ['capture','translate','preflight','evaluate','summarize']:
                print(time.strftime('%F %T'),'START',action,flush=True)
                subprocess.run([sys.executable,'-u',str(ROOT/'run_experiment.py'),action],cwd=ROOT,check=True)
            save(ROOT/'runner_status.json',dict(status='completed',pid=os.getpid(),completed_at=time.strftime('%F %T')))
        except Exception as exc:
            save(ROOT/'runner_status.json',dict(status='failed',pid=os.getpid(),error=str(exc),time=time.strftime('%F %T')))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    elif args.launch:launch()
    else:parser.error('Use --launch or --run')
