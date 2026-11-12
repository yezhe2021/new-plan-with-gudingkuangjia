"""Launch E4 detached and leave a status file alongside the results."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def status(state, **fields):
    temp = ROOT / "status.tmp"
    temp.write_text(json.dumps(dict(status=state, pid=os.getpid(), time=time.strftime("%F %T"), **fields)) + "\n")
    temp.replace(ROOT / "status.json")


def run():
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase = "prepare"
        try:
            status("running", phase=phase)
            if not (ROOT / "data_manifest.json").exists():
                subprocess.run([sys.executable, "-u", "prepare.py", "quality_dev.jsonl"], cwd=ROOT, check=True)
            phase = "benchmark"
            while True:
                free = int(subprocess.check_output(
                    ["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True,
                ).strip())
                if free >= 30720:
                    break
                status("waiting_for_gpu", phase=phase, free_mib=free)
                time.sleep(30)
            status("running", phase=phase)
            subprocess.run([sys.executable, "-u", "benchmark.py"], cwd=ROOT, check=True)
            status("completed", phase=phase)
        except Exception as error:
            status("failed", phase=phase, error=repr(error))
            raise


def launch():
    with (ROOT / "launch.lock").open("a") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / "run.lock").open("a") as run_lock:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = ROOT / "status.json"
            if path.exists() and json.loads(path.read_text()).get("status") == "completed":
                raise SystemExit("E4 already completed")
        with (ROOT / "run.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "run.py"), "run"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        print("Detached E4 PID", child.pid, flush=True)


if __name__ == "__main__":
    {"launch": launch, "run": run}[sys.argv[1]]()
