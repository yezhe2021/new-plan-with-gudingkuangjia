"""Detached server A workflow for Gemma receiver DST and its evaluation."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(state, **details):
    temporary = ROOT / "runner_status.tmp"
    temporary.write_text(json.dumps(dict(status=state, pid=os.getpid(), time=time.strftime("%F %T"), **details)) + "\n")
    temporary.replace(ROOT / "runner_status.json")


def run():
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = [
            ["prepare.py"],
            ["experiment.py", "smoke", "stage1"],
            ["experiment.py", "train", "stage1"],
            ["experiment.py", "smoke", "stage2"],
            ["experiment.py", "train", "stage2"],
            ["experiment.py", "evaluate"],
        ]
        try:
            for action in plan:
                if action[0] != "prepare.py":
                    while True:
                        if (ROOT / "controls/pause_requested.json").exists():
                            save("paused", action=action)
                            return
                        output = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.free",
                                                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
                        if output.returncode == 0 and int(output.stdout.strip()) >= 30720:
                            break
                        save("waiting_for_gpu", action=action, nvml=output.stdout.strip() or output.stderr.strip())
                        time.sleep(30)
                save("running", action=action)
                result = subprocess.run([sys.executable, "-u", *action], cwd=ROOT)
                if result.returncode == 75:
                    save("paused", action=action)
                    return
                result.check_returncode()
        except Exception as error:
            save("failed", action=action, error=repr(error))
            raise
        save("completed")


def launch():
    with (ROOT / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / "run.lock").open("a") as worker:
            fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (ROOT / "run.log").open("a") as log:
                child = subprocess.Popen([sys.executable, "-u", str(ROOT / "run.py"), "run"],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (ROOT / "pid").write_text(str(child.pid) + "\n")
            print("Detached Qwen→Gemma DST PID", child.pid, flush=True)


if __name__ == "__main__":
    {"launch": launch, "run": run}[sys.argv[1]]()
