"""Run the smoke, training, and evaluation stages on server B."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "static_memory"
CFG = json.loads((REFERENCE / "config.json").read_text())


def save(name, **details):
    path = ROOT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(pid=os.getpid(), time=time.strftime("%F %T"), **details), indent=2) + "\n")
    temp.replace(path)


def launch():
    with (ROOT / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / "launch.json").exists(), "Inspect the existing launch first"
        with (ROOT / "run.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", __file__, "run"], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        save("launch.json", child_pid=child.pid)
        print("Detached server B job", child.pid, flush=True)


def run():
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for phase in ("prepare", "smoke", "train", "evaluate"):
                while phase != "prepare":
                    if (ROOT / "controls/pause_requested.json").exists():
                        save("runner_status.json", status="paused", phase=phase)
                        return
                    try:
                        output = subprocess.check_output(
                            ["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            text=True, timeout=15,
                        )
                        free = int(output.strip())
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        save("runner_status.json", status="waiting_for_gpu", phase=phase, error=str(error))
                    else:
                        if free >= CFG["minimum_free_gpu_mib"]:
                            break
                        save("runner_status.json", status="waiting_for_gpu", phase=phase, free_mib=free)
                    time.sleep(30)
                save("runner_status.json", status="running", phase=phase)
                result = subprocess.run([sys.executable, "-u", "experiment.py", phase], cwd=ROOT)
                if result.returncode == 75:
                    save("runner_status.json", status="paused", phase=phase)
                    return
                result.check_returncode()
        except Exception as error:
            save("runner_status.json", status="failed", phase=phase, error=repr(error))
            raise
        save("runner_status.json", status="completed")


if __name__ == "__main__":
    {"launch": launch, "run": run}[sys.argv[1]]()
