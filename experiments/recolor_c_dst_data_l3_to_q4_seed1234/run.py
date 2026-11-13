"""Run the Recolor and C calibration pipeline detached on server B."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def status(phase, state, **details):
    path = ROOT / "runner_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(time=time.strftime("%F %T"), phase=phase, status=state, **details)) + "\n")
    temp.replace(path)


def run():
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in ("prepare", "fit", "train", "evaluate"):
            status(phase, "running")
            result = subprocess.run([sys.executable, "-u", "experiment.py", phase], cwd=ROOT)
            if result.returncode:
                status(phase, "paused" if result.returncode == 75 else "failed", exit_code=result.returncode)
                return result.returncode
        status("evaluation", "completed")
    return 0


def launch():
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        ).strip())
        if free < 30720:
            raise RuntimeError(f"GPU has {free} MiB free")
        with (ROOT / "run.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", __file__, "run"], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / "pid").write_text(f"{child.pid}\n")
        print(f"Detached Recolor+C PID {child.pid}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "launch":
        launch()
    else:
        sys.exit(run())
