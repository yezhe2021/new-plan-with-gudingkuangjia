"""One detached run; failures stop for review, no model training or retries."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())


def run(script, *args):
    print(time.strftime("%F %T"), "START", script, args, flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / script), *args], cwd=ROOT, check=True)


with (ROOT / "pipeline.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "completed.json").exists():
        print("Already completed", flush=True)
        sys.exit(0)
    assert not (ROOT / "failure.json").exists(), "Review previous failure before resuming"
    if not (ROOT / "manifest.json").exists():
        run("experiment.py", "prepare")
    run("experiment.py", "verify")
    for family in CFG["models"]:
        if not (ROOT / f"smoke/{family}.json").exists():
            run("experiment.py", "smoke", family)
    for family in CFG["models"]:
        run("experiment.py", "capture", family)
    run("analyze.py", "match")
    run("analyze.py", "compare")
    for family in CFG["models"]:
        run("experiment.py", "ablate", family)
    run("analyze.py", "summarize")
