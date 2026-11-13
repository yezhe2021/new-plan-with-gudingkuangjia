"""Start the RULER/HELMET workflow in a detached server B process."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(name, **details):
    target = ROOT / name
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(pid=os.getpid(), time=time.strftime("%F %T"), **details)) + "\n")
    temp.replace(target)


def prepare_helmet():
    sources = ROOT / "sources"
    source = sources / "helmet_hotpotqa_k105.jsonl"
    while not source.exists():
        pid = int((sources / "helmet_download.pid").read_text())
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split(")", 1)[1].split()[0] == "Z":
            raise RuntimeError("HELMET download stopped before extraction. See sources/helmet_download.log")
        partial = sources / "helmet_data.tar.gz.part"
        archive = sources / "helmet_data.tar.gz"
        save("helmet_prepare_status.json", phase="download_or_extract",
             bytes=next((p.stat().st_size for p in (partial, archive) if p.exists()), 0))
        time.sleep(15)
    save("helmet_prepare_status.json", phase="construct_cohort")
    subprocess.run([sys.executable, "-u", "build_cohort.py", "helmet"], cwd=ROOT, check=True)
    save("helmet_prepare_status.json", phase="completed")


def run():
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase = "prepare"
        try:
            with (ROOT / "helmet_prepare.log").open("a") as log:
                helmet = subprocess.Popen([sys.executable, "-u", __file__, "prepare_helmet"], cwd=ROOT,
                                          stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            save("runner_status.json", status="running", phase=phase)
            subprocess.run([sys.executable, "-u", "build_cohort.py", "ruler"], cwd=ROOT, check=True)
            for task in ("ruler", "helmet"):
                phase = "evaluate_" + task
                if task == "helmet":
                    save("runner_status.json", status="waiting_for_helmet_cohort", phase=phase)
                    if helmet.wait():
                        raise RuntimeError("HELMET cohort preparation failed. See helmet_prepare.log")
                summary = ROOT / "evaluation" / task / "summary.json"
                if summary.exists() and json.loads(summary.read_text())["status"] == "completed":
                    continue
                while True:
                    if (ROOT / "controls/pause_requested.json").exists():
                        save("runner_status.json", status="paused", phase=phase)
                        return
                    free = int(subprocess.check_output(
                        ["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                        text=True, timeout=15).strip())
                    if free >= 30720:
                        break
                    save("runner_status.json", status="waiting_for_gpu", phase=phase, free_mib=free)
                    time.sleep(30)
                save("runner_status.json", status="running", phase=phase)
                result = subprocess.run([sys.executable, "-u", "evaluate.py", task], cwd=ROOT)
                if result.returncode == 75:
                    save("runner_status.json", status="paused", phase=phase)
                    return
                result.check_returncode()
            phase = "plot"
            subprocess.run([sys.executable, "-u", "plot.py"], cwd=ROOT, check=True)
        except Exception as error:
            save("runner_status.json", status="failed", phase=phase, error=repr(error))
            raise
        save("runner_status.json", status="completed", phase="all")


def launch():
    with (ROOT / "launch.lock").open("a") as lock, (ROOT / "run.lock").open("a") as worker:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / "run.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", __file__, "run"], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / "pid").write_text(str(child.pid) + "\n")
        print("Detached RULER + HELMET PID", child.pid, flush=True)


if __name__ == "__main__":
    {"launch": launch, "run": run, "prepare_helmet": prepare_helmet}[sys.argv[1]]()
