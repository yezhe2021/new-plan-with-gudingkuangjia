"""One background pipeline; train only the requested new Stage B and evaluate it."""
import fcntl
import json
import os
import subprocess
import sys
import time

from runtime import CONFIG, HERE, PARENT, log, save, verify_manifest


def status(phase, **kwargs):
    save(HERE / "status.json", {"phase": phase, "pid": os.getpid(),
                               "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kwargs})


def command(script, *args):
    log("START " + script + " " + " ".join(args))
    subprocess.run([sys.executable, "-u", str(HERE / script), *args], cwd=PARENT, check=True)
    log("DONE " + script + " " + " ".join(args))


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    lock = (HERE / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_manifest()
    if not json.loads((HERE / "smoke.json").read_text()).get("passed"):
        raise RuntimeError("GPU smoke gate has not passed")
    try:
        if not (HERE / "stage_b/completed.json").exists():
            status("training_new_stage_b")
            command("train_stage_b.py", "train")
        for family in CONFIG["evaluation_families"]:
            if not (HERE / f"evaluation/{family}/new_stage_b.json").exists():
                status("evaluation_" + family)
                command("evaluate_stage_b.py", "evaluate", "--family", family)
            command("evaluate_stage_b.py", "summarize")
        status("completed")
        log("ALL NEW STAGE B TRAINING AND EVALUATION COMPLETED")
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
