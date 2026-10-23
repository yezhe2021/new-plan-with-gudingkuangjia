"""One-shot checkpoint-triggered validation; no changes to the trainer."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    with (out / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait(args, out)


def wait(args, out):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch

    def status(**fields):
        path = out / "watcher_status.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(), pid=os.getpid(),
                                       expected_step=args.expected_step, **fields), indent=2) + "\n")
        tmp.replace(path)

    try:
        started = time.monotonic()
        observed_step = None
        stamp = None
        snapshot = out / "reader_snapshot.pt"
        while not snapshot.exists():
            assert time.monotonic() - started < 24 * 3600, "Checkpoint wait exceeded 24 hours"
            checkpoint = args.pilot_root / "runs/reader/last.pt"
            if checkpoint.exists():
                # Inspect a stable inode, even if the trainer atomically replaces last.pt.
                with checkpoint.open("rb") as stream:
                    st = os.fstat(stream.fileno())
                    current_stamp = (st.st_ino, st.st_mtime_ns, st.st_size)
                    if current_stamp != stamp:
                        value = torch.load(stream, map_location="cpu", weights_only=True)
                        observed_step = value["step"]
                        assert observed_step <= args.expected_step, ("Requested checkpoint missed", observed_step)
                        if observed_step == args.expected_step:
                            assert value["manifest_sha256"]
                            stream.seek(0)
                            temporary = snapshot.with_suffix(".pt.tmp")
                            with temporary.open("wb") as target:
                                shutil.copyfileobj(stream, target)
                            temporary.replace(snapshot)
                        del value
                        stamp = current_stamp
            if snapshot.exists():
                break
            state_path = args.pilot_root / "status.json"
            try:
                training = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                training = {}
            assert training.get("status") not in ["failed", "failed_or_incomplete"], training
            if training.get("phase") == "reader_training" and training.get("step", 0) > args.expected_step:
                # The save is before the status transition to the next step;
                # retry once immediately in case it raced our read above.
                if current_stamp == stamp:
                    with checkpoint.open("rb") as stream:
                        newest = torch.load(stream, map_location="cpu", weights_only=True)
                    assert newest["step"] >= args.expected_step, ("Target step passed without a checkpoint", training)
                    del newest
                    stamp = None
                    continue
            status(status="waiting_checkpoint", latest_saved_step=observed_step,
                   training_step=training.get("step"), training_phase=training.get("phase"))
            time.sleep(20)
        status(status="validating", snapshot=str(snapshot))
        subprocess.run([sys.executable, "-u", str(out / "run_validation.py"),
                        "--pilot-root", str(args.pilot_root), "--expected-step", str(args.expected_step),
                        "--baseline-dir", str(args.baseline_dir)], cwd=out, check=True)
        status(status="completed")
    except Exception as exc:
        status(status="failed", error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
