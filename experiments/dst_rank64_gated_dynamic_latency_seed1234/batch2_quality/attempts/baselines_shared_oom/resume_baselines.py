"""Resume baseline timing with an OOM retry that covers every Native trial."""
import fcntl
from pathlib import Path

import benchmark_baselines as B

torch = B.torch


def native_test(engine, rows):
    errors = []
    for block in (None, 4096, 2048, 1024):
        torch.cuda.empty_cache()
        try:
            warm = B.measure(engine, rows, block=block)
            trials = [B.measure(engine, rows, block=block, decode=True) for _ in range(B.B.TRIALS)]
            assert all(t["first_tokens"] == warm["first_tokens"] for t in trials)
        except torch.cuda.OutOfMemoryError as exc:
            errors.append(dict(block=block, error=str(exc)))
            print("NATIVE OOM RETRY", block, str(exc).split(".")[0], flush=True)
        else:
            return dict(native_prefill_block=block, trials=trials, oom_retries=errors,
                        resume_script_sha256=B.C.sha(Path(__file__)))
    raise RuntimeError("Native prefill or decode could not fit with the smallest block")


if __name__ == "__main__":
    B.native_test = native_test
    with (B.ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            B.main()
        except Exception as exc:
            B.status("failed", error=repr(exc))
            raise
