"""Measure numerically flagged arms for diagnosis, keeping parity flags."""

import fcntl
import json
from pathlib import Path
import random

import sweep
from sweep import ChunkBenchmark, MODES, TRIALS, probe, save, sha, torch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "diagnostic_timings"


@torch.no_grad()
def main():
    original = json.loads((ROOT / "summary.json").read_text())
    assert original["status"] == "completed"
    manifest = original["manifest"]
    assert manifest["script_sha256"] == sha(ROOT / "sweep.py")
    data = json.loads((ROOT.parent / "data_manifest.json").read_text())
    assert data["input_sha256"] == sha(ROOT.parent / "inputs.json")
    selected = {r["id"]: r for r in json.loads((ROOT.parent / "inputs.json").read_text()) if r["id"] in manifest["input_ids"]}
    bench = ChunkBenchmark()
    OUTPUT.mkdir(exist_ok=True)
    records = []
    for row_id in manifest["input_ids"]:
        record = json.loads((ROOT / (row_id + ".json")).read_text())
        row = selected[row_id]
        source = bench.capture(row, "dst_rank64")
        variants = record["variants"]
        flagged = [name for name, r in variants.items() if r["status"] == "parity_failed"]
        for name in flagged:
            variants[name].update(trials={m: [] for m in MODES}, numerical_flag="original tolerance exceeded")
        fresh_native = {m: [] for m in MODES}
        for trial in range(TRIALS):
            schedule = [(name, mode) for name in ["native", *flagged] for mode in MODES]
            random.Random(f"{row_id}:diagnostic:{trial}").shuffle(schedule)
            for name, mode in schedule:
                if name == "native":
                    native_mode = "ready" if mode == "ready" else "text_ttft"
                    fresh_native[mode].append(bench.measure(row, "native", native_mode))
                    continue
                arm = variants[name]
                if arm["status"] == "oom":
                    continue
                path, chunk = name.rsplit("_", 1)
                bench.path, bench.chunk = path, int(chunk)
                result = probe(lambda: bench.measure(row, "dst_rank64", mode, source))
                if result is None:
                    arm.update(status="oom", stage="diagnostic_timing")
                else:
                    arm["trials"][mode].append(result)
            print(row_id, "trial", trial + 1, "/", TRIALS, flush=True)
        for name in flagged:
            arm = variants[name]
            if arm["status"] == "oom":
                continue
            path, chunk = name.rsplit("_", 1)
            bench.path, bench.chunk = path, int(chunk)
            profile = probe(lambda: bench.measure(row, "dst_rank64", "handoff_ttft", source, split=True))
            if profile is None:
                arm.update(status="oom", stage="diagnostic_profile")
            else:
                arm.update(status="completed", profile=profile["phase_ms"])
        record["fresh_native"] = fresh_native
        save(OUTPUT / (row_id + ".json"), record)
        records.append(record)
        del source
    manifest["diagnostics"] = "Flagged variants timed in a separate pass; original parity measurements retained per article"
    sweep.ROOT = OUTPUT
    sweep.summarize(records, manifest, complete=True)
    summary = json.loads((OUTPUT / "summary.json").read_text())
    for name, values in summary["variants"].items():
        values["articles_with_numerical_flag"] = sum("numerical_flag" in r["variants"][name] for r in records)
    save(OUTPUT / "summary.json", summary)
    print("Supplement complete", flush=True)


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
