"""Independent, fixed-checkpoint validation alongside the unchanged pilot."""
import os

for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
import fcntl
import importlib.util
import json
import shutil
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=208)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    with (out / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out)


def run(args, out):
    spec = importlib.util.spec_from_file_location("checkpoint_validation_pilot", args.pilot_root / "experiment.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    C, U, P, torch = E.C, E.U, E.P, E.torch

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        temporary.replace(path)

    def status(**fields):
        save(out / "status.json", dict(time=time.strftime("%F %T"), pid=os.getpid(), **fields))

    def log(message):
        print(f"[{time.strftime('%F %T')}] {message}", flush=True)

    try:
        status(status="preparing")
        E.verify()
        validation = E.rows("validation")
        assert len(validation) == 64 and all(r["task"] == "gsm8k" for r in validation)
        keys = {r["question_key"] for r in validation}
        assert len(keys) == 64
        assert not keys & {r["question_key"] for r in E.rows("train")}
        assert not keys & {r["question_key"] for r in E.rows("test")}
        assert all(U.source_path(E.source_row(r)).exists() for r in validation)
        assert C.sha(E.initial_path()) == E.CFG["reader_initial_sha256"]

        # Read one inode, so an atomic training checkpoint replacement cannot
        # produce a mixed snapshot. Never modify or load weights in the trainer.
        snapshot = out / "reader_snapshot.pt"
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".pt.tmp")
            with (E.ROOT / "runs/reader/last.pt").open("rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            candidate = torch.load(temporary, map_location="cpu", weights_only=True)
            assert candidate["step"] == args.expected_step, candidate["step"]
            assert candidate["manifest_sha256"] == E.identity()
            del candidate
            temporary.replace(snapshot)
        checkpoint = torch.load(snapshot, map_location="cpu", weights_only=True)
        assert checkpoint["step"] == args.expected_step
        assert checkpoint["manifest_sha256"] == E.identity()
        identity = dict(
            pilot_manifest_sha256=E.identity(),
            validation_sha256=C.sha(E.ROOT / "prepared/validation.jsonl"),
            initial_reader_sha256=C.sha(E.initial_path()),
            checkpoint_step=checkpoint["step"],
            checkpoint_sha256=C.sha(snapshot),
            script_sha256=C.sha(Path(__file__)),
            ordered_question_ids=[r["id"] for r in validation],
            max_new_tokens=4096,
            temperature=0.0,
            gpu_allocator_limit_gib=11,
            scope="64 existing protected validation questions; diagnostic only, not final test or checkpoint selection",
            protocol="Existing FP16 model, Full-Sync KV, thinking=False, greedy generation, pinned numeric parser",
        )
        assert E.CFG["formal_max_new_tokens"] == identity["max_new_tokens"]
        baseline_manifest = json.loads((args.baseline_dir / "run.json").read_text())
        for field in ["pilot_manifest_sha256", "validation_sha256", "initial_reader_sha256",
                      "ordered_question_ids", "max_new_tokens", "temperature", "protocol"]:
            assert baseline_manifest[field] == identity[field], ("Baseline protocol differs", field)
        assert (args.baseline_dir / "completed.json").exists()
        baseline_records = {}
        baseline_hashes = {}
        for row in validation:
            path = args.baseline_dir / "per_sample" / (U.fname(row) + ".json")
            previous = json.loads(path.read_text())
            assert previous["run_id"] == baseline_manifest["run_id"]
            for field in ["id", "prompt", "gold"]:
                assert previous[field] == row[field], (field, row["id"])
            assert "initial_reader" in previous["conditions"]
            baseline_records[row["id"]] = previous["conditions"]["initial_reader"]
            baseline_hashes[row["id"]] = C.sha(path)
        identity["baseline_reuse"] = dict(run_id=baseline_manifest["run_id"],
                                          path=str(args.baseline_dir), per_sample_sha256=baseline_hashes)
        run_id = C.digest(identity)
        run_manifest = dict(identity, run_id=run_id)
        if (out / "run.json").exists():
            assert json.loads((out / "run.json").read_text()) == run_manifest
        else:
            save(out / "run.json", run_manifest)
        if args.prepare_only:
            status(status="prepared", checkpoint_step=checkpoint["step"], total=64, run_id=run_id)
            log(f"PREFLIGHT PASSED step={checkpoint['step']} validation=64 run_id={run_id}")
            return
        if (out / "completed.json").exists():
            assert json.loads((out / "completed.json").read_text())["run_id"] == run_id
            log("Already completed")
            return

        # Bound this process's allocator, leaving the trainer's current 17.3 GiB
        # plus >3 GiB spare. Refuse to start if initial headroom has disappeared.
        assert torch.cuda.is_available()
        free, total_memory = torch.cuda.mem_get_info()
        assert free / 2**30 >= 13, ("Insufficient headroom for concurrent validation", free / 2**30)
        torch.cuda.set_per_process_memory_fraction(11 * 2**30 / total_memory)
        status(status="loading_model", checkpoint_step=checkpoint["step"], total=64, run_id=run_id)
        tok, model, base, writer, reader = E.load_all()
        model.eval()
        for parameter in reader.parameters():
            parameter.requires_grad_(False)
        # Keep an immutable CPU baseline while switching the controller's state.
        initial = {name: value.detach().cpu().clone() for name, value in reader.state_dict().items()}
        states = [("initial_reader", initial), ("checkpoint_reader", checkpoint["state"])]
        runtime = C.runtime(model, tok)
        started = time.monotonic()

        def summarize():
            records = []
            for row in validation:
                path = out / "per_sample" / (U.fname(row) + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    assert record["run_id"] == run_id and record["id"] == row["id"]
                    records.append(record)
            metrics = {}
            for arm, _ in states:
                values = [r["conditions"][arm] for r in records if arm in r["conditions"]]
                if values:
                    metrics[arm] = dict(n=len(values), correct=sum(v["correct"] for v in values),
                                        accuracy=sum(v["correct"] for v in values) / len(values),
                                        cap_count=sum(v["reached_cap"] for v in values),
                                        mean_generated_tokens=sum(v["generated_tokens"] for v in values) / len(values))
            pairs = [(r["conditions"]["initial_reader"]["correct"], r["conditions"]["checkpoint_reader"]["correct"])
                     for r in records if len(r["conditions"]) == 2]
            paired = dict(n=len(pairs), both_correct=sum(a and b for a, b in pairs),
                          initial_only=sum(a and not b for a, b in pairs),
                          checkpoint_only=sum(b and not a for a, b in pairs),
                          both_wrong=sum(not a and not b for a, b in pairs))
            if pairs:
                paired["accuracy_delta_pp"] = 100 * sum(b - a for a, b in pairs) / len(pairs)
            summary = dict(run_id=run_id, checkpoint_step=checkpoint["step"], total=64,
                           completed_pairs=len(pairs), metrics=metrics, paired=paired,
                           elapsed_seconds=time.monotonic() - started,
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                           caveat="Existing validation set, partial figures provisional; final step256/test128 protocol unchanged")
            save(out / "summary.json", summary)
            return summary

        with torch.no_grad():
            for index, raw in enumerate(validation, 1):
                row = E.source_row(raw)
                path = out / "per_sample" / (U.fname(row) + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    assert record["run_id"] == run_id and record["id"] == row["id"]
                else:
                    record = dict(id=row["id"], prompt=row["prompt"], gold=row["gold"],
                                  conditions={"initial_reader": baseline_records[row["id"]]}, run_id=run_id,
                                  initial_reader_reused_from=baseline_manifest["run_id"])
                    save(path, record)
                if len(record["conditions"]) == 2:
                    continue
                status(status="running", sample=index, total=64, arm="fixed_state", run_id=run_id,
                       checkpoint_step=checkpoint["step"])
                native, k, v = P.fixed_state(model, base, writer, reader, row)
                for arm, state in states:
                    if arm in record["conditions"]:
                        continue
                    status(status="running", sample=index, total=64, arm=arm, run_id=run_id,
                           checkpoint_step=checkpoint["step"])
                    reader.load_state_dict(state)
                    with reader.mode(True):
                        text, count = C.R.generate(
                            runtime, torch.tensor([row["suffix_ids"]], device="cuda"), 4096,
                            past_key_values=U.translated_cache(model, row, native, k, v),
                            past_length=len(row["frames"]["qwen"]["body_ids"]), temperature=0.0)
                    prediction = C.S.extract_numeric_answer(text)
                    record["conditions"][arm] = dict(
                        generation=text, prediction=prediction,
                        correct=int(prediction == C.S.normalize_number(row["gold"])),
                        generated_tokens=count, reached_cap=int(count >= 4096))
                    save(path, record)
                    summary = summarize()
                    log(f"VALIDATION sample={index}/64 arm={arm} correct={record['conditions'][arm]['correct']} "
                        f"tokens={count} completed_pairs={summary['completed_pairs']}")
                del native, k, v
                torch.cuda.empty_cache()

        summary = summarize()
        assert summary["completed_pairs"] == 64
        save(out / "completed.json", dict(run_id=run_id, time=time.strftime("%F %T"), summary=summary))
        status(status="completed", total=64, completed_pairs=64, checkpoint_step=checkpoint["step"], run_id=run_id)
        log("ALL 64 PAIRED VALIDATION QUESTIONS COMPLETED " + json.dumps(summary["metrics"]))
    except Exception as exc:
        status(status="failed", error_type=type(exc).__name__, message=str(exc))
        save(out / "failure.json", dict(time=time.strftime("%F %T"), type=type(exc).__name__,
                                       message=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
