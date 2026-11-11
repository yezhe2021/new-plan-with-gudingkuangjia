"""Evaluate D with all Memory keys masked at their original cache positions."""

from collections import Counter
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import sys
import time

OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT.parent))

from experiment_common import ARM, BASE, C, CFG, ROOT, done, identity, rows, same_prediction, torch, verify
from runtime_new import load
from score import score

MASKED = "D_static_masked"
QA = {"hotpotqa", "2wiki", "musique"}


@contextmanager
def mask_memory(routing, mask_calls):
    original = routing.allowed

    def allowed(phase, layer, header, body_length, memory_length, query, key_length):
        visible = original(phase, layer, header, body_length, memory_length, query, key_length)
        if phase == "answer":
            visible[:, body_length:body_length + memory_length] = False
            mask_calls[layer] += 1
        return visible

    routing.allowed = allowed
    try:
        yield
    finally:
        del routing.allowed


def save_status(status, **details):
    C.save(OUTPUT / "status.json", dict(status=status, pid=os.getpid(), time=time.strftime("%F %T"), **details))


def reference(row, pins):
    path = ROOT / "evaluation" / row["task"] / "per_sample" / (row["id"] + ".json")
    record = C.read(path)
    assert record["manifest_sha256"] == pins["d_manifest_sha256"]
    assert record["b_checkpoint_sha256"] == pins["b_checkpoint_sha256"]
    assert record["memory_checkpoint_sha256"] == pins["memory_checkpoint_sha256"]
    assert all(record[key] == row[key] for key in ("id", "question", "gold"))
    return {name: record["conditions"][name] for name in (BASE, ARM)}


def metric_keys(task):
    if task in QA:
        return ("em", "f1")
    return ("correct", "legacy_correct") if task == "gsm8k" else ("correct",)


def summarize(task, records):
    metrics = {}
    comparisons = {}
    for name in (BASE, ARM, MASKED):
        metrics[name] = {
            key: sum(r["conditions"][name][key] for r in records) / len(records)
            for key in metric_keys(task)
        }
    for before, after in ((BASE, MASKED), (MASKED, ARM), (BASE, ARM)):
        contrasts = {}
        for key in metric_keys(task):
            delta = [r["conditions"][after][key] - r["conditions"][before][key] for r in records]
            contrasts[key] = dict(
                paired=C.paired(delta),
                improved_ids=[r["id"] for r, d in zip(records, delta) if d > 0],
                declined_ids=[r["id"] for r, d in zip(records, delta) if d < 0],
            )
        comparisons[f"{after}_minus_{before}"] = contrasts
    return dict(n=len(records), metrics=metrics, comparisons=comparisons)


@torch.no_grad()
def run():
    save_status("checking")
    verify()
    assert done("completed.json") and done("runs/completed.json")
    memory_hash = C.read(ROOT / "runs/completed.json")["checkpoint_sha256"]
    assert C.sha(ROOT / "runs/epoch4_step1536.pt") == memory_hash
    pins = dict(
        d_manifest_sha256=identity(),
        b_checkpoint_sha256=C.read(ROOT / "manifest.json")["b_checkpoint_sha256"],
        memory_checkpoint_sha256=memory_hash,
        script_sha256=C.sha(__file__),
        test_sha256=C.sha(ROOT / "prepared/test.jsonl"),
    )
    engine, correction, module = load(final=True)
    module.eval().requires_grad_(False)
    engine.mode = "test"
    versions = (C.versions(correction), C.versions(module))
    layer_count = engine.model.config.num_hidden_layers
    assert layer_count == 36 and module.M == 32
    cohort = rows("test")
    assert len(cohort) == len({row["id"] for row in cohort}) == sum(CFG["test_counts"].values())
    tasks = {}
    total = len(cohort)
    progress = 0
    parity_tasks = {"arc", "hotpotqa"}
    for task, count in CFG["test_counts"].items():
        selected = [row for row in cohort if row["task"] == task]
        assert len(selected) == count
        results = []
        for sample, row in enumerate(selected, 1):
            save_status("running", task=task, sample=sample, task_total=count, completed=progress, total=total)
            output = OUTPUT / "evaluation" / task / "per_sample" / (row["id"] + ".json")
            if output.exists():
                record = C.read(output)
                assert record["pins"] == pins and record["id"] == row["id"]
            else:
                conditions = reference(row, pins)
                body = engine.cache(row, engine.base(row), correction)
                body_length = body.get_seq_length()
                cache = module.construct(row, body)
                assert body_length == len(engine.prefix_ids) + len(row["content_ids"])
                assert cache.get_seq_length() == body_length + module.M
                if task in parity_tasks and sample == 1:
                    same_prediction(score(module, row, body, rn=False), conditions[BASE], task, BASE)
                    same_prediction(score(module, row, cache, rn=True), conditions[ARM], task, ARM)
                mask_calls = Counter()
                with mask_memory(module.routing, mask_calls):
                    conditions[MASKED] = score(module, row, cache, rn=True)
                assert set(mask_calls) == set(range(1, layer_count + 1))
                assert len(set(mask_calls.values())) == 1
                record = dict(
                    pins=pins, id=row["id"], task=task, question=row["question"], gold=row["gold"],
                    body_length=body_length, answer_suffix_position=body_length + module.M,
                    masked_attention_calls_per_layer=mask_calls[1], conditions=conditions,
                )
                engine.unchanged()
                assert versions == (C.versions(correction), C.versions(module))
                assert engine.student_test_native_prefills == 0
                C.save(output, record)
                del body, cache
            results.append(record)
            progress += 1
            if sample == 1 or sample % 8 == 0:
                print(time.strftime("%F %T"), task, f"{sample}/{count}", f"total {progress}/{total}", flush=True)
        tasks[task] = summarize(task, results)
        C.save(OUTPUT / "evaluation" / task / "summary.json", dict(pins=pins, **tasks[task]))
    C.save(OUTPUT / "summary.json", dict(status="completed", pins=pins, tasks=tasks))
    save_status("completed", completed=progress, total=total)


if __name__ == "__main__":
    with (OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except Exception as error:
            save_status("failed", error=repr(error))
            raise
