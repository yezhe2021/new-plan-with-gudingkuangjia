"""Smoke checks, matched training, and paired held-out evaluation."""

from contextlib import contextmanager
import sys

import gated_common as G
from gated_common import ARM, C, CFG, ROOT, baseline, done, mark, paused, reference, rows, status, torch, verify
from gated_memory import GatedMemory, load
from score import score


def matched_runtime():
    runtime = reference.Z.scoped_module("gated_matched_runtime", reference.ROOT / "runtime_new.py", G)
    runtime.load = load
    return runtime


@contextmanager
def training_runtime(runtime):
    previous = sys.modules.get("runtime_new")
    sys.modules["runtime_new"] = runtime
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["runtime_new"]
        else:
            sys.modules["runtime_new"] = previous


def smoke():
    verify()
    if done("smoke/completed.json"):
        return
    engine, correction, module = load()
    original = reference.Z.DE.BodyReader(engine)
    engine.mode = "test"
    parity_tasks = ("arc", "hotpotqa")
    module.eval()
    with torch.no_grad():
        assert torch.count_nonzero(module.gates) == 0
        for task in parity_tasks:
            row = next(row for row in rows("test") if row["task"] == task)
            body = engine.cache(row, engine.base(row), correction)
            attached = module.construct(row, body)
            assert attached.body is body
            native_path = original.logits(row, body, positions=(0,), rn=False)
            gated_path = module.logits(row, attached, positions=(0,), rn=True)
            assert torch.equal(gated_path, native_path), (task, float((gated_path - native_path).abs().max()))
            actual = score(module, row, attached, rn=True)
            reference.same_prediction(actual, baseline(row)[reference.BASE], task, reference.BASE)
            print("Zero-gate parity:", task, flush=True)
            del body, attached, native_path, gated_path
    engine.mode = "train"
    module.train()
    runtime = matched_runtime()
    optimizer = torch.optim.AdamW(module.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = G.T.scaler_for()
    updates = []
    initial_gate = module.gates.detach().clone()
    for step, batch in enumerate(runtime.batches(1)[:2], 1):
        if paused():
            raise SystemExit(75)
        _, outcome, retries = runtime.backward_batch(engine, correction, module, optimizer, scaler, batch, 1, step, scope="smoke")
        assert torch.isfinite(module.gates).all() and not torch.equal(initial_gate, module.gates)
        updates.append(dict(step=step, retries=retries, **outcome))
        print("Gated smoke update:", step, "gradient norm:", outcome["gradient_norm"], flush=True)

    # Use the longest training row of each task to verify the actual full
    # rationale path and gradient flow through both checkpointed forwards.
    gradients = {}
    for task in CFG["train_counts"]:
        row = max((row for row in rows("train") if row["task"] == task),
                  key=lambda row: len(row["content_ids"]) + len(row["gold_tokens"]))
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            body = engine.cache(row, engine.base(row), correction)
            native = engine.native_cache(row)
        loss, _ = runtime.losses(engine, module, row, body, native, "gold", 1)
        loss.backward()
        parameter_gradients = {
            name: float(param.grad.float().norm()) for name, param in module.named_parameters()
            if param.grad is not None
        }
        assert set(parameter_gradients) == {name for name, _ in module.named_parameters()}
        assert all(torch.isfinite(param.grad).all() for param in module.parameters())
        assert parameter_gradients["gates"] > 0 and parameter_gradients["memory.prompt"] > 0
        assert any(norm > 0 for name, norm in parameter_gradients.items() if name.startswith("memory.adapters."))
        gradients[task] = dict(id=row["id"], norms=parameter_gradients)
        del loss, body, native
        engine.unchanged()
        print("Gated longest-row gradients:", task, flush=True)
    mark("smoke/completed.json", status="completed", zero_gate_exact_logits=list(parity_tasks),
         disposable_updates=updates, gradients=gradients, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def train():
    runtime = matched_runtime()
    with training_runtime(runtime):
        trainer = reference.Z.scoped_module("gated_matched_training", reference.ROOT / "train_new.py", G)
    trainer.main()


@torch.no_grad()
def evaluate():
    verify()
    assert done("runs/completed.json")
    if done("completed.json"):
        return
    engine, correction, module = load(final=True)
    module.eval().requires_grad_(False)
    engine.mode = "test"
    versions = (C.versions(correction), C.versions(module))
    gates = module.gates.tanh().tolist()
    names = (reference.BASE, reference.ARM, ARM, "native")
    tasks = {}
    lines = ["# Independent gated Static Memory", "",
             "| Dataset | Metric | B | D | Gated | Qwen Native |",
             "|---|---|---:|---:|---:|---:|"]
    for task, count in CFG["test_counts"].items():
        selected = [row for row in rows("test") if row["task"] == task]
        assert len(selected) == count
        records = []
        for sample, row in enumerate(selected, 1):
            if paused():
                raise SystemExit(75)
            path = f'evaluation/{task}/per_sample/{row["id"]}.json'
            if not (ROOT / path).exists():
                body = engine.cache(row, engine.base(row), correction)
                cache = module.construct(row, body)
                conditions = baseline(row)
                conditions[ARM] = score(module, row, cache, rn=True)
                mark(path, id=row["id"], question=row["question"], gold=row["gold"], conditions=conditions)
                del body, cache
            record = C.read(ROOT / path)
            assert record["manifest_sha256"] == G.identity() and record["id"] == row["id"]
            records.append(record)
            engine.unchanged()
            assert versions == (C.versions(correction), C.versions(module)) and engine.student_test_native_prefills == 0
            status(status="running", phase="evaluation", task=task, sample=sample, total=count)
            if sample == 1 or sample % 8 == 0:
                print("Gated eval", task, sample, count, flush=True)
        keys = ("em", "f1") if task in {"hotpotqa", "2wiki", "musique"} else (
            ("correct", "legacy_correct") if task == "gsm8k" else ("correct",))
        metrics = {name: {key: sum(r["conditions"][name][key] for r in records) / count for key in keys} for name in names}
        changes = {}
        for baseline_name in (reference.BASE, reference.ARM):
            changes[baseline_name] = {}
            for key in keys:
                deltas = [r["conditions"][ARM][key] - r["conditions"][baseline_name][key] for r in records]
                changes[baseline_name][key] = dict(
                    paired=C.paired(deltas),
                    improved_ids=[r["id"] for r, delta in zip(records, deltas) if delta > 0],
                    declined_ids=[r["id"] for r, delta in zip(records, deltas) if delta < 0],
                )
        tasks[task] = dict(n=count, metrics=metrics, gated_minus=changes)
        mark(f"evaluation/{task}/summary.json", status="completed", **tasks[task])
        for key in keys:
            scores = " | ".join(f"{metrics[name][key]:.2%}" for name in names)
            lines.append(f"| {task} | {key} | {scores} |")
    mark("summary.json", status="completed", tasks=tasks, gates=gates)
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    mark("completed.json", status="completed")
    status(status="completed")


if __name__ == "__main__":
    {"prepare": G.prepare, "smoke": smoke, "train": train, "evaluate": evaluate}[sys.argv[1]]()
