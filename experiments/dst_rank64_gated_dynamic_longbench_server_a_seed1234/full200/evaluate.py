"""Finish all 200 LongBench questions per task using the 64-question pilot."""
import fcntl
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "pilot64"
spec = importlib.util.spec_from_file_location("longbench_four_arm_pilot", PILOT / "evaluate.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
L, C, torch = P.L, P.C, P.torch


def status(state, **details):
    C.save(ROOT / "status.json", dict(status=state, time=L.strftime("%F %T"), **details))


def summarize(task, records, pin, target):
    full_text = [r for r in records if not r["clipped"]]
    value = dict(status="completed" if len(records) == target else "evaluating",
                 manifest_sha256=pin, overall=P.metrics(records),
                 fully_retained=P.metrics(full_text) if full_text else None)
    C.save(ROOT / f"evaluation/{task}/summary.json", value)
    return value


def main():
    parent = C.read(L.ROOT / "manifest.json")
    parent_pin = C.sha(L.ROOT / "manifest.json")
    pilot_manifest = C.read(PILOT / "manifest.json")
    pilot_pin = C.sha(PILOT / "manifest.json")
    cap = parent["input_cap"]
    status("loading", input_cap=cap)
    assert pilot_manifest["parent"] == parent_pin and pilot_manifest["cap"] == cap
    assert C.sha(PILOT / "evaluate.py") == pilot_manifest["script"]
    assert tuple(pilot_manifest["arms"]) == P.ARMS
    assert C.sha(L.ROOT / "evaluate.py") == parent["pins"]["script"]
    assert C.sha(L.ROOT / "config.json") == parent["pins"]["config"]
    for name, digest in parent["pins"]["fullsync"].items():
        assert C.sha(L.ROOT / name) == digest
    L.G.G.verify()
    assert L.G.G.identity() == parent["pins"]["reference"]
    assert C.sha(L.G.FINAL) == parent["pins"]["gated_checkpoint"]
    examples = L.load_examples()
    for task, rows in examples.items():
        assert [row["_id"] for row in rows[:P.N]] == pilot_manifest["cohorts"][task]
    map_path = Path(C.CFG["recolor"])
    checkpoint = C.ROOT / "runs/C/epoch4_step800.pt"
    assert C.sha(map_path) == C.CFG["recolor_sha256"] == pilot_manifest["recolor_map_sha256"]
    assert C.sha(checkpoint) == pilot_manifest["c_checkpoint_sha256"]
    settings = dict(parent=parent_pin, pilot=pilot_pin, script=C.sha(Path(__file__)), cap=cap,
                    arms=list(P.ARMS), cohorts={task: [row["_id"] for row in rows] for task, rows in examples.items()})
    manifest = ROOT / "manifest.json"
    if manifest.exists():
        assert C.read(manifest) == settings
    else:
        C.save(manifest, settings)
    pin = C.sha(manifest)

    engine, rank, memory = L.G.load(final=True)
    memory.eval().requires_grad_(False)
    reader = L.G.BodyReader(engine)
    engine.mode = "test"
    protocol = L.Protocol(engine)
    recolor = P.RecolorC(engine, map_path, checkpoint)
    total = sum(map(len, examples.values()))
    completed = 0
    output = {}
    for task, items in examples.items():
        records = []
        generation_cap = L.CFG["tasks"][task]["output_cap"]
        for i, item in enumerate(items, 1):
            if (ROOT / "controls/pause_requested.json").exists():
                status("paused", task=task, task_completed=len(records), total_completed=completed)
                raise SystemExit(75)
            path = P.sample_path(ROOT, task, item)
            if path.exists():
                record = C.read(path)
                assert record["manifest_sha256"] == pin and record["id"] == item["_id"]
            else:
                pilot_path = P.sample_path(PILOT, task, item)
                if i <= P.N:
                    record = C.read(pilot_path)
                    assert record["manifest_sha256"] == pilot_pin and record["id"] == item["_id"]
                    assert record["question"] == item["input"] and record["answers"] == item["answers"]
                    assert tuple(record["conditions"]) == P.ARMS
                    record.update(reuse_source="pilot64", manifest_sha256=pin)
                else:
                    torch.cuda.reset_peak_memory_stats()
                    row = protocol.row(task, item, cap)
                    scores = P.reuse(row, item, task, parent_pin)
                    reused = scores is not None
                    if not reused:
                        scores, _ = L.eval_row(engine, rank, memory, reader, row, item["answers"], generation_cap)
                    scores["recolor_c"] = recolor.run(reader, row, item["answers"], generation_cap)
                    engine.unchanged()
                    record = {k: row[k] for k in ("id", "clipped", "qwen_tokens", "sender_tokens")}
                    record.update(manifest_sha256=pin, question=item["input"], answers=item["answers"],
                                  conditions=scores, reuse_source="original_three_arm" if reused else None,
                                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
                C.save(path, record)
            records.append(record)
            completed += 1
            if i >= P.N:
                status("evaluating", task=task, task_completed=i, task_total=len(items),
                       total_completed=completed, total=total, reuse_source=record["reuse_source"])
            if i >= P.N and (i % 8 == 0 or i == len(items)):
                value = summarize(task, records, pin, len(items))
                print(task, f"{i}/{len(items)}", value["overall"], flush=True)
        output[task] = summarize(task, records, pin, len(items))
    C.save(ROOT / "summary.json", dict(status="completed", manifest_sha256=pin, tasks=output))
    status("completed", total_completed=completed, total=total)


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            status("failed", error=repr(exc))
            raise
