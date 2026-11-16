"""Resume the first 64 questions per task and add the frozen Recolor+C arm."""
import fcntl
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
import evaluate as L

C, torch = L.C, L.torch
N = 64
ARMS = (*L.ARMS, "recolor_c")


def status(state, **details):
    C.save(ROOT / "status.json", dict(status=state, time=L.strftime("%F %T"), **details))


def sample_path(root, task, item):
    identifier = hashlib.sha256(item["_id"].encode()).hexdigest()[:20]
    return root / f"evaluation/{task}/per_sample/{identifier}.json"


class RecolorC:
    def __init__(self, engine, map_path, checkpoint):
        self.engine = engine
        self.builder = C.HC.ExampleBuilder((engine.sender_tok, engine.sender),
                                           (engine.tok, engine.model), C.load_map(map_path), "cuda")
        assert self.builder.prefix_ids.tolist() == engine.prefix_ids
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert weights["manifest_sha256"] == C.identity() and weights["arm"] == "C"
        assert weights["step"] == 800 and weights["recolor_sha256"] == C.CFG["recolor_sha256"]
        self.correction = C.fresh("C")
        self.correction.load_state_dict(weights["state"], strict=True)
        self.correction.eval().requires_grad_(False)
        self.versions = C.versions(self.correction)

    @torch.no_grad()
    def run(self, reader, row, answers, cap):
        encoded = self.engine.tok(row["prompt"], add_special_tokens=False,
                                  return_offsets_mapping=True, verbose=False)
        assert encoded["input_ids"] == row["content_ids"]
        def make_cache():
            base = self.builder.recolor_outputs(row["prompt"], encoded["offset_mapping"])
            assert all(x.shape == (36, len(row["content_ids"]), 1024) for x in base)
            return C.Engine.cache(self.engine, row, base, self.correction)

        cache, cache_seconds = L.measured(make_cache)
        assert cache.get_seq_length() == len(self.engine.prefix_ids) + len(row["content_ids"])
        tokens, decode_seconds = L.measured(lambda: reader.rollout(row, cache, rn=False, cap=cap))
        pred = self.engine.tok.decode(tokens, skip_special_tokens=True).strip()
        assert C.versions(self.correction) == self.versions and self.engine.student_test_native_prefills == 0
        return dict(prediction=pred, f1=max(L.qa_f1(pred, gold) for gold in answers),
                    generated_tokens=len(tokens), seconds=cache_seconds + decode_seconds)


def reuse(row, item, task, pin):
    path = sample_path(PARENT, task, item)
    if not path.exists():
        return None
    record = C.read(path)
    assert record["manifest_sha256"] == pin and record["id"] == row["id"]
    assert record["question"] == item["input"] and record["answers"] == item["answers"]
    assert all(record[key] == row[key] for key in (
        "clipped", "qwen_tokens", "sender_tokens", "original_qwen_tokens", "original_sender_tokens",
    ))
    assert tuple(record["conditions"]) == L.ARMS
    return record["conditions"]


def metrics(records):
    paired = np.array([[r["conditions"][arm]["f1"] * 100 for arm in ARMS] for r in records])
    ix = np.random.default_rng(1234).integers(len(records), size=(10000, len(records)))
    comparisons = {}
    for reference in ("dst_rank64", "recolor_c", "native"):
        diff = paired[:, 2] - paired[:, ARMS.index(reference)]
        comparisons["gated_minus_" + reference] = {
            "delta_pp": float(diff.mean()), "ci95_pp": np.quantile(diff[ix].mean(1), [.025, .975]).tolist(),
        }
    return dict(n=len(records), clipped=sum(r["clipped"] for r in records),
                f1={arm: float(paired[:, i].mean()) for i, arm in enumerate(ARMS)}, paired=comparisons)


def summarize(task, records, pin):
    available = [r for r in records if not r["clipped"]]
    value = dict(status="completed" if len(records) == N else "evaluating",
                 manifest_sha256=pin, overall=metrics(records),
                 fully_retained=metrics(available) if available else None)
    C.save(ROOT / f"evaluation/{task}/summary.json", value)
    return value


def main():
    parent_manifest = PARENT / "manifest.json"
    parent = C.read(parent_manifest)
    assert C.sha(PARENT / "evaluate.py") == parent["pins"]["script"]
    assert C.sha(PARENT / "config.json") == parent["pins"]["config"]
    for name, digest in parent["pins"]["fullsync"].items():
        assert C.sha(PARENT / name) == digest
    parent_pin = C.sha(parent_manifest)
    cap = parent["input_cap"]
    status("loading", input_cap=cap)
    L.G.G.verify()
    assert L.G.G.identity() == parent["pins"]["reference"]
    assert C.sha(L.G.FINAL) == parent["pins"]["gated_checkpoint"]
    examples = {task: rows[:N] for task, rows in L.load_examples().items()}
    map_path = Path(C.CFG["recolor"])
    map_sha = C.sha(map_path)
    assert map_sha == C.CFG["recolor_sha256"]
    checkpoint = C.ROOT / "runs/C/epoch4_step800.pt"
    settings = dict(parent=parent_pin, script=C.sha(Path(__file__)), cap=cap, arms=list(ARMS),
                    recolor_map_sha256=map_sha, c_checkpoint_sha256=C.sha(checkpoint),
                    cohorts={task: [row["_id"] for row in rows] for task, rows in examples.items()})
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
    recolor = RecolorC(engine, map_path, checkpoint)
    total = 0
    output = {}
    for task, rows in examples.items():
        records = []
        generation_cap = L.CFG["tasks"][task]["output_cap"]
        for item in rows:
            if (ROOT / "controls/pause_requested.json").exists():
                status("paused", task=task, task_completed=len(records), total_completed=total)
                raise SystemExit(75)
            path = sample_path(ROOT, task, item)
            if path.exists():
                record = C.read(path)
                assert record["manifest_sha256"] == pin and record["id"] == item["_id"]
            else:
                torch.cuda.reset_peak_memory_stats()
                row = protocol.row(task, item, cap)
                scores = reuse(row, item, task, parent_pin)
                reused = scores is not None
                if not reused:
                    scores, _ = L.eval_row(engine, rank, memory, reader, row, item["answers"], generation_cap)
                scores["recolor_c"] = recolor.run(reader, row, item["answers"], generation_cap)
                engine.unchanged()
                record = {k: row[k] for k in ("id", "clipped", "qwen_tokens", "sender_tokens")}
                record.update(manifest_sha256=pin, question=item["input"], answers=item["answers"],
                              conditions=scores, reused=reused, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
                C.save(path, record)
            records.append(record)
            total += 1
            status("evaluating", task=task, task_completed=len(records), task_total=N, total_completed=total,
                   total=2 * N, peak_gpu_gib=record["peak_gpu_gib"], reused=record["reused"])
            if len(records) == 1 or len(records) % 8 == 0 or len(records) == N:
                value = summarize(task, records, pin)
                print(task, f"{len(records)}/{N}", value["overall"], flush=True)
        output[task] = summarize(task, records, pin)
    C.save(ROOT / "summary.json", dict(status="completed", manifest_sha256=pin, tasks=output))
    status("completed", total_completed=total)


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            status("failed", error=repr(exc))
            raise
