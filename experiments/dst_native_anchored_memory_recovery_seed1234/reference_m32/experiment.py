"""Train one Gated Dynamic arm and compare it to the saved matched runs."""

from collections import Counter
from pathlib import Path
import sys

import experiment_common as G
from experiment_common import C, CFG, INPUTS, ROOT, SETTINGS, mark, rows, status, torch
import runtime_new as runtime
from dst_engine import BodyReader, Engine as BaseEngine
from residual import ResidualTranslator
from translator import DSTConfig, DepthStateTranslator
from gated_memory import GatedDynamicMemory
from score import score

ARMS = ("native", "B_rank64", "D_static", "rank64_dynamic", "gated_static", "gated_dynamic")
FINAL = ROOT / "runs/epoch4_step1536.pt"


def prepare():
    if (ROOT / "manifest.json").exists():
        G.verify()
        return
    for name, expected in SETTINGS["inputs_sha256"].items():
        assert C.sha(INPUTS / name) == expected, name
    static = C.read(INPUTS / "static_manifest.json")
    dynamic = C.read(INPUTS / "dynamic_manifest.json")
    assert static["b_checkpoint_sha256"] == dynamic["b_checkpoint_sha256"] == SETTINGS["inputs_sha256"]["rank64.pt"]
    for split in ("train", "validation", "test"):
        file = INPUTS / f"prepared/{split}.jsonl"
        key = next(k for k in dynamic["sources"] if k.endswith(f"/static_memory/prepared/{split}.jsonl"))
        assert C.sha(file) == dynamic["sources"][key]
        assert Counter(row["task"] for row in rows(split)) == CFG[f"{split}_counts"]
    schedules = {
        str(epoch): C.digest([[(row["id"], kind) for row, kind in batch] for batch in runtime.batches(epoch)])
        for epoch in range(1, CFG["epochs"] + 1)
    }
    assert schedules == static["batch_schedules"] == dynamic["batch_schedules"]
    assert CFG["updates"] == SETTINGS["updates"] == 1536
    for row in rows("test"):
        G.baseline(row)
    pins = {}
    for name, expected in dynamic["native_prefix_sources"].items():
        path = INPUTS / "native_prefix" / Path(name).name
        assert C.sha(path) == expected, path
        pins[path.name] = expected
    files = [*ROOT.glob("*.py"), ROOT / "config.json"]
    files.extend(p for p in INPUTS.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p.parent.name != "native_prefix")
    C.save(ROOT / "manifest.json", dict(settings=SETTINGS, sources={str(p): C.sha(p) for p in files},
                                       native_prefix_sources=pins, batch_schedules=schedules,
                                       trainable_parameters=409636))
    C.log("GATED DYNAMIC PREPARED", G.identity())


class Engine(BaseEngine):
    def __init__(self):
        super().__init__()
        cfg = G.DST_CONFIG
        self.dst = DepthStateTranslator(DSTConfig(**cfg["model"]), cfg["seed"]).cuda().float()
        cp = torch.load(INPUTS / "dst.pt", map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == SETTINGS["inputs_sha256"]["dst_manifest.json"]
        assert cp["stage"] == "stage1" and cp["step"] == 1536
        self.dst.load_state_dict(cp["state"], strict=True)
        self.correction = ResidualTranslator(self.dst, seed=CFG["seed"]).cuda().float()
        cp = torch.load(INPUTS / "rank64.pt", map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == SETTINGS["inputs_sha256"]["rank64_manifest.json"]
        assert cp["stage"] == "stage2" and cp["step"] == 1536
        self.correction.load_state_dict(cp["state"], strict=True)
        self.correction.eval().requires_grad_(False)
        self.translator_versions = (C.versions(self.dst), C.versions(self.correction))

    @torch.no_grad()
    def base(self, row):
        return self.source(row)

    @torch.no_grad()
    def cache(self, row, base, correction):
        assert correction is self.correction
        return self.translated(row, correction, base, checkpointed=False)

    def unchanged(self):
        super().unchanged()
        assert self.translator_versions == (C.versions(self.dst), C.versions(self.correction))
        assert all(not p.requires_grad and p.grad is None for m in (self.dst, self.correction) for p in m.parameters())


def load(final=False):
    engine = Engine()
    module = GatedDynamicMemory(engine, CFG)
    assert sum(p.numel() for p in module.parameters()) == 409636
    if final:
        assert C.sha(FINAL) == C.read(ROOT / "runs/completed.json")["checkpoint_sha256"]
        cp = torch.load(FINAL, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == G.identity() and cp["step"] == CFG["updates"]
        module.load_state_dict(cp["state"], strict=True)
    return engine, engine.correction, module


@torch.no_grad()
def native_prefix(module, row, native):
    name = row["id"] + ".json"
    pins = C.read(ROOT / "manifest.json")["native_prefix_sources"]
    if name in pins:
        source = INPUTS / "native_prefix" / name
        assert C.sha(source) == pins[name]
        record = C.read(source)
        assert record["manifest_sha256"] == G.N.identity() and record["id"] == row["id"]
        return record["tokens"]
    return runtime_prefix(module, row, native)


runtime_prefix = runtime.native_prefix
runtime.native_prefix = native_prefix
runtime.load = load


def smoke():
    G.verify()
    if G.done("smoke/completed.json"):
        return
    engine, correction, module = load()
    reader = BodyReader(engine)
    probes = (next(row for row in rows("test") if row["task"] == task) for task in ("arc", "hotpotqa"))
    module.eval()
    checked = []
    with torch.no_grad():
        assert torch.count_nonzero(module.gates) == 0
        for row in probes:
            body = engine.cache(row, engine.base(row), correction)
            bank = module.construct(row, body)
            assert bank.body is body and len(bank.bank) == 72
            assert all(x.shape[-2] == module.M and torch.isfinite(x).all() for x in bank.bank)
            expected = reader.logits(row, body, rn=False)
            actual = module.logits(row, bank, rn=True)
            assert torch.equal(expected, actual), float((expected - actual).abs().max())
            plain, gated = score(reader, row, body, rn=False), score(module, row, bank, rn=True)
            for key in ("prediction", "correct", "generation", "scores"):
                if key in plain:
                    assert plain[key] == gated[key], (row["id"], key)
                    prior = G.baseline(row)["B_rank64"].get(key)
                    if key == "scores":
                        assert max(abs(a-b) for a,b in zip(plain[key], prior)) < 1e-4
                    else:
                        assert plain[key] == prior
            checked.append(row["task"])
            C.log("ZERO GATE PARITY", row["task"])
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = G.T.scaler_for()
    _, outcome, _ = runtime.backward_batch(engine, correction, module, optimizer, scaler,
                                            runtime.batches(1)[0], 1, 1, scope="smoke")
    assert torch.count_nonzero(module.gates) > 0
    C.log("GATE UPDATE", outcome["gradient_norm"])
    train_rows = rows("train")
    for task in CFG["train_counts"]:
        row = max((r for r in train_rows if r["task"] == task),
                  key=lambda r: len(r["content_ids"]) + len(r["gold_tokens"]))
        body = engine.cache(row, engine.base(row), correction)
        native = engine.native_cache(row)
        loss, _ = runtime.losses(engine, module, row, body, native, "gold", 1)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert module.gates.grad.norm() > 0 and module.memory.prompt.grad.norm() > 0
        assert any(p.grad.norm() > 0 for p in module.memory.adapters.parameters())
        optimizer.zero_grad(set_to_none=True)
        engine.unchanged()
        C.log("LONG BACKWARD", task, row["id"])
        del body, native, loss
    mark("smoke/completed.json", status="completed", zero_gate_exact_logits=checked,
         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)


def train():
    import train_new

    train_new.main()


@torch.no_grad()
def evaluate():
    G.verify()
    assert G.done("runs/completed.json")
    if G.done("summary.json"):
        return
    engine, correction, module = load(final=True)
    engine.mode = "test"
    module.eval().requires_grad_(False)
    tasks = {}
    test = rows("test")
    lines = ["# Independent Gated Dynamic Memory", "",
             "| Task | Metric | Native | Rank64 | Static Keep-both | Dynamic Keep-both | Gated Static | Gated Dynamic |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, count in CFG["test_counts"].items():
        records = []
        for i, row in enumerate((r for r in test if r["task"] == task), 1):
            if G.paused():
                raise SystemExit(75)
            rel = f'evaluation/{task}/per_sample/{row["id"]}.json'
            path = ROOT / rel
            if not path.exists():
                body = engine.cache(row, engine.base(row), correction)
                conditions = G.baseline(row)
                conditions["gated_dynamic"] = score(module, row, module.construct(row, body), rn=True)
                assert engine.student_test_native_prefills == 0
                engine.unchanged()
                mark(rel, id=row["id"], question=row["question"], gold=row["gold"], conditions=conditions)
            item = C.read(path)
            assert item["manifest_sha256"] == G.identity() and item["id"] == row["id"]
            records.append(item)
            if i == 1 or i % 8 == 0:
                status(status="running", phase="evaluation", task=task, sample=i, total=count)
                C.log("GATED DYNAMIC EVAL", task, i, count)
        assert len(records) == count
        keys = ("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else ("correct",)
        if task == "gsm8k":
            keys += ("legacy_correct",)
        metrics = {arm: {k: sum(r["conditions"][arm][k] for r in records) / count for k in keys} for arm in ARMS}
        changes = {}
        for arm in ("B_rank64", "rank64_dynamic", "gated_static"):
            by_key = {}
            for key in keys:
                differences = [r["conditions"]["gated_dynamic"][key] - r["conditions"][arm][key] for r in records]
                by_key[key] = dict(paired=C.paired(differences),
                                   improved_ids=[r["id"] for r,d in zip(records, differences) if d > 0],
                                   declined_ids=[r["id"] for r,d in zip(records, differences) if d < 0])
            changes[arm] = by_key
        tasks[task] = dict(n=count, metrics=metrics, gated_dynamic_minus=changes)
        mark(f"evaluation/{task}/summary.json", status="completed", **tasks[task])
        for key in keys:
            lines.append(f"| {task} | {key} | " + " | ".join(f"{metrics[arm][key]:.2%}" for arm in ARMS) + " |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    mark("summary.json", status="completed", tasks=tasks, gates=module.gates.tanh().tolist())
    status(status="completed")


if __name__ == "__main__":
    {"prepare": prepare, "smoke": smoke, "train": train, "evaluate": evaluate}[sys.argv[1]]()
