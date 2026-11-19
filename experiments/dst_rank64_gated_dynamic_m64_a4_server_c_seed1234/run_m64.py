"""Matched Gated Dynamic Memory training with 64 learned slots."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference_m32"
ORIGINAL = ROOT.parent / "dst_rank64_gated_dynamic_m32_a4_server_a_seed1234"
ARM = "gated_dynamic_m64"
SLOTS = 64
EXTRA_SEED = 7634
GPU_FRACTION = 0.48
FINAL = ROOT / "runs/epoch4_step1536.pt"


def expand(memory, torch):
    assert memory.M == 32
    prompt = memory.memory.prompt
    rng = torch.Generator(device="cpu").manual_seed(EXTRA_SEED)
    extra = 0.02 * torch.randn(SLOTS-32, prompt.shape[1], generator=rng, dtype=torch.float32)
    memory.memory.prompt = torch.nn.Parameter(torch.cat((prompt.detach(), extra.to(prompt)), dim=0))
    memory.cfg["memory_tokens"] = SLOTS
    assert torch.equal(memory.memory.prompt[:32], prompt)
    assert sum(p.numel() for p in memory.parameters()) == 491556
    return memory


def load_final(core, torch, memory):
    record = core.read(ROOT / "runs/completed.json")
    assert record["status"] == "completed" and record["updates"] == 1536
    digest = core.sha(FINAL)
    manifest = core.sha(ROOT / "manifest.json")
    assert record["checkpoint_sha256"] == digest and record["manifest_sha256"] == manifest
    data = torch.load(FINAL, map_location="cpu", weights_only=True)
    assert data["manifest_sha256"] == manifest and data["step"] == 1536
    memory.load_state_dict(data["state"], strict=True)
    memory.eval().requires_grad_(False)
    return digest


def init_training_reference():
    sys.path.insert(0, str(REFERENCE))
    import experiment as old

    G = old.G
    G.ROOT = old.ROOT = old.runtime.ROOT = ROOT
    return old


def prepare(old):
    G, C = old.G, old.C
    m32 = C.read(REFERENCE / "manifest.json")
    sources = {}
    for name, digest in m32["sources"].items():
        path = REFERENCE / Path(name).relative_to(ORIGINAL)
        assert C.sha(path) == digest, path
        sources[str(path)] = digest
    for name, digest in m32["native_prefix_sources"].items():
        assert C.sha(G.INPUTS / "native_prefix" / name) == digest
    schedules = {str(ep): C.digest([[(r["id"], kind) for r, kind in batch] for batch in old.runtime.batches(ep)])
                 for ep in range(1, 5)}
    assert schedules == m32["batch_schedules"] and old.CFG["memory_tokens"] == 32
    assert old.CFG["updates"] == 1536 and old.CFG["lambda_kl"] == 0.5
    for path in (Path(__file__), ROOT / "run_pipeline.sh", REFERENCE / "manifest.json"):
        sources[str(path)] = C.sha(path)
    manifest = dict(sources=sources, native_prefix_sources=m32["native_prefix_sources"], batch_schedules=schedules,
                    reference_manifest_sha256=C.sha(REFERENCE / "manifest.json"), rank64_sha256=sources[str(G.INPUTS/'rank64.pt')],
                    memory_tokens=SLOTS, additional_slot_seed=EXTRA_SEED, trainable_parameters=491556,
                    updates=1536, lambda_kl=0.5, gpu_fraction_limit=GPU_FRACTION)
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)


def load(old, final=False):
    engine = old.Engine()
    memory = expand(old.GatedDynamicMemory(engine, old.CFG), old.torch)
    if final:
        load_final(old.C, old.torch, memory)
        engine.mode = "test"
    return engine, engine.correction, memory


def smoke(old):
    G, torch = old.G, old.torch
    if G.done("smoke/completed.json"):
        return
    G.status(status="running", phase="smoke_loading")
    engine, correction, memory = load(old)
    reader = old.BodyReader(engine)
    train_rows = G.rows("train")
    first = min(train_rows, key=lambda row: len(row["content_ids"]))
    longest = max(train_rows, key=lambda row: len(row["content_ids"]))
    with torch.no_grad():
        body = engine.cache(first, engine.base(first), correction)
        bank = memory.construct(first, body)
        assert bank.body is body and len(bank.bank) == 72 and all(k.shape[-2] == SLOTS for k in bank.bank)
        assert torch.count_nonzero(memory.gates) == 0
        assert torch.equal(reader.logits(first, body, rn=False), memory.logits(first, bank, rn=True))
        del bank, body
    # Warm through one real batch so the zero-initialized gates open before measuring slot gradients.
    opt = torch.optim.AdamW(memory.parameters(), lr=old.CFG["learning_rate"], weight_decay=old.CFG["weight_decay"])
    scaler = G.T.scaler_for()
    G.status(status="running", phase="smoke_gate_batch")
    old.runtime.backward_batch(engine, correction, memory, opt, scaler, old.runtime.batches(1)[0], 1, 1, scope="smoke")
    assert torch.count_nonzero(memory.gates) > 0
    opt.zero_grad(set_to_none=True)
    G.status(status="running", phase="smoke_longest_backward", body_tokens=len(longest["content_ids"]))
    with torch.no_grad():
        body = engine.cache(longest, engine.base(longest), correction)
        native = engine.native_cache(longest)
    loss, _ = old.runtime.losses(engine, memory, longest, body, native, "gold", 1)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    extra = memory.memory.prompt.grad[32:].float().norm().item()
    assert extra > 0 and memory.gates.grad.norm() > 0
    engine.unchanged()
    G.mark("smoke/completed.json", status="completed", longest_input_tokens=len(longest["content_ids"]),
           extra_slot_grad_norm=extra, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def train(old):
    G, torch = old.G, old.torch
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    prepare(old)
    smoke(old)


def formal_train(old):
    old.torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    old.runtime.load = lambda final=False: load(old, final)
    trainer = old.G.load_module("gated_m64_original_train", old.G.INPUTS / "train_new.py")
    trainer.main()


def short(old):
    C, G, torch = old.C, old.G, old.torch
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    engine, correction, memory = load(old, True)
    baseline = C.sha(REFERENCE / "manifest.json")
    checkpoint = C.sha(FINAL)
    summary = {}
    with torch.no_grad():
        for task, count in old.CFG["test_counts"].items():
            records = []
            for i, row in enumerate((x for x in G.rows("test") if x["task"] == task), 1):
                if G.paused():
                    raise SystemExit(75)
                rel = Path(task) / "per_sample" / (row["id"] + ".json")
                prior = C.read(REFERENCE / "evaluation" / rel)
                assert prior["manifest_sha256"] == baseline and all(prior[k] == row[k] for k in ("id", "question", "gold"))
                path = ROOT / "evaluation/short" / rel
                if path.exists():
                    item = C.read(path)
                    assert item["checkpoint_sha256"] == checkpoint
                else:
                    body = engine.cache(row, engine.base(row), correction)
                    result = old.score(memory, row, memory.construct(row, body), rn=True)
                    item = dict(id=row["id"], task=task, checkpoint_sha256=checkpoint,
                                conditions=dict(prior["conditions"], **{ARM: result}))
                    C.save(path, item)
                    del body
                records.append(item["conditions"])
                G.status(status="running", phase="eval_short", task=task, sample=i, total=count)
            assert len(records) == count
            keys = ("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else ("correct",)
            if task == "gsm8k":
                keys += ("legacy_correct",)
            scores = {arm: {k: sum(r[arm][k] for r in records)/count for k in keys} for arm in records[0]}
            changes = {k: C.paired([r[ARM][k]-r["gated_dynamic"][k] for r in records]) for k in keys}
            summary[task] = dict(n=count, scores=scores, paired_minus_m32=changes)
            C.save(ROOT / "evaluation/short" / task / "summary.json", summary[task])
    C.save(ROOT / "short_summary.json", dict(status="completed", checkpoint_sha256=checkpoint, tasks=summary))


if __name__ == "__main__":
    phase = sys.argv[1]
    old = init_training_reference()
    actions = {"smoke": train, "train": formal_train, "short": short}
    actions[phase](old)
