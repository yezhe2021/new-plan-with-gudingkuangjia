"""Train four post-block Recovery Adapters on the original DST Memory protocol."""
from pathlib import Path
import sys
from types import SimpleNamespace

from recovery import ADAPTER_RANK, HIDDEN_WEIGHT, LAYERS, attach, hidden_loss

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference_m32"
ORIGINAL = ROOT.parent / "dst_rank64_gated_dynamic_m32_a4_server_a_seed1234"
M32_HASH = "4547b751d7d486efb803ce2b7df65ecad5df0b91739d3aa6830831be8ca914f6"
FINAL = ROOT / "runs/epoch4_step1536.pt"
ARM = "recovery_memory"


def reference():
    sys.path.insert(0, str(REFERENCE))
    import experiment as old

    old.G.ROOT = old.ROOT = old.runtime.ROOT = ROOT
    return old


def prepare(old):
    G, C = old.G, old.C
    swap = C.read(ROOT / "oracle/summary.json")
    assert swap["status"] == "completed" and all(swap["tasks"][x]["n"] == 64 for x in ("quality", "coursera", "sfiction"))
    base = C.read(REFERENCE / "manifest.json")
    sources = {}
    for name, digest in base["sources"].items():
        path = REFERENCE / Path(name).relative_to(ORIGINAL)
        assert C.sha(path) == digest, path
        sources[str(path)] = digest
    for name, digest in base["native_prefix_sources"].items():
        assert C.sha(G.INPUTS / "native_prefix" / name) == digest
    schedules = {str(ep): C.digest([[(r["id"], kind) for r, kind in batch] for batch in old.runtime.batches(ep)])
                 for ep in range(1, 5)}
    assert schedules == base["batch_schedules"]
    assert old.CFG["updates"] == 1536 and old.CFG["lambda_kl"] == 0.5
    assert C.sha(REFERENCE / "m32.pt") == M32_HASH
    for path in (Path(__file__), ROOT / "recovery.py", ROOT / "run_pipeline.sh", REFERENCE / "manifest.json", REFERENCE / "m32.pt"):
        sources[str(path)] = C.sha(path)
    manifest = dict(sources=sources, native_prefix_sources=base["native_prefix_sources"], batch_schedules=schedules,
                    reference_manifest_sha256=C.sha(REFERENCE / "manifest.json"), oracle_manifest_sha256=swap["identity"],
                    rank64_sha256=sources[str(G.INPUTS/"rank64.pt")], m32_sha256=M32_HASH, frozen_memory_slots=32,
                    recovery_after_layers=list(LAYERS), adapter_rank=ADAPTER_RANK, trainable_parameters=327680,
                    updates=1536, lambda_kl=0.5, lambda_hidden=HIDDEN_WEIGHT)
    p = ROOT / "manifest.json"
    if p.exists():
        assert C.read(p) == manifest
    else:
        C.save(p, manifest)


def load(old, final=False):
    C, torch = old.C, old.torch
    engine = old.Engine()
    memory = old.GatedDynamicMemory(engine, old.CFG)
    path = REFERENCE / "m32.pt"
    assert C.sha(path) == M32_HASH
    base = torch.load(path, map_location="cpu", weights_only=True)
    assert base["manifest_sha256"] == C.sha(REFERENCE / "manifest.json") and base["step"] == 1536
    memory.load_state_dict(base["state"], strict=True)
    attach(memory)
    assert sum(p.numel() for p in memory.parameters() if p.requires_grad) == 327680
    if final:
        done = C.read(ROOT / "runs/completed.json")
        assert C.sha(FINAL) == done["checkpoint_sha256"]
        data = torch.load(FINAL, map_location="cpu", weights_only=True)
        assert data["manifest_sha256"] == old.G.identity() and data["step"] == 1536
        memory.load_state_dict(data["state"], strict=True)
        memory.eval().requires_grad_(False)
        engine.mode = "test"
    return engine, engine.correction, memory


def loss(old, engine, memory, row, translated, native, kind, epoch):
    C, torch = old.C, old.torch
    assert translated.get_seq_length() == native.get_seq_length()
    with torch.no_grad(), memory.recovery.capture(teacher=True) as native_hidden:
        memory.construct(row, native)
    with memory.recovery.capture(teacher=False) as recovered_hidden:
        bank = memory.construct(row, translated)
    hidden = hidden_loss(recovered_hidden, native_hidden)
    target = row["gold_tokens"]
    logits = memory.token_logits(row, bank, target, rn=True)
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=logits.device))
    del logits
    with torch.no_grad():
        if kind == "gold":
            tokens = target
        elif kind == "native":
            tokens = old.runtime.native_prefix(memory, row, native)
        else:
            tokens = memory.rollout(row, bank, rn=True, cap=old.CFG["prefix_max_new_tokens"])
        assert tokens
        positions = C.positions_for(tokens, row, epoch, validation=epoch == 0)
        prefix = tokens[:max(positions)]
        teacher = memory.logits(row, native, prefix, positions, rn=False)
    student = memory.logits(row, bank, prefix, positions, rn=True)
    kl = C.vocab_kl(teacher, student)
    total = ce + 0.5*kl + HIDDEN_WEIGHT*hidden
    assert torch.isfinite(total)
    return total, dict(loss=float(total.detach()), task_loss=float(ce.detach()), full_kl=float(kl.detach()),
                       hidden_nmse=float(hidden.detach()), prefix_kind=kind, positions=len(positions),
                       prefix_tokens=len(tokens), gold_tokens=len(target), task=row["task"])


def smoke(old):
    G, torch = old.G, old.torch
    prepare(old)
    if G.done("smoke/completed.json"):
        return
    G.status(status="running", phase="recovery_smoke")
    engine, correction, memory = load(old)
    dataset = G.rows("train")
    small = min(dataset, key=lambda row: len(row["content_ids"]))
    longest = max(dataset, key=lambda row: len(row["content_ids"]))
    with torch.no_grad():
        body = engine.cache(small, engine.base(small), correction)
        with memory.recovery.capture(teacher=True):
            original = memory.construct(small, body)
        zero = memory.construct(small, body)
        assert all(torch.equal(a,b) for a,b in zip(original.bank, zero.bank))
        del original, zero, body
        translated = engine.cache(longest, engine.base(longest), correction)
        native = engine.native_cache(longest)
    value, metrics = loss(old, engine, memory, longest, translated, native, "gold", 1)
    value.backward()
    for n,p in memory.named_parameters():
        if n.startswith("recovery."):
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        else:
            assert not p.requires_grad and p.grad is None, n
    norms = {l: memory.recovery.blocks[str(l)][-1].weight.grad.float().norm().item() for l in LAYERS}
    assert all(x > 0 for x in norms.values())
    engine.unchanged()
    G.mark("smoke/completed.json", status="completed", exact_zero_adapter_bank=True, gradient_norms=norms,
           input_tokens=len(longest["content_ids"]), losses=metrics, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def train(old):
    G, C = old.G, old.C
    runtime = old.runtime
    runtime.load = lambda final=False: load(old, final)
    runtime.losses = lambda *args: loss(old, *args)
    runtime.O = SimpleNamespace(finish_backward=lambda memory, opt, scaler: G.O.finish_backward(memory.recovery, opt, scaler))
    original_backward = runtime.backward_batch

    def with_hidden(engine, correction, memory, opt, scaler, batch, epoch, batch_index, *, scope):
        samples, outcome, retries = original_backward(engine, correction, memory, opt, scaler, batch, epoch,
                                                      batch_index, scope=scope)
        C.save(ROOT / "checks/latest_hidden_loss.json", dict(epoch=epoch, batch=batch_index,
               hidden_nmse=sum(x["hidden_nmse"] for x in samples)/len(samples)))
        return samples, outcome, retries

    runtime.backward_batch = with_hidden
    trainer = G.load_module("native_recovery_original_train", G.INPUTS / "train_new.py")
    trainer.main()


def short(old):
    G, C, torch = old.G, old.C, old.torch
    engine, correction, memory = load(old, True)
    checkpoint = C.sha(FINAL)
    reference_pin = C.sha(REFERENCE / "manifest.json")
    output = {}
    with torch.no_grad():
        for task, count in old.CFG["test_counts"].items():
            records = []
            for i,row in enumerate((r for r in G.rows("test") if r["task"] == task),1):
                if G.paused():
                    raise SystemExit(75)
                rel = Path(task) / "per_sample" / (row["id"]+".json")
                prior = C.read(REFERENCE / "evaluation" / rel)
                assert prior["manifest_sha256"] == reference_pin and all(prior[k] == row[k] for k in ("id", "question", "gold"))
                path = ROOT / "evaluation/short" / rel
                if path.exists():
                    result = C.read(path)
                    assert result["checkpoint_sha256"] == checkpoint
                else:
                    body = engine.cache(row, engine.base(row), correction)
                    score = old.score(memory, row, memory.construct(row, body), rn=True)
                    result = dict(id=row["id"], checkpoint_sha256=checkpoint,
                                  conditions={"gated_dynamic":prior["conditions"]["gated_dynamic"], ARM:score})
                    C.save(path,result)
                    del body
                assert engine.student_test_native_prefills == 0
                records.append(result["conditions"])
                G.status(status="running", phase="eval_short", task=task, sample=i, total=count)
            assert len(records) == count
            keys = ("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else ("correct",)
            metrics = {arm: {k:sum(r[arm][k] for r in records)/count for k in keys} for arm in records[0]}
            deltas = {k:C.paired([r[ARM][k]-r["gated_dynamic"][k] for r in records]) for k in keys}
            output[task] = dict(n=count, scores=metrics, paired_minus_m32=deltas)
            C.save(ROOT / "evaluation/short" / task / "summary.json", output[task])
    C.save(ROOT / "short_summary.json", dict(status="completed", checkpoint_sha256=checkpoint, tasks=output))
    G.status(status="completed", phase="completed")


if __name__ == "__main__":
    {"smoke": smoke, "train": train, "short": short}[sys.argv[1]](reference())
