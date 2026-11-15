"""One continuation epoch of joint Rank64 and Dynamic Memory optimization."""

import sys

import experiment_common as G
from experiment_common import C, CFG, K, ROOT, SETTINGS, mark, status, torch
from memory_dynamic import DynamicMemory
from runtime_new import batches, losses
from score import score
from torch.utils.checkpoint import checkpoint

REF = G.REFERENCE
FINAL = ROOT / "runs/joint_step384.pt"
LAST = ROOT / "runs/last.pt"
EPOCH = SETTINGS["training_epoch"]
TOTAL = SETTINGS["updates"]
MEMORY = REF / "runs/epoch4_step1536.pt"
RANK64 = G.reference.RANK64
ARMS = ("native", "B_rank64", "D_static", "rank64_dynamic", "joint")


def verify():
    record = C.read(ROOT / "manifest.json")
    for name, expected in record["pins"].items():
        assert C.sha(name) == expected, name


def prepare():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    ref = G.reference
    ref.verify()
    assert ref.identity() == SETTINGS["reference_manifest_sha256"] and ref.done("completed.json")
    assert C.sha(MEMORY) == SETTINGS["dynamic_checkpoint_sha256"]
    assert C.sha(RANK64) == SETTINGS["rank64_checkpoint_sha256"]
    assert EPOCH == 5 and TOTAL == len(batches(EPOCH))
    plan = [[(row["id"], kind) for row,kind in batch] for batch in batches(EPOCH)]
    pinned = [ROOT / "config.json", *ROOT.glob("*.py"), RANK64, MEMORY, REF / "manifest.json"]
    parent_manifest = C.read(REF / "manifest.json")
    pins = {**parent_manifest["sources"], **{str(path): C.sha(path) for path in pinned}}
    for row in G.rows("test"):
        path = REF / f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
        v = C.read(path)
        assert v["manifest_sha256"] == ref.identity() and all(row[k] == v[k] for k in ("id", "question", "gold"))
        assert v["memory_checkpoint_sha256"] == SETTINGS["dynamic_checkpoint_sha256"]
        assert v["rank64_checkpoint_sha256"] == SETTINGS["rank64_checkpoint_sha256"]
        pins[str(path)] = C.sha(path)
    C.save(ROOT / "manifest.json", dict(
        settings=SETTINGS, pins=pins, batch_schedule_sha256=C.digest(plan),
        native_prefix_sources=parent_manifest["native_prefix_sources"],
        trainable_parameters={"rank64": 9437184, "memory": 409600},
    ))
    C.log("JOINT PREPARED", G.identity())


class JointMemory(DynamicMemory):
    def construct(self, row, translated, *, gradient="both"):
        length = translated.get_seq_length()
        header = len(self.engine.prefix_ids)
        model = self.engine.model
        body = K.flat(translated)
        construction = tuple(x.detach() for x in body) if gradient == "direct" else body

        def build(prompt, *cache_values):
            cache = K.from_values(cache_values, model)
            positions = torch.arange(length, length + self.M, device=prompt.device)
            with self.routing.context("construct", header, length, self.M, positions):
                result = C.decoder(model)(
                    inputs_embeds=prompt[None].to(model.dtype), past_key_values=cache,
                    attention_mask=torch.ones(1, length + self.M, device=prompt.device, dtype=torch.long),
                    position_ids=positions[None], use_cache=True,
                )
            full = K.flat(result.past_key_values)
            assert all(x.shape[2] == length + self.M for x in full)
            return tuple(x[:, :, length:, :].clone() for x in full)

        inputs = (self.memory.prompt, *construction)
        if self.activation_checkpointing and torch.is_grad_enabled():
            memory = checkpoint(build, *inputs, use_reentrant=False, preserve_rng_state=True)
        else:
            memory = build(*inputs)
        if gradient == "memory":
            body = tuple(x.detach() for x in body)
        return C.DynamicCache(ddp_cache_data=[
            (torch.cat((body[i], memory[i]), 2), torch.cat((body[i + 1], memory[i + 1]), 2))
            for i in range(0, len(body), 2)
        ], config=model.config)


def load():
    ref = G.reference
    engine_class = ref.Z.scoped_module("joint_rank64_engine", ref.B.ROOT / "engine.py", ref.B).Engine
    engine = engine_class()
    rank = engine.correction
    memory = JointMemory(engine, CFG)
    for module, path, manifest in ((rank, RANK64, ref.B.identity()), (memory, MEMORY, ref.identity())):
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == manifest and cp["step"] == 1536
        module.load_state_dict(cp["state"], strict=True)
    assert sum(p.numel() for p in rank.parameters()) == 9437184
    assert sum(p.numel() for p in memory.parameters()) == 409600
    engine.unchanged()
    return engine, rank, memory


def translated(engine, rank, row):
    source = engine.source(row)
    return engine.translated(row, rank, source, checkpointed=True)


def optimizer_for(rank, memory):
    return torch.optim.AdamW([
        dict(params=rank.parameters(), lr=SETTINGS["rank64_lr"]),
        dict(params=memory.parameters(), lr=SETTINGS["memory_lr"]),
    ], weight_decay=CFG["weight_decay"])


def train_batch(engine, rank, memory, opt, scaler, batch, step):
    joint = torch.nn.ModuleDict({"rank64": rank, "memory": memory})
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    for attempt in range(9):
        opt.zero_grad(set_to_none=True)
        details = []
        for row, kind in batch:
            body = translated(engine, rank, row)
            native = engine.native_cache(row)
            loss, detail = losses(engine, memory, row, body, native, kind, EPOCH)
            scaler.scale(loss / len(batch)).backward()
            details.append(detail)
            del loss, body, native
        outcome = G.O.finish_backward(joint, opt, scaler)
        if outcome["accepted"]:
            engine.unchanged()
            return ({key: sum(entry[key] for entry in details) / len(batch)
                     for key in ("loss", "task_loss", "full_kl")}, outcome, attempt)
        C.log("JOINT AMP RETRY", step, attempt + 1, outcome["new_scale"])
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
    raise FloatingPointError(f"Joint gradients remain nonfinite at step {step}")


def smoke():
    verify()
    if G.done("checks/completed.json"):
        return
    engine, rank, memory = load()
    rows = G.rows("train")
    probe = next(row for row in rows if row["task"] == "arc")
    with torch.no_grad():
        body = translated(engine, rank, probe)
        old = DynamicMemory(engine, CFG)
        old.load_state_dict(memory.state_dict(), strict=True)
        previous = old.construct(probe, body)
        old.routing.close()
        current = memory.construct(probe, body)
        assert all(torch.equal(a, b) for a,b in zip(K.flat(previous), K.flat(current)))
        del old, body, previous, current
    path_norms = {}
    for mode in ("direct", "memory"):
        body = translated(engine, rank, probe)
        cache = memory.construct(probe, body, gradient=mode)
        logits = memory.token_logits(probe, cache, probe["gold_tokens"], rn=True)
        target = torch.tensor(probe["gold_tokens"], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        gradients = torch.autograd.grad(loss, tuple(rank.parameters()), allow_unused=True)
        nonzero = [g for g in gradients if g is not None]
        assert nonzero and all(torch.isfinite(g).all() for g in nonzero)
        norm = sum(float(g.square().sum()) for g in nonzero) ** 0.5
        assert norm > 0, mode
        path_norms[mode] = norm
        C.log("JOINT GRADIENT PATH", mode, norm)
        del logits, target, loss, gradients, nonzero, cache, body
    opt, scaler = optimizer_for(rank, memory), G.T.scaler_for()
    before = C.versions(rank), C.versions(memory)
    longest = max(rows, key=lambda r: len(r["content_ids"]) + len(r["gold_tokens"]))
    metrics, _, _ = train_batch(engine, rank, memory, opt, scaler, [(longest, "gold")], 0)
    assert all(a != b for a,b in zip(before, (C.versions(rank), C.versions(memory))))
    C.log("JOINT LONG BACKWARD PASSED", longest["id"], metrics)
    train_batch(engine, rank, memory, opt, scaler, batches(EPOCH)[0], 1)
    mark("checks/completed.json", status="completed", cache_parity=True, gradients=path_norms,
         longest=longest["id"], peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    C.log("JOINT GPU CHECKS PASSED")


@torch.no_grad()
def validate(engine, rank, memory):
    if G.done("validation/summary.json"):
        return
    metrics = []
    rows = G.rows("validation")
    for i,row in enumerate(rows, 1):
        if G.paused():
            raise SystemExit(75)
        path = f'validation/per_sample/{row["id"]}.json'
        if (ROOT / path).exists():
            saved = C.read(ROOT / path)
            assert saved["manifest_sha256"] == G.identity()
        else:
            body = translated(engine, rank, row)
            native = engine.native_cache(row)
            _, details = losses(engine, memory, row, body, native, "gold" if i % 2 else "native", 0)
            mark(path, metrics=details)
            saved = C.read(ROOT / path)
        metrics.append(saved["metrics"])
        if i == 1 or i % 32 == 0:
            status("validation", sample=i, total=len(rows))
            C.log("JOINT VALIDATE", i, len(rows))
    mark("validation/summary.json", status="completed", n=len(rows),
         metrics={key: sum(x[key] for x in metrics) / len(metrics) for key in ("loss", "task_loss", "full_kl")})


def restore(cp, rank, memory):
    assert cp["manifest_sha256"] == G.identity()
    rank.load_state_dict(cp["rank64"], strict=True)
    memory.load_state_dict(cp["memory"], strict=True)


def train():
    verify()
    assert G.done("checks/completed.json")
    if G.done("runs/completed.json"):
        return
    engine, rank, memory = load()
    opt, scaler = optimizer_for(rank, memory), G.T.scaler_for()
    start = 0
    if LAST.exists():
        cp = torch.load(LAST, map_location="cpu", weights_only=True)
        restore(cp, rank, memory)
        opt.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"])
        torch.set_rng_state(cp["rng"])
        torch.cuda.set_rng_state(cp["cuda_rng"])
        start = cp["step"]
        del cp
    path = ROOT / "runs/steps.jsonl"
    prior = [r for r in C.read_jsonl(path) if r["step"] <= start] if path.exists() else []
    assert [r["step"] for r in prior] == list(range(1, start + 1))
    C.write_jsonl(path, prior)

    def save(step, final=False):
        state = dict(manifest_sha256=G.identity(), step=step, rank64=rank.state_dict(), memory=memory.state_dict())
        if final:
            C.atomic_save(state, FINAL)
        else:
            state.update(optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
            C.atomic_save(state, LAST)

    if not LAST.exists():
        save(start)
    with path.open("a") as stream:
        for step,batch in enumerate(batches(EPOCH), 1):
            if step <= start:
                continue
            if G.paused():
                save(step - 1)
                raise SystemExit(75)
            metrics, outcome, retries = train_batch(engine, rank, memory, opt, scaler, batch, step)
            record = dict(step=step, total=TOTAL, metrics=metrics, gradient_norm=outcome["gradient_norm"],
                          amp_retries=retries, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            stream.write(G.json.dumps(record) + "\n")
            stream.flush()
            status("training", **record)
            if step == 1 or step % 4 == 0:
                C.log("JOINT TRAIN", record)
            if step <= 4 or step % CFG["checkpoint_every"] == 0 or step == TOTAL:
                save(step)
    rank.eval()
    memory.eval()
    validate(engine, rank, memory)
    save(TOTAL, final=True)
    mark("runs/completed.json", status="completed", updates=TOTAL, checkpoint_sha256=C.sha(FINAL))


@torch.no_grad()
def evaluate():
    verify()
    assert G.done("runs/completed.json")
    if G.done("summary.json"):
        return
    engine, rank, memory = load()
    assert C.sha(FINAL) == C.read(ROOT / "runs/completed.json")["checkpoint_sha256"]
    cp = torch.load(FINAL, map_location="cpu", weights_only=True)
    assert cp["step"] == TOTAL
    restore(cp, rank, memory)
    rank.eval().requires_grad_(False)
    memory.eval().requires_grad_(False)
    engine.mode = "test"
    test = G.rows("test")
    tasks = {}
    for task, count in CFG["test_counts"].items():
        records = []
        for i,row in enumerate((r for r in test if r["task"] == task), 1):
            if G.paused():
                raise SystemExit(75)
            rel = f'evaluation/{task}/per_sample/{row["id"]}.json'
            path = ROOT / rel
            if not path.exists():
                ref = C.read(REF / rel)
                prior = {arm: ref["conditions"][arm] for arm in ARMS[:-1]}
                body = translated(engine, rank, row)
                cache = memory.construct(row, body)
                result = score(memory, row, cache, rn=True)
                del body, cache
                assert engine.student_test_native_prefills == 0
                mark(rel, id=row["id"], question=row["question"], gold=row["gold"], conditions={**prior, "joint": result})
            record = C.read(path)
            assert record["manifest_sha256"] == G.identity() and record["id"] == row["id"]
            records.append(record["conditions"])
            if i == 1 or i % 8 == 0:
                status("evaluation", task=task, sample=i, total=count)
                C.log("JOINT EVAL", task, i, count)
        assert len(records) == count
        keys = ("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else ("correct",)
        if task == "gsm8k":
            keys += ("legacy_correct",)
        metrics = {arm: {k: sum(r[arm][k] for r in records) / count for k in keys} for arm in ARMS}
        diffs = {arm: {k: C.paired([r["joint"][k] - r[arm][k] for r in records]) for k in keys}
                 for arm in ARMS[:-1]}
        tasks[task] = dict(n=count, metrics=metrics, joint_minus=diffs)
        mark(f"evaluation/{task}/summary.json", status="completed", **tasks[task])
    lines = ["# DST+Rank64 and Dynamic Memory joint fine-tuning (384 updates)", "",
             "| Task | Metric | Native | Rank64 | Rank64+Static | Rank64+Dynamic before joint | Joint |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task, vals in tasks.items():
        for key in vals["metrics"]["joint"]:
            values = " | ".join(f'{vals["metrics"][arm][key]:.2%}' for arm in ARMS)
            lines.append(f"| {task} | {key} | {values} |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    mark("summary.json", status="completed", tasks=tasks)
    status("completed")


if __name__ == "__main__":
    {"prepare": prepare, "smoke": smoke, "train": train, "evaluate": evaluate}[sys.argv[1]]()
