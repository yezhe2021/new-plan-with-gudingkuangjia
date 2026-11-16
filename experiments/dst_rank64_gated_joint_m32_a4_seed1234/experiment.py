"""One 384-update continuation of the completed Gated Dynamic model."""

import json
import sys

import experiment_common as G
from experiment_common import C, CFG, K, ROOT, SETTINGS, mark, status, torch
from fresh_cohort import prepare as prepare_cohort
from gated_joint import GatedJointMemory, original
from runtime_new import batches, losses
from score import score

RANK = G.reference.RANK64
INITIAL = ROOT / "initial/runs/epoch4_step1536.pt"
LAST = ROOT / "runs/last.pt"
FINAL = ROOT / "runs/step384.pt"
TOTAL = SETTINGS["updates"]
EPOCH = SETTINGS["training_epoch"]


def verify():
    for path, expected in C.read(ROOT / "manifest.json")["pins"].items():
        assert C.sha(path) == expected, path


def prepare():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    ref = G.reference
    ref.verify()
    assert ref.done("completed.json") and ref.identity() == SETTINGS["reference_manifest_sha256"]
    assert C.sha(RANK) == SETTINGS["rank64_checkpoint_sha256"]
    assert C.sha(INITIAL) == SETTINGS["gated_checkpoint_sha256"]
    gated_manifest = ROOT / "initial/manifest.json"
    assert C.sha(gated_manifest) == SETTINGS["gated_manifest_sha256"]
    assert C.read(gated_manifest)["settings"]["inputs_sha256"]["rank64.pt"] == C.sha(RANK)
    order = [[(row["id"], kind) for row, kind in group] for group in batches(EPOCH)]
    assert len(order) == TOTAL == 384
    fresh = prepare_cohort()
    paths = [*ROOT.glob("*.py"), ROOT / "config.json", ROOT / "prepared/test.jsonl",
             ROOT / "initial/gated_memory.py", gated_manifest, INITIAL, RANK, ref.ROOT / "manifest.json"]
    parent = C.read(ref.ROOT / "manifest.json")
    C.save(ROOT / "manifest.json", dict(
        config=SETTINGS, pins={**parent["sources"], **{str(p): C.sha(p) for p in paths}},
        fresh_cohort=fresh, native_prefix_sources=parent["native_prefix_sources"],
        batch_schedule_sha256=C.digest(order), trainable_parameters=9846820,
    ))
    C.log("GATED JOINT PREPARED", G.identity())


def load():
    ref = G.reference
    engine_class = ref.Z.scoped_module("gated_joint_engine", ref.B.ROOT / "engine.py", ref.B).Engine
    engine = engine_class()
    rank = engine.correction
    memory = GatedJointMemory(engine, CFG)
    for module, path, identity in (
        (rank, RANK, ref.B.identity()), (memory, INITIAL, SETTINGS["gated_manifest_sha256"]),
    ):
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == identity and cp["step"] == 1536
        module.load_state_dict(cp["state"], strict=True)
    assert torch.count_nonzero(memory.gates) > 0
    joint = torch.nn.ModuleDict(dict(rank64=rank, gated=memory))
    assert sum(p.numel() for p in joint.parameters()) == 9846820
    engine.unchanged()
    return engine, rank, memory, joint


def translated(engine, rank, row, source=None):
    if source is None:
        source = engine.source(row)
    return engine.translated(row, rank, source, checkpointed=True)


def group_norm(gradients):
    values = [g.float().norm().square() for g in gradients if g is not None]
    assert values
    return float(torch.stack(values).sum().sqrt())


def diagnostic(engine, rank, memory, row, step):
    name = f"diagnostics/step{step}.json"
    if G.done(name):
        return
    norms = {}
    parameters = tuple(rank.parameters())
    target = row["gold_tokens"]
    for path in ("body", "memory"):
        body = translated(engine, rank, row)
        logits = memory.token_logits(row, memory.construct(row, body, gradient=path), target, rn=True)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=logits.device))
        gradients = torch.autograd.grad(loss * 64, parameters, allow_unused=True)
        norms[path] = group_norm(gradients) / 64
        assert 0 < norms[path] < float("inf"), (step, path, norms)
        del body, logits, loss, gradients

    starts, contributions, handles = {}, {}, []
    for i, layer in enumerate(C.layers_of(engine.model)):
        def before(_module, inputs, layer=i):
            starts[layer] = inputs[0].float()
        def after(_module, inputs, layer=i):
            base = starts.pop(layer)
            delta = inputs[0].float() - base
            contributions[layer] = dict(delta=float(delta.norm()), body=float(base.norm()))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(before, prepend=True))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(after))
    try:
        with torch.no_grad():
            body = translated(engine, rank, row)
            bank = memory.construct(row, body)
            memory.logits(row, bank, rn=True)
    finally:
        for handle in handles:
            handle.remove()
    assert len(contributions) == 36
    with torch.no_grad():
        initial = torch.load(RANK, map_location="cuda", weights_only=True)["state"]
        def base_rank(*args):
            return torch.func.functional_call(rank, initial, args, strict=True)
        baseline = translated(engine, base_rank, row)
        diffs, sizes = [], []
        for a, b in zip(K.flat(body), K.flat(baseline)):
            diffs.append((a.float() - b.float()).norm().square())
            sizes.append(b.float().norm().square())
        relative_change = float((torch.stack(diffs).sum() / torch.stack(sizes).sum()).sqrt())
    mark(name, status="completed", step=step, id=row["id"], rank64_path_gradients=norms,
         per_layer_projection_input=contributions, tanh_gates=memory.gates.detach().tanh().tolist(),
         translated_kv_relative_l2=relative_change)
    C.log("GATED JOINT DIAGNOSTIC", step, norms, "KV relative L2", relative_change)


def optimizer_for(rank, memory):
    return torch.optim.AdamW([
        dict(params=rank.parameters(), lr=SETTINGS["rank64_lr"]),
        dict(params=memory.memory.parameters(), lr=SETTINGS["memory_lr"]),
        dict(params=[memory.gates], lr=SETTINGS["gate_lr"]),
    ], weight_decay=CFG["weight_decay"])


def update(engine, rank, memory, joint, optimizer, scaler, batch, step):
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    for retry in range(9):
        optimizer.zero_grad(set_to_none=True)
        details = []
        for row, kind in batch:
            body = translated(engine, rank, row)
            native = engine.native_cache(row)
            loss, values = losses(engine, memory, row, body, native, kind, EPOCH)
            scaler.scale(loss / len(batch)).backward()
            details.append(values)
            del loss, body, native
        scale = scaler.get_scale()
        groups = {name: group_norm(p.grad for p in module.parameters()) / scale
                  for name, module in (("rank64", rank), ("memory", memory.memory))}
        groups["gate"] = float(memory.gates.grad.float().norm()) / scale
        result = G.O.finish_backward(joint, optimizer, scaler)
        if result["accepted"]:
            assert all(value > 0 for value in groups.values()), groups
            engine.unchanged()
            means = {key: sum(v[key] for v in details)/len(batch) for key in ("loss", "task_loss", "full_kl")}
            return dict(metrics=means, gradient_norms=groups, amp_retries=retry)
        C.log("GATED JOINT AMP RETRY", step, retry + 1, result["new_scale"])
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
    raise FloatingPointError(f"Gradient overflow after retries at step {step}")


def smoke():
    verify()
    if G.done("checks/completed.json"):
        return
    engine, rank, memory, joint = load()
    rows = G.rows("train")
    probe = next(row for row in rows if row["task"] == "arc")
    with torch.no_grad():
        body = translated(engine, rank, probe)
        baseline = original.GatedDynamicMemory(engine, CFG)
        baseline.load_state_dict(memory.state_dict(), strict=True)
        expected = baseline.construct(probe, body)
        actual = memory.construct(probe, body)
        assert all(torch.equal(a,b) for a,b in zip(expected.bank, actual.bank))
        assert torch.equal(baseline.logits(probe, expected), memory.logits(probe, actual))
        gates = memory.gates.detach().clone()
        memory.gates.zero_()
        assert torch.equal(memory.logits(probe, actual), memory.logits(probe, body, rn=False))
        memory.gates.copy_(gates)
        baseline.routing.close()
        for handle in baseline._handles:
            handle.remove()
        del baseline, body, expected, actual
    diagnostic(engine, rank, memory, probe, 0)
    opt, scaler = optimizer_for(rank, memory), G.T.scaler_for()
    before = [C.versions(rank), C.versions(memory.memory), memory.gates._version]
    longest = max(rows, key=lambda r: len(r["content_ids"]) + len(r["gold_tokens"]))
    update(engine, rank, memory, joint, opt, scaler, [(longest, "gold")], 0)
    after = [C.versions(rank), C.versions(memory.memory), memory.gates._version]
    assert all(a != b for a,b in zip(before, after))
    update(engine, rank, memory, joint, opt, scaler, batches(EPOCH)[0], 1)
    mark("checks/completed.json", status="completed", original_and_gate_off_logits_exact=True,
         longest=longest["id"], peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    C.log("GATED JOINT GPU CHECKS PASSED")


def train():
    verify()
    assert G.done("checks/completed.json")
    if G.done("runs/completed.json"):
        return
    engine, rank, memory, joint = load()
    opt, scaler = optimizer_for(rank, memory), G.T.scaler_for()
    step = 0
    if LAST.exists():
        cp = torch.load(LAST, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == G.identity()
        joint.load_state_dict(cp["state"], strict=True)
        opt.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"])
        torch.set_rng_state(cp["rng"])
        torch.cuda.set_rng_state(cp["cuda_rng"])
        step = cp["step"]
        del cp
    path = ROOT / "runs/steps.jsonl"
    history = [r for r in C.read_jsonl(path) if r["step"] <= step] if path.exists() else []
    assert [r["step"] for r in history] == list(range(1, step+1))
    C.write_jsonl(path, history)

    def checkpoint():
        C.atomic_save(dict(manifest_sha256=G.identity(), step=step, state=joint.state_dict(),
            optimizer=opt.state_dict(), scaler=scaler.state_dict(),
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), LAST)

    schedule = batches(EPOCH)
    probe = next(row for row in G.rows("train") if row["task"] == "arc")
    with path.open("a") as stream:
        if step and step % SETTINGS["diagnostic_every"] == 0:
            diagnostic(engine, rank, memory, probe, step)
        for batch in schedule[step:]:
            if G.paused():
                checkpoint()
                raise SystemExit(75)
            record = update(engine, rank, memory, joint, opt, scaler, batch, step+1)
            step += 1
            record.update(step=step, total=TOTAL, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            status("training", **record)
            if step == 1 or step % 4 == 0:
                C.log("GATED JOINT TRAIN", record)
            if step <= 4 or step % 16 == 0:
                checkpoint()
            if step % SETTINGS["diagnostic_every"] == 0:
                diagnostic(engine, rank, memory, probe, step)
    assert step == TOTAL
    C.atomic_save(dict(manifest_sha256=G.identity(), step=step, state=joint.state_dict()), FINAL)
    mark("runs/completed.json", status="completed", updates=step, checkpoint_sha256=C.sha(FINAL))


@torch.no_grad()
def evaluate():
    verify()
    assert G.done("runs/completed.json")
    if G.done("summary.json"):
        return
    engine, rank, memory, joint = load()
    engine.mode = "test"
    test = C.read_jsonl(ROOT / "prepared/test.jsonl")
    initial = dict(state={k:v.cpu().clone() for k,v in joint.state_dict().items()})
    final = torch.load(FINAL, map_location="cpu", weights_only=True)
    assert final["manifest_sha256"] == G.identity() and final["step"] == TOTAL
    assert C.sha(FINAL) == C.read(ROOT / "runs/completed.json")["checkpoint_sha256"]
    joint.eval().requires_grad_(False)
    for arm, version in (("A", initial), ("D", final)):
        joint.load_state_dict(version["state"], strict=True)
        for i,row in enumerate(test, 1):
            if G.paused():
                raise SystemExit(75)
            name = f'evaluation/{arm}/{row["task"]}/{row["id"]}.json'
            if not (ROOT / name).exists():
                body = translated(engine, rank, row)
                prediction = score(memory, row, memory.construct(row, body), rn=True)
                assert engine.student_test_native_prefills == 0
                engine.unchanged()
                mark(name, id=row["id"], question=row["question"], gold=row["gold"], result=prediction)
                del body
            if i == 1 or i % 16 == 0:
                status("evaluation", arm=arm, sample=i, total=len(test), task=row["task"])
                C.log("GATED JOINT EVAL", arm, i, len(test))
    tasks, aggregate = {}, {"A": [], "D": []}
    lines = ["# Gated Dynamic joint continuation on fresh evaluation questions", "",
             "| Task | Metric | A starting Gated Dynamic | D joint 384 | D minus A (points) |",
             "|---|---|---:|---:|---:|"]
    for task in CFG["test_counts"]:
        selected = [r for r in test if r["task"] == task]
        n = len(selected)
        assert n == SETTINGS["evaluation_per_task"]
        values = {}
        for arm in ("A", "D"):
            values[arm] = []
            for row in selected:
                record = C.read(ROOT / f'evaluation/{arm}/{task}/{row["id"]}.json')
                assert record["manifest_sha256"] == G.identity() and record["id"] == row["id"]
                values[arm].append(record["result"])
        binary = "em" if task in ("hotpotqa","2wiki","musique") else "correct"
        primary = "f1" if binary == "em" else binary
        keys = (binary, "f1") if binary == "em" else (binary,)
        if task == "gsm8k":
            keys += ("legacy_correct",)
        metrics = {arm: {k:sum(r[k] for r in v)/n for k in keys} for arm,v in values.items()}
        paired = {k:C.paired([d[k]-a[k] for a,d in zip(values["A"],values["D"])]) for k in keys}
        flips = [d[binary]-a[binary] for a,d in zip(values["A"],values["D"])]
        details = dict(n=n, metrics=metrics, d_minus_a=paired,
                       rescued=[r["id"] for r,d in zip(selected,flips) if d>0],
                       harmed=[r["id"] for r,d in zip(selected,flips) if d<0])
        tasks[task] = details
        mark(f"evaluation/{task}_summary.json", status="completed", **details)
        for arm in aggregate:
            aggregate[arm].append(metrics[arm][primary])
        for key in keys:
            lines.append(f'| {task} | {key} | {metrics["A"][key]:.2%} | {metrics["D"][key]:.2%} | {100*paired[key]["mean"]:+.2f} |')
    (ROOT / "RESULTS.md").write_text("\n".join(lines)+"\n")
    mark("summary.json", status="completed", tasks=tasks,
         macro_average={arm:sum(vals)/len(vals) for arm,vals in aggregate.items()})
    status("completed")


if __name__ == "__main__":
    {"prepare": prepare, "smoke": smoke, "train": train, "evaluate": evaluate}[sys.argv[1]]()
