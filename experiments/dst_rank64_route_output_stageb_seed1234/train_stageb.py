"""Train the fresh Rank64 residual using the pinned short-data Stage B schedule."""
from collections import Counter
import json
from time import monotonic

from objective import ROOT, REFERENCE, C, CFG, G, METRICS, PROBES, BodyReader, Engine, components, objective, original, torch

STAGE = CFG["stage2"]
STEPS = CFG["updates_per_epoch"]
TOTAL = STAGE["epochs"] * STEPS
RUN = ROOT / "runs"
FINAL = RUN / f"epoch{STAGE['epochs']}_step{TOTAL}.pt"
TARGET_SHARE = 0.10
MAX_PROBE_SHARE = 0.20
PREPARED = REFERENCE / "prepared"


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def check_pause():
    if paused():
        raise SystemExit(75)


def register():
    old = C.read(REFERENCE / "manifest.json")
    splits = [PREPARED / f"{name}.jsonl" for name in ("train", "validation", "test")]
    for path in splits:
        assert C.sha(path) == old["sources"][str(path)]
        assert Counter(row["task"] for row in C.read_jsonl(path)) == Counter(CFG[f"{path.stem}_counts"])
    score_source = REFERENCE / "runs/stage2/epoch4_step1536.pt"
    assert C.sha(score_source) == C.read(REFERENCE / "runs/stage2/completed.json")["checkpoint_sha256"]
    paths = [*splits, REFERENCE / "manifest.json", REFERENCE / "config.json", score_source,
             *[REFERENCE / name for name in ("engine.py", "runtime.py", "residual.py", "score.py")]]
    manifest = dict(protocol="dst-fullbody-chat-v1", body_training_data="original DST Stage B", rank=CFG["rank"],
                    frozen_stage1_sha256=CFG["dst_stage1_sha256"], trainable_parameters=9437184,
                    sources={str(p): C.sha(p) for p in paths}, updates=TOTAL,
                    effective_batch=CFG["effective_batch"], prefix_quotas=CFG["prefix_quotas_per_batch"],
                    ce_weight=1., vocab_kl_weight=STAGE["lambda_kl"], learning_rate=STAGE["learning_rate"],
                    max_queries_per_layer=PROBES, aux_gradient_target=TARGET_SHARE, max_smoke_gradient_share=MAX_PROBE_SHARE,
                    implementation={name: C.sha(ROOT / name) for name in ("objective.py", "train_stageb.py")})
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)
    return C.sha(path)


def gradient_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [g.float().square().sum() for g in grads if g is not None]
    norm = float(torch.stack(squares).sum().sqrt())
    assert 0 < norm < float("inf")
    return norm


def calibrate(engine, reader, pin):
    path = ROOT / "checks" / "weights.json"
    if path.exists():
        record = C.read(path)
        assert record["manifest_sha256"] == pin
        return record["weights"]
    # Use only training rows. Zero-initialized up projections are the active parameters at step 0.
    training = G.rows("train")
    selected = [next(r for r in training if r["task"] == label) for label in CFG["train_counts"]]
    selected.append(max(training, key=lambda r: len(r["content_ids"])))
    correction = engine.correction
    key_up = tuple(block[-1].weight for block in correction.keys)
    value_up = tuple(block[-1].weight for block in correction.values)
    params = key_up + value_up
    samples = []
    for row in selected:
        check_pause()
        source = engine.source(row)
        parts = components(engine, reader, row, source, "gold", 1)
        ce, kl, route, output = parts
        norms = dict(functional=gradient_norm(ce+STAGE["lambda_kl"]*kl, params),
                     route=gradient_norm(route, key_up), output=gradient_norm(output, params),
                     output_to_v=gradient_norm(output, value_up))
        samples.append(dict(id=row["id"], task=row["task"], body_tokens=len(row["content_ids"]), norms=norms))
        del parts, ce, kl, route, output, source
        C.log("AUX GRADIENT SMOKE", row["id"], norms)
    combined = {k: sum(x["norms"][k]**2 for x in samples)**0.5 for k in samples[0]["norms"]}
    weights = {}
    for term in ("route", "output"):
        pooled_ratio = combined["functional"] / combined[term]
        smallest_ratio = min(x["norms"]["functional"] / x["norms"][term] for x in samples)
        weights[term] = min(TARGET_SHARE * pooled_ratio, MAX_PROBE_SHARE * smallest_ratio)
    C.save(path, dict(status="completed", manifest_sha256=pin, weights=weights, samples=samples,
                      pooled_gradient_shares={k: weights[k] * combined[k] / combined["functional"] for k in weights},
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    return weights


def backward_batch(engine, reader, opt, scaler, batch, epoch, weights):
    module = engine.correction
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True)
        details = []
        for row, kind in batch:
            source = engine.source(row)
            loss, metrics = objective(components(engine, reader, row, source, kind, epoch), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite objective on {row['id']}")
            scaler.scale(loss / len(batch)).backward()
            details.append(metrics)
            del source, loss
        scaler.unscale_(opt)
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads)
        if not all(torch.isfinite(g).all() for g in grads):
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale:
                raise FloatingPointError("Gradient scale did not decrease")
            C.log("RETRY BATCH", epoch, retry, scale, scaler.get_scale())
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda)
            continue
        norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), CFG["gradient_clip"]))
        assert 0 < norm < float("inf")
        scaler.step(opt)
        scaler.update()
        return mean_metrics(details), norm, retry
    raise FloatingPointError("Gradient retries exhausted")


def mean_metrics(records):
    return {k: sum(r[k] for r in records) / len(records) for k in METRICS}


def status(phase, **fields):
    C.save(ROOT / "status.json", dict(phase=phase, total_steps=TOTAL, **fields))


@torch.no_grad()
def validate(engine, reader, epoch, step, weights, pin):
    module = engine.correction
    module.eval()
    records = []
    rows = G.rows("validation")
    for i, row in enumerate(rows, 1):
        check_pause()
        path = ROOT / "validation" / f"epoch{epoch}" / (row["id"] + ".json")
        if path.exists():
            record = C.read(path)
            assert record["manifest_sha256"] == pin and record["step"] == step
        else:
            source = engine.source(row)
            _, metrics = objective(components(engine, reader, row, source, "native" if i%2 else "gold", 0), weights)
            record = dict(manifest_sha256=pin, step=step, id=row["id"], task=row["task"], metrics=metrics)
            C.save(path, record)
        records.append(record["metrics"])
        status("validation", epoch=epoch, step=step, sample=i, total_rows=len(rows))
    result = mean_metrics(records)
    C.save(ROOT / "validation" / f"epoch{epoch}" / "summary.json", dict(step=step, metrics=result, manifest_sha256=pin))
    return result


def save(module, opt, scaler, epoch, batch, step, pin):
    weights = dict(manifest_sha256=pin, epoch=epoch, batch=batch, step=step, state=module.state_dict())
    if batch == STEPS:
        C.atomic_save(weights, RUN / f"epoch{epoch}_step{step}.pt")
    C.atomic_save(dict(**weights, optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                       cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), RUN / "last.pt")


def train():
    pin = register()
    complete = RUN / "completed.json"
    if complete.exists():
        record = C.read(complete)
        assert record["manifest_sha256"] == pin and record["checkpoint_sha256"] == C.sha(FINAL)
        return
    check_pause()
    status("loading", step=0)
    engine = Engine()
    reader = BodyReader(engine)
    module = engine.correction
    assert sum(p.numel() for p in module.parameters()) == 9437184
    opt = torch.optim.AdamW(module.parameters(), lr=STAGE["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64., growth_interval=1000000)
    last = RUN / "last.pt"
    logs = RUN / "steps.jsonl"
    epoch_start, batch_start, step = 1, 0, 0
    status("calibrating_auxiliary_gradients", step=step)
    weights = calibrate(engine, reader, pin)
    if last.exists():
        cp = torch.load(last, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == pin
        module.load_state_dict(cp["state"], strict=True)
        opt.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"])
        epoch_start, batch_start, step = cp["epoch"], cp["batch"], cp["step"]
        torch.set_rng_state(cp["cpu_rng"])
        torch.cuda.set_rng_state(cp["cuda_rng"])
        del cp
    if logs.exists():
        records = C.read_jsonl(logs)
        assert [r["step"] for r in records[:step]] == list(range(1, step+1))
        if len(records) > step:
            C.write_jsonl(logs, records[:step])
    if not last.exists():
        save(module, opt, scaler, 1, 0, 0, pin)
    started = monotonic()
    with logs.open("a") as log:
        for epoch in range(epoch_start, STAGE["epochs"]+1):
            module.train()
            for index, batch in enumerate(original.batches(epoch, "stage2"), 1):
                if epoch == epoch_start and index <= batch_start:
                    continue
                if paused():
                    save(module, opt, scaler, epoch, index-1, step, pin)
                    check_pause()
                assert len(batch) == CFG["effective_batch"]
                metrics, norm, retries = backward_batch(engine, reader, opt, scaler, batch, epoch, weights)
                engine.unchanged()
                step += 1
                record = dict(step=step, epoch=epoch, batch=index, metrics=metrics, gradient_norm=norm,
                              amp_retries=retries, elapsed_seconds=monotonic()-started,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                log.write(json.dumps(record)+"\n")
                log.flush()
                status("training", **record)
                if step <= 2 or step % 4 == 0:
                    C.log("STAGE B TRAIN", json.dumps(record))
                if step <= 2 or step % CFG["checkpoint_every"] == 0 or index == STEPS:
                    save(module, opt, scaler, epoch, index, step, pin)
            assert step == STEPS * epoch
            validate(engine, reader, epoch, step, weights, pin)
    assert step == TOTAL and FINAL.exists()
    C.save(complete, dict(status="completed", manifest_sha256=pin, steps=step, weights=weights, checkpoint_sha256=C.sha(FINAL)))
    status("training_completed", step=step)


if __name__ == "__main__":
    train()
