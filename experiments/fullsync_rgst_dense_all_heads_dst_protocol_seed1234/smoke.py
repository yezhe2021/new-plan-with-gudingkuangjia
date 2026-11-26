"""Exercise the two existing RGST training objectives with a single dense rank."""
import argparse
import fcntl
import gc
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT.parent / "fullsync_rgst_r16_r32_dst_protocol_seed1234"
sys.path.insert(0, str(CONTROL))
import train as locked

dense = locked.fixed.load_module("rgst_dense_model", ROOT / "model.py")
DenseAllHeads, DenseRidge = dense.DenseAllHeads, dense.DenseRidge
HEADS, HIDDEN = dense.HEADS, dense.HIDDEN

C, torch, CFG = locked.C, locked.torch, locked.CFG
RANK = 32
DEVICE = "cuda"
GIB = 2**30


class Experiment(locked.Experiment):
    def cached_features(self, source):
        # Keep memory bounded by one token chunk during activation checkpoint replay.
        return None


@torch.no_grad()
def verify_mapping(experiment, model, row):
    sample = tuple(tensor[:, :8] for tensor in experiment.source(row))
    old_x, old_base = experiment.ridge.ridge.features(*sample)
    x, base = experiment.ridge.features(*sample)
    torch.testing.assert_close(base, old_base, rtol=0, atol=0)
    shuffled = x.reshape(x.size(0), x.size(1), 4, HEADS, -1)
    by_head = shuffled.transpose(2, 3).permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(-2)
    torch.testing.assert_close(by_head, old_x, rtol=0, atol=0)
    actual = model.coordinates(x, experiment.stats)
    normalized = (x.float() - experiment.stats["center_x"]) / experiment.stats["scale_x"]
    group, head = 35, 7
    hidden_slice = slice(head * HIDDEN, (head + 1) * HIDDEN)
    output_index = group * HEADS + head
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = torch.nn.functional.gelu(
            normalized[group] @ model.down[group, :, hidden_slice] + model.bias_down[group, :, hidden_slice]
        )
        expected = hidden @ model.up[output_index] + model.bias_up[output_index]
    torch.testing.assert_close(actual[output_index], expected, rtol=0.02, atol=0.015)
    delta = model(x, base, experiment.stats) - base
    basis = experiment.stats["axes"][..., :model.rank]
    orthogonal = delta - (delta @ basis) @ basis.transpose(-1, -2)
    relative_error = float(orthogonal.norm() / delta.norm())
    assert relative_error < 1e-5, relative_error
    return dict(ridge_exact=True, head_order_exact=True, independent_head_parity=True,
                complement_relative_error=relative_error)


def peak():
    return dict(allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                reserved_gib=torch.cuda.max_memory_reserved() / GIB)


def run_batch(experiment, model, optimizer, scale, stage, batch):
    models = {str(RANK): model}
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for retry in range(9):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        def backward(_, loss):
            scale.scale(loss / len(batch)).backward()
        for row, kind in batch:
            source = experiment.source(row)
            if stage == "stage1":
                metrics = locked.reconstruct(experiment, models, row, source, experiment.target(row), backward)
            else:
                metrics = locked.functional(experiment, models, row, source, kind, 1, backward)
            losses.append(dict(id=row["id"], kind=kind, body=len(row["content_ids"]),
                               gold=len(row["gold_tokens"]), metrics=metrics[str(RANK)]))
            print(f"{stage} {len(losses)}/{len(batch)} {row['id']} {kind}", flush=True)
        scale.unscale_(optimizer)
        assert all(p.grad is not None for p in model.parameters())
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip"]))
        if norm > 0 and norm < float("inf"):
            break
        scale.update(new_scale=scale.get_scale() / 2)
    else:
        raise FloatingPointError(f"{stage} exceeded the RGST retry budget")
    before = model.down[0, :8, :8].detach().clone()
    scale.step(optimizer)
    scale.update()
    assert len(optimizer.state) == len(list(model.parameters()))
    assert not torch.equal(before, model.down[0, :8, :8])
    torch.cuda.synchronize()
    return dict(seconds=time.monotonic() - start, free_before_gib=free/GIB, total_gib=total/GIB,
                gradient_norm=norm, retries=retry, adamw_updated=True, samples=losses, **peak())


def main(offload_layers):
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(CFG["seed"])
    experiment = Experiment()
    experiment.ridge = DenseRidge(experiment.ridge)
    original = C.read(CONTROL / "manifest.json")
    revision = C.read(CONTROL / "runtime_revision.json")
    assert revision["original_manifest_sha256"] == C.sha(CONTROL / "manifest.json")
    assert all(C.sha(CONTROL / name) == digest for name, digest in revision["code"].items())
    assert revision["check_sha256"] == C.sha(CONTROL / "speed_check.json")
    assert experiment.ridge_sha == original["ridge_sha256"]
    stats = CONTROL / "statistics.pt"
    all_heads = ROOT.parent / "fullsync_rgst_all_heads_dst_protocol_seed1234/manifest.json"
    assert C.sha(stats) == C.read(all_heads)["statistics_sha256"]
    experiment.stats = dense.input_statistics(torch.load(stats, map_location=DEVICE, weights_only=True))
    experiment.identity = C.sha(CONTROL / "manifest.json")
    experiment.fast_capture = revision["enabled"]["capture"]
    experiment.fast_functional = revision["enabled"]["functional"]
    experiment.fast_decode = revision["enabled"]["decoder"]
    model = DenseAllHeads(RANK)
    assert sum(p.numel() for p in model.parameters()) == 304_441_344
    corpus = locked.fixed.rows("train")
    longest_body = max(corpus, key=lambda row: len(row["content_ids"]))
    longest_answer = max(corpus, key=lambda row: len(row["gold_tokens"]))
    batch_size = CFG["effective_batch"]
    check_row = min(corpus, key=lambda row: len(row["content_ids"]))
    engine = experiment.engine
    reference = None
    if offload_layers:
        from accelerate import cpu_offload

        reference = experiment.source(check_row)
        layers = C.layers_of(engine.sender)
        assert 0 < offload_layers < len(layers)
        for block in layers[:offload_layers]:
            cpu_offload(block, execution_device=torch.device("cuda:0"))
        torch.cuda.empty_cache()
        captured = experiment.source(check_row)
        assert all(torch.equal(a, b) for a, b in zip(reference, captured))
    frozen_models = (engine.model, engine.sender)
    versions = [
        [(name, id(p), p._version) for name, p in frozen.named_parameters() if not p.is_meta]
        for frozen in frozen_models
    ]
    results = dict(rank=RANK, gpu=torch.cuda.get_device_name(), parity=verify_mapping(experiment, model, check_row),
                   chunk=locked.CHUNK, ridge_sha256=experiment.ridge_sha, pca_sha256=C.sha(stats),
                   sender_cpu_offload_layers=offload_layers, sender_capture_exact=reference is not None)
    C.save(ROOT / "smoke.json", results)

    for stage in ("stage1", "stage2"):
        batch = next(iter(locked.sampler.batches(1, stage)))
        assert len(batch) == batch_size == 8
        learning_rate = CFG[stage]["learning_rate"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=CFG["weight_decay"], fused=True)
        scale = torch.amp.GradScaler(DEVICE, init_scale=64., growth_interval=1000000)
        results[stage] = run_batch(experiment, model, optimizer, scale, stage, batch)
        C.save(ROOT / "smoke.json", results)
        if stage == "stage2":
            # The first update has allocated both AdamW moments when the stress pass begins.
            extremes = [(longest_body, "student"), (longest_answer, "gold")]
            remaining = [item for item in batch if item[0]["id"] not in {row["id"] for row, _ in extremes}]
            stress_batch = [*remaining[:batch_size - len(extremes)], *extremes]
            assert len(stress_batch) == batch_size == 8
            results["stage2_stress"] = run_batch(experiment, model, optimizer, scale, stage, stress_batch)
            C.save(ROOT / "smoke.json", results)
        model.zero_grad(set_to_none=True)
        del optimizer, scale
        gc.collect()
        torch.cuda.empty_cache()
    assert versions == [
        [(name, id(p), p._version) for name, p in frozen.named_parameters() if not p.is_meta]
        for frozen in frozen_models
    ]
    assert all(not p.requires_grad and p.grad is None for frozen in frozen_models for p in frozen.parameters())
    C.save(ROOT / "smoke.json", dict(status="passed", **results))
    print("DENSE RGST SMOKE PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-sender-layers", type=int, default=0)
    args = parser.parse_args()
    with (ROOT / "smoke.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.offload_sender_layers)
