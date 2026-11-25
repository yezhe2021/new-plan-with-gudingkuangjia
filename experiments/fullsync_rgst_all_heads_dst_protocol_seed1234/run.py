"""Run the parameter-matched RGST sender-head ablation on the locked DST suite."""

import argparse
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT.parent / "fullsync_rgst_r16_r32_dst_protocol_seed1234"
sys.path.insert(0, str(CONTROL))
import train as locked

C, CFG, fixed, torch = locked.C, locked.CFG, locked.fixed, locked.torch
scoring = fixed.load_module("rgst_control_scoring", CONTROL / "evaluate.py")
SCORER_TASKS, reader_tools = scoring.SCORER_TASKS, scoring.reader_tools
L, H = locked.model_code.LAYERS, locked.model_code.HEADS
INPUT, HIDDEN = (getattr(locked.model_code, k) for k in ("INPUT", "HIDDEN"))
GROUP = HIDDEN // H
COHORT = ROOT.parent / "fullsync_value_subspace_oracle_seed1234" / "manifest.json"


class AllHeadsRGST(locked.RGST):
    """Each output head sees all input heads, with 16 hidden units per input head."""

    def forward(self, features, ridge, stats):
        x = (features.float() - stats["center_x"]) / stats["scale_x"]
        x = x.unflatten(0, (2, L, H))
        # Store the same number of weights as the local-head control. Each output
        # head has independent weights for each input head; hidden groups are
        # sparse across input heads and are mixed by the original output layer.
        down = self.down.view(2, L, H, INPUT, H, GROUP)
        down = down.permute(0, 1, 4, 3, 2, 5).flatten(-2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = (x @ down).unflatten(-1, (H, GROUP))
            hidden = hidden.permute(0, 1, 4, 3, 2, 5).flatten(0, 2).flatten(-2)
            z = torch.nn.functional.gelu(hidden + self.bias_down) @ self.up + self.bias_up
        basis = stats["axes"][..., :self.rank]
        current = (ridge.float() - stats["center_y"]) @ basis
        return ridge.float() + (z.float() - current) @ basis.transpose(-1, -2)


def prepare():
    original = C.sha(CONTROL / "manifest.json")
    revision = C.read(CONTROL / "runtime_revision.json")
    assert revision["original_manifest_sha256"] == original
    assert all(C.sha(CONTROL / k) == v for k, v in revision["code"].items())
    assert revision["check_sha256"] == C.sha(CONTROL / "speed_check.json")
    stats = CONTROL / "statistics.pt"
    manifest = dict(
        control_manifest_sha256=original,
        code_sha256=C.sha(__file__),
        runtime_revision_sha256=C.sha(CONTROL / "runtime_revision.json"),
        statistics_sha256=C.sha(stats),
        cohort_sha256=C.sha(COHORT),
        control_stage2_sha256={str(r): C.sha(CONTROL / "stage2" / f"final_r{r}.pt") for r in locked.RANKS},
        features_per_output_head=INPUT * H,
        hidden_units_per_source_head=GROUP,
        trainable_parameters={"16": 39011328, "32": 40200192},
        stage1=CFG["stage1"],
        stage2=CFG["stage2"],
        selection="new lowest validation loss and paired final epoch",
    )
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)
    experiment = locked.Experiment()
    assert experiment.ridge_sha == C.read(CONTROL / "manifest.json")["ridge_sha256"]
    experiment.identity = C.sha(path)
    experiment.stats = torch.load(stats, map_location="cuda", weights_only=True)
    experiment.fast_capture, experiment.fast_decode, experiment.fast_functional = (
        revision["enabled"][key] for key in ("capture", "decoder", "functional")
    )
    models = {}
    for rank in locked.RANKS:
        torch.manual_seed(CFG["seed"])
        model = AllHeadsRGST(rank)
        assert sum(p.numel() for p in model.parameters()) == manifest["trainable_parameters"][str(rank)]
        models[str(rank)] = model
    return experiment, models


def snapshot_on_validation(experiment, models, stage, epoch):
    step = epoch * CFG["updates_per_epoch"]
    for rank, model in models.items():
        path = ROOT / stage / f"r{rank}_epoch{epoch}.pt"
        if not path.exists():
            locked.save(path, dict(manifest_sha256=experiment.identity, step=step, model=model.state_dict()))
    validate(experiment, models, stage, epoch)


def load_model(model, path, manifest_sha256):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    assert state["manifest_sha256"] == manifest_sha256
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(experiment, models):
    cohort = C.read(COHORT)["test"]
    assert len(cohort) == 8 and all(len(ids) == 16 for ids in cohort.values())
    end = CFG["stage2"]["epochs"]
    losses = {e: C.read(ROOT / "stage2" / f"validation_epoch{e}.json") for e in range(1, end+1)}
    best = {r: min(losses, key=lambda e: losses[e][r]["loss"]) for r in models}
    models = {f"allheads{r}_final": load_model(m, ROOT / "stage2" / f"r{r}_epoch{end}.pt",
              experiment.identity) for r, m in models.items()}
    for rank, epoch in best.items():
        if epoch != end:
            models[f"allheads{rank}_best"] = load_model(
                AllHeadsRGST(int(rank)), ROOT / "stage2" / f"r{rank}_epoch{epoch}.pt", experiment.identity)
    control_manifest = C.sha(CONTROL / "manifest.json")
    for rank in locked.RANKS:
        models[f"localhead{rank}_final"] = load_model(
            locked.RGST(rank), CONTROL / "stage2" / f"final_r{rank}.pt", control_manifest)
    engine, reader = experiment.engine, experiment.reader
    engine.mode = "test"
    rows = {r["id"]: r for r in fixed.rows("test")}
    results = {}
    for task, ids in cohort.items():
        results[task] = []
        for sample_id in ids:
            row = rows[sample_id]
            assert row["task"] == task
            path = ROOT / "per_sample" / task / (sample_id + ".json")
            if path.exists():
                record = C.read(path)
                assert record["manifest_sha256"] == experiment.identity and record["id"] == sample_id
            else:
                source = experiment.source(row)
                features = experiment.cached_features(source)
                # Use single-cache generation for each new method, including long GSM8K answers.
                scores = {}
                for name, model in models.items():
                    cache = experiment.make_cache(experiment.predict(source, model, prepared=features))
                    scores[name] = fixed.score_locked(reader, row, cache)
                    del cache
                for rank, epoch in best.items():
                    if epoch == end:
                        scores[f"allheads{rank}_best"] = scores[f"allheads{rank}_final"]
                record = dict(id=sample_id, task=task, question=row["question"], gold=row["gold"],
                              manifest_sha256=experiment.identity, results=scores)
                C.save(path, record)
            results[task].append(record["results"])
            C.save(ROOT / "status.json", dict(stage="evaluation", task=task,
                   completed=sum(map(len, results.values())), total=128))

    def score_group(tasks):
        samples = [sample for task in tasks for sample in results[task]]
        return {name: {field: mean(sample[name][field] for sample in samples)
                       for field in ("correct", "f1", "em", "reached_cap") if field in samples[0][name]}
                for name in samples[0]}

    C.save(ROOT / "summary.json", dict(status="completed", samples=128,
        manifest_sha256=experiment.identity, best_epochs=best, validation=losses,
        tasks={task: score_group([task]) for task in results},
        choice=score_group(sorted(SCORER_TASKS)), multi_hop=score_group(("hotpotqa", "2wiki", "musique")),
        gsm8k=score_group(["gsm8k"])))
    C.save(ROOT / "status.json", dict(stage="completed", evaluated=128))


validate = locked.validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    experiment, models = prepare()
    locked.ROOT = ROOT
    if args.smoke:
        row = min(fixed.rows("validation"), key=lambda r: len(r["content_ids"]))
        source = experiment.source(row)
        target = experiment.target(row)
        backward = lambda _, loss: loss.backward()
        for trial in (
            lambda: locked.reconstruct(experiment, models, row, source, target, backward),
            lambda: locked.functional(experiment, models, row, source, "gold", 0, backward),
        ):
            print(trial(), flush=True)
            for model in models.values():
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                model.zero_grad(set_to_none=True)
        C.log("all-head reconstruction and functional smoke passed", row["id"])
        return
    locked.validate = snapshot_on_validation
    for stage in ("stage1", "stage2"):
        (ROOT / stage).mkdir(exist_ok=True)
        locked.train(experiment, models, stage)
    evaluate(experiment, models)


if __name__ == "__main__":
    main()
