"""Train Dense All R32 on the frozen RGST protocol and evaluate on the fixed DST suite."""

import argparse
from contextlib import contextmanager
import gc
from statistics import mean
import sys

from smoke import C, CFG, CONTROL, DenseAllHeads, DenseRidge, Experiment, RANK, ROOT, dense, locked, torch

GROUPED = ROOT.parent / "fullsync_rgst_all_heads_dst_protocol_seed1234"
GROUPED_CHECKPOINT = GROUPED / "stage2/final_r32.pt"
RANK_KEY = str(RANK)
STAGES = ("stage1", "stage2")
METHODS = ("dense_r32", "grouped_r32")
BASE_VALIDATE = locked.validate


def prepare():
    original = CONTROL / "manifest.json"
    revision = CONTROL / "runtime_revision.json"
    statistics = CONTROL / "statistics.pt"
    grouped_manifest = GROUPED / "manifest.json"
    control = C.read(original)
    runtime = C.read(revision)
    smoke = C.read(ROOT / "smoke.json")
    assert smoke["status"] == "passed" and smoke["rank"] == RANK == 32
    assert smoke["chunk"] == locked.CHUNK
    assert runtime["original_manifest_sha256"] == C.sha(original)
    for name, expected in runtime["code"].items():
        assert C.sha(CONTROL / name) == expected, name
    assert runtime["check_sha256"] == C.sha(CONTROL / "speed_check.json")
    assert C.read(grouped_manifest)["code_sha256"] == C.sha(GROUPED / "run.py")
    stats_sha = C.sha(statistics)
    assert stats_sha == C.read(grouped_manifest)["statistics_sha256"] == smoke["pca_sha256"]
    assert C.sha(locked.RIDGE) == control["ridge_sha256"] == smoke["ridge_sha256"]
    train_data = locked.fixed.BASE / "prepared"
    for split, expected in control["data_hashes"].items():
        assert C.sha(train_data / (split + ".jsonl")) == expected

    manifest = dict(
        rank=RANK, trainable_parameters=304_441_344, control_manifest_sha256=C.sha(original),
        grouped_manifest_sha256=C.sha(grouped_manifest), grouped_final_sha256=C.sha(GROUPED_CHECKPOINT),
        runtime_revision_sha256=C.sha(revision), statistics_sha256=stats_sha,
        ridge_sha256=control["ridge_sha256"], data_hashes=control["data_hashes"],
        stage1=CFG["stage1"], stage2=CFG["stage2"], effective_batch=CFG["effective_batch"],
        updates_per_epoch=CFG["updates_per_epoch"], checkpoint_every=CFG["checkpoint_every"],
        test_counts=CFG["test_counts"], seed=CFG["seed"], chunk=locked.CHUNK,
        checkpoint_selection="fixed final epoch 4 in both methods",
        code={name: C.sha(ROOT / name) for name in ("model.py", "smoke.py", "run.py", "run.sh")},
    )
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)
    return C.sha(path), runtime["enabled"], statistics


def frozen_versions(engine):
    return [[(name, id(p), p._version) for name, p in model.named_parameters() if not p.is_meta]
            for model in (engine.model, engine.sender)]


class TrainingExperiment(Experiment):
    def __init__(self, identity, flags, statistics, offload_layers):
        super().__init__()
        self.identity = identity
        self.original_ridge = self.ridge
        self.original_stats = torch.load(statistics, map_location="cuda", weights_only=True)
        self.ridge = DenseRidge(self.original_ridge)
        self.stats = dense.input_statistics(self.original_stats)
        self.fast_capture = flags["capture"]
        self.fast_decode = flags["decoder"]
        self.fast_functional = flags["functional"]

        from accelerate import cpu_offload

        layers = C.layers_of(self.engine.sender)
        assert 0 < offload_layers < len(layers)
        for block in layers[:offload_layers]:
            cpu_offload(block, execution_device=torch.device("cuda:0"))
        torch.cuda.empty_cache()
        pin = frozen_versions(self.engine)

        def verify_frozen():
            assert frozen_versions(self.engine) == pin
            assert all(not p.requires_grad and p.grad is None
                       for frozen in (self.engine.model, self.engine.sender) for p in frozen.parameters())

        self.engine.unchanged = verify_frozen

    @contextmanager
    def grouped_features(self):
        original = self.ridge, self.stats
        self.ridge, self.stats = self.original_ridge, self.original_stats
        try:
            yield
        finally:
            self.ridge, self.stats = original


def validate_with_snapshot(experiment, models, stage, epoch):
    path = ROOT / stage / f"r{RANK}_epoch{epoch}.pt"
    if not path.exists():
        locked.save(path, dict(manifest_sha256=experiment.identity, stage=stage, epoch=epoch,
                              step=epoch * CFG["updates_per_epoch"], model=models[RANK_KEY].state_dict()))
    BASE_VALIDATE(experiment, models, stage, epoch)


@torch.no_grad()
def subspace_diagnostics(experiment, model, stage):
    output = ROOT / stage / "kv_validation.json"
    if output.exists():
        assert C.read(output)["manifest_sha256"] == experiment.identity
        return
    sums = torch.zeros((4, 2), dtype=torch.float64, device="cuda")
    basis = experiment.stats["axes"][..., :RANK]
    center = experiment.stats["center_y"]
    cohort = locked.fixed.rows("validation")

    def energy(tensor):
        return tensor.reshape(2, -1).square().sum(dim=1, dtype=torch.float64)

    for index, row in enumerate(cohort, 1):
        source = experiment.source(row)
        target = experiment.target(row)
        for start in range(0, len(row["content_ids"]), locked.CHUNK):
            interval = slice(start, start + locked.CHUNK)
            x, ridge = experiment.ridge.features(*(t[:, interval] for t in source))
            native = locked.pack_native([t[:, interval].cuda() for t in target]).float()
            predicted = model(x, ridge, experiment.stats).half().float()
            sums += torch.stack((
                energy(predicted - native),
                energy(native),
                energy((predicted - native) @ basis),
                energy((native - center) @ basis),
            ))
        if index % 32 == 0 or index == len(cohort):
            C.log("Dense RGST KV validation", stage, index, len(cohort))

    total_error, total_energy, projected_error, projected_energy = sums.cpu().tolist()
    C.save(output, dict(
        manifest_sha256=experiment.identity, n=len(cohort), rank=RANK,
        metrics={role: dict(full_nmse=total_error[i] / total_energy[i],
                            subspace_nmse=projected_error[i] / projected_energy[i])
                 for i, role in enumerate(("key", "value"))},
    ))


def summarize(records):
    fields = ("correct", "em", "f1", "legacy_correct", "raw_correct", "reached_cap")
    score = {method: {field: mean(r[method][field] for r in records)
                      for field in fields if field in records[0][method]} for method in METHODS}
    return dict(n=len(records), metrics=score, dense_minus_grouped={
        field: C.paired([r["dense_r32"][field] - r["grouped_r32"][field] for r in records])
        for field in score["dense_r32"] if field in score["grouped_r32"]
    })


@torch.no_grad()
def evaluate(experiment, model):
    output = ROOT / "summary.json"
    if output.exists():
        assert C.read(output)["manifest_sha256"] == experiment.identity
        return
    grouped_module = locked.fixed.load_module("rgst_grouped_evaluation_control", GROUPED / "run.py")
    grouped = grouped_module.AllHeadsRGST(RANK)
    payload = torch.load(GROUPED_CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    assert payload["manifest_sha256"] == C.sha(GROUPED / "manifest.json")
    assert payload["step"] == CFG["updates_per_epoch"] * CFG["stage2"]["epochs"]
    grouped.load_state_dict(payload["model"], strict=True)
    del payload
    model.eval().requires_grad_(False)
    grouped.eval().requires_grad_(False)
    experiment.engine.mode = "test"
    records = {task: [] for task in CFG["test_counts"]}
    cohort = locked.fixed.rows("test")
    assert [row["id"] for row in cohort] == C.read(CONTROL / "manifest.json")["test_ids"]
    for index, row in enumerate(cohort, 1):
        sample, task = row["id"], row["task"]
        path = ROOT / "per_sample" / task / (sample + ".json")
        if path.exists():
            record = C.read(path)
            assert record["manifest_sha256"] == experiment.identity
            assert record["id"] == sample and record["gold"] == row["gold"]
        else:
            source = experiment.source(row)
            scores = {}
            for name, writer in zip(METHODS, (model, grouped)):
                if name == "dense_r32":
                    packed = experiment.predict(source, model=writer)
                else:
                    with experiment.grouped_features():
                        packed = experiment.predict(source, model=writer)
                cache = experiment.make_cache(packed)
                scores[name] = locked.fixed.score_locked(experiment.reader, row, cache)
                del cache, packed
            record = dict(id=sample, task=task, question=row["question"], gold=row["gold"], results=scores,
                          manifest_sha256=experiment.identity)
            C.save(path, record)
        records[task].append(record["results"])
        C.save(ROOT / "status.json", dict(stage="evaluation", task=task, completed=index, total=len(cohort)))
        if index == 1 or index % 8 == 0 or index == len(cohort):
            C.log("Dense RGST evaluation", task, index, len(cohort))

    assert {task: len(rows) for task, rows in records.items()} == CFG["test_counts"]
    C.save(output, dict(
        status="completed", n=len(cohort), manifest_sha256=experiment.identity,
        tasks={task: summarize(rows) for task, rows in records.items()},
        multi_hop=summarize([record for task in ("hotpotqa", "2wiki", "musique") for record in records[task]]),
        kv_validation={stage: C.read(ROOT / stage / "kv_validation.json") for stage in STAGES},
    ))
    C.save(ROOT / "status.json", dict(stage="completed", evaluated=len(cohort)))


def main(offload_layers):
    torch.set_float32_matmul_precision("highest")
    identity, flags, statistics = prepare()
    locked.ROOT = ROOT
    locked.validate = validate_with_snapshot
    C.save(ROOT / "runtime.json", dict(sender_cpu_offload_layers=offload_layers, manifest_sha256=identity))
    C.log("Dense RGST start", f"R{RANK}", "sender CPU offload", offload_layers)

    experiment = TrainingExperiment(identity, flags, statistics, offload_layers)
    torch.manual_seed(CFG["seed"])
    model = DenseAllHeads(RANK)
    assert sum(p.numel() for p in model.parameters()) == 304_441_344
    for stage in STAGES:
        (ROOT / stage).mkdir(exist_ok=True)
        locked.train(experiment, {RANK_KEY: model}, stage)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        subspace_diagnostics(experiment, model, stage)
    evaluate(experiment, model)
    experiment.engine.unchanged()
    C.log("Dense RGST completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-sender-layers", type=int, required=True)
    args = parser.parse_args()
    try:
        main(args.offload_sender_layers)
    except torch.OutOfMemoryError as exc:
        C.log("Dense RGST CUDA OOM", str(exc))
        sys.exit(75)
