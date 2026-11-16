"""Use the original trainers with carry_state=False and an isolated run root."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / "config.json").read_text())
REFERENCE_ROOT = ROOT.parent if (ROOT.parent / SETTINGS["dst"]).is_dir() else ROOT.parent.parent
DST = REFERENCE_ROOT / SETTINGS["dst"]
RANK = REFERENCE_ROOT / SETTINGS["rank64"]
HELDOUT = REFERENCE_ROOT / SETTINGS["heldout"]
NO = ROOT / "no_carry"


@contextmanager
def imports(directory, **aliases):
    paths = list(sys.path)
    saved = {name: sys.modules.get(name) for name in aliases}
    sys.path.insert(0, str(directory))
    sys.modules.update(aliases)
    try:
        yield
    finally:
        sys.path[:] = paths
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_module(name, path, **aliases):
    with imports(path.parent, **aliases):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


reference = load_module("depth_full_dst_reference", DST / "experiment_common.py")
C, K, torch = reference.C, reference.K, reference.torch
ARCH = load_module("depth_full_dst_architecture", DST / "translator.py")
ALIGN = load_module("depth_full_dst_alignment", DST / "alignment.py")
RESIDUAL = load_module("depth_rank64_architecture", RANK / "residual.py")
CFG = {**reference.CFG, "model": {**reference.CFG["model"], "carry_state": False}, "rank": 64}
STAGES = ("stage1", "stage2")
ARMS = ("carry", "no_carry")


def checkpoint_path(arm, stage):
    root = NO if arm == "no_carry" else DST if stage == "stage1" else RANK
    return root / f"runs/{stage}/epoch4_step1536.pt"


def identity():
    return C.sha(ROOT / "manifest.json")


def folder(arm="no_carry"):
    return ROOT / arm


def mark(path, arm="no_carry", **values):
    C.save(folder(arm) / path, dict(manifest_sha256=identity(), **values))


def done(path, arm="no_carry"):
    full = folder(arm) / path
    if not full.exists():
        return False
    value = C.read(full)
    assert value["manifest_sha256"] == identity() and value["status"] == "completed", full
    return True


def paused():
    return (ROOT / "controls/pause_requested.json").exists() or (NO / "controls/pause_requested.json").exists()


def status(**details):
    value = dict(time=time.strftime("%F %T"), **details)
    C.save(NO / "status.json", value)
    C.save(ROOT / "status.json", value)


def verify():
    for path, expected in C.read(ROOT / "manifest.json")["pins"].items():
        assert C.sha(path) == expected, path


def prepare():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    reference.verify()
    assert reference.identity() == SETTINGS["dst_manifest_sha256"]
    assert C.sha(RANK / "manifest.json") == SETTINGS["rank64_manifest_sha256"]
    rank_config = C.read(RANK / "config.json")
    for name in ("seed", "model", "effective_batch", "updates_per_epoch", "token_chunk", "weight_decay",
                 "gradient_clip", "prefix_quotas_per_batch", "prefix_max_new_tokens", "supervised_positions_max",
                 "stage1", "stage2", "train_counts", "validation_counts"):
        assert rank_config[name] == reference.CFG[name], name
    assert rank_config["rank"] == 64
    assert reference.CFG["model"]["carry_state"] is True
    assert C.sha(HELDOUT) == SETTINGS["heldout_sha256"]
    heldout = C.read_jsonl(HELDOUT)
    assert Counter(row["task"] for row in heldout) == {task: SETTINGS["heldout_per_task"] for task in CFG["test_counts"]}
    questions = {C.norm(row["question"]) for row in heldout}
    assert len(questions) == len(heldout)
    paths = [*ROOT.glob("*.py"), ROOT / "config.json", DST / "manifest.json", RANK / "manifest.json", HELDOUT]
    for split in ("train", "validation", "test"):
        path = DST / f"prepared/{split}.jsonl"
        assert C.sha(path) == C.sha(RANK / f"prepared/{split}.jsonl"), split
        assert not questions.intersection(C.norm(row["question"]) for row in reference.rows(split)), split
        paths.extend((path, RANK / f"prepared/{split}.jsonl"))
    for stage in STAGES:
        path = checkpoint_path("carry", stage)
        assert C.sha(path) == SETTINGS[f"{stage}_sha256"]
        info = C.read(path.parent / "completed.json")
        assert info["updates"] == 1536 and info["checkpoint_sha256"] == SETTINGS[f"{stage}_sha256"]
        paths.extend((path, path.parent / "completed.json"))
    paths.extend(DST / name for name in ("train.py", "runtime.py", "engine.py", "checks.py", "alignment.py", "translator.py"))
    paths.extend(RANK / name for name in ("train.py", "runtime.py", "residual.py", "score.py"))
    paths.extend((C.ROOT / "core.py", C.ROOT / "manifest.json"))
    for arm in ARMS:
        config = {**CFG, "model": {**CFG["model"], "carry_state": arm == "carry"},
                  "test_counts": {task: SETTINGS["heldout_per_task"] for task in CFG["test_counts"]}}
        C.save(folder(arm) / "config.json", config)
        paths.append(folder(arm) / "config.json")
    C.save(ROOT / "manifest.json", dict(settings=SETTINGS, pins={str(p): C.sha(p) for p in paths},
        comparison="Fresh No-Carry versus immutable Carry checkpoints, no Memory, final step1536 for both stages",
        heldout_provenance="Reuses the locked Gated Joint cohort; exact-question disjoint from both training/validation and older 64-question DST regression cohort."))
    mark("checkpoints.json", arm="carry", status="completed",
         checkpoints={stage: str(checkpoint_path("carry", stage)) for stage in STAGES})
    C.log("DEPTH ABLATION PREPARED", identity(), len(heldout))


# Populate a scoped common module for the original trainer and runtime. Only the
# choice of architecture flag, checkpoint destination and evaluation rows vary.
scope = ModuleType("depth_ablation_training_common")
scope.__dict__.update({name: value for name, value in vars(reference).items() if not name.startswith("_")})
scope.__dict__.update(ROOT=NO, CFG=CFG, mark=mark, status=status, paused=paused,
                      done=done, identity=identity, verify=verify)
engine_module = load_module("depth_ablation_engine", DST / "engine.py", experiment_common=scope)
Engine, BodyReader = engine_module.Engine, engine_module.BodyReader


def stage1_model(arm, *, device):
    config = ARCH.DSTConfig(**{**CFG["model"], "carry_state": arm == "carry"})
    return ARCH.DepthStateTranslator(config, CFG["seed"]).to(device=device, dtype=torch.float32)


def load_weights(model, arm, stage):
    path = checkpoint_path(arm, stage)
    cp = torch.load(path, map_location="cpu", weights_only=True)
    expected = identity() if arm == "no_carry" else SETTINGS["dst_manifest_sha256" if stage == "stage1" else "rank64_manifest_sha256"]
    assert cp["manifest_sha256"] == expected and cp["step"] == 1536 and cp["stage"] == stage
    model.load_state_dict(cp["state"], strict=True)
    return model


def rank64_model(base):
    return RESIDUAL.ResidualTranslator(base, rank=CFG["rank"], seed=CFG["seed"]).cuda().float()


class RankEngine(Engine):
    def __init__(self):
        super().__init__()
        self.frozen_dst = load_weights(stage1_model("no_carry", device="cuda"), "no_carry", "stage1")
        self.correction = rank64_model(self.frozen_dst)
        self.base_versions = C.versions(self.frozen_dst)

    def unchanged(self):
        super().unchanged()
        assert C.versions(self.frozen_dst) == self.base_versions
        assert all(not p.requires_grad and p.grad is None for p in self.frozen_dst.parameters())


def stage_modules(stage):
    root = DST if stage == "stage1" else RANK
    engine = engine_module if stage == "stage1" else ModuleType("depth_ablation_rank_engine")
    if stage == "stage2":
        engine.Engine, engine.BodyReader = RankEngine, BodyReader
    aliases = dict(experiment_common=scope, engine=engine, translator=ARCH, residual=RESIDUAL)
    runtime = load_module(f"depth_ablation_runtime_{stage}", root / "runtime.py", **aliases)
    trainer = load_module(f"depth_ablation_trainer_{stage}", root / "train.py", runtime=runtime, **aliases)
    return runtime, trainer


def reference_checks(stage):
    runtime, _ = stage_modules(stage)
    return load_module("depth_ablation_reference_checks", DST / "checks.py", experiment_common=scope,
                       translator=ARCH, alignment=ALIGN, engine=engine_module, runtime=runtime)


def cpu():
    verify()
    allocation = "checks/recurrence.json"
    if not done(allocation):
        carry, reset = (stage1_model(arm, device="cpu") for arm in ARMS)
        assert carry.specification()["parameters"] == reset.specification()["parameters"]
        assert list(carry.state_dict()) == list(reset.state_dict())
        assert all(torch.equal(a,b) for a,b in zip(carry.state_dict().values(), reset.state_dict().values()))
        tiny = ARCH.DSTConfig(sender_layers=3, receiver_layers=6, source_width=8, target_width=8, state_width=4)
        rng = torch.Generator().manual_seed(1234)
        source = tuple(tuple(torch.randn(24,8,generator=rng) for _ in range(3)) for _ in range(2))
        norms = {}
        for arm in ARMS:
            model = ARCH.DepthStateTranslator(replace(tiny, carry_state=arm == "carry"))
            full = model(*source, range(24))
            pieces = [model(*source, range(i,i+4)) for i in range(0,24,4)]
            for j,out in enumerate(full):
                torch.testing.assert_close(out, torch.cat([p[j] for p in pieces], 1), atol=2e-6, rtol=1e-5)
            changed = tuple(tuple(x.clone() for x in group) for group in source)
            changed[0][0][7].add_(4)
            modified = model(*changed, range(24))
            keep = [i for i in range(24) if i != 7]
            assert all(torch.equal(a[:,keep],b[:,keep]) for a,b in zip(full, modified))
            (full[0][-1].sum()+full[1][-1].sum()).backward()
            grad = model.layers[0].input.weight.grad
            norms[arm] = float(grad.norm()) if grad is not None else 0.0
        assert norms["carry"] > 1e-10 and norms["no_carry"] == 0
        mark(allocation, status="completed", full_initial_weights_identical=True, gradient_norms=norms,
             token_independence=True, chunk_parity=True)
        del carry, reset
    if not done("checks/cpu_completed.json"):
        reference_checks("stage1").cpu()


def gpu():
    # Use the No-Carry configuration in the original real-model smoke test.
    runtime, _ = stage_modules("stage1")
    with imports(DST, runtime=runtime):
        reference_checks("stage1").gpu()


def rank_smoke():
    verify()
    if done("checks/rank64_completed.json"):
        return
    assert done("runs/stage1/completed.json")
    runtime, _ = stage_modules("stage2")
    engine = RankEngine()
    module = engine.correction
    reader = BodyReader(engine)
    row = max(reference.rows("train"), key=lambda r: len(r["content_ids"])+len(r["gold_tokens"]))
    with torch.no_grad():
        source = engine.source(row)
        base = engine.translated(row, engine.frozen_dst, source)
        engine.norm_counts = [0] * 36
        residual = engine.translated(row, module, source)
        assert engine.norm_counts == [(len(row["content_ids"])+CFG["token_chunk"]-1)//CFG["token_chunk"]]*36
        engine.norm_counts = None
        assert all(torch.equal(a,b) for a,b in zip(K.flat(base), K.flat(residual)))
        del source, base, residual
    opt = torch.optim.AdamW(module.parameters(), lr=CFG["stage2"]["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64., growth_interval=1000000)
    _, outcome = runtime.backward_batch(engine, reader, module, opt, scaler, [(row,"gold")], "stage2", 1)
    engine.unchanged()
    mark("checks/rank64_completed.json", status="completed", id=row["id"], outcome=outcome,
         zero_init_cache_exact=True, k_norm_once_per_chunk=True)
    C.log("NO CARRY RANK64 SMOKE PASSED", row["id"])


def train(stage):
    assert stage in STAGES
    if stage == "stage2":
        assert done("checks/rank64_completed.json")
    _, trainer = stage_modules(stage)
    trainer.main(stage)


if __name__ == "__main__":
    {"prepare": prepare, "cpu": cpu, "gpu": gpu, "rank_smoke": rank_smoke, "train": train}[sys.argv[1]](*sys.argv[2:])
