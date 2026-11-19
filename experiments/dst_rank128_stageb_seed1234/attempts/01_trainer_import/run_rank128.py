"""Run the original DST Stage B protocol with a Rank128 residual on server C."""
import argparse
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "dst_stage1_rank64_then_static_m32_a4_seed1234"
RANK = 128
BASELINE_MANIFEST = "28800234ee77722a274642a5c09a74c7cceb77b4dba2110913fa09ac778bceee"
BASELINE_CHECKPOINT = "aaae0362577395681095d5f72fbc07c4bc9920e0506f79dc21e531ac618958a9"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(stage="stage2"):
    assert stage == "stage2"
    return ROOT / "runs/stage2/epoch4_step1536.pt"


def training():
    sys.path.insert(0, str(REFERENCE))
    import experiment_common as base
    import runtime as original_runtime
    import train as original_train
    from engine import Engine
    from residual import ResidualTranslator

    C, torch, cfg = base.C, base.torch, base.CFG
    assert base.identity() == BASELINE_MANIFEST and cfg["rank"] == 64
    assert cfg["stage2"]["epochs"] == 4 and cfg["updates_per_epoch"] == 384
    reference = C.read(REFERENCE / "manifest.json")
    old_checkpoint = REFERENCE / "runs/stage2/epoch4_step1536.pt"
    assert C.sha(old_checkpoint) == BASELINE_CHECKPOINT
    stage1 = base.D.ROOT / "runs/stage1/epoch4_step1536.pt"
    assert C.sha(stage1) == cfg["dst_stage1_sha256"]
    prepared = {}
    for split in ("train", "validation", "test"):
        path = REFERENCE / "prepared" / (split + ".jsonl")
        prepared[split] = C.sha(path)
        assert prepared[split] == reference["sources"][str(path)]
    scripts = {}
    for name in ("train.py", "runtime.py", "engine.py", "score.py", "residual.py"):
        scripts[name] = C.sha(REFERENCE / name)
        assert scripts[name] == reference["code"][name]
    assert C.sha(REFERENCE / "config.json") == reference["config_sha256"]
    gpu = torch.cuda.get_device_name()
    manifest = dict(rank=RANK, frozen_stage1_sha256=cfg["dst_stage1_sha256"], trainable_parameters=18874368,
                    original_manifest_sha256=BASELINE_MANIFEST, original_rank64_sha256=BASELINE_CHECKPOINT,
                    code_sha256=C.sha(Path(__file__)), original_scripts=scripts, prepared_sha256=prepared,
                    train_counts=cfg["train_counts"], stage2=cfg["stage2"], effective_batch=cfg["effective_batch"],
                    updates=1536, prefix_quotas_per_batch=cfg["prefix_quotas_per_batch"], device=gpu,
                    pytorch=str(torch.__version__), training_host="server-c")
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        assert C.read(manifest_path) == manifest
    else:
        C.save(manifest_path, manifest)
    pin = C.sha(manifest_path)

    def mark(name, **fields):
        C.save(ROOT / name, dict(manifest_sha256=pin, **fields))

    def status(**fields):
        C.save(ROOT / "status.json", fields)

    def done(name):
        path = ROOT / name
        if not path.exists():
            return False
        record = C.read(path)
        assert record["manifest_sha256"] == pin and record["status"] == "completed"
        return True

    def verify():
        assert C.sha(manifest_path) == pin and C.sha(Path(__file__)) == manifest["code_sha256"]

    class Rank128Engine(Engine):
        def __init__(self):
            super().__init__()
            self.correction = ResidualTranslator(self.frozen_dst, rank=RANK, seed=cfg["seed"]).cuda().float()
            parameters = sum(p.numel() for p in self.correction.parameters() if p.requires_grad)
            assert parameters == manifest["trainable_parameters"]
            assert all(torch.count_nonzero(layer[-1].weight) == 0
                       for layer in (*self.correction.keys, *self.correction.values))
            self.unchanged()
            mark("checks/gpu_completed.json", status="completed", gpu=gpu, trainable_parameters=parameters)

    @torch.no_grad()
    def native_prefix(reader, row, native):
        folders = ((base.REFERENCE, base.N.identity()), (REFERENCE, BASELINE_MANIFEST), (ROOT, pin))
        for folder, identity in folders:
            path = folder / "native_prefix" / (row["id"] + ".json")
            if path.exists():
                record = C.read(path)
                assert record["manifest_sha256"] == identity and record["id"] == row["id"]
                return record["tokens"]
        tokens = reader.rollout(row, native, rn=False, cap=cfg["prefix_max_new_tokens"])
        mark(Path("native_prefix") / (row["id"] + ".json"), id=row["id"], tokens=tokens)
        return tokens

    if done("runs/stage2/completed.json"):
        return
    status(status="loading", stage="stage2", total_steps=1536)
    engine = Rank128Engine()
    with patch.multiple(original_train, ROOT=ROOT, identity=lambda: pin, mark=mark, status=status,
                        verify=verify, done=done, Engine=lambda: engine,
                        paused=lambda: (ROOT / "controls/pause_requested.json").exists(),
                        checkpoint_path=checkpoint_path), patch.object(original_runtime, "native_prefix", native_prefix):
        original_train.main("stage2")


def evaluation(split):
    reference = ROOT.parent / "dst_rank64_route_output_stageb_seed1234"
    sys.path.append(str(reference))
    replay = load("rank128_cohort_replay", reference / "evaluate_stageb.py")
    replay.ROOT = ROOT
    replay.ARM = "dst_rank128"

    def load_rank128(core, torch, engine):
        residual = load("rank128_eval_residual", REFERENCE / "residual.py")
        completion = core.read(ROOT / "runs/stage2/completed.json")
        pin = core.sha(ROOT / "manifest.json")
        path = checkpoint_path()
        digest = core.sha(path)
        assert completion["manifest_sha256"] == pin and completion["checkpoint_sha256"] == digest
        assert completion["status"] == "completed" and completion["updates"] == 1536
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == pin and cp["step"] == 1536
        engine.correction = residual.ResidualTranslator(engine.frozen_dst, rank=RANK).cuda().float()
        engine.correction.load_state_dict(cp["state"], strict=True)
        engine.correction.eval().requires_grad_(False)
        engine.mode = "test"
        return digest

    replay.checkpoint = load_rank128
    {"short": replay.short, "long": replay.long}[split]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "short", "long"))
    phase = parser.parse_args().phase
    if phase == "train":
        training()
    else:
        evaluation(phase)
