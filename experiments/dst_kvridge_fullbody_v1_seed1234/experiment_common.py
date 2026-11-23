"""Local run state for the original DST Rank64 training and scoring modules."""

import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BASE = PROJECT / "dst_stage1_rank64_then_static_m32_a4_seed1234"
PILOT = PROJECT / "fullsync_kvridge_task_adapters_seed1234"
CALIBRATION = PILOT / "data/fineweb_calibration.jsonl.gz"
RIDGE = ROOT / "stage_a/ridge.pt"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Rank = load_module("fixed_dst_rank_reference", BASE / "experiment_common.py")
sys.path.insert(0, str(ROOT))
sys.path.append(str(BASE))
CFG, C, D, Z, K, N, F, torch = (getattr(Rank, name) for name in
                               ("CFG", "C", "D", "Z", "K", "N", "F", "torch"))
REFERENCE = Rank.REFERENCE
ORIGINAL, GSM = Rank.ORIGINAL, Rank.GSM
MODEL_REFERENCE = PROJECT / "herb_evidence_kv_pilot_20261119/cap192/summary.json"
MODEL_FILES = C.read(MODEL_REFERENCE)["model_files"]
C.CFG["models"] = {role: MODEL_FILES[name]["path"]
                   for role, name in (("sender", "llama"), ("receiver", "qwen"))}


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(name, **values):
    C.save(ROOT / name, dict(manifest_sha256=identity(), **values))


def status(**values):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), **values))


@lru_cache(maxsize=3)
def rows(split):
    return Rank.rows(split)


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def done(name):
    path = ROOT / name
    if not path.exists():
        return False
    record = C.read(path)
    assert record["manifest_sha256"] == identity() and record["status"] == "completed", name
    return True


def verify():
    record = C.read(ROOT / "manifest.json")
    for path, expected in record["sources"].items():
        assert C.sha(path) == expected, path


def preflight():
    lock_path = ROOT / "DST_PROTOCOL_V1.lock.json"
    lock = C.read(lock_path)
    assert lock["protocol"] == "dst-fullbody-chat-v1"
    assert D.identity() == lock["reference_manifest_sha256"]["llama_dst"]
    assert N.identity() == lock["data_reference_manifest_sha256"]
    rank_manifest = C.read(BASE / "manifest.json")
    assert C.sha(BASE / "config.json") == rank_manifest["config_sha256"]
    for filename in ("experiment_common.py", "train.py", "runtime.py", "residual.py", "score.py"):
        assert C.sha(BASE / filename) == rank_manifest["code"][filename], filename
    dst_manifest = C.read(D.ROOT / "manifest.json")
    for filename in ("engine.py", "alignment.py"):
        assert C.sha(D.ROOT / filename) == dst_manifest["code"][filename], filename
    assert C.read(C.ROOT / "manifest.json")["prefix_ids"] == lock["public_header_ids"]
    for filename, expected in lock["receiver_tokenizer_sha256"].items():
        assert C.sha(Path(C.CFG["models"]["receiver"]) / filename) == expected, filename
    paths = {}
    for role, name in (("sender", "llama"), ("receiver", "qwen")):
        for filename, expected in MODEL_FILES[name]["sha256"].items():
            path = Path(C.CFG["models"][role]) / filename
            assert C.sha(path) == expected, path
            paths[str(path)] = expected
    for split, expected in lock["prepared_sha256"]["llama"].items():
        path = BASE / "prepared" / f"{split}.jsonl"
        assert C.sha(path) == expected, path
        paths[str(path)] = expected
    assert C.sha(D.ROOT / "runs/stage1/epoch4_step1536.pt") == lock["reference_stage1_checkpoint_sha256"]["llama_dst"]
    assert C.sha(BASE / "runs/stage2/epoch4_step1536.pt") == C.read(BASE / "runs/stage2/completed.json")["checkpoint_sha256"]
    assert C.sha(CALIBRATION) == C.read(PILOT / "stage_a/summary.json")["source_sha256"]
    paths.update({str(p): C.sha(p) for p in (
        lock_path, MODEL_REFERENCE, BASE / "manifest.json", BASE / "config.json", BASE / "train.py", BASE / "runtime.py",
        BASE / "residual.py", BASE / "score.py", D.ROOT / "engine.py", D.ROOT / "alignment.py",
        BASE / "runs/stage2/epoch4_step1536.pt", D.ROOT / "runs/stage1/epoch4_step1536.pt",
        CALIBRATION, PILOT / "common.py", PILOT / "ridge_stage_a.py",
    )})
    manifest = dict(protocol=lock["protocol"], sources=paths, checkpoint_selection="fixed epoch 4",
                    ridge=dict(calibration="FineWeb-Edu sample-10BT", documents=500, context=1024, stride=4,
                               probes=32, top_k=4, ridge_lambda=0.01, key_space="pre-k_norm, pre-RoPE"),
                    train=CFG["train_counts"], validation=CFG["validation_counts"], test=CFG["test_counts"])
    if (ROOT / "manifest.json").exists():
        assert C.read(ROOT / "manifest.json") == manifest
    else:
        C.save(ROOT / "manifest.json", manifest)
    verify()
    mark("checks/preflight.json", status="completed", split_counts={split: len(rows(split)) for split in
                                                                  ("train", "validation", "test")})
    C.log("Ridge DST protocol preflight passed")


if __name__ == "__main__":
    preflight()
