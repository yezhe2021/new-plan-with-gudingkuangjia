"""Bindings for reusing the existing Dynamic M32 trainer on Gemma DST."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "inputs/gemma"
RECIPE = ROOT.parent / "dst_stage2_dynamic_memory_m32_a4_3task_seed1234"


def scoped(name, path, common=None):
    paths, old = list(sys.path), sys.modules.get("experiment_common")
    sys.path.insert(0, str(path.parent))
    if common is not None:
        sys.modules["experiment_common"] = common
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path[:] = paths
        if old is None:
            sys.modules.pop("experiment_common", None)
        else:
            sys.modules["experiment_common"] = old


G = scoped("gemma_frozen_common", ASSETS / "experiment_common.py")
C, torch, K, M, N = G.C, G.torch, G.K, G.M, G.N
R, T, O, AR = N.R, N.T, N.O, N.AR
S = SimpleNamespace(ORIGINAL=N.ORIGINAL, GSM=N.GSM)
PARENT = G.REFERENCE
CFG = C.read(RECIPE / "config.json")
CFG = {key: value for key, value in CFG.items() if key not in (
    "caveats", "dst_reference", "dst_manifest_sha256", "dst_checkpoint_sha256",
    "static_reference", "static_checkpoint_sha256", "static_manifest_sha256",
    "static_reference_manifest_sha256", "static_reference_checkpoint_sha256",
    "static_source", "transfer_reference", "transfer_manifest_sha256",
)}
CFG["translator"] = "Gemma-3-4B-IT → Qwen3-4B; frozen final DST Stage1 and Rank64 Stage2"
CFG["minimum_free_gpu_mib"] = 24_000
CFG["allocator_limit_mib"] = 23_500
CFG["gemma_offloaded_layers"] = 14
CFG["sender_placement"] = "FP32 Gemma; language layers execute on CUDA; inactive vision and 14 layers' weights stay on CPU between forwards"


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(path, **values):
    C.save(ROOT / path, dict(manifest_sha256=identity(), **values))


def status(**values):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T %z"), **values))


def rows(split):
    return G.rows(split)


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def done(name):
    path = ROOT / name
    if not path.exists():
        return False
    record = C.read(path)
    assert record["manifest_sha256"] == identity() and record["status"] == "completed", path
    return True


def verify():
    manifest = C.read(ROOT / "manifest.json")
    for name, expected in manifest["sources"].items():
        assert C.sha(name) == expected, name
    assert manifest["config"] == CFG
