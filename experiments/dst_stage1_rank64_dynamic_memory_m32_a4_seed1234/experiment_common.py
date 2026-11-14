"""Use the matched Static protocol with the previous dynamic construction."""

import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / "config.json").read_text())
STATIC_ROOT = ROOT.parent / SETTINGS["static_reference"]
DYNAMIC_ROOT = ROOT.parent / SETTINGS["dynamic_implementation"]
spec = importlib.util.spec_from_file_location("rank64_static_matched", STATIC_ROOT / "experiment_common.py")
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)

B, Z, C, torch, K, SM = F.B, F.Z, F.C, F.torch, F.K, F.SM
S, M, O, AR, T, N = F.S, F.M, F.O, F.AR, F.T, F.N
PARENT = F.PARENT
BASE, STATIC, ARM = F.BASE, F.ARM, "rank64_dynamic"
CFG = {**F.CFG, **SETTINGS}
RANK64 = B.ROOT / "runs/stage2/epoch4_step1536.pt"
STATIC_CHECKPOINT = STATIC_ROOT / "runs/epoch4_step1536.pt"
same_prediction = F.same_prediction


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(name, **details):
    C.save(ROOT / name, dict(manifest_sha256=identity(), **details))


def status(**details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), **details))


def rows(split):
    return F.rows(split)


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
    F.verify()
    assert F.identity() == SETTINGS["static_reference_manifest_sha256"]
    assert F.done("completed.json") and B.done("runs/stage2/completed.json")
    manifest = C.read(ROOT / "manifest.json")
    for name, expected in manifest["sources"].items():
        assert C.sha(name) == expected, name


def baseline_path(row):
    return STATIC_ROOT / f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'


def baseline(row):
    path = baseline_path(row)
    record = C.read(path)
    assert record["manifest_sha256"] == F.identity()
    assert all(record[key] == row[key] for key in ("id", "gold", "question")), path
    assert record["b_checkpoint_sha256"] == SETTINGS["rank64_checkpoint_sha256"]
    assert record["memory_checkpoint_sha256"] == SETTINGS["static_reference_checkpoint_sha256"]
    return record["conditions"]
