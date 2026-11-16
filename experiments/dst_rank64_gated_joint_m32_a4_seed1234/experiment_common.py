"""Reuse the completed Rank64 and Dynamic Memory experiment's protocol."""

import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / "config.json").read_text())
REFERENCE = ROOT.parent / SETTINGS["reference"]
spec = importlib.util.spec_from_file_location("joint_reference", REFERENCE / "experiment_common.py")
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)

# The prior runtime imports these protocol objects from experiment_common.
C, torch, K, M, AR = reference.C, reference.torch, reference.K, reference.M, reference.AR
S, T, O, N, PARENT = reference.S, reference.T, reference.O, reference.N, reference.PARENT
rows = reference.rows
CFG = {**reference.CFG, "prefix_quotas_per_batch": [
    *reference.CFG["prefix_quotas_per_batch"], reference.CFG["prefix_quotas_per_batch"][-1],
]}
sys.path.append(str(reference.DYNAMIC_ROOT))


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(name, **details):
    C.save(ROOT / name, dict(manifest_sha256=identity(), **details))


def status(phase, **details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), phase=phase, **details))


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def done(name):
    path = ROOT / name
    if not path.exists():
        return False
    v = C.read(path)
    assert v["manifest_sha256"] == identity() and v["status"] == "completed", name
    return True
