"""Pinned inputs and access to the existing Rank64 and Memory modules."""

import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
SETTINGS = json.loads((ROOT / "config.json").read_text())
CFG = json.loads((INPUTS / "training_config.json").read_text())
PARENT = Path("/hy-tmp/yezhe/异构模型/rnkv_keep_both_all36_arc_hotpot_gsm_seed1234")
sys.path.append(str(INPUTS))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N = load_module("gated_dynamic_original_protocol", PARENT / "experiment_common.py")
C, torch, K, M, AR, R, O, T = N.C, N.torch, N.K, N.M, N.AR, N.R, N.O, N.T
S = SimpleNamespace(ORIGINAL=N.ORIGINAL, GSM=N.GSM)
DST_CONFIG = C.read(INPUTS / "dst_manifest.json")["config"]
CFG = {**DST_CONFIG, **CFG}
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(INPUTS))


def rows(split):
    return C.read_jsonl(INPUTS / f"prepared/{split}.jsonl")


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(name, **details):
    C.save(ROOT / name, dict(manifest_sha256=identity(), **details))


def status(**details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), **details))


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def done(name):
    path = ROOT / name
    if not path.exists():
        return False
    record = C.read(path)
    assert record["status"] == "completed" and record["manifest_sha256"] == identity()
    return True


def verify():
    manifest = C.read(ROOT / "manifest.json")
    for name, expected in manifest["sources"].items():
        assert C.sha(name) == expected, name


def baseline(row):
    rel = f'{row["task"]}/per_sample/{row["id"]}.json'
    values = {}
    for kind, arms in (("dynamic", ("native", "B_rank64", "D_static", "rank64_dynamic")),
                       ("gated_static", ("gated_static",))):
        record = C.read(INPUTS / f"{kind}_evaluation" / rel)
        expected = SETTINGS["inputs_sha256"][f"{kind}_manifest.json"]
        assert record["manifest_sha256"] == expected
        assert all(record[k] == row[k] for k in ("id", "question", "gold"))
        values.update({arm: record["conditions"][arm] for arm in arms})
    return values
