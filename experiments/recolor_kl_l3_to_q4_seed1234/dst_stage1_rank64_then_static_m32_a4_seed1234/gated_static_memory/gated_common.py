"""Use D's pinned cohorts and schedule with a separate output directory."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "static_memory"))
import experiment_common as reference
from experiment_common import C, CFG, N, O, PARENT, T, torch

ARM = "gated_static"
PARAMETERS = 409600 + 36


def prepare():
    reference.verify()
    assert reference.done("completed.json")
    manifest = {
        "d_manifest_sha256": reference.identity(),
        "b_checkpoint_sha256": C.read(reference.ROOT / "manifest.json")["b_checkpoint_sha256"],
        "d_checkpoint_sha256": C.read(reference.ROOT / "runs/completed.json")["checkpoint_sha256"],
        "native_prefix_sources": C.read(reference.ROOT / "manifest.json")["native_prefix_sources"],
        "code": {p.name: C.sha(p) for p in sorted(ROOT.glob("*.py"))},
        "training_config": CFG,
        "trainable_parameters": PARAMETERS,
        "memory_attention": "Separate GQA using pre-RoPE normalized Q/K, tanh gate per layer initialized to zero, added before o_proj",
        "answer_positions": "Same as B",
    }
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)


def verify():
    assert (ROOT / "manifest.json").exists()
    prepare()


def identity():
    return C.sha(ROOT / "manifest.json")


def rows(split):
    return reference.rows(split)


def mark(name, **details):
    C.save(ROOT / name, dict(manifest_sha256=identity(), **details))


def status(**details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), **details))


def done(name):
    path = ROOT / name
    if not path.exists():
        return False
    record = C.read(path)
    assert record["manifest_sha256"] == identity() and record["status"] == "completed", name
    return True


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def baseline(row):
    record = C.read(reference.ROOT / "evaluation" / row["task"] / "per_sample" / (row["id"] + ".json"))
    manifest = C.read(ROOT / "manifest.json")
    assert record["manifest_sha256"] == manifest["d_manifest_sha256"]
    assert record["b_checkpoint_sha256"] == manifest["b_checkpoint_sha256"]
    assert record["memory_checkpoint_sha256"] == manifest["d_checkpoint_sha256"]
    assert all(record[key] == row[key] for key in ("id", "question", "gold"))
    return {name: record["conditions"][name] for name in (reference.BASE, reference.ARM, "native")}
