"""Configuration and scoped reuse of the previous two-stage DST trainer."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "gemma_dst_rank64_stage12_server_a_seed1234"
BASE_CONFIG = json.loads((TEMPLATE / "config.json").read_text())
REFERENCE = Path(BASE_CONFIG["data_reference"])


@contextmanager
def imports(directory, **aliases):
    paths = list(sys.path)
    saved = {key: sys.modules.get(key) for key in aliases}
    sys.path.insert(0, str(directory))
    sys.modules.update(aliases)
    try:
        yield
    finally:
        sys.path[:] = paths
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


def scoped(name, path, **aliases):
    with imports(path.parent, **aliases):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


N = scoped("reverse_dst_source_data", REFERENCE / "experiment_common.py")
C, M, K, torch = N.C, N.M, N.K, N.torch
ARCH = scoped("reverse_dst_architecture", TEMPLATE / "translator.py")
RESIDUAL = scoped("reverse_dst_residual", TEMPLATE / "residual.py")
ALIGN = scoped("reverse_dst_alignment", TEMPLATE / "alignment.py")

CFG = dict(BASE_CONFIG)
for key in ("caveats", "baseline_manifest_sha256", "protocol", "launch_scope"):
    CFG.pop(key, None)
CFG.update(
    sender=BASE_CONFIG["receiver"], receiver=BASE_CONFIG["sender"],
    source_dtype="float16", receiver_dtype="float32",
    model=dict(BASE_CONFIG["model"], sender_layers=36, receiver_layers=34),
    protocol="Qwen RAW body to Gemma Full-Sync; Gemma public header and suffix; native sliding and full attention; no student native body prefill",
    launch_scope="Server A, Stage1 then frozen-base Rank64 Stage2, same raw rows, no Memory",
    minimum_free_gpu_mib=30720,
)


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(name, **details):
    C.save(ROOT / name, dict(manifest_sha256=identity(), **details))


def status(**details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), **details))


def rows(split):
    return C.read_jsonl(ROOT / "prepared" / (split + ".jsonl"))


def paused():
    return (ROOT / "controls/pause_requested.json").exists()


def done(name):
    path = ROOT / name
    if not path.exists():
        return False
    data = C.read(path)
    assert data["status"] == "completed" and data["manifest_sha256"] == identity()
    return True


def verify():
    manifest = C.read(ROOT / "manifest.json")
    assert manifest["config"] == CFG
    assert N.identity() == CFG["data_reference_manifest_sha256"]
    for path, digest in manifest["pins"].items():
        assert C.sha(path) == digest, path
