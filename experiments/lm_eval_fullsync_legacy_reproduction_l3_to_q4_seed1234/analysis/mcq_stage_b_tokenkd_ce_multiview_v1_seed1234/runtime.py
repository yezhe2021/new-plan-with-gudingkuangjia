"""Immutable provenance and filesystem helpers for the new Stage B only."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]
sys.path.insert(0, str(PARENT))
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S UTC] ", time.gmtime()) + message, flush=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def plain(value):
    return value.item() if hasattr(value, "item") else str(value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=plain) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def identities():
    import lm_eval
    from kv_backend import CFG
    source = [HERE / name for name in ("config.json", "runtime.py", "data_views.py", "train_stage_b.py",
                                      "evaluate_stage_b.py", "run_pipeline.py", "tests.py")]
    source += [PARENT / name for name in ("config.json", "kv_backend.py", "tasks_native.py",
                                        "base_code/common.py", "base_code/protocol.py", "base_code/translator.py")]
    template = Path(lm_eval.__file__).parent / "tasks/mmlu/default/_default_template_yaml"
    source.append(template)
    baseline = PARENT / CONFIG["mmlu_baseline_directory"]
    source += [baseline / "run_mmlu.py", baseline / "manifest.json"]
    data = [PARENT / CONFIG[name] for name in ("parent_training_data", "parent_validation_data")]
    provenance = json.loads((baseline / "datasets/mmlu/provenance.json").read_text())
    data += [baseline / "datasets/mmlu" / item["file"] for item in provenance["files"]]
    return {"configuration": CONFIG, "parent_protocol": CFG["protocol"], "models": CFG["models"],
            "max_length": CFG["max_length"], "attention_implementation": CFG["attention_implementation"],
            "source_sha256": {str(path): digest(path) for path in source},
            "data_sha256": {str(path): digest(path) for path in data},
            "stage_a": {"path": str((PARENT / CONFIG["parent_stage_a"]).resolve()),
                        "sha256": digest(PARENT / CONFIG["parent_stage_a"]),
                        "selection": json.loads((PARENT / "runs/mcq/stage_a/selection.json").read_text())["best"]},
            "versions": {name: importlib.metadata.version(name)
                         for name in ("lm_eval", "torch", "transformers", "datasets")}}


def verify_manifest():
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["identities"] != identities():
        raise RuntimeError("Code, data, or frozen Stage A changed since preparation")
    for relative, expected in manifest["prepared_sha256"].items():
        if digest(HERE / relative) != expected:
            raise RuntimeError("Prepared training/validation data changed: " + relative)
    return manifest


def gpu_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    log(f"GPU {torch.cuda.get_device_name(0)}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB")
    if free / 2**30 < CONFIG["min_free_gpu_gib"]:
        raise RuntimeError("Not enough free GPU memory; unrelated jobs were not changed")


def load_new_adapter(path, trainable=False):
    import torch
    from kv_backend import ResidualKVAdapter
    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = json.loads((HERE / "manifest.json").read_text())
    if payload["protocol"] != CONFIG["protocol"] or payload["stage_a_sha256"] != manifest["identities"]["stage_a"]["sha256"]:
        raise RuntimeError("Wrong new Stage B checkpoint or frozen Stage A identity")
    model = ResidualKVAdapter(rank=CONFIG["adapter_rank"]).cuda()
    model.load_state_dict(payload["state"], strict=True)
    return model.train(trainable).requires_grad_(trainable)
