"""Regenerate the immutable-file manifest and machine-built results index."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SOURCE_COMMIT = "fb2719fe43fd8737e3e9019f5bf669ff7429badc"
SOURCE_REPO = "https://github.com/yezhe2021/weichaxun"
PREREQUISITES = {
    "sync_boundary_nativekv_full28_diagonal_translator_llama3_2_3b_to_qwen3_4b_seed1234",
    "multidataset_obqa_arc_mmlupro_train1024_full28_mlp_l3_to_q4_seed1234",
    "multidataset_obqa_arc_mmlupro_train1024_full34_headmix_g3_to_q4_seed1234",
    "multidataset_obqa_arc_mmlupro_train1024_full36_headmix_q4_to_g3_seed1234",
    "multidataset_obqa_arc_mmlupro_train1024_full36_mlp_q4_to_l3_seed1234",
}
SERVER_ONLY = "fulltoken_sync_copydrop_v0_q4_to_g3_seed1234"
SERVER_LOG_SUPPLEMENTS = {
    "frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234",
    "fulltoken_sync_copydrop_v0_g3_to_q4_seed1234",
    "fulltoken_sync_copydrop_v0_q4_to_l3_seed1234",
}
CHINESE_SOURCE = {
    "fulltoken_sync_copydrop_v0_l3_to_q4_seed1234",
    "fulltoken_sync_copydrop_v0_g3_to_q4_seed1234",
    "fulltoken_sync_copydrop_v0_q4_to_l3_seed1234",
    "frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234",
    "sync_boundary_nativekv_full28_diagonal_translator_llama3_2_3b_to_qwen3_4b_seed1234",
}


def source_path(name):
    if name == SERVER_ONLY:
        return "/home/yezhe/异构模型/" + name
    if name in CHINESE_SOURCE:
        return "异构模型/" + name
    if name in PREREQUISITES:
        return "runs/" + name
    return name


def classify(path):
    parts = path.parts
    if path.name == "config.json" or path.name == "training_config.json":
        return "config"
    if path.suffix == ".jsonl" and any(x in path.name for x in ("sample", "generation", "per_sample")):
        return "per_sample"
    if path.name.endswith("summary.json") or path.name.endswith("metrics.json"):
        return "summary"
    if path.suffix == ".log":
        return "log"
    return "other"


def metrics(data, prefix=""):
    if not isinstance(data, dict):
        return []
    output = []
    for key, value in data.items():
        label = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and label.count(".") < 3:
            output.extend(metrics(value, label))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if any(word in key.lower() for word in ("accuracy", "correct", "exact_match", "oracle", "count")):
                output.append((label, value))
    return output


def main():
    entries = []
    for directory in sorted(EXPERIMENTS.iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name
        files = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            source = "server-a" if (name == SERVER_ONLY or
                       (name in SERVER_LOG_SUPPLEMENTS and
                        path.relative_to(directory).parts[0] == "logs")) else SOURCE_REPO
            files.append({"path": relative, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "kind": classify(path.relative_to(directory)),
                          "source": source})
        entries.append({
            "name": name,
            "role": "code_prerequisite" if name in PREREQUISITES else "completed_experiment",
            "original_path": source_path(name),
            "source_commit": None if name == SERVER_ONLY else SOURCE_COMMIT,
            "source_repository": "server-a" if name == SERVER_ONLY else SOURCE_REPO,
            "files": files,
        })

    payload = {"schema_version": 1, "source_commit": SOURCE_COMMIT,
               "checkpoint_and_cache_policy": "excluded", "entries": entries}
    (ROOT / "MIGRATION_MANIFEST.json").write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    lines = ["# Experiment results", "",
             "Machine-generated from archived raw files by `tools/build_catalog.py`.",
             "This index does not reinterpret or replace each experiment's original results.", ""]
    for entry in entries:
        if entry["role"] != "completed_experiment":
            continue
        name = entry["name"]
        files = entry["files"]
        lines.extend([f"## {name}", "", "- Status: `completed` (archived result artifacts)",
                      f"- Original path: `{entry['original_path']}`"])
        config_file = next((file for file in files if file["kind"] == "config" and
                            "smoke" not in file["path"]), None)
        if config_file:
            try:
                config = json.loads((ROOT / config_file["path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                config = {}
            fields = ("seed", "train_samples", "val_samples", "test_samples",
                      "stage_a_epochs", "stage_b_epochs")
            fixed = [f"{key}={config[key]}" for key in fields if key in config]
            if fixed:
                lines.append("- Recorded configuration: " + ", ".join(fixed))
        for kind, limit in (("config", 2), ("summary", 6), ("per_sample", 4), ("log", 2)):
            selected = [file for file in files if file["kind"] == kind]
            # Ignore smoke summaries when a formal summary exists.
            if kind == "summary" and any("smoke" not in file["path"] for file in selected):
                selected = [file for file in selected if "smoke" not in file["path"]]
            for file in selected[:limit]:
                lines.append(f"- {kind.replace('_', ' ').title()}: [{Path(file['path']).name}]({file['path']})")
        result_md = next((file for file in files if file["path"].endswith("/RESULTS.md")), None)
        if result_md:
            lines.append(f"- Original report: [RESULTS.md]({result_md['path']})")
        selections = [file for file in files if file["path"].endswith("selection.json") and
                      "smoke" not in file["path"]]
        for file in selections[:2]:
            lines.append(f"- Checkpoint selection: [{Path(file['path']).parent.name}/selection.json]({file['path']})")
        primary = next((file for file in files if file["kind"] == "summary" and
                        "smoke" not in file["path"] and
                        (file["path"].endswith("test128_summary.json") or
                         file["path"].endswith("results/summary.json"))), None)
        if primary is None:
            primary = next((file for file in files if file["kind"] == "summary" and
                            "smoke" not in file["path"]), None)
        if primary:
            try:
                raw = json.loads((ROOT / primary["path"]).read_text(encoding="utf-8"))
                picked = metrics(raw)[:12]
            except (OSError, ValueError):
                picked = []
            if picked:
                lines.extend(["", "Automatically extracted metrics from the linked JSON:", "",
                              "| JSON field | Value |", "|---|---:|"])
                lines.extend(f"| `{key}` | {value} |" for key, value in picked)
        lines.append("")
    (ROOT / "EXPERIMENT_RESULTS.md").write_bytes(
        ("\n".join(lines).rstrip() + "\n").encode("utf-8"))
    print(f"Indexed {sum(x['role'] == 'completed_experiment' for x in entries)} experiments "
          f"and {sum(x['role'] == 'code_prerequisite' for x in entries)} code prerequisites")


if __name__ == "__main__":
    main()
