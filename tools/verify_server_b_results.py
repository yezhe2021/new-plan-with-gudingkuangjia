"""Recompute archived results and verify source fingerprints without a GPU."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
READER = EXPERIMENTS / "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234"
REWARD = READER / "reward_aligned_reader_gsm8k_pilot_seed1234"
GROUPED = EXPERIMENTS / "grouped_headmix4_l3_to_q4_seed1234"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_metrics(records, metrics):
    assert len({r["id"] for r in records}) == len(records), "Duplicate sample IDs"
    for arm, expected in metrics.items():
        values = [r["conditions"][arm] for r in records]
        correct = sum(v["correct"] for v in values)
        assert len(values) == expected["n"], (arm, "n")
        assert correct == expected["correct"], (arm, "correct")
        assert math.isclose(correct / len(values), expected["accuracy"], abs_tol=1e-12)
        if "cap_count" in expected:
            assert sum(v["reached_cap"] for v in values) == expected["cap_count"]
        if "mean_generated_tokens" in expected:
            mean = sum(v["generated_tokens"] for v in values) / len(values)
            assert math.isclose(mean, expected["mean_generated_tokens"], abs_tol=1e-10)


def check_pairs(records, expected, initial="initial_reader", final="reward_reader"):
    pairs = [(r["conditions"][initial]["correct"], r["conditions"][final]["correct"]) for r in records]
    actual = dict(both_correct=sum(a and b for a, b in pairs),
                  initial_only=sum(a and not b for a, b in pairs),
                  reward_only=sum(b and not a for a, b in pairs),
                  both_wrong=sum(not a and not b for a, b in pairs))
    for name, value in actual.items():
        key = "checkpoint_only" if name == "reward_only" and final == "checkpoint_reader" else name
        assert expected[key] == value, (key, value, expected[key])


def main():
    manifests = list(EXPERIMENTS.glob("*/manifest.json")) + [REWARD / "manifest.json"]
    checked_sources = 0
    for path in manifests:
        manifest = read(path)
        if "config_sha256" in manifest:
            assert sha(path.parent / "config.json") == manifest["config_sha256"], path
        for name, expected in manifest.get("code", {}).items():
            assert sha(path.parent / name) == expected, path.parent / name
            checked_sources += 1

    for folder, section in [(GROUPED, "results"), (READER, "tasks")]:
        summary = read(folder / "summary.json")
        records = rows(folder / "evaluation/per_sample.jsonl")
        assert len(records) == 544
        for task, values in summary[section].items():
            selected = [r for r in records if r["task"] == task]
            check_metrics(selected, values if section == "results" else values["metrics"])

    summary = read(REWARD / "summary.json")
    records = rows(REWARD / "evaluation/per_sample.jsonl")
    assert len(records) == 128
    for name, values in summary["subsets"].items():
        selected = records if name == "all128" else [r for r in records if r["subset"] == name]
        check_metrics(selected, values["metrics"])
        check_pairs(selected, values["paired"])

    for step in [160, 208]:
        folder = REWARD / "diagnostics" / f"validation_step{step:04d}_20261022"
        summary = read(folder / "summary.json")
        manifest = read(folder / "run.json")
        assert sha(folder / "run_validation.py") == manifest["script_sha256"]
        assert summary["checkpoint_step"] == step and summary["completed_pairs"] == 64
        records = rows(folder / "per_sample.jsonl")
        assert len(records) == 64
        assert {r["id"] for r in records} == set(manifest["ordered_question_ids"])
        assert all(r["run_id"] == manifest["run_id"] for r in records)
        check_metrics(records, summary["metrics"])
        check_pairs(records, summary["paired"], final="checkpoint_reader")

    inventory = ROOT / "experiments/SERVER_B_ARCHIVE_SHA256.json"
    if inventory.exists():
        for name, expected in read(inventory)["files"].items():
            assert sha(ROOT / name) == expected, name
    print(f"PASS: {checked_sources} source fingerprints, 544 grouped results, 544 Reader results, "
          "128 final reward results, and 2 x 64 intermediate validation results.")


if __name__ == "__main__":
    main()
