"""Require exact greedy continuations from HF Qwen and custom native Qwen."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sample_file(relative):
    files = sorted((HERE / relative).rglob("samples_gsm8k_local_*.jsonl"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one GSM8K sample file below {relative}, found {files}")
    return files[0]


def first_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            result = first_text(item)
            if result is not None:
                return result
    return None


def read(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row.get("doc_id"), first_text(row.get("resps"))) for row in rows]


hf = read(sample_file("results/qwen_native_hf_eager_0shot_limit10"))
custom = read(sample_file("results/qwen_native_custom_eager_eosfix_0shot_limit10"))
if len(hf) != len(custom):
    raise RuntimeError(f"Native sample count mismatch: HF={len(hf)} custom={len(custom)}")
records = []
for left, right in zip(hf, custom):
    record = {"doc_id_hf": left[0], "doc_id_custom": right[0],
              "hf_generation": left[1], "custom_generation": right[1],
              "exact_generation_match": left == right}
    records.append(record)
(HERE / "results/native_qwen_exact_comparison.json").write_text(
    json.dumps({"count": len(records), "exact_matches": sum(r["exact_generation_match"] for r in records),
                "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not all(row["exact_generation_match"] for row in records):
    raise RuntimeError("Custom native-Qwen generation does not exactly reproduce HF Qwen")
print(f"Native-Qwen exact generation audit passed: {len(records)}/{len(records)}")
