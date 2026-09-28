"""Greedy generation evaluation for Stage-A, old MCQ-B and new Gen-B."""

import argparse
import gc
import json
from pathlib import Path

import torch

import generation_common as common
from generation_common import (CFG, HERE, adapted_cache, load_adapter, native_token0,
    prepare_base_cache, read_jsonl, save_json)


def condition(text, ids, gold):
    prediction = common.gen.extract_final_answer(text)
    return {"generated_text": text, "generated_tokens": len(ids),
            "pred_final_answer": prediction, "gold_final_answer": gold,
            "exact_match": prediction == gold, "contains_gold": common.gen.contains_gold(text, gold)}


@torch.no_grad()
def generate_with_adapter(qwen, tok, entry, adapter, token0_k, token0_v, max_tokens):
    key, value = adapted_cache(adapter, entry, token0_k, token0_v)
    continuation = list(tok.encode("Answer:", add_special_tokens=False))
    return common.gen.greedy_generate(qwen, tok, continuation, max_tokens, key, value)


@torch.no_grad()
def generate_stage_a(qwen, tok, entry, token0_k, token0_v, max_tokens):
    key = torch.cat((token0_k, entry["base_k"].cuda()), 1)
    value = torch.cat((token0_v, entry["base_v"].cuda()), 1)
    continuation = list(tok.encode("Answer:", add_special_tokens=False))
    return common.gen.greedy_generate(qwen, tok, continuation, max_tokens, key, value)


def summarize(records, conditions):
    output = {"count": len(records), "protocol": "standard GSM8K free generation"}
    for name in conditions:
        exact = sum(row["conditions"][name]["exact_match"] for row in records)
        output[name] = {"correct": exact, "accuracy": exact / len(records),
                        "contains_gold": sum(row["conditions"][name]["contains_gold"] for row in records) / len(records),
                        "mean_generated_tokens": sum(row["conditions"][name]["generated_tokens"] for row in records) / len(records)}
    return output


def evaluate(rows, label, new_checkpoint, conditions, limit):
    rows = rows[:limit]
    cache, _, tok, stage_a_path, module = prepare_base_cache(rows, label)
    qwen = common.ref.load_model("qwen")
    token0_k, token0_v, _ = native_token0(qwen, tok)
    old_adapter = load_adapter(module, common.old_stage_b_checkpoint()).eval() if "old_mcq_stage_b" in conditions else None
    new_adapter = load_adapter(module, new_checkpoint).eval() if "new_generation_stage_b" in conditions else None
    records = []
    try:
        for index, (row, entry) in enumerate(zip(rows, cache), 1):
            gold = common.gen.extract_final_answer(row["answer"])
            values = {}
            if "qwen_native" in conditions:
                _, _, prompt = common.gen.prompt_parts(row["question"])
                ids, text = common.gen.greedy_generate(qwen, tok, common.gen.encode_prefix(tok, prompt), CFG["max_new_tokens"])
                values["qwen_native"] = condition(text, ids, gold)
            if "native_cache_oracle" in conditions:
                # Native-cache equivalence is a hard preflight audit; reuse the
                # identical native trajectory rather than regenerate it.
                if "qwen_native" not in values: raise RuntimeError("Oracle reuse requires qwen_native")
                values["native_cache_oracle"] = dict(values["qwen_native"])
            if "stage_a" in conditions:
                ids, text = generate_stage_a(qwen, tok, entry, token0_k, token0_v, CFG["max_new_tokens"])
                values["stage_a"] = condition(text, ids, gold)
            if "old_mcq_stage_b" in conditions:
                ids, text = generate_with_adapter(qwen, tok, entry, old_adapter, token0_k, token0_v, CFG["max_new_tokens"])
                values["old_mcq_stage_b"] = condition(text, ids, gold)
            if "new_generation_stage_b" in conditions:
                ids, text = generate_with_adapter(qwen, tok, entry, new_adapter, token0_k, token0_v, CFG["max_new_tokens"])
                values["new_generation_stage_b"] = condition(text, ids, gold)
            records.append({"id": f"{label}_{index-1}", "question": row["question"],
                            "gold_answer": row["answer"], "conditions": values})
            if index % 8 == 0 or index == len(rows): print(f"{label} generation {index}/{len(rows)}", flush=True)
    finally:
        del qwen, old_adapter, new_adapter, cache; gc.collect(); torch.cuda.empty_cache()
    return records, summarize(records, conditions), stage_a_path


def write_records(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def preflight_old():
    rows = read_jsonl(CFG["data_test"])
    records, summary, _ = evaluate(rows, "test", None, ("old_mcq_stage_b",), CFG["test_samples"])
    root = HERE / "runs/preflight_old_stage_b"
    write_records(root / "per_sample.jsonl", records); save_json(root / "summary.json", summary)
    if summary["old_mcq_stage_b"]["correct"] != 0:
        raise RuntimeError(f"Old Stage-B no longer reproduces 0/32: {summary}")
    print(json.dumps(summary, indent=2), flush=True)


def final_evaluate(smoke=False):
    root = HERE / "runs" / ("smoke" if smoke else "study")
    checkpoint = root / "checkpoints/final.pt"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    test_rows = read_jsonl(CFG["data_test"])
    conditions = ("qwen_native", "native_cache_oracle", "stage_a", "old_mcq_stage_b", "new_generation_stage_b")
    count = 2 if smoke else CFG["test_samples"]
    test_records, test_summary, stage_a = evaluate(test_rows, "test", checkpoint, conditions, count)
    train_manifest = json.loads((root / "train_manifest.json").read_text())
    all_train = read_jsonl(CFG["data_train"])
    train_rows = [all_train[index] for index in train_manifest["selected_indices"]]
    train_count = 2 if smoke else CFG["train_generation_samples"]
    train_records, train_summary, _ = evaluate(train_rows, "train", checkpoint,
                                                ("new_generation_stage_b",), train_count)
    write_records(root / "results/test_per_sample.jsonl", test_records)
    write_records(root / "results/train_per_sample.jsonl", train_records)
    result = {"status": "completed", "test": test_summary, "train": train_summary,
              "stage_a_checkpoint": stage_a, "old_stage_b_checkpoint": common.old_stage_b_checkpoint(),
              "new_stage_b_checkpoint": str(checkpoint), "test_rows": count, "train_generation_rows": train_count}
    save_json(root / "results/summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight_old", "evaluate"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    preflight_old() if args.action == "preflight_old" else final_evaluate(args.smoke)


if __name__ == "__main__":
    main()
