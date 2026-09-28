"""Official-test128 free-generation comparison for Stage-B v2."""

import gc
import json
from pathlib import Path

import torch

import generation_common as C


def evaluate():
    rows = C.read_jsonl(C.CFG["data_test"])[:C.CFG["test_samples"]]
    indices = list(range(len(rows)))
    cache, llama_tok, tok, stage_a, module = C.prepare_base_cache(
        rows, indices, "test", sender_generation=True)
    qwen = C.ref.load_model("qwen").requires_grad_(False)
    token0_k, token0_v, _ = C.native_token0(qwen, tok, cache)
    adapters = {"old_gen_b": C.load_adapter(module, C.CFG["old_gen_b_checkpoint"]).eval()}
    selections = {}
    for objective in ("ce", "kl", "hybrid"):
        selection = json.loads((C.HERE / "runs" / objective / "selection.json").read_text())
        selections[objective] = selection["best"]
        adapters[objective] = C.load_adapter(module, selection["best"]["checkpoint"]).eval()
    names = ("llama_native", "qwen_native", "stage_a", "old_gen_b", "ce", "kl", "hybrid")
    per_condition = {name: [] for name in names}; records = []
    try:
        for position, (row, entry) in enumerate(zip(rows, cache), 1):
            conditions = {}
            sender_text, sender_count, sender_reason = entry["sender_result"]
            conditions["llama_native"] = C.generation_record(
                sender_text, sender_count, sender_reason, row["answer"])
            _, _, full_prompt = C.prompt_parts(row["question"])
            text, count, reason = C.greedy_generate(
                qwen, tok, C.encode_prefix(tok, full_prompt), C.CFG["max_new_tokens"])
            conditions["qwen_native"] = C.generation_record(text, count, reason, row["answer"])
            suffix = list(tok.encode("Answer:", add_special_tokens=False))
            stage_key = torch.cat((token0_k, entry["base_k"].cuda()), 1)
            stage_value = torch.cat((token0_v, entry["base_v"].cuda()), 1)
            text, count, reason = C.greedy_generate(qwen, tok, suffix, C.CFG["max_new_tokens"],
                                                   stage_key, stage_value)
            conditions["stage_a"] = C.generation_record(text, count, reason, row["answer"])
            for name, adapter in adapters.items():
                key, value = C.adapted_cache(adapter, entry, token0_k, token0_v)
                text, count, reason = C.greedy_generate(qwen, tok, suffix, C.CFG["max_new_tokens"], key, value)
                conditions[name] = C.generation_record(text, count, reason, row["answer"])
                del key, value
            for name in names: per_condition[name].append(conditions[name])
            records.append({"id": f"test_{position-1}", "question": row["question"],
                            "gold_answer": row["answer"], "conditions": conditions})
            if position % 8 == 0:
                print(f"test generation {position}/{len(rows)}", flush=True)
            del stage_key, stage_value
    finally:
        del qwen, adapters, cache; gc.collect(); torch.cuda.empty_cache()
    summary = {name: C.summarize_generations(per_condition[name]) for name in names}
    summary.update(status="completed", test_samples=len(rows), prompt="Question: {question}\\nAnswer:",
                   max_new_tokens=C.CFG["max_new_tokens"], stage_a_checkpoint=stage_a,
                   old_gen_b_checkpoint=C.CFG["old_gen_b_checkpoint"], selections=selections,
                   official_test_untouched_by_training=True)
    C.write_jsonl(C.HERE / "runs/test128_per_sample.jsonl", records)
    C.save_json(C.HERE / "runs/test128_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    evaluate()
