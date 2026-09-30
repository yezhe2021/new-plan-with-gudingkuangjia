"""Evaluate the four-epoch KL-only Hub checkpoint with the legacy GSM8K protocol."""

import gc
import json
import re
from pathlib import Path

import torch

import unified_hub as U


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "evaluation_legacy_gsm8k"
STOP = ("Question:", "</s>", "<|im_end|>")


def legacy_rows(cfg, toks):
    raw = U.read_jsonl(cfg["datasets"]["gsm8k_test"])[: cfg["test_counts"]["gsm8k"]]
    rows = []
    for index, item in enumerate(raw):
        body = f"Question: {str(item['question']).strip()}\n"
        serialized = {
            "body": body,
            "answer_prefix": "Answer:",
            "full_prompt": body + "Answer:",
            "continuation_text": str(item["answer"]),
            "gold_text": str(item["answer"]),
            "labels": [],
            "gold_index": None,
        }
        encoded = {name: U.encode_row(tok, serialized, "gsm8k") for name, tok in toks.items()}
        rows.append({
            "id": f"gsm8k_test_{index}", "dataset": "gsm8k", "task": "generation",
            "question": item["question"], "gold_answer": item["answer"],
            **serialized, "encoded": encoded,
        })
    return rows


def normalize(value):
    return str(value).replace(",", "").replace("$", "").strip().rstrip(".")


def gold_number(answer):
    values = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
    return normalize(values[-1]) if values else ""


def predictions(text):
    strict = re.findall(r"#### (\-?[0-9\.\,]+)", text)
    strict = normalize(strict[0]) if strict else "[invalid]"
    first_hash = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", text)
    first_hash = normalize(first_hash.group(1)) if first_hash else "[invalid]"
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if flexible:
        flexible = normalize(next((x for x in flexible[-1] if x), "[invalid]"))
    else:
        flexible = "[invalid]"
    return strict, flexible, first_hash


@torch.no_grad()
def generate(model, tok, ids, max_tokens, key=None, value=None):
    past = None if key is None else U.make_cache(
        model, key, value, torch.arange(key.shape[1], device="cuda")
    )
    current = torch.tensor([ids], device="cuda", dtype=torch.long)
    eos = model.generation_config.eos_token_id
    eos = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
    generated, text, reason = [], "", "max_tokens"
    for _ in range(max_tokens):
        prefix = 0 if past is None else int(past.get_seq_length())
        output = U.backbone(model)(
            input_ids=current,
            attention_mask=torch.ones((1, prefix + current.shape[1]), device="cuda", dtype=torch.long),
            position_ids=torch.arange(prefix, prefix + current.shape[1], device="cuda")[None],
            past_key_values=past, use_cache=True, return_dict=True,
        )
        token = int(model.lm_head(output.last_hidden_state[:, -1])[0].argmax())
        past = output.past_key_values
        if token in eos:
            reason = "eos"
            break
        generated.append(token)
        text = tok.decode(generated, skip_special_tokens=True)
        hits = [(text.find(marker), marker) for marker in STOP if text.find(marker) >= 0]
        if hits:
            first, marker = min(hits, key=lambda pair: pair[0])
            text, reason = text[:first], f"until:{marker}"
            break
        current = torch.tensor([[token]], device="cuda")
    return text, len(generated), reason


def record(text, count, reason, answer):
    gold = gold_number(answer)
    strict, flexible, first_hash = predictions(text)
    return {
        "generated_text": text, "generated_tokens": count, "stop_reason": reason, "gold": gold,
        "strict_prediction": strict, "flexible_prediction": flexible,
        "first_hash_prediction": first_hash, "strict_correct": strict == gold,
        "flexible_correct": flexible == gold, "first_hash_correct": first_hash == gold,
    }


def summarize(records):
    count = len(records)
    return {
        "count": count,
        "strict_correct": sum(x["strict_correct"] for x in records),
        "strict_accuracy": sum(x["strict_correct"] for x in records) / count,
        "first_hash_correct": sum(x["first_hash_correct"] for x in records),
        "first_hash_accuracy": sum(x["first_hash_correct"] for x in records) / count,
        "flexible_correct": sum(x["flexible_correct"] for x in records),
        "flexible_accuracy": sum(x["flexible_correct"] for x in records) / count,
        "mean_generated_tokens": sum(x["generated_tokens"] for x in records) / count,
        "eos_rate": sum(x["stop_reason"] == "eos" for x in records) / count,
        "max_token_rate": sum(x["stop_reason"] == "max_tokens" for x in records) / count,
    }


def main():
    cfg = U.runtime_config(False)
    U.seed_all(cfg["seed"])
    toks = U.tokenizers(cfg)
    rows = legacy_rows(cfg, toks)
    root = ROOT / "runs"
    base = U.NativeKVTranslator("full28_mlp", hidden_dim=cfg["mlp_hidden_dim"]).cuda().eval().requires_grad_(False)
    U.load_checkpoint(root / "stage_a" / "best.pt", cfg, "stage_a", base)
    adapter = U.ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().eval().requires_grad_(False)
    U.load_checkpoint(root / "universal_kl" / "best.pt", cfg, "universal_kl", adapter)
    llama, qwen = U.load_model(U.model_config(cfg), "llama"), U.load_model(U.model_config(cfg), "qwen")
    names = ("qwen_full_native", "stage_a", "universal_kl")
    all_records, by_condition = [], {name: [] for name in names}
    for number, row in enumerate(rows, 1):
        item = U.pair(llama, qwen, toks["llama"], toks["qwen"], row)
        base_k, base_v = U.mapped_base(base, item, cfg["chunk_tokens"])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            key, value, _, _ = adapter(base_k[None], base_v[None])
        caches = {"stage_a": (base_k, base_v), "universal_kl": (key[0], value[0])}
        conditions = {}
        for name in names:
            if name == "qwen_full_native":
                cache_key = cache_value = None
                prompt = row["encoded"]["qwen"]["body"] + row["encoded"]["qwen"]["suffix"]
            else:
                key0, value0 = caches[name]
                cache_key = torch.cat((item["token0_k"].cuda(), key0), 1)
                cache_value = torch.cat((item["token0_v"].cuda(), value0), 1)
                prompt = row["encoded"]["qwen"]["suffix"]
            text, count, reason = generate(
                qwen, toks["qwen"], prompt, cfg["gsm8k_max_new_tokens"], cache_key, cache_value
            )
            conditions[name] = record(text, count, reason, row["gold_answer"])
            by_condition[name].append(conditions[name])
        all_records.append({
            "id": row["id"], "question": row["question"],
            "gold_answer": row["gold_answer"], "conditions": conditions,
        })
        if number % 8 == 0 or number == len(rows):
            U.log(f"legacy GSM8K KL4 evaluation {number}/{len(rows)}")
        del item, base_k, base_v, caches, key, value
    metrics = {name: summarize(records) for name, records in by_condition.items()}
    for name in ("stage_a", "universal_kl"):
        both = sum(a["flexible_correct"] and b["flexible_correct"] for a, b in zip(by_condition[name], by_condition["qwen_full_native"]))
        method_only = sum(a["flexible_correct"] and not b["flexible_correct"] for a, b in zip(by_condition[name], by_condition["qwen_full_native"]))
        native_only = sum(not a["flexible_correct"] and b["flexible_correct"] for a, b in zip(by_condition[name], by_condition["qwen_full_native"]))
        metrics[name]["vs_qwen_native"] = {
            "both_correct": both, "method_only_correct": method_only,
            "qwen_only_correct": native_only, "both_wrong": len(rows) - both - method_only - native_only,
        }
    U.write_jsonl(OUT / "per_sample.jsonl", all_records)
    U.save_json(OUT / "summary.json", {
        "status": "completed",
        "checkpoint_protocol": cfg["protocol"],
        "evaluation_protocol": "legacy_single_gsm8k",
        "prompt": "Question: {question}\\nAnswer:",
        "stop_markers": list(STOP),
        "conditions": metrics,
    })
    U.log("LEGACY GSM8K KL4 EVALUATION COMPLETED")
    del llama, qwen, base, adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
