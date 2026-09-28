"""Standard free-generation GSM8K evaluation for the two frozen Writers.

This intentionally does not convert GSM8K to multiple choice.  It reuses the
existing benchmark prompt, greedy decoding, stopping rule and numeric answer
extraction.  Receiver token0 remains native; all remaining question-prefix KV
is Full-Sync aligned and translated by the frozen Stage-A/Stage-B modules.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from evaluate import (CFG, HERE, backbone, capture, fullsync_map, load_model,
                      load_writer, make_cache, save_json, translate)


def normalize_number(text):
    return str(text).replace(",", "").strip()


def extract_final_answer(text):
    text = str(text)
    matches = re.findall(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if matches:
        return normalize_number(matches[-1])
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", boxed[-1])
        if numbers:
            return normalize_number(numbers[-1])
    for pattern in (r"(?:final answer|answer|therefore|so|thus)\s*(?:is|:)?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else ""


def has_complete_final_answer(text):
    patterns = (
        r"####\s*\$?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*[\n\r.;,)]",
        r"\\boxed\{\s*\$?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\}",
        r"(?:final answer|answer|therefore|so|thus)\s*(?:is|:)?\s*\$?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*[\n\r.;,)]",
    )
    return any(re.search(pattern, str(text), flags=re.IGNORECASE) for pattern in patterns)


def contains_gold(text, gold):
    raw = normalize_number(gold)
    variants = {raw}
    if re.fullmatch(r"[-+]?\d+", raw):
        sign, digits = (raw[0], raw[1:]) if raw.startswith(("-", "+")) else ("", raw)
        groups = []
        while len(digits) > 3:
            groups.append(digits[-3:]); digits = digits[:-3]
        groups.append(digits)
        variants.add(sign + ",".join(reversed(groups)))
    return any(re.search(r"(?<![\d,])" + re.escape(value) + r"(?![\d,])", str(text)) for value in variants)


def prompt_parts(question):
    # The blank-line boundary is the same compositional boundary used by the
    # established standard prompt: prefix IDs + Answer IDs must exactly equal
    # the native full-prompt IDs for every tokenizer.
    source = f"Question:\n{question}\n\n"
    continuation = "Answer:"
    return source, continuation, source + continuation


def encode_prefix(tok, text):
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return bos + list(tok.encode(text, add_special_tokens=False))


def alignment_row(sample_id, source_text, family, source_tok, qwen_tok):
    encoded = {}
    for name, tok in ((family, source_tok), ("qwen", qwen_tok)):
        ids = encode_prefix(tok, source_text)
        answer_ids = list(tok.encode("Answer:", add_special_tokens=False))
        full_ids = encode_prefix(tok, source_text + "Answer:")
        if ids + answer_ids != full_ids:
            raise RuntimeError(f"Noncompositional GSM8K prefix boundary for {name}: {sample_id}")
        # Full-Sync protocols reserve sequence position 0 as the native
        # receiver anchor even for tokenizers without an explicit BOS token.
        encoded[name] = {"body": ids, "option_token_indices": list(range(1, len(ids)))}
    return {"id": sample_id, "body": source_text, "encoded": encoded}


@torch.no_grad()
def greedy_generate(model, tok, prompt_ids, max_new_tokens, key=None, value=None):
    if key is None:
        past, prefix = None, 0
    else:
        positions = torch.arange(key.shape[1], device="cuda")
        past, prefix = make_cache(model, key, value, positions), key.shape[1]
    current = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    generated = []
    for _ in range(max_new_tokens):
        length = int(past.get_seq_length()) if past is not None else prefix
        mask = torch.ones((1, length + current.shape[1]), device="cuda", dtype=torch.long)
        positions = torch.arange(length, length + current.shape[1], device="cuda")[None]
        output = backbone(model)(input_ids=current, attention_mask=mask, position_ids=positions,
                                 past_key_values=past, use_cache=True, return_dict=True)
        token = int(model.lm_head(output.last_hidden_state[:, -1])[0].argmax().item())
        generated.append(token)
        past = output.past_key_values
        if token == tok.eos_token_id:
            break
        text = tok.decode(generated, skip_special_tokens=True)
        if has_complete_final_answer(text):
            break
        current = torch.tensor([[token]], device="cuda", dtype=torch.long)
    return generated, tok.decode(generated, skip_special_tokens=True)


def load_rows(limit):
    rows = [json.loads(line) for line in Path(CFG["datasets"]["gsm8k"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit]


def condition_record(text, generated, gold):
    prediction = extract_final_answer(text)
    return {"generated_text": text, "generated_tokens": len(generated),
            "pred_final_answer": prediction, "gold_final_answer": gold,
            "exact_match": prediction == gold, "contains_gold": contains_gold(text, gold)}


def summarize(records, conditions):
    result = {"count": len(records), "protocol": "standard GSM8K free generation; no constructed choices"}
    for name in conditions:
        exact = sum(row["conditions"][name]["exact_match"] for row in records)
        result[name] = {"correct": exact, "accuracy": exact / len(records),
                        "contains_gold": sum(row["conditions"][name]["contains_gold"] for row in records) / len(records),
                        "mean_generated_tokens": sum(row["conditions"][name]["generated_tokens"] for row in records) / len(records)}
    for name in ("stage_a", "stage_b"):
        if name not in conditions or "native_cache_oracle" not in conditions:
            continue
        both = sum(row["conditions"][name]["exact_match"] and row["conditions"]["native_cache_oracle"]["exact_match"] for row in records)
        result[name].update(both_with_oracle_correct=both,
            writer_only_correct=result[name]["correct"] - both,
            oracle_only_correct=result["native_cache_oracle"]["correct"] - both,
            exact_generation_agreement=sum(row["conditions"][name]["generated_text"] == row["conditions"]["native_cache_oracle"]["generated_text"] for row in records) / len(records))
    if "qwen_native" in conditions and "native_cache_oracle" in conditions:
        result["native_text_vs_native_cache_exact_generation_agreement"] = sum(
            row["conditions"]["qwen_native"]["generated_text"] == row["conditions"]["native_cache_oracle"]["generated_text"] for row in records) / len(records)
    return result


@torch.no_grad()
def evaluate_family(family, limit, max_new_tokens, conditions):
    rows = load_rows(limit)
    source_tok = AutoTokenizer.from_pretrained(CFG["models"][family], local_files_only=True)
    qwen_tok = AutoTokenizer.from_pretrained(CFG["models"]["qwen"], local_files_only=True)
    source_cache = []
    sender = load_model(family)
    try:
        for index, item in enumerate(rows):
            source_text, _, native_prompt = prompt_parts(item["question"])
            row = alignment_row(f"gsm8k_{index}", source_text, family, source_tok, qwen_tok)
            target_indices, source_indices, counts = fullsync_map(row, source_tok, qwen_tok, family)
            fields = row["encoded"][family]
            source_k, source_v, _ = capture(sender, fields["body"], len(fields["body"]), [0])
            if "sender_native" in conditions:
                native_ids = encode_prefix(source_tok, native_prompt)
                generated, text = greedy_generate(sender, source_tok, native_ids, max_new_tokens)
                sender_native = (generated, text)
            else:
                sender_native = None
            source_cache.append({"k": source_k[:, source_indices].half(), "v": source_v[:, source_indices].half(),
                "target_indices": target_indices, "mapping_counts": counts,
                "sender_native": sender_native})
            if (index + 1) % 8 == 0 or index + 1 == len(rows):
                print(f"{family} GSM8K sender {index + 1}/{len(rows)}", flush=True)
    finally:
        del sender; gc.collect(); torch.cuda.empty_cache()

    qwen = load_model("qwen")
    base, adapter, path_a, path_b = load_writer(family)
    records = []
    try:
        for index, item in enumerate(rows):
            source_text, continuation, native_prompt = prompt_parts(item["question"])
            qids = encode_prefix(qwen_tok, source_text)
            native_k, native_v, _ = capture(qwen, qids, len(qids), [0])
            cached = source_cache[index]
            ak, av, bk, bv = translate(base, adapter, cached["k"], cached["v"])
            # Receiver-native token0 plus one translated KV per remaining Qwen prompt token.
            anchor_k, anchor_v = native_k[:, :1].cuda(), native_v[:, :1].cuda()
            oracle_k, oracle_v = native_k.cuda(), native_v.cuda()
            ak, av = torch.cat((anchor_k, ak), 1), torch.cat((anchor_v, av), 1)
            bk, bv = torch.cat((anchor_k, bk), 1), torch.cat((anchor_v, bv), 1)
            continuation_ids = list(qwen_tok.encode(continuation, add_special_tokens=False))
            native_ids = encode_prefix(qwen_tok, native_prompt)
            outputs = {}
            gold = extract_final_answer(item["answer"])
            if "sender_native" in conditions:
                sender_generated, sender_text = cached["sender_native"]
                outputs["sender_native"] = condition_record(sender_text, sender_generated, gold)
            candidates = (
                    ("qwen_native", native_ids, None, None),
                    ("native_cache_oracle", continuation_ids, oracle_k, oracle_v),
                    ("stage_a", continuation_ids, ak, av),
                    ("stage_b", continuation_ids, bk, bv))
            for name, prompt, key, value in candidates:
                if name not in conditions:
                    continue
                generated, text = greedy_generate(qwen, qwen_tok, prompt, max_new_tokens, key, value)
                outputs[name] = condition_record(text, generated, gold)
            records.append({"id": f"gsm8k_{index}", "question": item["question"],
                "gold_answer": item["answer"], "mapping_counts": cached["mapping_counts"],
                "qwen_prefix_tokens": len(qids), "conditions": outputs})
            if (index + 1) % 8 == 0 or index + 1 == len(rows):
                print(f"{family} GSM8K receiver {index + 1}/{len(rows)}", flush=True)
            del native_k, native_v, oracle_k, oracle_v, ak, av, bk, bv
    finally:
        del qwen, base, adapter, source_cache; gc.collect(); torch.cuda.empty_cache()
    out = HERE / "runs/gsm8k_generation" / family
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, conditions)
    summary.update(status="completed", family=family, stage_a_checkpoint=path_a,
                   stage_b_checkpoint=path_b, max_new_tokens=max_new_tokens)
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("llama", "gemma"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--conditions", default="sender_native,qwen_native,native_cache_oracle,stage_a,stage_b")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    allowed = {"sender_native", "qwen_native", "native_cache_oracle", "stage_a", "stage_b"}
    if not conditions or not set(conditions) <= allowed:
        raise ValueError(f"Invalid conditions: {conditions}")
    evaluate_family(args.family, args.limit, args.max_new_tokens, conditions)


if __name__ == "__main__":
    main()
