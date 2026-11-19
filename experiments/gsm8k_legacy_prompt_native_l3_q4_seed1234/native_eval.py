"""Native-only GSM8K audit using the frozen legacy generation protocol."""

import gc
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalize_number(text):
    return str(text).replace(",", "").replace("$", "").strip().rstrip(".")


def gold_number(answer):
    found = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
    return normalize_number(found[-1]) if found else ""


def predictions(text):
    strict = re.findall(r"#### (\-?[0-9\.\,]+)", text)
    strict = normalize_number(strict[0]) if strict else "[invalid]"
    first_hash = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", text)
    first_hash = normalize_number(first_hash.group(1)) if first_hash else "[invalid]"
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if flexible:
        pair = flexible[-1]
        flexible = normalize_number(next((part for part in pair if part), "[invalid]"))
    else:
        flexible = "[invalid]"
    return strict, flexible, first_hash


@torch.no_grad()
def greedy_generate(model, tok, prompt_ids):
    current = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    past = None
    eos = model.generation_config.eos_token_id
    eos = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
    generated, reason, text = [], "max_tokens", ""
    for _ in range(CFG["max_new_tokens"]):
        output = model(input_ids=current, past_key_values=past, use_cache=True, return_dict=True)
        token = int(output.logits[0, -1].argmax())
        past = output.past_key_values
        if token in eos:
            reason = "eos"
            break
        generated.append(token)
        text = tok.decode(generated, skip_special_tokens=True)
        hits = [(text.find(marker), marker) for marker in CFG["stop_markers"] if text.find(marker) >= 0]
        if hits:
            first, marker = min(hits, key=lambda item: item[0])
            text, reason = text[:first], f"until:{marker}"
            break
        current = torch.tensor([[token]], device="cuda", dtype=torch.long)
    return text, len(generated), reason


def evaluate_model(name, path, rows):
    log(f"loading {name}: {path}")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True
    ).cuda().eval().requires_grad_(False)
    records = []
    for index, row in enumerate(rows):
        prompt = CFG["prompt"].format(question=str(row["question"]).strip())
        ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + list(
            tok.encode(prompt, add_special_tokens=False)
        )
        text, count, reason = greedy_generate(model, tok, ids)
        gold = gold_number(row["answer"])
        strict, flexible, first_hash = predictions(text)
        records.append({
            "id": f"test_{index}", "question": row["question"], "gold_answer": row["answer"],
            "prompt": prompt, "generated_text": text, "generated_tokens": count,
            "stop_reason": reason, "gold": gold, "strict_prediction": strict,
            "flexible_prediction": flexible, "first_hash_prediction": first_hash,
            "strict_correct": strict == gold, "flexible_correct": flexible == gold,
            "first_hash_correct": first_hash == gold
        })
        if (index + 1) % 8 == 0 or index + 1 == len(rows):
            log(f"{name}: {index + 1}/{len(rows)}")
    count = len(records)
    summary = {
        "model": name, "count": count, "prompt": CFG["prompt"],
        "strict_correct": sum(x["strict_correct"] for x in records),
        "strict_accuracy": sum(x["strict_correct"] for x in records) / count,
        "first_hash_correct": sum(x["first_hash_correct"] for x in records),
        "first_hash_accuracy": sum(x["first_hash_correct"] for x in records) / count,
        "flexible_correct": sum(x["flexible_correct"] for x in records),
        "flexible_accuracy": sum(x["flexible_correct"] for x in records) / count,
        "mean_generated_tokens": sum(x["generated_tokens"] for x in records) / count,
        "eos_rate": sum(x["stop_reason"] == "eos" for x in records) / count,
        "max_token_rate": sum(x["stop_reason"] == "max_tokens" for x in records) / count
    }
    write_jsonl(HERE / "results" / f"{name}_per_sample.jsonl", records)
    save_json(HERE / "results" / f"{name}_summary.json", summary)
    log(f"{name} summary: {summary}")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    rows = read_jsonl(CFG["dataset"])[:CFG["samples"]]
    summaries = {name: evaluate_model(name, path, rows) for name, path in CFG["models"].items()}
    save_json(HERE / "results" / "summary.json", {
        "status": "completed", "protocol": "legacy_gsm8k_native_generation_v1",
        "dataset": CFG["dataset"], "sample_policy": "first_128_official_test", "models": summaries
    })
    log("ALL EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    main()
