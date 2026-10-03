"""Evaluate Llama3.2-3B full-native on the four unified-training test sets."""

import gc
import json
import re
from collections import Counter
from pathlib import Path

import torch

import unified_hub as U


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "llama_full_native_four_datasets"
STOP = ("Question:", "</s>", "<|im_end|>")


def normalize(value):
    return str(value).replace(",", "").replace("$", "").strip().rstrip(".")


def gold_number(answer):
    values = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
    return normalize(values[-1]) if values else ""


def numeric_predictions(text):
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
def generate(model, tokenizer, prompt, max_tokens):
    current = torch.tensor([prompt], device="cuda", dtype=torch.long)
    past = None
    eos = model.generation_config.eos_token_id
    eos = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
    generated, text, reason = [], "", "max_tokens"
    for _ in range(max_tokens):
        prefix = 0 if past is None else int(past.get_seq_length())
        output = U.backbone(model)(
            input_ids=current,
            attention_mask=torch.ones((1, prefix + current.shape[1]), device="cuda", dtype=torch.long),
            position_ids=torch.arange(prefix, prefix + current.shape[1], device="cuda")[None],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        token = int(model.lm_head(output.last_hidden_state[:, -1])[0].argmax())
        past = output.past_key_values
        if token in eos:
            reason = "eos"
            break
        generated.append(token)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        hits = [(text.find(marker), marker) for marker in STOP if text.find(marker) >= 0]
        if hits:
            first, marker = min(hits, key=lambda pair: pair[0])
            text, reason = text[:first], f"until:{marker}"
            break
        current = torch.tensor([[token]], device="cuda", dtype=torch.long)
    return text, len(generated), reason


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def evaluate_mcq(model, row):
    fields = row["encoded"]["llama"]
    logits = U.next_token_logits(model, fields["body"] + fields["suffix"])
    choice_ids = torch.tensor(fields["choice_ids"], device="cuda")
    choice_logits = logits[choice_ids].float().cpu()
    prediction = int(choice_logits.argmax())
    return {
        "prediction": prediction,
        "gold": int(row["gold_index"]),
        "correct": prediction == int(row["gold_index"]),
        "choice_logits": choice_logits.tolist(),
    }


def evaluate_gsm(model, tokenizer, row, max_tokens):
    fields = row["encoded"]["llama"]
    text, count, reason = generate(model, tokenizer, fields["body"] + fields["suffix"], max_tokens)
    strict, flexible, first_hash = numeric_predictions(text)
    gold = gold_number(row["gold_answer"])
    return {
        "generated_text": text,
        "generated_tokens": count,
        "stop_reason": reason,
        "gold": gold,
        "strict_prediction": strict,
        "flexible_prediction": flexible,
        "first_hash_prediction": first_hash,
        "strict_correct": strict == gold,
        "flexible_correct": flexible == gold,
        "first_hash_correct": first_hash == gold,
        "correct": flexible == gold,
    }


def main():
    cfg = U.runtime_config(False)
    U.seed_all(cfg["seed"])
    tokenizers = U.tokenizers(cfg)
    rows = U.prepare_rows(cfg, "test", tokenizers)
    datasets = ("openbookqa", "arc_challenge", "mmlu_pro", "gsm8k")
    counts = Counter(row["dataset"] for row in rows)
    expected = {name: cfg["test_counts"][name] for name in datasets}
    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected test manifest counts: {dict(counts)} != {expected}")

    llama = U.load_model(U.model_config(cfg), "llama")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "running",
        "model": cfg["models"]["llama"],
        "checkpoint": None,
        "protocol": cfg["protocol"],
        "gsm8k_evaluation": {
            "prompt": "Question:\\n{question}\\n\\nAnswer:",
            "stop_markers": list(STOP),
            "parser": "flexible last number after truncation",
            "max_new_tokens": cfg["gsm8k_max_new_tokens"],
        },
        "datasets": {},
    }
    U.save_json(OUT / "summary.json", summary)

    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        correct = 0
        strict_correct = first_hash_correct = 0
        generated_tokens = 0
        stop_counts = Counter()
        with (OUT / f"{dataset}_per_sample.jsonl").open("w", encoding="utf-8") as stream:
            for number, row in enumerate(selected, 1):
                if row["task"] == "mcq":
                    result = evaluate_mcq(llama, row)
                else:
                    result = evaluate_gsm(
                        llama, tokenizers["llama"], row, cfg["gsm8k_max_new_tokens"]
                    )
                    strict_correct += int(result["strict_correct"])
                    first_hash_correct += int(result["first_hash_correct"])
                    generated_tokens += result["generated_tokens"]
                    stop_counts[result["stop_reason"]] += 1
                correct += int(result["correct"])
                write_record(stream, {
                    "id": row["id"],
                    "dataset": dataset,
                    "task": row["task"],
                    "question": row["question"],
                    "llama_full_native": result,
                })
                if number % 8 == 0 or number == len(selected):
                    U.log(
                        f"Llama full native {dataset} {number}/{len(selected)} "
                        f"correct={correct} accuracy={correct / number:.4f}"
                    )
        metrics = {"count": len(selected), "correct": correct, "accuracy": correct / len(selected)}
        if dataset == "gsm8k":
            metrics.update({
                "strict_correct": strict_correct,
                "strict_accuracy": strict_correct / len(selected),
                "first_hash_correct": first_hash_correct,
                "first_hash_accuracy": first_hash_correct / len(selected),
                "flexible_correct": correct,
                "flexible_accuracy": correct / len(selected),
                "mean_generated_tokens": generated_tokens / len(selected),
                "stop_reasons": dict(stop_counts),
            })
        summary["datasets"][dataset] = metrics
        U.save_json(OUT / "summary.json", summary)

    summary["status"] = "completed"
    U.save_json(OUT / "summary.json", summary)
    U.log("LLAMA FULL NATIVE FOUR-DATASET EVALUATION COMPLETED")
    del llama
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
