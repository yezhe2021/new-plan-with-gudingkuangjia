import json
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
RESULTS = ROOT / "results"


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def gold_number(answer):
    found = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
    return found[-1].replace(",", "") if found else ""


def flexible_number(text):
    values = re.findall(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    return values[-1].replace(",", "").rstrip(".") if values else "[invalid]"


def encode_prompt(tokenizer, question):
    text = f"Question:\n{str(question).strip()}\n\nAnswer:"
    body = list(tokenizer.encode(text, add_special_tokens=False))
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return bos + body, text


def text_backbone(model):
    inner = model.model
    return getattr(inner, "language_model", inner)


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt_ids):
    backbone = text_backbone(model)
    current = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    past = None
    eos = model.generation_config.eos_token_id
    eos = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
    generated = []
    reason = "max_tokens"
    for _ in range(CFG["max_new_tokens"]):
        prefix = 0 if past is None else int(past.get_seq_length())
        output = backbone(
            input_ids=current,
            attention_mask=torch.ones((1, prefix + current.shape[1]), device="cuda", dtype=torch.long),
            position_ids=torch.arange(prefix, prefix + current.shape[1], device="cuda")[None],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        token = int(model.lm_head(output.last_hidden_state[:, -1])[0].argmax().item())
        past = output.past_key_values
        if token in eos:
            reason = "eos"
            break
        generated.append(token)
        current = torch.tensor([[token]], device="cuda", dtype=torch.long)
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated), reason


def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    rows = [json.loads(line) for line in Path(CFG["test_data"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: CFG["test_samples"]]
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=CFG["attention_implementation"],
    )
    del model.model.vision_tower
    del model.model.multi_modal_projector
    model = model.to("cuda").eval().requires_grad_(False)

    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / "per_sample.jsonl"
    correct = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            prompt_ids, prompt = encode_prompt(tokenizer, row["question"])
            text, generated_tokens, stop_reason = greedy_generate(model, tokenizer, prompt_ids)
            gold = gold_number(str(row["answer"]))
            prediction = flexible_number(text)
            is_correct = prediction == gold
            correct += int(is_correct)
            record = {
                "id": f"gsm8k_test_{index}",
                "prompt": prompt,
                "generated_text": text,
                "generated_tokens": generated_tokens,
                "stop_reason": stop_reason,
                "prediction": prediction,
                "gold": gold,
                "correct": is_correct,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            atomic_json(RESULTS / "summary.json", {
                "status": "running" if index + 1 < len(rows) else "completed",
                "model": CFG["model"],
                "dtype": CFG["dtype"],
                "prompt_protocol": CFG["prompt_protocol"],
                "completed": index + 1,
                "total": len(rows),
                "correct": correct,
                "accuracy": correct / (index + 1),
            })
            if (index + 1) % 8 == 0 or index + 1 == len(rows):
                log(f"Gemma native GSM8K {index + 1}/{len(rows)} correct={correct} accuracy={correct / (index + 1):.4f}")
    log("ALL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
