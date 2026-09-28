"""Shared Full-Sync Generation Stage-B v2 protocol."""

import gc
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
REF = Path(CFG["reference_eval"])
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))

import evaluate as ref  # noqa: E402
import evaluate_gsm8k_generation as old_gen  # noqa: E402


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prompt_parts(question):
    source = f"Question: {question}\n"
    suffix = "Answer:"
    return source, suffix, source + suffix


def encode_prefix(tok, text):
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return bos + list(tok.encode(text, add_special_tokens=False))


def prepare_manifest():
    rows = read_jsonl(CFG["data_train"])
    order = list(range(len(rows)))
    random.Random(CFG["seed"]).shuffle(order)
    old_train_count = 1024
    val_indices = order[old_train_count:old_train_count + CFG["val_samples"]]
    train_indices = order[:old_train_count] + order[
        old_train_count + CFG["val_samples"]:
        old_train_count + CFG["val_samples"] + CFG["train_samples"] - old_train_count
    ]
    if len(train_indices) != CFG["train_samples"] or set(train_indices) & set(val_indices):
        raise RuntimeError("Insufficient train rows or overlap with frozen validation split")
    old = json.loads(Path(CFG["old_train_manifest"]).read_text(encoding="utf-8"))
    previous = json.loads(Path(CFG["previous_four_epoch_manifest"]).read_text(encoding="utf-8"))
    if old["selected_indices"] != train_indices[:old_train_count]:
        raise RuntimeError("Original train1024 indices changed")
    if previous["val_indices"] != val_indices:
        raise RuntimeError("Frozen validation indices changed")
    payload = {
        "seed": CFG["seed"], "source": CFG["data_train"], "source_count": len(rows),
        "train_count": len(train_indices), "val_count": len(val_indices),
        "train_indices": train_indices, "val_indices": val_indices,
        "train_indices_sha256": digest(train_indices), "val_indices_sha256": digest(val_indices),
        "old_manifest": CFG["old_train_manifest"], "old_train_indices_exact_match": True,
        "previous_four_epoch_manifest": CFG["previous_four_epoch_manifest"],
        "previous_validation_indices_exact_match": True,
        "test_not_used": True,
    }
    save_json(HERE / "split_manifest.json", payload)
    return ([rows[i] for i in train_indices], [rows[i] for i in val_indices], payload)


def load_tokenizers():
    return (AutoTokenizer.from_pretrained(ref.CFG["models"]["llama"], local_files_only=True),
            AutoTokenizer.from_pretrained(ref.CFG["models"]["qwen"], local_files_only=True))


def load_stage_a():
    module = ref.load_module("stageb_v2_translator", Path(CFG["translator_base"]) / "translator.py")
    base = module.NativeKVTranslator("full28_mlp", hidden_dim=1024)
    selection = json.loads((Path(CFG["stage_a_experiment"]) / "runs/retrain/stage_a/selection.json").read_text())["best"]
    payload = torch.load(selection["path"], map_location="cpu", weights_only=True)
    base.load_state_dict(payload["state"], strict=True)
    return base.cuda().eval().requires_grad_(False), selection["path"], module


def alignment_row(sample_id, source_text, source_tok, qwen_tok):
    encoded = {}
    for name, tok in (("llama", source_tok), ("qwen", qwen_tok)):
        body = encode_prefix(tok, source_text)
        suffix = list(tok.encode("Answer:", add_special_tokens=False))
        if body + suffix != encode_prefix(tok, source_text + "Answer:"):
            raise RuntimeError(f"Noncompositional boundary: {sample_id}/{name}")
        encoded[name] = {"body": body, "option_token_indices": list(range(1, len(body)))}
    return {"id": sample_id, "body": source_text, "encoded": encoded}


@torch.no_grad()
def greedy_generate(model, tok, prompt_ids, max_new_tokens, key=None, value=None):
    if key is None:
        past = None
    else:
        past = ref.make_cache(model, key, value, torch.arange(key.shape[1], device="cuda"))
    current = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    eos = model.generation_config.eos_token_id
    eos = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos}
    generated, reason, text = [], "max_tokens", ""
    until = ("Question:", "</s>", "<|im_end|>")
    for _ in range(max_new_tokens):
        prefix = int(past.get_seq_length()) if past is not None else 0
        mask = torch.ones((1, prefix + current.shape[1]), device="cuda", dtype=torch.long)
        positions = torch.arange(prefix, prefix + current.shape[1], device="cuda")[None]
        output = ref.backbone(model)(input_ids=current, attention_mask=mask, position_ids=positions,
                                     past_key_values=past, use_cache=True, return_dict=True)
        token = int(model.lm_head(output.last_hidden_state[:, -1])[0].argmax().item())
        past = output.past_key_values
        if token in eos:
            reason = "eos"; break
        generated.append(token)
        text = tok.decode(generated, skip_special_tokens=True)
        hits = [(text.find(marker), marker) for marker in until if text.find(marker) >= 0]
        if hits:
            first, marker = min(hits, key=lambda item: item[0])
            text, reason = text[:first], f"until:{marker}"; break
        current = torch.tensor([[token]], device="cuda", dtype=torch.long)
    return text, len(generated), reason


def source_entry(row, source_index, llama_tok, qwen_tok, sender, sender_generation=False):
    source_text, suffix, full_prompt = prompt_parts(row["question"])
    aligned = alignment_row(f"gsm8k_train_{source_index}", source_text, llama_tok, qwen_tok)
    target, source, counts = ref.fullsync_map(aligned, llama_tok, qwen_tok, "llama")
    fields = aligned["encoded"]["llama"]
    key, value, _ = ref.capture(sender, fields["body"], len(fields["body"]), [0])
    sender_result = None
    if sender_generation:
        sender_result = greedy_generate(sender, llama_tok, encode_prefix(llama_tok, full_prompt),
                                        CFG["max_new_tokens"])
    return {
        "source_k": key[:, source].contiguous().half(),
        "source_v": value[:, source].contiguous().half(),
        "target_indices": target, "mapping_counts": counts,
        "qwen_prefix_tokens": len(aligned["encoded"]["qwen"]["body"]),
        "first_qwen_token": aligned["encoded"]["qwen"]["body"][0],
        "source_text": source_text, "question": row["question"], "answer": row["answer"],
        "id": f"gsm8k_{source_index}", "sender_result": sender_result,
    }


@torch.no_grad()
def map_stage_a(base, entry):
    out_k, out_v = [], []
    for begin in range(0, entry["source_k"].shape[1], CFG["chunk_tokens"]):
        stop = begin + CFG["chunk_tokens"]
        sk = entry["source_k"][:, begin:stop][None].cuda()
        sv = entry["source_v"][:, begin:stop][None].cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            key, value = base(sk, sv)
        out_k.append(key[0].cpu().half()); out_v.append(value[0].cpu().half())
    mapped = {name: value for name, value in entry.items() if name not in {"source_k", "source_v"}}
    mapped.update(base_k=torch.cat(out_k, 1), base_v=torch.cat(out_v, 1))
    if mapped["base_k"].shape[1] + 1 != mapped["qwen_prefix_tokens"]:
        raise RuntimeError("Stage-A cache length does not match native Qwen prefix")
    return mapped


def prepare_base_cache(rows, indices, label, sender_generation=False):
    llama_tok, qwen_tok = load_tokenizers()
    sender = ref.load_model("llama")
    source = []
    try:
        for position, (row, source_index) in enumerate(zip(rows, indices), 1):
            source.append(source_entry(row, source_index, llama_tok, qwen_tok, sender, sender_generation))
            if position % 32 == 0 or position == len(rows):
                print(f"{label} Llama capture {position}/{len(rows)}", flush=True)
    finally:
        del sender; gc.collect(); torch.cuda.empty_cache()
    base, path, module = load_stage_a()
    mapped = []
    try:
        for position, entry in enumerate(source, 1):
            mapped.append(map_stage_a(base, entry))
            if position % 32 == 0 or position == len(source):
                print(f"{label} Stage-A map {position}/{len(source)}", flush=True)
    finally:
        del base, source; gc.collect(); torch.cuda.empty_cache()
    return mapped, llama_tok, qwen_tok, path, module


def load_adapter(module, checkpoint=None):
    adapter = module.ResidualKVAdapter(rank=CFG["adapter_rank"])
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        adapter.load_state_dict(payload["state"], strict=True)
    return adapter.cuda()


@torch.no_grad()
def native_token0(qwen, qwen_tok, entries):
    ids = {entry["first_qwen_token"] for entry in entries}
    if len(ids) != 1:
        raise RuntimeError(f"Qwen token0 is not shared: {ids}")
    token = next(iter(ids))
    key, value, _ = ref.capture(qwen, [token], 1, [0])
    return key.cuda().half(), value.cuda().half(), token


def adapted_cache(adapter, entry, token0_k, token0_v):
    base_k, base_v = entry["base_k"][None].cuda(), entry["base_v"][None].cuda()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        key, value, _, _ = adapter(base_k, base_v)
    key, value = torch.cat((token0_k, key[0]), 1), torch.cat((token0_v, value[0]), 1)
    if key.shape[1] != entry["qwen_prefix_tokens"]:
        raise RuntimeError("Adapted cache length mismatch")
    return key, value


def cached_trajectory(qwen, tok, key, value, answer_text):
    answer = list(tok.encode(answer_text, add_special_tokens=False))
    suffix = list(tok.encode("Answer:", add_special_tokens=False))
    if not answer or not suffix:
        raise RuntimeError("Empty suffix/answer tokenization")
    ids = suffix + answer
    cache = ref.make_cache(qwen, key, value, torch.arange(key.shape[1], device="cuda"))
    tensor = torch.tensor([ids], device="cuda")
    mask = torch.ones((1, key.shape[1] + len(ids)), device="cuda", dtype=torch.long)
    positions = torch.arange(key.shape[1], key.shape[1] + len(ids), device="cuda")[None]
    output = ref.backbone(qwen)(input_ids=tensor, attention_mask=mask, position_ids=positions,
              past_key_values=cache, use_cache=False, return_dict=True)
    logits = qwen.lm_head(output.last_hidden_state)[0]
    begin = len(suffix) - 1
    return logits[begin:begin + len(answer)], logits[begin + len(answer)], torch.tensor(answer, device="cuda")


@torch.no_grad()
def native_trajectory(qwen, tok, source_text, answer_text):
    source = encode_prefix(tok, source_text)
    suffix = list(tok.encode("Answer:", add_special_tokens=False))
    answer = list(tok.encode(answer_text, add_special_tokens=False))
    ids = source + suffix + answer
    tensor = torch.tensor([ids], device="cuda")
    output = ref.backbone(qwen)(input_ids=tensor, attention_mask=torch.ones_like(tensor),
              position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False, return_dict=True)
    logits = qwen.lm_head(output.last_hidden_state)[0]
    begin = len(source) + len(suffix) - 1
    return logits[begin:begin + len(answer)], logits[begin + len(answer)], torch.tensor(answer, device="cuda")


def loss_components(student_answer, student_eos, teacher_answer, targets, eos_id):
    answer_ce = F.cross_entropy(student_answer.float(), targets)
    eos_ce = F.cross_entropy(student_eos[None].float(), torch.tensor([eos_id], device="cuda"))
    teacher_prob = F.softmax(teacher_answer.float(), dim=-1)
    trajectory_kl = F.kl_div(F.log_softmax(student_answer.float(), dim=-1), teacher_prob,
                             reduction="batchmean")
    return trajectory_kl, answer_ce, eos_ce


def objective_loss(objective, trajectory_kl, answer_ce, eos_ce):
    if objective == "ce":
        return answer_ce + CFG["eos_weight"] * eos_ce
    if objective == "kl":
        return trajectory_kl + CFG["eos_weight"] * eos_ce
    if objective == "hybrid":
        return trajectory_kl + CFG["hybrid_ce_weight"] * answer_ce + CFG["eos_weight"] * eos_ce
    raise ValueError(objective)


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


def generation_record(text, count, reason, answer):
    gold = gold_number(answer)
    strict, flexible, first_hash = predictions(text)
    return {"generated_text": text, "generated_tokens": count, "stop_reason": reason,
            "gold": gold, "strict_prediction": strict, "flexible_prediction": flexible,
            "first_hash_prediction": first_hash, "strict_correct": strict == gold,
            "flexible_correct": flexible == gold, "first_hash_correct": first_hash == gold}


def summarize_generations(records):
    count = len(records)
    return {"count": count,
            "strict_accuracy": sum(r["strict_correct"] for r in records) / count,
            "flexible_accuracy": sum(r["flexible_correct"] for r in records) / count,
            "first_hash_accuracy": sum(r["first_hash_correct"] for r in records) / count,
            "mean_generated_tokens": sum(r["generated_tokens"] for r in records) / count,
            "eos_rate": sum(r["stop_reason"] == "eos" for r in records) / count,
            "stop_rate": sum(r["stop_reason"] != "max_tokens" for r in records) / count,
            "max_token_rate": sum(r["stop_reason"] == "max_tokens" for r in records) / count}


def save_checkpoint(path, adapter, objective, epoch, step, metrics):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "generation_stage_b_v2", "objective": objective, "epoch": epoch,
                "step": step, "metrics": metrics,
                "state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()}}, path)
