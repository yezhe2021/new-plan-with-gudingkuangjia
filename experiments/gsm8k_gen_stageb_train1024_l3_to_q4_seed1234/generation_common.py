"""Shared protocol for generation-only GSM8K Stage-B training/evaluation."""

import gc
import hashlib
import importlib.util
import json
import random
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
import evaluate_gsm8k_generation as gen  # noqa: E402


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_manifest(smoke=False):
    rows = read_jsonl(CFG["data_train"])
    order = list(range(len(rows)))
    random.Random(CFG["seed"]).shuffle(order)
    # Three samples satisfy the requested 3--5 sample protocol audit while
    # keeping the GPU smoke path inexpensive.
    count = 3 if smoke else CFG["train_samples"]
    selected = order[:count]
    records = [{"selected_rank": rank, "source_index": index,
                "question_sha256": hashlib.sha256(rows[index]["question"].encode()).hexdigest()}
               for rank, index in enumerate(selected)]
    payload = {"seed": CFG["seed"], "source": CFG["data_train"], "count": count,
               "source_count": len(rows), "selected_indices": selected,
               "selected_indices_sha256": digest(selected), "records": records,
               "test_not_used": True}
    root = HERE / "runs" / ("smoke" if smoke else "study")
    save_json(root / "train_manifest.json", payload)
    return [rows[index] for index in selected], root


def load_tokenizers():
    return (AutoTokenizer.from_pretrained(ref.CFG["models"]["llama"], local_files_only=True),
            AutoTokenizer.from_pretrained(ref.CFG["models"]["qwen"], local_files_only=True))


def source_entry(row, index, llama, qwen, model):
    source_text, _, _ = gen.prompt_parts(row["question"])
    aligned = gen.alignment_row(f"gsm8k_train_{index}", source_text, "llama", llama, qwen)
    target, source, counts = ref.fullsync_map(aligned, llama, qwen, "llama")
    fields = aligned["encoded"]["llama"]
    key, value, _ = ref.capture(model, fields["body"], len(fields["body"]), [0])
    return {"source_k": key[:, source].contiguous().half(),
            "source_v": value[:, source].contiguous().half(),
            "target_indices": target, "mapping_counts": counts,
            "question": row["question"], "answer": row["answer"], "id": f"gsm8k_{index}"}


def load_stage_a():
    module = ref.load_module("generation_stage_a_translator", Path(CFG["translator_base"]) / "translator.py")
    base = module.NativeKVTranslator("full28_mlp", hidden_dim=1024)
    selection = json.loads((Path(CFG["stage_a_experiment"]) / "runs/retrain/stage_a/selection.json").read_text())["best"]
    payload = torch.load(selection["path"], map_location="cpu", weights_only=True)
    base.load_state_dict(payload["state"], strict=True)
    return base.cuda().eval().requires_grad_(False), selection["path"], module


def map_stage_a(base, entry):
    out_k, out_v = [], []
    count = entry["source_k"].shape[1]
    with torch.no_grad():
        for begin in range(0, count, CFG["chunk_tokens"]):
            stop = begin + CFG["chunk_tokens"]
            sk = entry["source_k"][:, begin:stop][None].cuda()
            sv = entry["source_v"][:, begin:stop][None].cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                key, value = base(sk, sv)
            out_k.append(key[0].cpu().half()); out_v.append(value[0].cpu().half())
    return {"base_k": torch.cat(out_k, 1), "base_v": torch.cat(out_v, 1),
            "mapping_counts": entry["mapping_counts"], "question": entry["question"],
            "answer": entry["answer"], "id": entry["id"]}


def prepare_base_cache(rows, label):
    llama, qwen = load_tokenizers()
    sender = ref.load_model("llama")
    source = []
    try:
        for index, row in enumerate(rows, 1):
            source.append(source_entry(row, index - 1, llama, qwen, sender))
            if index % 32 == 0 or index == len(rows):
                print(f"{label} Llama capture {index}/{len(rows)}", flush=True)
    finally:
        del sender; gc.collect(); torch.cuda.empty_cache()
    base, path, module = load_stage_a()
    mapped = []
    try:
        for index, entry in enumerate(source, 1):
            mapped.append(map_stage_a(base, entry))
            if index % 32 == 0 or index == len(source):
                print(f"{label} frozen Stage-A map {index}/{len(source)}", flush=True)
    finally:
        del base, source; gc.collect(); torch.cuda.empty_cache()
    return mapped, llama, qwen, path, module


def load_adapter(module, checkpoint=None):
    adapter = module.ResidualKVAdapter(rank=CFG["adapter_rank"])
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        adapter.load_state_dict(payload["state"], strict=True)
    return adapter.cuda()


def old_stage_b_checkpoint():
    selection = json.loads((Path(CFG["stage_a_experiment"]) / "runs/retrain/stage_b/selection.json").read_text())
    return selection["best"]["path"]


@torch.no_grad()
def native_token0(qwen_model, qwen_tok):
    source, _, _ = gen.prompt_parts("placeholder")
    token = gen.encode_prefix(qwen_tok, source)[0]
    key, value, _ = ref.capture(qwen_model, [token], 1, [0])
    return key.cuda().half(), value.cuda().half(), token


def adapted_cache(adapter, entry, token0_k, token0_v):
    base_k, base_v = entry["base_k"][None].cuda(), entry["base_v"][None].cuda()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        key, value, _, _ = adapter(base_k, base_v)
    return torch.cat((token0_k, key[0]), 1), torch.cat((token0_v, value[0]), 1)


def teacher_forcing(qwen_model, qwen_tok, key, value, answer_text):
    answer_ids = list(qwen_tok.encode(answer_text, add_special_tokens=False))
    prefix_ids = list(qwen_tok.encode("Answer:", add_special_tokens=False))
    if not answer_ids or not prefix_ids:
        raise RuntimeError("Empty Answer:/gold tokenization")
    input_ids = prefix_ids + answer_ids[:-1]
    cache = ref.make_cache(qwen_model, key, value, torch.arange(key.shape[1], device="cuda"))
    tensor = torch.tensor([input_ids], device="cuda", dtype=torch.long)
    mask = torch.ones((1, key.shape[1] + len(input_ids)), device="cuda", dtype=torch.long)
    positions = torch.arange(key.shape[1], key.shape[1] + len(input_ids), device="cuda")[None]
    output = ref.backbone(qwen_model)(input_ids=tensor, attention_mask=mask, position_ids=positions,
              past_key_values=cache, use_cache=False, return_dict=True)
    logits = qwen_model.lm_head(output.last_hidden_state)[0]
    begin = len(prefix_ids) - 1
    answer_logits = logits[begin:begin + len(answer_ids)]
    targets = torch.tensor(answer_ids, device="cuda", dtype=torch.long)
    if answer_logits.shape[0] != targets.shape[0]:
        raise RuntimeError((answer_logits.shape, targets.shape))
    return answer_logits, targets, F.cross_entropy(answer_logits.float(), targets)


@torch.no_grad()
def native_full_teacher_forcing(qwen_model, qwen_tok, question, answer_text):
    source, continuation, _ = gen.prompt_parts(question)
    source_ids = gen.encode_prefix(qwen_tok, source)
    prefix_ids = list(qwen_tok.encode(continuation, add_special_tokens=False))
    answer_ids = list(qwen_tok.encode(answer_text, add_special_tokens=False))
    ids = source_ids + prefix_ids + answer_ids[:-1]
    tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = ref.backbone(qwen_model)(input_ids=tensor, attention_mask=torch.ones_like(tensor),
              position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False, return_dict=True)
    logits = qwen_model.lm_head(output.last_hidden_state)[0]
    begin = len(source_ids) + len(prefix_ids) - 1
    selected = logits[begin:begin + len(answer_ids)]
    targets = torch.tensor(answer_ids, device="cuda")
    return selected, targets, F.cross_entropy(selected.float(), targets)


@torch.no_grad()
def native_prefix_cache(qwen_model, qwen_tok, question):
    source, _, _ = gen.prompt_parts(question)
    ids = gen.encode_prefix(qwen_tok, source)
    key, value, _ = ref.capture(qwen_model, ids, len(ids), [0])
    return key.cuda(), value.cuda()


def save_checkpoint(path, adapter, epoch, step, metrics):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "generation_stage_b", "protocol": "gsm8k_generation_only_stage_b_l3_to_q4_seed1234",
                "epoch": epoch, "step": step, "metrics": metrics,
                "state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()}}, path)
