"""Frozen GSM8K-trained Full-Sync Writer transfer to four choice benchmarks.

The trained Llama->Qwen and Gemma->Qwen Stage-A/Stage-B modules are never
updated. Native source KV is retained only in CPU RAM while one dataset is
evaluated, then released. No KV tensor is written to disk.
"""

import argparse
import gc
import importlib.util
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
LABELS = "ABCDEFGHIJ"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Span:
    index: int
    start: int
    end: int


def token_spans(tok, text, expected_ids):
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + list(encoded["input_ids"])
    if ids != list(expected_ids):
        raise RuntimeError("Offset tokenization mismatch")
    offsets = [(0, 0)] * len(bos) + [tuple(map(int, x)) for x in encoded["offset_mapping"]]
    return [Span(i, a, b) for i, (a, b) in enumerate(offsets)]


def source_rank(source_count, target_count, target_rank):
    if source_count > target_count:
        return math.ceil((target_rank + 1) * source_count / target_count) - 1
    return target_rank * source_count // target_count


def fullsync_map(row, source_tok, qwen_tok, family):
    text = row["body"]
    source = token_spans(source_tok, text, row["encoded"][family]["body"])
    target = token_spans(qwen_tok, text, row["encoded"]["qwen"]["body"])
    char_bytes = [0]
    for character in text:
        char_bytes.append(char_bytes[-1] + len(character.encode("utf-8")))
    def byte_tokens(spans):
        return [(x.index, char_bytes[x.start], char_bytes[x.end]) for x in spans
                if x.index != 0 and x.start != x.end]
    source_b, target_b = byte_tokens(source), byte_tokens(target)
    shared = sorted(({end for _, _, end in source_b} & {end for _, _, end in target_b}) | {0})
    option_indices = list(row["encoded"]["qwen"]["option_token_indices"])
    option_set, mapping, counts = set(option_indices), {}, Counter()
    for left, right in zip(shared, shared[1:]):
        si = [index for index, _, end in source_b if left < end <= right]
        ti = [index for index, _, end in target_b if left < end <= right]
        if not si or not ti:
            continue
        touched = [index for index in ti if index in option_set]
        if not touched:
            continue
        kind = "one_to_one" if len(si) == len(ti) == 1 else "copy" if len(si) == 1 else "drop" if len(ti) == 1 else "many_to_many"
        counts[kind + "_units"] += 1
        counts[kind + "_target_tokens"] += len(touched)
        for rank, target_index in enumerate(ti):
            if target_index in option_set:
                mapping[target_index] = si[source_rank(len(si), len(ti), rank)]
    if set(mapping) != option_set:
        raise RuntimeError(f"Unmapped Qwen option tokens for {row['id']}: {len(option_set - set(mapping))}")
    return option_indices, [mapping[index] for index in option_indices], dict(counts)


def canonical_number(value):
    value = Decimal(value)
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def gsm_options(answer_text, seed):
    match = re.search(r"####\s*([-+]?[$]?[\d,]+(?:\.\d+)?)", answer_text)
    if not match:
        return None
    raw = match.group(1).replace("$", "").replace(",", "")
    try:
        gold = Decimal(raw)
    except InvalidOperation:
        return None
    scale = max(Decimal(1), abs(gold) * Decimal("0.1"))
    pool = [gold + 1, gold - 1, gold + scale, gold - scale, gold * 2, gold / 2 if gold else Decimal(2)]
    choices, seen = [gold], {canonical_number(gold)}
    for value in pool:
        text = canonical_number(value)
        if text not in seen:
            choices.append(value); seen.add(text)
        if len(choices) == 4:
            break
    rng = random.Random(seed)
    rng.shuffle(choices)
    options = [canonical_number(x) for x in choices]
    return options, options.index(canonical_number(gold))


def round_robin(items, key, limit, seed):
    groups = defaultdict(list)
    for item in items:
        groups[str(item.get(key, "unknown"))].append(item)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    output = []
    while len(output) < limit and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(output) < limit:
                output.append(groups[name].pop())
    return output


def raw_rows(dataset):
    count, seed = CFG["samples_per_dataset"], CFG["seed"]
    if dataset == "hellaswag":
        items = [json.loads(x) for x in Path(CFG["datasets"][dataset]).read_text(encoding="utf-8").splitlines() if x.strip()]
        items = [x for x in items if str(x.get("label", "")).isdigit()]
        random.Random(seed + 11).shuffle(items)
        return [{"id": f"hellaswag_{x['ind']}", "dataset": dataset,
                 "question": "Complete the following passage:\n" + x["ctx"],
                 "options": list(x["endings"]), "gold_index": int(x["label"]),
                 "category": x.get("activity_label", "unknown")} for x in items[:count]]
    if dataset == "gsm8k":
        items = [json.loads(x) for x in Path(CFG["datasets"][dataset]).read_text(encoding="utf-8").splitlines() if x.strip()]
        random.Random(seed + 12).shuffle(items)
        output = []
        for index, item in enumerate(items):
            built = gsm_options(item["answer"], seed + index)
            if built is None:
                continue
            options, gold = built
            output.append({"id": f"gsm8k_{index}", "dataset": dataset,
                           "question": item["question"], "options": options,
                           "gold_index": gold, "category": "deterministic_numeric_mc"})
            if len(output) == count:
                break
        return output
    if dataset == "longbench_v2":
        items = json.loads(Path(CFG["datasets"][dataset]).read_text(encoding="utf-8"))
        items = [x for x in items if str(x.get("answer", "")) in "ABCD"]
        items = round_robin(items, "domain", count, seed + 13)
        return [{"id": f"longbench_v2_{x['_id']}", "dataset": dataset,
                 "question": "Context:\n" + x["context"] + "\n\nQuestion:\n" + x["question"],
                 "options": [x[f"choice_{label}"] for label in "ABCD"],
                 "gold_index": "ABCD".index(x["answer"]),
                 "category": x.get("domain", "unknown"), "longbench_length": x.get("length")}
                for x in items]
    raise ValueError(dataset)


def serialize(row):
    question = str(row["question"]).strip()
    qhead = "Question:\n"
    ohead = "\n\nOptions:\n"
    options_text = "\n".join(f"{LABELS[i]}. {str(option).strip()}" for i, option in enumerate(row["options"]))
    options_start = len(qhead) + len(question) + len(ohead)
    options_end = options_start + len(options_text)
    body = qhead + question + ohead + options_text + "\n\n"
    answer = "Answer:"
    return {"body": body, "question_prefix": body[:options_start],
            "receiver_answer": body[options_end:] + answer, "full_prompt": body + answer,
            "options_char_span": [options_start, options_end], "labels": list(LABELS[:len(row["options"])])}


def encode(tok, text):
    plain = lambda value: list(tok.encode(value, add_special_tokens=False))
    body_enc = tok(text["body"], add_special_tokens=False, return_offsets_mapping=True)
    body_plain = list(body_enc["input_ids"])
    full_plain = plain(text["full_prompt"])
    answer_plain = plain("Answer:")
    question_plain = plain(text["question_prefix"])
    receiver_answer = plain(text["receiver_answer"])
    if body_plain + answer_plain != full_plain or full_plain[:len(question_plain)] != question_plain:
        raise RuntimeError("Noncompositional tokenizer boundary")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    choice_ids = []
    for label in text["labels"]:
        continuation = plain(text["full_prompt"] + " " + label)
        if continuation[:len(full_plain)] != full_plain or len(continuation) != len(full_plain) + 1:
            raise RuntimeError(f"Unstable choice token: {label}")
        choice_ids.append(continuation[-1])
    start, end = text["options_char_span"]
    option_indices = [len(bos) + i for i, (a, b) in enumerate(body_enc["offset_mapping"])
                      if min(int(b), end) > max(int(a), start)]
    return {"body": bos + body_plain, "full": bos + full_plain,
            "question_prefix_length": len(bos) + len(question_plain),
            "receiver_answer": receiver_answer, "option_token_indices": option_indices,
            "choice_ids": choice_ids}


def prepare_rows(dataset, tokenizers):
    output, audit = [], []
    for row in raw_rows(dataset):
        original_question = row["question"]
        truncated = False
        if dataset == "longbench_v2":
            # Keep the question and balanced context ends under the historical 2048-token ceiling.
            marker = "\n\nQuestion:\n"
            context, question = original_question.split(marker, 1)
            # Some LongBench-v2 contexts exceed 150k tokens. Starting the
            # exact tokenizer search from the full string needlessly encodes
            # that text many times. At a 2,048-token ceiling, 12k characters
            # is a conservative upper bracket; the loop still enforces the
            # exact per-tokenizer limit below.
            lo, hi, best = 0, min(len(context), 12000), ""
            while lo <= hi:
                mid = (lo + hi) // 2
                left, right = (mid + 1) // 2, mid // 2
                candidate = context[:left] + ("\n...[TRUNCATED]...\n" if mid < len(context) else "") + (context[-right:] if right else "")
                row["question"] = candidate + marker + question
                text = serialize(row)
                lengths = {name: len(encode(tok, text)["body"]) for name, tok in tokenizers.items()}
                if max(lengths.values()) <= CFG["max_body_tokens"]:
                    best = row["question"]; lo = mid + 1
                else:
                    hi = mid - 1
            if not best:
                raise RuntimeError(f"Cannot fit LongBench question: {row['id']}")
            row["question"] = best
            truncated = len(best) < len(original_question)
        text = serialize(row)
        encoded = {name: encode(tok, text) for name, tok in tokenizers.items()}
        lengths = {name: len(fields["body"]) for name, fields in encoded.items()}
        if max(lengths.values()) > CFG["max_body_tokens"]:
            raise RuntimeError(f"Body budget exceeded: {row['id']} {lengths}")
        prepared = {**row, **text, "encoded": encoded}
        output.append(prepared)
        audit.append({"id": row["id"], "category": row.get("category"),
                      "original_question_chars": len(original_question),
                      "used_question_chars": len(row["question"]), "truncated": truncated,
                      "body_tokens": lengths})
    return output, audit


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def backbone(model):
    return getattr(model.model, "language_model", model.model)


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(model, x, positions, layer_index):
    base = backbone(model)
    positions = positions.to(x.device).unsqueeze(0)
    config = text_config(model)
    types = getattr(config, "layer_types", None)
    if types is not None and isinstance(getattr(base.rotary_emb, "rope_type", None), dict):
        cos, sin = base.rotary_emb(x.unsqueeze(0), positions, types[layer_index])
    else:
        cos, sin = base.rotary_emb(x.unsqueeze(0), positions)
    return x * cos[0, :, None] + rotate_half(x) * sin[0, :, None]


@torch.no_grad()
def capture(model, ids, body_length, choice_ids):
    captured_k, captured_v, handles = {}, {}, []
    base = backbone(model)
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"):
                try: key = module.k_norm(key.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError): key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            captured_k[index], captured_v[index] = key[0, :body_length].cpu(), value[0, :body_length].cpu()
        return apply
    for index, layer in enumerate(base.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
        output = base(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                      position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False)
        logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
    finally:
        for handle in handles: handle.remove()
    return torch.stack([captured_k[i] for i in range(len(base.layers))]), torch.stack([captured_v[i] for i in range(len(base.layers))]), logits[torch.tensor(choice_ids, device="cuda")].cpu()


def make_cache(model, key, value, positions):
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions, index) for index, layer in enumerate(key)])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0), value[layer].permute(1, 0, 2).unsqueeze(0))
            for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=text_config(model))


@torch.no_grad()
def readout(model, ids, key, value):
    positions = torch.arange(key.shape[1], device="cuda")
    cache = make_cache(model, key, value, positions)
    suffix = torch.tensor([ids], device="cuda", dtype=torch.long)
    mask = torch.ones((1, key.shape[1] + len(ids)), device="cuda", dtype=torch.long)
    output = backbone(model)(input_ids=suffix, attention_mask=mask,
                             position_ids=torch.arange(key.shape[1], key.shape[1] + len(ids), device="cuda")[None],
                             past_key_values=cache, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1])[0].float()


def load_model(family):
    dtype = torch.float32 if family == "gemma" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(CFG["models"][family], local_files_only=True,
              dtype=dtype, attn_implementation=CFG["attention_implementation"])
    if family == "gemma":
        del model.model.vision_tower
        del model.model.multi_modal_projector
    return model.to("cuda").eval().requires_grad_(False)


def load_writer(family, stage_b_kind):
    base_path = Path(CFG["bases"][family])
    module = load_module(f"translator_{family}", base_path / "translator.py")
    if family == "llama":
        base = module.NativeKVTranslator("full28_mlp", hidden_dim=1024)
    else:
        base = module.NativeKVTranslator("full34_headmix256", hidden_dim=1024,
                depth_output_dim=256, head_mapping="full_head")
    adapter = module.ResidualKVAdapter(rank=64)
    paths = CFG["checkpoints"][family]
    path_a, path_b = paths["stage_a"], paths[stage_b_kind + "_stage_b"]
    base.load_state_dict(torch.load(path_a, map_location="cpu", weights_only=True)["state"], strict=True)
    adapter.load_state_dict(torch.load(path_b, map_location="cpu", weights_only=True)["state"], strict=True)
    return base.cuda().eval().requires_grad_(False), adapter.cuda().eval().requires_grad_(False), path_a, path_b


def translate(base, adapter, source_k, source_v):
    out_k, out_v = [], []
    for begin in range(0, source_k.shape[1], CFG["chunk_tokens"]):
        stop = begin + CFG["chunk_tokens"]
        sk, sv = source_k[:, begin:stop][None].cuda(), source_v[:, begin:stop][None].cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = base(sk, sv)
        out_k.append(pk[0]); out_v.append(pv[0])
    stage_a_k, stage_a_v = torch.cat(out_k, 1), torch.cat(out_v, 1)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        stage_b_k, stage_b_v, _, _ = adapter(stage_a_k[None], stage_a_v[None])
    return stage_a_k, stage_a_v, stage_b_k[0], stage_b_v[0]


def component(prediction, target):
    prediction, target = prediction.float(), target.float()
    return ((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).item(), F.cosine_similarity(prediction, target, dim=-1).mean().item()


def summarize(records):
    conditions = ("sender_full_native", "qwen_full_native", "native_oracle", "stage_a", "stage_b")
    result = {"count": len(records)}
    for name in conditions:
        correct = sum(r["conditions"][name]["correct"] for r in records)
        result[name] = {"correct": correct, "accuracy": correct / len(records)}
    for name in ("stage_a", "stage_b"):
        both = sum(r["conditions"][name]["correct"] and r["conditions"]["native_oracle"]["correct"] for r in records)
        result[name]["both_with_oracle_correct"] = both
        result[name]["writer_only_correct"] = result[name]["correct"] - both
        result[name]["oracle_only_correct"] = result["native_oracle"]["correct"] - both
        result[name]["native_agreement"] = sum(r["conditions"][name]["prediction"] == r["conditions"]["native_oracle"]["prediction"] for r in records) / len(records)
        result[name]["choice_kl"] = sum(r["representation"][name]["choice_kl"] for r in records) / len(records)
    result["truncated_count"] = sum(r["truncated"] for r in records)
    result["mean_qwen_transferred_tokens"] = sum(r["qwen_transferred_tokens"] for r in records) / len(records)
    return result


@torch.no_grad()
def crossbench_rows(dataset, tokenizers):
    if dataset == "hellaswag":
        rows, audit = prepare_rows(dataset, tokenizers)
    else:
        manifest = json.loads(Path(CFG["old_test_manifest"]).read_text(encoding="utf-8"))
        rows = [row for row in manifest["rows"] if row["dataset"] == dataset]
        if len(rows) != CFG["samples_per_dataset"]:
            raise RuntimeError(f"Historical test manifest count mismatch: {dataset}={len(rows)}")
        audit = []
        for row in rows:
            row["encoded"] = {name: encode(tok, row) for name, tok in tokenizers.items()}
            lengths = {name: len(value["body"]) for name, value in row["encoded"].items()}
            if max(lengths.values()) > CFG["max_body_tokens"]:
                raise RuntimeError(f"Historical sample exceeds body budget: {row['id']} {lengths}")
            audit.append({"id": row["id"], "body_tokens": lengths, "truncated": False})
    for row in rows:
        # Full-Sync GSM8K protocol: transfer every body token except receiver-native token0.
        qwen = row["encoded"]["qwen"]
        qwen["option_token_indices"] = list(range(1, len(qwen["body"])))
        qwen["answer_only"] = list(tokenizers["qwen"].encode("Answer:", add_special_tokens=False))
    return rows, audit


def evaluate_family(family, stage_b_kind, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out = HERE / "runs" / stage_b_kind / family
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True)
                  for name, path in CFG["models"].items()}
    datasets = ["openbookqa", "arc_challenge", "mmlu_pro", "hellaswag"]
    all_summary = {"status": "running", "family": family, "datasets": {},
                   "note": "GSM8K-trained frozen Writer; full Question+Options body replaces all but Qwen-native token0."}
    for dataset in datasets:
        rows, audit = crossbench_rows(dataset, {family: tokenizers[family], "qwen": tokenizers["qwen"]})
        if smoke: rows, audit = rows[:2], audit[:2]
        save_json(out / f"{dataset}_manifest_audit.json", {"count": len(rows), "records": audit})
        source_cache = {}
        sender = load_model(family)
        try:
            for number, row in enumerate(rows, 1):
                target_indices, source_indices, counts = fullsync_map(row, tokenizers[family], tokenizers["qwen"], family)
                fields = row["encoded"][family]
                key, value, logits = capture(sender, fields["full"], len(fields["body"]), fields["choice_ids"])
                source_cache[row["id"]] = {"k": key[:, source_indices].half(), "v": value[:, source_indices].half(),
                    "sender_logits": logits, "target_indices": target_indices, "counts": counts}
                if number % 16 == 0 or number == len(rows): log(f"{family}/{dataset} sender capture {number}/{len(rows)}")
        finally:
            del sender; gc.collect(); torch.cuda.empty_cache()
        qwen, base, adapter = load_model("qwen"), None, None
        base, adapter, path_a, path_b = load_writer(family, stage_b_kind)
        records = []
        try:
            for number, row in enumerate(rows, 1):
                cached, fields = source_cache[row["id"]], row["encoded"]["qwen"]
                native_k, native_v, full_logits = capture(qwen, fields["full"], len(fields["body"]), fields["choice_ids"])
                indices = cached["target_indices"]
                question_k, question_v = native_k[:, :1].cuda(), native_v[:, :1].cuda()
                oracle_k, oracle_v = native_k[:, indices].cuda(), native_v[:, indices].cuda()
                ak, av, bk, bv = translate(base, adapter, cached["k"], cached["v"])
                choices = torch.tensor(fields["choice_ids"], device="cuda")
                def condition_logits(ok, ov):
                    logits = readout(qwen, fields["answer_only"], torch.cat((question_k, ok), 1), torch.cat((question_v, ov), 1))
                    return logits.index_select(0, choices).cpu()
                logits = {"sender_full_native": cached["sender_logits"], "qwen_full_native": full_logits,
                          "native_oracle": condition_logits(oracle_k, oracle_v),
                          "stage_a": condition_logits(ak, av), "stage_b": condition_logits(bk, bv)}
                gold = row["gold_index"]
                conditions = {name: {"prediction": int(value.argmax()), "correct": int(value.argmax()) == gold,
                                      "choice_logits": value.float().tolist()} for name, value in logits.items()}
                representation = {}
                teacher_log = F.log_softmax(logits["native_oracle"].float(), -1)
                for name, key, value in (("stage_a", ak, av), ("stage_b", bk, bv)):
                    kn, kc = component(key, oracle_k); vn, vc = component(value, oracle_v)
                    student_log = F.log_softmax(logits[name].float(), -1)
                    representation[name] = {"k_nmse": kn, "k_cosine": kc, "v_nmse": vn, "v_cosine": vc,
                        "choice_kl": F.kl_div(student_log, teacher_log, reduction="sum", log_target=True).item()}
                audit_row = audit[number - 1]
                records.append({"id": row["id"], "dataset": dataset, "category": row.get("category"),
                    "gold_index": gold, "truncated": audit_row["truncated"],
                    "body_tokens": audit_row["body_tokens"], "qwen_transferred_tokens": len(indices),
                    "mapping_counts": cached["counts"], "conditions": conditions, "representation": representation})
                if number % 16 == 0 or number == len(rows): log(f"{family}/{dataset} receiver evaluation {number}/{len(rows)}")
                del native_k, native_v, question_k, question_v, oracle_k, oracle_v, ak, av, bk, bv
        finally:
            del qwen, base, adapter, source_cache; gc.collect(); torch.cuda.empty_cache()
        summary = summarize(records)
        all_summary["datasets"][dataset] = summary
        path = out / f"{dataset}_per_sample.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in records: stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_json(out / f"{dataset}_summary.json", summary)
        log(f"DONE {family}/{dataset}: Stage-B={summary['stage_b']['accuracy']:.4f} Oracle={summary['native_oracle']['accuracy']:.4f}")
    all_summary.update(status="completed", stage_a_checkpoint=path_a, stage_b_checkpoint=path_b,
                       samples_per_dataset=len(rows))
    save_json(out / ("smoke_summary.json" if smoke else "summary.json"), all_summary)


def audit_only():
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True)
                  for name, path in CFG["models"].items()}
    summary = {}
    for family in ("llama", "gemma"):
        summary[family] = {}
        for dataset in ("hellaswag", "gsm8k", "longbench_v2"):
            rows, audit = prepare_rows(dataset, {family: tokenizers[family], "qwen": tokenizers["qwen"]})
            for row in rows:
                fullsync_map(row, tokenizers[family], tokenizers["qwen"], family)
            summary[family][dataset] = {"count": len(rows), "truncated": sum(x["truncated"] for x in audit),
                "max_body_tokens": {name: max(x["body_tokens"][name] for x in audit) for name in (family, "qwen")}}
            log(f"AUDIT {family}/{dataset}: {summary[family][dataset]}")
    save_json(HERE / "runs/audit.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("llama", "gemma"))
    parser.add_argument("--stage-b", choices=("hybrid", "purekl"), default="hybrid")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    evaluate_family(args.action, args.stage_b, args.smoke)


if __name__ == "__main__":
    main()
