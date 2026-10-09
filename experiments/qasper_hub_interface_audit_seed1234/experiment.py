"""QASPER dev interface audit. No training; plain text, full evidence, no truncation."""
import collections
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234"
sys.path.insert(0, str(LEGACY))
from kv_backend import CFG, load_model, load_base, load_adapter, pair, map_base, assemble, seed_all, save_json
from protocol import make_cache

DATA = Path("/hy-tmp/yezhe/all_models/datasets/allenai/qasper/extracted")
MAX_NEW = 128


def evidence(p):
    return ("Read the paper and answer the question using only the paper. "
        "Give a concise answer without reasoning. If not answerable, output Unanswerable. "
        "For yes/no questions output Yes or No.\n\nTitle: " + p["title"] +
        "\n\nAbstract: " + p["abstract"] + "\n\n" + "\n\n".join(
            (s["section_name"] or "") + "\n" + "\n\n".join(s["paragraphs"]) for s in p["full_text"])).rstrip()


def freeze_cache(cache):
    return [(layer.keys.detach().cpu(), layer.values.detach().cpu()) for layer in cache.layers]


def restore(model, state):
    return DynamicCache(ddp_cache_data=[(k.cuda(), v.cuda()) for k, v in state], config=model.config)


@torch.no_grad()
def prefill(model, ids, cache=None):
    start = cache.get_seq_length() if cache is not None else 0
    tensor = torch.tensor([ids], device="cuda")
    out = model.model(input_ids=tensor, past_key_values=cache, use_cache=True,
        attention_mask=torch.ones((1, start+len(ids)), device="cuda", dtype=torch.long),
        position_ids=torch.arange(start, start+len(ids), device="cuda")[None])
    logits = model.lm_head(out.last_hidden_state[:, -1])[0].float()
    return logits, out.past_key_values


@torch.no_grad()
def generate(model, logits, cache, tokenizer):
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    ids = []
    for _ in range(MAX_NEW):
        token = int(logits.argmax())
        if token in eos:
            return {"text": tokenizer.decode(ids, skip_special_tokens=True), "ids": ids, "stop": "eos"}
        ids.append(token)
        logits, cache = prefill(model, [token], cache)
    return {"text": tokenizer.decode(ids, skip_special_tokens=True), "ids": ids, "stop": "max_tokens"}


def main():
    seed_all(1234)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("qasper_official", DATA / "qasper_evaluator.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    data = json.loads((DATA / "qasper-dev-v0.3.json").read_text())
    references = evaluator.get_answers_and_evidence(data, True)
    tok = AutoTokenizer.from_pretrained(CFG["models"]["qwen"], local_files_only=True)
    buckets = collections.defaultdict(list)
    excluded = collections.Counter()
    for pid, paper in data.items():
        text = evidence(paper)
        ei = tok.encode(text, add_special_tokens=False)
        if not 512 <= len(ei) <= 3072:
            excluded["outside_512_3072_evidence_tokens"] += len(paper["qas"])
            continue
        for qa in paper["qas"]:
            suffix = "\n\nQuestion: " + qa["question"] + "\nAnswer:"
            ids = tok.encode(text+suffix, add_special_tokens=False)
            if ids[:len(ei)-1] != ei[:-1]:
                excluded["question_token_boundary"] += 1
                continue
            typ = references[qa["question_id"]][0]["type"]
            buckets[typ].append({"paper_id": pid, "question_id": qa["question_id"],
                "question": qa["question"], "type": typ, "evidence": text,
                "evidence_ids": ei, "split": len(ei)-1, "suffix": suffix, "ids": ids})
    rng = random.Random(1234)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected = []
    for typ in ("extractive", "abstractive", "boolean", "none"):
        selected.extend(buckets[typ][:4])
    if len(selected) != 16:
        raise RuntimeError(f"Need four questions per type; got {len(selected)}")
    save_json(output / "manifest.json", {"split": "dev", "seed": 1234, "no_truncation": True,
        "max_new_tokens": MAX_NEW, "chat_template": False, "excluded": dict(excluded),
        "eligible": {k: len(v) for k,v in buckets.items()}, "rows": selected})
    cfg = {"models": CFG["models"], "attention_implementation": CFG["attention_implementation"]}
    receiver = load_model(cfg, "qwen")
    native_states, native_logits, records = {}, {}, []
    all_predictions = collections.defaultdict(dict)
    for row in selected:
        pid, qid = row["paper_id"], row["question_id"]
        ei, ids = row["evidence_ids"], row["ids"]
        if pid not in native_states:
            _, ec = prefill(receiver, ids[:row["split"]])
            native_states[pid] = freeze_cache(ec)
            del ec
        start = time.perf_counter()
        full_logits, full_cache = prefill(receiver, ids)
        native_logits[qid] = full_logits.cpu()
        full = generate(receiver, full_logits, full_cache, tok)
        del full_cache
        full_seconds = time.perf_counter()-start
        start = time.perf_counter()
        cached_logits, cached_cache = prefill(receiver, ids[row["split"]:], restore(receiver, native_states[pid]))
        cached = generate(receiver, cached_logits, cached_cache, tok)
        del cached_cache
        delta = cached_logits-native_logits[qid].cuda()
        result = {"question_id": qid, "paper_id": pid, "type": row["type"], "evidence_tokens": len(ei),
            "cache_tokens": row["split"], "full": full, "native_cache": cached, "generation_identical": full["ids"] == cached["ids"],
            "first_logit_max_abs": float(delta.abs().max()), "first_logit_mean_abs": float(delta.abs().mean()),
            "first_argmax_identical": int(cached_logits.argmax()) == int(native_logits[qid].argmax()),
            "full_seconds": full_seconds, "cache_question_seconds": time.perf_counter()-start}
        records.append(result)
        for name, pred in (("full", full), ("native_cache", cached)):
            all_predictions[name][qid] = {"answer": pred["text"].strip(), "evidence": []}
        print(f"NATIVE {len(records)}/16 parity={result['generation_identical']} delta={result['first_logit_max_abs']}", flush=True)
        save_json(output / "per_sample.json", records)
    # Old checkpoints are cross-protocol zero-shot diagnostics, NOT a new method comparison.
    sender = load_model(cfg, "llama")
    source_tok = AutoTokenizer.from_pretrained(CFG["models"]["llama"], local_files_only=True)
    for group in ("mcq", "gsm8k"):
        base = load_base(LEGACY / f"runs/{group}/stage_a/best.pt")
        adapter = load_adapter(LEGACY / f"runs/{group}/stage_b/best.pt")
        for index, row in enumerate(selected):
            item = pair(sender, receiver, source_tok, tok, row["evidence"])
            assert item["ids"] == row["evidence_ids"]
            bk, bv = map_base(base, item)
            for stage, residual in (("a", None), ("b", adapter)):
                k, v = assemble(item, bk, bv, residual)
                assert k.shape[1] == row["split"]
                cache = make_cache(receiver, k, v, torch.arange(k.shape[1], device="cuda"))
                logits, cache = prefill(receiver, row["ids"][k.shape[1]:], cache)
                pred = generate(receiver, logits, cache, tok)
                del cache
                name = group+"_stage_"+stage
                records[index][name] = pred
                all_predictions[name][row["question_id"]] = {"answer": pred["text"].strip(), "evidence": []}
            print(f"DIAGNOSTIC {group} {index+1}/16", flush=True)
            save_json(output / "per_sample.json", records)
        del base, adapter
        torch.cuda.empty_cache()
    gold = {r["question_id"]: references[r["question_id"]] for r in selected}
    metrics = {}
    for name, predictions in all_predictions.items():
        report = evaluator.evaluate(gold, predictions)
        report.pop("Evidence F1", None)
        report["evidence_scoring"] = "not evaluated; answer-only experiment"
        report["max_token_rate"] = sum(r[name]["stop"] == "max_tokens" for r in records)/16
        metrics[name] = report
        save_json(output / (name+"_predictions.json"), predictions)
    summary = {"count": 16, "metrics": metrics,
        "generation_exact_agreement": sum(r["generation_identical"] for r in records)/16,
        "first_argmax_agreement": sum(r["first_argmax_identical"] for r in records)/16,
        "max_first_logit_error": max(r["first_logit_max_abs"] for r in records),
        "status": "completed", "protocol": "plain-text evidence prefix then native question; dev pilot; no training",
        "old_writer_note": "Evidence except native token0 and final boundary token translated; boundary token and question native. Off-distribution checkpoint diagnostics only."}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    print("EXPERIMENT 0 COMPLETED", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
