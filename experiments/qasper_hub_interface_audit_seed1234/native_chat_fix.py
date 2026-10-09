"""Versioned native-only chat repair on the frozen dev manifest. No training."""
import hashlib
import importlib.util
import json
import time
import torch
from transformers import AutoTokenizer
import experiment as e

SYSTEM = ("Answer the question using only the supplied paper. Return only a concise "
          "answer, not an explanation or reasoning. For a yes/no question return exactly "
          "Yes or No. If the paper does not contain the answer return exactly Unanswerable. "
          "Do not repeat the question, add an Answer: label, or continue with another question.")


def run():
    e.seed_all(1234)
    output = e.ROOT / "results_native_chat_v1"
    output.mkdir(exist_ok=True)
    rows = json.loads((e.ROOT/"results/manifest.json").read_text())["rows"]
    tok = AutoTokenizer.from_pretrained(e.CFG["models"]["qwen"], local_files_only=True)
    model = e.load_model({"models": e.CFG["models"],
        "attention_implementation": e.CFG["attention_implementation"]}, "qwen")
    eos = model.generation_config.eos_token_id
    eos_set = {eos} if isinstance(eos, int) else set(eos)
    e.save_json(output/"config.json", {"source_manifest": "results/manifest.json", "split": "dev",
        "count": len(rows), "chat_template": tok.chat_template, "enable_thinking": False,
        "system": SYSTEM, "max_new_tokens": 128, "do_sample": False, "EOS": eos,
        "dtype": "float16", "seed": 1234, "no_truncation": True,
        "scope": "chat template + explicit format + no-thinking bundle; NOT single-variable ablation"})
    records, cached_papers = [], {}
    for row in rows:
        paper = "Title: " + row["evidence"].split("Title: ", 1)[1]
        marker = "\n\nQuestion: " + row["question"]
        user = "Paper:\n" + paper + marker
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        boundary = prompt.rindex(marker)
        split = sum(b <= boundary for a, b in encoded["offset_mapping"])
        if not 0 < split < len(ids):
            raise RuntimeError("Invalid chat evidence/question boundary")
        tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
        args = {"input_ids": tensor, "attention_mask": torch.ones_like(tensor),
                "max_new_tokens": 128, "do_sample": False, "use_cache": True,
                "pad_token_id": tok.pad_token_id or tok.eos_token_id}
        start = time.perf_counter()
        native = model.generate(**args)[0, len(ids):].tolist()
        full_seconds = time.perf_counter()-start
        key = hashlib.sha256(json.dumps(ids[:split]).encode()).hexdigest()
        if key not in cached_papers:
            _, cache = e.prefill(model, ids[:split])
            cached_papers[key] = e.freeze_cache(cache)
            del cache
        start = time.perf_counter()
        # HF receives the identical full sequence and handles its cache positions.
        reused = model.generate(**args, past_key_values=e.restore(model, cached_papers[key]))[0, len(ids):].tolist()
        record = {"question_id": row["question_id"], "paper_id": row["paper_id"], "type": row["type"],
            "prompt": prompt, "input_ids": ids, "cache_split": split,
            "full_text": tok.decode(native, skip_special_tokens=True).strip(),
            "cache_text": tok.decode(reused, skip_special_tokens=True).strip(),
            "full_ids": native, "cache_ids": reused,
            "full_eos": bool(native and native[-1] in eos_set),
            "cache_eos": bool(reused and reused[-1] in eos_set),
            "generation_identical": native == reused,
            "full_seconds": full_seconds, "cache_seconds": time.perf_counter()-start}
        records.append(record)
        e.save_json(output/"per_sample.json", records)
        print(f"CHAT NATIVE {len(records)}/{len(rows)} EOS={record['full_eos']} length={len(native)} parity={record['generation_identical']}", flush=True)
    spec = importlib.util.spec_from_file_location("qasper_eval", e.DATA/"qasper_evaluator.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    data = json.loads((e.DATA/"qasper-dev-v0.3.json").read_text())
    refs = evaluator.get_answers_and_evidence(data, True)
    gold = {r["question_id"]: refs[r["question_id"]] for r in rows}
    metrics = {}
    for name in ("full", "cache"):
        predictions = {r["question_id"]: {"answer": r[name+"_text"], "evidence": []} for r in records}
        report = evaluator.evaluate(gold, predictions)
        report.pop("Evidence F1")
        report.update(EOS_rate=sum(r[name+"_eos"] for r in records)/len(rows),
            mean_generated_tokens=sum(len(r[name+"_ids"]) for r in records)/len(rows),
            max_token_rate=sum(not r[name+"_eos"] and len(r[name+"_ids"])==128 for r in records)/len(rows))
        metrics[name] = report
        e.save_json(output/(name+"_predictions.json"), predictions)
    old = json.loads((e.ROOT/"results/summary.json").read_text())
    summary = {"status": "completed", "count": len(rows), "metrics": metrics,
        "generation_agreement": sum(r["generation_identical"] for r in records)/len(rows),
        "previous_plaintext_full_answer_f1": old["metrics"]["full"]["Answer F1"],
        "note": "Same dev sample IDs and official raw-answer F1; no gold-based parser tuning; no training. Evidence F1 not evaluated."}
    e.save_json(output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    print("NATIVE CHAT REPAIR COMPLETED", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        run()
