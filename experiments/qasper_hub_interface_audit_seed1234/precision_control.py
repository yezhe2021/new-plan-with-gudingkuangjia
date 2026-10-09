"""FP32 numerical control on two dev questions; separate from FP16 task scores."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import experiment as e


def main():
    e.seed_all(1234)
    output = e.ROOT / "results"
    manifest = json.loads((output/"manifest.json").read_text())
    previous = json.loads((output/"per_sample.json").read_text())
    candidates = [r["question_id"] for r in previous if not r["generation_identical"]][:2]
    if not candidates:
        candidates = [previous[0]["question_id"]]
    rows = [r for r in manifest["rows"] if r["question_id"] in candidates]
    model = AutoModelForCausalLM.from_pretrained(e.CFG["models"]["qwen"], local_files_only=True,
        dtype=torch.float32, attn_implementation=e.CFG["attention_implementation"]).cuda().eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(e.CFG["models"]["qwen"], local_files_only=True)
    e.MAX_NEW = 32
    records = []
    for row in rows:
        full_logits, fc = e.prefill(model, row["ids"])
        full = e.generate(model, full_logits, fc, tok)
        del fc
        _, ec = e.prefill(model, row["ids"][:row["split"]])
        cached_logits, cc = e.prefill(model, row["ids"][row["split"]:], ec)
        cached = e.generate(model, cached_logits, cc, tok)
        del cc, ec
        records.append({"question_id": row["question_id"], "dtype": "float32", "max_new_tokens": 32,
            "first_logit_max_abs": float((full_logits-cached_logits).abs().max()),
            "first_argmax_identical": int(full_logits.argmax())==int(cached_logits.argmax()),
            "generation_identical": full["ids"]==cached["ids"], "full": full, "native_cache": cached})
        e.save_json(output/"precision_control.json", {"scope": "FP32 control only, not replacement benchmark", "records": records})
        print(json.dumps(records[-1]), flush=True)
    print("PRECISION CONTROL COMPLETED", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
