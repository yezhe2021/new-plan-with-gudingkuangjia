"""Fetch only the official MMLU-Pro validation demonstrations, never repurpose test."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"config.json").read_text())
path=ROOT/CFG["datasets"]["mmlu_pro_validation"]
url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/validation-00000-of-00001.parquet"
if not path.exists():
    path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(url,timeout=30) as response: content=response.read()
    if not content.startswith(b"PAR1"): raise RuntimeError("Invalid parquet bytes, do not use as demonstrations")
    tmp=path.with_suffix(".tmp"); tmp.write_bytes(content); tmp.replace(path)
import pyarrow.parquet as pq
rows=pq.read_table(path).to_pylist()
if not rows or "cot_content" not in rows[0]: raise RuntimeError("Official few-shot CoT schema missing")
(path.parent/"provenance.json").write_text(json.dumps({"url":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"count":len(rows)}),encoding="utf-8")
print(f"Official MMLU validation ready: {len(rows)} rows",flush=True)
