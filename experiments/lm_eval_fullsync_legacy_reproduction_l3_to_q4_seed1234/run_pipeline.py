"""One resumable serialized pipeline. No loss/architecture experimentation before legacy completes."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def log(text):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ")+text,flush=True)


def command(script,*args):
    argv=[sys.executable,"-u",str(ROOT/script),*args]
    log("START "+" ".join(argv)); subprocess.run(argv,cwd=ROOT,check=True)
    log("DONE "+" ".join(argv))


def phase(name):
    path=ROOT/"pipeline_status.json"; tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"phase":name,"pid":os.getpid(),"updated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}),encoding="utf-8")
    tmp.replace(path)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    (ROOT/"logs").mkdir(exist_ok=True)
    lock=(ROOT/"pipeline.lock").open("w")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Another experiment pipeline is already running")
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    command("tests.py")
    phase("prepare")
    for group in ("mcq","gsm8k"):
        run=ROOT/"runs"/group; smoke=ROOT/"runs_smoke"/group
        if not (run/"train.jsonl").exists(): command("train.py","prepare","--group",group)
        if not (smoke/"native_parity.json").exists():
            phase(f"{group}_native_parity"); command("train.py","parity","--group",group)
        for stage in ("a","b"):
            if not (smoke/f"stage_{stage}/completed.json").exists():
                phase(f"{group}_stage_{stage}_smoke"); command("train.py",f"stage_{stage}","--group",group,"--smoke")
    for group in ("mcq","gsm8k"):
        run=ROOT/"runs"/group
        for stage in ("a","b"):
            if not (run/f"stage_{stage}/completed.json").exists():
                phase(f"{group}_stage_{stage}"); command("train.py",f"stage_{stage}","--group",group)
        for condition in ("llama_native","qwen_native","native_oracle","stage_a","stage_b"):
            if not (run/f"evaluation/{condition}.json").exists():
                phase(f"{group}_evaluate_{condition}"); command("evaluate.py","--group",group,"--condition",condition)
        command("summarize.py")
    phase("core_completed")
    # OOD requires official MMLU validation bytes. Missing data is a genuine blocker, never test-as-demo.
    command("prepare_ood.py")
    for condition in ("llama_native","qwen_native","stage_a","stage_b"):
        if not (ROOT/f"runs/mcq/evaluation_ood/{condition}.json").exists():
            phase(f"ood_evaluate_{condition}"); command("evaluate.py","--group","mcq","--condition",condition,"--ood")
    command("summarize.py")
    phase("ALL_LEGACY_EXPERIMENTS_COMPLETED")
    log("ALL LEGACY EXPERIMENTS COMPLETED; follow-up shared-adapter experiment is a separate recorded phase")


if __name__=="__main__":
    try: main()
    except Exception as exc:
        log(f"PIPELINE FAILED: {type(exc).__name__}: {exc}")
        raise
