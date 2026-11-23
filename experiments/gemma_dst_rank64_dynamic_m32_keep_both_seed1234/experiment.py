"""Train only fresh all-layer Keep-both M32 on the completed Gemma DST+Rank64."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def launch():
    import json

    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs/pipeline.log").open("a") as log:
        child = subprocess.Popen([sys.executable, "-u", str(__file__), "run"], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / "launch.json").write_text(json.dumps({"pid": child.pid, "server": "server-b"}) + "\n")
    print("LAUNCHED", child.pid, flush=True)


if __name__ == "__main__" and sys.argv[1] == "launch":
    launch()
    sys.exit(0)

import experiment_common as G
from experiment_common import ASSETS, C, CFG, PARENT, RECIPE, done, identity, mark, paused, rows, status, torch, verify

# These are the same training, routing and scoring implementations used by the
# original Llama DST comparison. Only their engine and frozen source change.
sys.path.insert(0, str(RECIPE))
import runtime_new as runtime
from memory_dynamic import DynamicMemory
from score import score

gemma_engine = G.scoped("gemma_frozen_engine", ASSETS / "engine.py", G.G)
gemma_translator = G.scoped("gemma_frozen_translator", ASSETS / "translator.py", G.G)
gemma_residual = G.scoped("gemma_frozen_residual", ASSETS / "residual.py", G.G)
CHECKPOINTS = {stage: ASSETS / f"runs/{stage}/epoch4_step1536.pt" for stage in ("stage1", "stage2")}
ASSET_SIZES = {
    CHECKPOINTS["stage1"]: 472468292,
    CHECKPOINTS["stage1"].with_name("completed.json"): 888,
    CHECKPOINTS["stage2"]: 37808590,
    CHECKPOINTS["stage2"].with_name("completed.json"): 1054,
}
FINAL = ROOT / "runs/epoch4_step1536.pt"
ARM = "DST_rank64_dynamic_m32"
BASELINE = "DST_rank64"
QA = ("hotpotqa", "2wiki", "musique")


def model_paths():
    paths = {}
    for role in ("sender", "receiver"):
        path = Path(G.G.CFG[role])
        if not path.is_dir():
            path = Path("/hy-tmp/yezhe/data/all_models") / path.relative_to("/hy-tmp/yezhe/all_models")
        assert path.is_dir(), path
        paths[role] = path
    return paths


def prepare():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    sources = {str(path): C.sha(path) for path in [
        Path(__file__), ROOT / "experiment_common.py", ASSETS / "config.json",
        ASSETS / "manifest.json", ASSETS / "engine.py", ASSETS / "translator.py",
        ASSETS / "residual.py", RECIPE / "config.json",
        *[RECIPE / f"{name}.py" for name in ("train_new", "runtime_new", "memory_dynamic", "score")],
    ]}
    for stage, path in CHECKPOINTS.items():
        completed = C.read(path.with_name("completed.json"))
        assert completed["status"] == "completed" and completed["updates"] == 1536
        assert completed["manifest_sha256"] == G.G.identity()
        assert C.sha(path) == completed["checkpoint_sha256"]
        sources[str(path)] = completed["checkpoint_sha256"]
    fields = ("id", "task", "question", "prompt", "gold", "gold_tokens", "content_ids", "suffix_ids", "candidates", "answers")
    prefix_sources = {}
    for split in ("train", "validation", "test"):
        source_path = ASSETS / "prepared" / f"{split}.jsonl"
        original_path = PARENT / "prepared" / f"{split}.jsonl"
        data = rows(split)
        original = {row["id"]:row for row in C.read_jsonl(original_path)}
        assert len(data) == len(original) and Counter(row["task"] for row in data) == CFG[f"{split}_counts"]
        for row in data:
            assert all(row.get(key) == original[row["id"]].get(key) for key in fields), row["id"]
            if split in ("train", "validation"):
                path = PARENT / "native_prefix" / (row["id"] + ".json")
                if path.exists():
                    prefix_sources[str(path)] = C.sha(path)
        sources[str(source_path)] = C.sha(source_path)
        sources[str(original_path)] = C.sha(original_path)
    for path in model_paths().values():
        for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
            sources[str(path / name)] = C.sha(path / name)
    schedule = {str(epoch):C.digest([[(row["id"], kind) for row,kind in batch] for batch in runtime.batches(epoch)])
                for epoch in range(1,5)}
    C.save(ROOT / "manifest.json", dict(
        sources=sources, config=CFG, native_prefix_sources=prefix_sources, batch_schedules=schedule,
        gemma_base_manifest_sha256=G.G.identity(), training="fresh 409600 Memory parameters only",
    ))
    print("VERIFIED GEMMA DST weights and aligned 3072/192/512 rows", flush=True)


class Engine(gemma_engine.Engine):
    def __init__(self):
        from accelerate import cpu_offload
        from transformers import Gemma3ForConditionalGeneration

        self.tok, self.model = C.load_model(G.G.CFG["receiver"], "cuda", dtype=torch.float16)
        self.sender_tok = C.AutoTokenizer.from_pretrained(G.G.CFG["sender"], local_files_only=True)
        # Text-only capture never calls the vision tower or multimodal projector.
        # Offloaded text blocks execute with FP32 GPU weights during their forward.
        self.sender = Gemma3ForConditionalGeneration.from_pretrained(
            G.G.CFG["sender"], dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        ).eval().requires_grad_(False)
        language = C.decoder(self.sender)
        assert len(language.layers) == G.G.CFG["model"]["sender_layers"] == 34
        device = torch.device("cuda", 0)
        for name, component in language.named_children():
            if name != "layers":
                component.to(device)
        self.sender.lm_head.to(device)
        for index, block in enumerate(language.layers):
            if index < CFG["gemma_offloaded_layers"]:
                cpu_offload(block, execution_device=device)
            else:
                block.to(device)
        assert self.sender.get_input_embeddings().weight.device == device
        self.frozen_base = None
        self.base_versions = None
        self.prefix_ids = C.read(C.ROOT / "manifest.json")["prefix_ids"]
        with torch.no_grad():
            self.header = C.decoder(self.model)(torch.tensor([self.prefix_ids], device=device),
                                               use_cache=True).past_key_values
        self.mode = "train"
        self.student_test_native_prefills = 0
        self.resident_versions = self.resident_parameter_versions()
        self.meta_count = sum(p.is_meta for p in self.sender.parameters())
        assert self.meta_count > 0

    def resident_parameter_versions(self):
        return [[(name, id(p), p._version) for name, p in model.named_parameters() if not p.is_meta]
                for model in (self.model, self.sender)]

    def unchanged(self):
        assert self.resident_parameter_versions() == self.resident_versions
        assert sum(p.is_meta for p in self.sender.parameters()) == self.meta_count
        assert C.versions(self.frozen_base) == self.base_versions
        assert all(not p.requires_grad and p.grad is None
                   for model in (self.model, self.sender, self.frozen_base) for p in model.parameters())

    def base(self, row):
        return self.source(row)

    @torch.no_grad()
    def cache(self, row, source, correction):
        return self.translated(row, correction, source=source, checkpointed=False)


def checked_state(module, stage):
    checkpoint = torch.load(CHECKPOINTS[stage], map_location="cpu", weights_only=True)
    assert checkpoint["manifest_sha256"] == G.G.identity()
    assert checkpoint["stage"] == stage and checkpoint["step"] == 1536 and checkpoint["epoch"] == 4
    module.load_state_dict(checkpoint["state"], strict=True)
    module.eval().requires_grad_(False)
    return module


def load(final=False):
    for role,path in model_paths().items():
        G.G.CFG[role] = str(path)
    engine = Engine()
    params = gemma_translator.DSTConfig(**G.G.CFG["model"])
    base = checked_state(gemma_translator.DepthStateTranslator(params, CFG["seed"]).cuda().float(), "stage1")
    engine.frozen_base, engine.base_versions = base, C.versions(base)
    ranked = checked_state(gemma_residual.ResidualTranslator(base, rank=64, seed=CFG["seed"]).cuda().float(), "stage2")
    assert sum(p.numel() for p in ranked.parameters()) == 9437184
    memory = DynamicMemory(engine, CFG)
    assert sum(p.numel() for p in memory.parameters()) == 409600
    if final:
        completed = C.read(ROOT / "runs/completed.json")
        assert C.sha(FINAL) == completed["checkpoint_sha256"]
        checkpoint = torch.load(FINAL, map_location="cpu", weights_only=True)
        assert checkpoint["manifest_sha256"] == identity() and checkpoint["step"] == 1536
        memory.load_state_dict(checkpoint["state"], strict=True)
        memory.eval().requires_grad_(False)
    return engine, ranked, memory


@torch.no_grad()
def smoke_forward(engine, ranked, memory, row):
    body = engine.cache(row, engine.base(row), ranked)
    baseline = score(memory, row, body, rn=False)
    built = memory.construct(row, body)
    assert body.get_seq_length() == len(engine.prefix_ids) + len(row["content_ids"])
    assert built.get_seq_length() == body.get_seq_length() + 32
    result = score(memory, row, built, rn=True)
    assert all(key in baseline and key in result for key in ("em","f1"))
    return dict(id=row["id"], body_tokens=len(row["content_ids"]), built_positions=built.get_seq_length())


def smoke():
    if done("smoke/completed.json"):
        return
    status(status="running",phase="smoke")
    engine, ranked, memory = load()
    snapshot = C.versions(ranked)
    hotpot = next(r for r in rows("validation") if r["task"]=="hotpotqa")
    info = smoke_forward(engine, ranked, memory, hotpot)
    opt = torch.optim.AdamW(memory.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = G.T.scaler_for()
    for index,batch in enumerate(runtime.batches(1)[:2],1):
        _, outcome, retries = runtime.backward_batch(engine, ranked, memory, opt, scaler, batch, 1, index, scope="smoke")
        assert outcome["accepted"] and outcome["gradient_norm"] > 0
        engine.unchanged()
        assert C.versions(ranked)==snapshot
        print("SMOKE accepted batch",index,"AMP retries",retries,flush=True)
    longest = max(rows("train"), key=lambda r:len(r["content_ids"]))
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        body = engine.cache(longest, engine.base(longest), ranked)
        native = engine.native_cache(longest)
    loss,_ = runtime.losses(engine,memory,longest,body,native,"gold",1)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    mark("smoke/completed.json",status="completed",forward=info,longest_row=longest["id"],
         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.no_grad()
def evaluate():
    if done("completed.json"):
        return
    assert done("runs/completed.json")
    engine, ranked, memory = load(final=True)
    engine.mode = "test"
    before = [C.versions(m) for m in (ranked,memory)]
    cohort = rows("test")
    lines = ["# Gemma → Qwen DST+Rank64 + Dynamic M32 Keep-both", "",
             "| Task | Metric | DST+Rank64 | +M32 | Δ pp |", "|---|---|---:|---:|---:|"]
    tasks = {}
    for task,n in CFG["test_counts"].items():
        records = []
        for index,row in enumerate((r for r in cohort if r["task"]==task),1):
            if paused():
                raise SystemExit(75)
            rel = Path("evaluation") / task / "per_sample" / (row["id"]+".json")
            if not (ROOT/rel).exists():
                body = engine.cache(row,engine.base(row),ranked)
                baseline = score(memory,row,body,rn=False)
                output = score(memory,row,memory.construct(row,body),rn=True)
                if task == "hellaswag":
                    for scored in (baseline, output):
                        raw = scored["raw_sum_scores"]
                        scored["raw_correct"] = int(max(range(len(raw)), key=raw.__getitem__) == row["gold"])
                assert engine.student_test_native_prefills == 0
                engine.unchanged()
                assert before == [C.versions(m) for m in (ranked,memory)]
                mark(rel,id=row["id"],task=task,question=row["question"],gold=row["gold"],
                     conditions={BASELINE:baseline, ARM:output})
                del body
            record = C.read(ROOT/rel)
            assert record["manifest_sha256"] == identity() and record["id"] == row["id"]
            records.append(record["conditions"])
            status(status="running",phase="evaluation",task=task,sample=index,total=n)
            if index==1 or index%8==0:
                print("GEMMA M32 EVAL",task,index,n,flush=True)
        assert len(records)==n
        keys = ("em","f1") if task in QA else ("correct","legacy_correct") if task=="gsm8k" else (
            ("correct","raw_correct") if task=="hellaswag" else ("correct",)
        )
        scores = {arm:{key:statistics.mean(r[arm][key] for r in records) for key in keys} for arm in (BASELINE,ARM)}
        differences = {k:C.paired([r[ARM][k]-r[BASELINE][k] for r in records]) for k in keys}
        tasks[task] = dict(n=n,scores=scores,memory_minus_base=differences)
        for k in keys:
            a,b=scores[BASELINE][k],scores[ARM][k]
            lines.append(f"| {task} | {k} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |")
        mark(Path("evaluation")/task/"summary.json",**tasks[task])
    mark("summary.json",status="completed",tasks=tasks,
         macro_acc_em={arm:statistics.mean(t["scores"][arm]["em" if task in QA else "correct"]
                                           for task,t in tasks.items()) for arm in (BASELINE,ARM)})
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    mark("completed.json",status="completed")
    status(status="completed",phase="complete")


def run():
    while True:
        if paused():
            raise SystemExit(75)
        copied = {str(path.relative_to(ASSETS)):path.stat().st_size if path.exists() else 0
                  for path in ASSET_SIZES}
        if list(copied.values()) == list(ASSET_SIZES.values()):
            break
        status(status="waiting_inputs",phase="asset_sync",copied_bytes=sum(copied.values()),
               expected_bytes=sum(ASSET_SIZES.values()))
        time.sleep(20)
    status(status="running",phase="prepare")
    prepare()
    if done("completed.json"):
        return
    while True:
        if paused():
            raise SystemExit(75)
        detected = subprocess.run(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits","-i","0"],
                                  capture_output=True,text=True,timeout=10)
        free = int(detected.stdout.strip()) if detected.returncode==0 else 0
        if free >= CFG["minimum_free_gpu_mib"]:
            break
        status(status="waiting_gpu",phase="queue",free_mib=free,required_mib=CFG["minimum_free_gpu_mib"])
        time.sleep(30)
    budget = min(CFG["allocator_limit_mib"], free - 1536)
    torch.cuda.set_per_process_memory_fraction(budget * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    print("SHARED GPU allocator budget MiB", budget, flush=True)
    smoke()
    torch.cuda.empty_cache()
    runtime.load = load
    import train_new
    train_new.main()
    torch.cuda.empty_cache()
    evaluate()


if __name__ == "__main__":
    with (ROOT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run()
        except SystemExit as error:
            if error.code==75:
                status(status="paused",phase="paused")
            raise
        except Exception as error:
            status(status="failed",phase="failed",error=repr(error))
            raise
