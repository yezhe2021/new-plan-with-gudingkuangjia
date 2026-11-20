"""Train and evaluate only full-depth conditional two-step reading."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
CFG = json.loads(CONFIG_PATH.read_text())
BASE = ROOT.parent / CFG["base"]
ARCHIVE = ROOT.parent / CFG["gated_memory_archive"]
FORMER_MODEL_ROOT = Path("/hy-tmp/yezhe/all_models")
sys.path.insert(0, str(BASE))

from experiment_common import C, K, rows, torch
from engine import BodyReader, Engine
from runtime import batches, checkpoint_path, load_stage
from score import score
from reader import Reader

RUNS = ROOT / "runs"
MANIFEST = ROOT / "manifest.json"
FINAL = RUNS / f"epoch{CFG['epochs']}_step{CFG['updates']}.pt"
TASKS = ("arc", "hotpotqa", "2wiki", "musique", "sciq", "mmlu_pro", "hellaswag", "gsm8k")
QA = ("hotpotqa", "2wiki", "musique")
ARMS = ("B_rank64", "gated_dynamic", "D_two_step_all36")


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def status(phase, **details):
    C.save(ROOT / "status.json", dict(time=stamp(), phase=phase, **details))


def identified(path, **data):
    C.save(ROOT / path, dict(manifest_sha256=C.sha(MANIFEST), **data))


def matching(path, **expected):
    data = C.read(ROOT / path)
    assert data["manifest_sha256"] == C.sha(MANIFEST)
    assert all(data[key] == value for key, value in expected.items()), path
    return data


def paused():
    if (ROOT / "controls/pause_requested.json").exists():
        raise SystemExit(75)


def prepare():
    sources = {
        BASE / "manifest.json": CFG["base_manifest_sha256"],
        checkpoint_path("stage2"): CFG["rank64_sha256"],
        **{BASE / "prepared" / f"{split}.jsonl": expected for split, expected in CFG["split_sha256"].items()},
        ARCHIVE / "inputs/prepared/test.jsonl": CFG["split_sha256"]["test"],
    }
    for path, expected in sources.items():
        assert C.sha(path) == expected, path
    archive_manifest = C.read(ARCHIVE / "manifest.json")
    assert archive_manifest["settings"]["inputs_sha256"]["rank64.pt"] == CFG["rank64_sha256"]
    for model in ("sender", "receiver"):
        path = resolved_model(model)
        for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
            sources[path / name] = C.sha(path / name)
    expected_test = dict.fromkeys(TASKS, 64)
    for split, expected in (("train", dict.fromkeys(("arc", "hotpotqa", "gsm8k"), 1024)),
                            ("validation", dict.fromkeys(("arc", "hotpotqa", "gsm8k"), 64)),
                            ("test", expected_test)):
        dataset = rows(split)
        assert Counter(row["task"] for row in dataset) == expected
        assert len({row["id"] for row in dataset}) == len(dataset)
    for row in rows("test"):
        archived_scores(row)

    schedule = [batches(epoch, "stage2") for epoch in range(1, CFG["epochs"] + 1)]
    assert sum(len(epoch) for epoch in schedule) == CFG["updates"]
    assert all(len(batch) == CFG["effective_batch"] for epoch in schedule for batch in epoch)
    manifest = dict(
        protocol=CFG["protocol"],
        objective="full gold CE on original Qwen suffix and gold answer tokens",
        batch_hashes=[C.digest([[row["id"] for row, _ in batch] for batch in epoch]) for epoch in schedule],
        sources={str(path): value for path, value in sources.items()},
        gated_archive_sha256=C.sha(ARCHIVE / "manifest.json"),
        code={path.name: C.sha(path) for path in (*ROOT.glob("*.py"), CONFIG_PATH)},
        settings=CFG,
    )
    if MANIFEST.exists() and (RUNS / "last.pt").exists():
        assert C.read(MANIFEST) == manifest, "Inputs or code changed during an existing run"
    else:
        C.save(MANIFEST, manifest)
    return schedule


def resolved_model(role):
    path = Path(C.CFG["models"][role])
    if path.is_dir():
        return path
    return Path(CFG["model_root"]) / path.relative_to(FORMER_MODEL_ROOT)


def load():
    torch.manual_seed(CFG["seed"])
    for role in ("sender", "receiver"):
        C.CFG["models"][role] = str(resolved_model(role))
    engine = Engine()
    load_stage(engine.correction, "stage2")
    engine.correction.eval().requires_grad_(False)
    reader = Reader(engine, CFG)
    assert sum(p.numel() for p in reader.parameters()) == 7_667_784
    return engine, reader, C.versions(engine.correction)


def frozen(engine, correction_versions):
    engine.unchanged()
    assert C.versions(engine.correction) == correction_versions
    assert all(not param.requires_grad and param.grad is None for param in engine.correction.parameters())


@torch.no_grad()
def translation(engine, row):
    return engine.translated(row, engine.correction, engine.source(row), checkpointed=False)


def task_loss(reader, row, cache):
    target = row["gold_tokens"]
    logits = reader.token_logits(row, cache, target, rn=False)
    labels = torch.tensor(target, device=logits.device, dtype=torch.long)
    return torch.nn.functional.cross_entropy(logits, labels)


def smoke(engine, reader, correction_versions, schedule):
    if (ROOT / "smoke/completed.json").exists():
        data = C.read(ROOT / "smoke/completed.json")
        if data["manifest_sha256"] == C.sha(MANIFEST):
            print("SMOKE already completed", flush=True)
            return
    status("smoke", status="running")
    torch.cuda.reset_peak_memory_stats()
    probes = [next(row for row in rows("validation") if row["task"] == task) for task in ("arc", "hotpotqa", "gsm8k")]
    original = BodyReader(engine)
    parity = {}
    with torch.no_grad():
        for row in probes:
            cache = translation(engine, row)
            expected = original.token_logits(row, cache, row["gold_tokens"], rn=False)
            reader.strength = 0.0
            actual = reader.token_logits(row, cache, row["gold_tokens"], rn=False)
            reader.strength = 1.0
            assert torch.equal(expected, actual), (row["id"], (expected - actual).abs().max().item())
            assert cache.get_seq_length() == len(engine.prefix_ids) + len(row["content_ids"])
            parity[row["task"]] = row["id"]
            print("SMOKE zero side path parity", row["task"], row["id"], flush=True)
            del cache, expected, actual
    train_rows = rows("train")
    grad_probes = {
        "longest_body": max(train_rows, key=lambda row: len(row["content_ids"])),
        "longest_gold": max(train_rows, key=lambda row: len(row["gold_tokens"])),
    }
    baseline_state = {key: value.detach().cpu().clone() for key, value in reader.state_dict().items()}
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64, growth_interval=1_000_000)
    for batch_index, batch in enumerate(schedule[0][:2], 1):
        loss, grad, retries = update(engine, reader, optimizer, scaler, batch)
        for layer in reader.readers:
            assert torch.count_nonzero(layer.scales.grad) == 2, "A read is disconnected"
            assert layer.query.up.weight.grad.norm() > 0 and layer.output.up.weight.grad.norm() > 0
        print("SMOKE consecutive accepted batch", batch_index, "loss", loss, "gradient", grad, "retries", retries, flush=True)
    checked = {}
    for name, row in grad_probes.items():
        reader.zero_grad(set_to_none=True)
        cache = translation(engine, row)
        loss = task_loss(reader, row, cache)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())
        checked[name] = dict(id=row["id"], body_tokens=len(row["content_ids"]), gold_tokens=len(row["gold_tokens"]))
        print("SMOKE", name, checked[name], flush=True)
        del loss, cache
    reader.load_state_dict(baseline_state, strict=True)
    reader.zero_grad(set_to_none=True)
    frozen(engine, correction_versions)
    identified("smoke/completed.json", status="completed", parity=parity, probes=checked,
               trainable_parameters=sum(param.numel() for param in reader.parameters()),
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def update(engine, reader, optimizer, scaler, batch):
    for attempt in range(9):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row, _ in batch:
            cache = translation(engine, row)
            loss = task_loss(reader, row, cache)
            value = float(loss.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite gold CE for {row['id']}")
            scaler.scale(loss / len(batch)).backward()
            losses.append(value)
            del loss, cache
        scaler.unscale_(optimizer)
        assert all(param.grad is not None for param in reader.parameters())
        norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG["gradient_clip"])
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale:
            print("AMP retry same batch", attempt + 1, scale, scaler.get_scale(), flush=True)
            continue
        assert torch.isfinite(norm) and norm > 0, norm
        return statistics.mean(losses), float(norm), attempt
    raise FloatingPointError("Exhausted FP16 scale backoff on the same batch")


@torch.no_grad()
def validate(engine, reader, epoch, step):
    reader.eval()
    val = rows("validation")
    losses = {task: [] for task in ("arc", "hotpotqa", "gsm8k")}
    for index, row in enumerate(val, 1):
        paused()
        path = Path("validation") / f"epoch{epoch}" / f"{row['id']}.json"
        if not (ROOT / path).exists():
            cache = translation(engine, row)
            value = float(task_loss(reader, row, cache))
            identified(path, id=row["id"], loss=value)
            del cache
        else:
            value = matching(path, id=row["id"])["loss"]
        losses[row["task"]].append(value)
        if index == 1 or index % 16 == 0:
            status("validation", status="running", epoch=epoch, step=step, sample=index, total=len(val))
    result = {task: statistics.mean(values) for task, values in losses.items()}
    identified(f"validation/epoch{epoch}/summary.json", status="completed", gold_ce=result)
    print("VALIDATION", epoch, result, flush=True)
    reader.train()


def train(engine, reader, correction_versions, schedule):
    if FINAL.exists() and (RUNS / "completed.json").exists():
        marker = matching("runs/completed.json", status="completed")
        assert C.sha(FINAL) == marker["checkpoint_sha256"]
        return
    RUNS.mkdir(exist_ok=True)
    opt = torch.optim.AdamW(reader.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64, growth_interval=1_000_000)
    last = RUNS / "last.pt"
    step = 0
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        assert saved["manifest_sha256"] == C.sha(MANIFEST)
        step = saved["step"]
        reader.load_state_dict(saved["state"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])

    log_path = RUNS / "steps.jsonl"
    if log_path.exists():
        committed = [record for record in C.read_jsonl(log_path) if record["step"] <= step]
        assert [record["step"] for record in committed] == list(range(1, step + 1))
        C.write_jsonl(log_path, committed)
    resume_step = step
    elapsed_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def checkpoint():
        C.atomic_save(dict(manifest_sha256=C.sha(MANIFEST), state=reader.state_dict(), step=step,
                           optimizer=opt.state_dict(), scaler=scaler.state_dict()), last)

    if not last.exists():
        checkpoint()
    with log_path.open("a") as stream:
        logical_step = 0
        reader.train()
        for epoch, batches_in_epoch in enumerate(schedule, 1):
            for batch in batches_in_epoch:
                logical_step += 1
                if logical_step <= resume_step:
                    continue
                if (ROOT / "controls/pause_requested.json").exists():
                    checkpoint()
                    paused()
                batch_start = time.perf_counter()
                ce, grad, retries = update(engine, reader, opt, scaler, batch)
                frozen(engine, correction_versions)
                step += 1
                record = dict(step=step, total_steps=CFG["updates"], epoch=epoch, gold_ce=ce,
                              gradient_norm=grad, amp_retries=retries, step_seconds=time.perf_counter() - batch_start,
                              elapsed_session_seconds=time.perf_counter() - elapsed_start,
                              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if step <= 2 or step % CFG["checkpoint_every"] == 0:
                    checkpoint()
                status("training", status="running", **record)
                if step <= 2 or step % 4 == 0:
                    print("TRAIN", json.dumps(record), flush=True)
            if step == logical_step:
                checkpoint()
                validate(engine, reader, epoch, step)
    assert step == CFG["updates"]
    C.atomic_save(dict(manifest_sha256=C.sha(MANIFEST), state=reader.state_dict(), step=step), FINAL)
    identified("runs/completed.json", status="completed", checkpoint_sha256=C.sha(FINAL), step=step)


def archived_scores(row):
    relative = Path("evaluation") / row["task"] / "per_sample" / (row["id"] + ".json")
    rank64 = C.read(BASE / relative)
    gated = C.read(ARCHIVE / relative)
    assert rank64["manifest_sha256"] == CFG["base_manifest_sha256"]
    assert gated["manifest_sha256"] == C.sha(ARCHIVE / "manifest.json")
    assert rank64["id"] == gated["id"] == row["id"] and rank64["gold"] == gated["gold"] == row["gold"]
    return dict(B_rank64=rank64["conditions"]["B_rank64"], gated_dynamic=gated["conditions"]["gated_dynamic"])


@torch.no_grad()
def evaluate(engine, reader, correction_versions):
    if (ROOT / "summary.json").exists():
        matching("summary.json", status="completed")
        return
    marker = matching("runs/completed.json", status="completed")
    assert C.sha(FINAL) == marker["checkpoint_sha256"]
    final = torch.load(FINAL, map_location="cpu", weights_only=True)
    assert final["manifest_sha256"] == C.sha(MANIFEST) and final["step"] == CFG["updates"]
    reader.load_state_dict(final["state"], strict=True)
    reader.eval()
    engine.mode = "test"
    tasks = {}
    lines = ["# Full-depth Two-Step KV Reading", "",
             "| Task | Metric | Rank64 | Gated Dynamic M32 | D full 36 layers |",
             "|---|---|---:|---:|---:|"]
    for task in TASKS:
        records = []
        cohort = [row for row in rows("test") if row["task"] == task]
        for index, row in enumerate(cohort, 1):
            paused()
            path = Path("evaluation") / task / "per_sample" / (row["id"] + ".json")
            if not (ROOT / path).exists():
                cache = translation(engine, row)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                reader.timing = dict(candidate_forward_tokens=0, decode_forward_tokens=0,
                                     native_body_tokens=0, synthetic_memory_tokens=0,
                                     body_tokens=len(row["content_ids"]), suffix_tokens=len(row["suffix_ids"]))
                started = time.perf_counter()
                result = score(reader, row, cache, rn=False)
                torch.cuda.synchronize()
                reader.timing.update(receiver_score_seconds=time.perf_counter() - started,
                                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                assert engine.student_test_native_prefills == 0
                frozen(engine, correction_versions)
                identified(path, id=row["id"], task=task, gold=row["gold"],
                           conditions={**archived_scores(row), "D_two_step_all36": result}, timing=reader.timing)
                reader.timing = None
                del cache
            records.append(matching(path, id=row["id"]))
            if index == 1 or index % 8 == 0:
                status("evaluation", status="running", task=task, sample=index, total=len(cohort))
                print("EVAL", task, index, len(cohort), flush=True)
        keys = ("em", "f1") if task in QA else ("correct", "legacy_correct") if task == "gsm8k" else ("correct",)
        metrics = {arm: {key: statistics.mean(r["conditions"][arm][key] for r in records) for key in keys} for arm in ARMS}
        comparisons = {
            arm: {key: C.paired([r["conditions"][ARMS[-1]][key] - r["conditions"][arm][key] for r in records])
                  for key in keys}
            for arm in ARMS[:-1]
        }
        timing = {}
        for key in ("receiver_ttft_seconds", "seconds_per_decode_token", "receiver_score_seconds", "peak_allocated_gib"):
            values = [r["timing"][key] for r in records if key in r["timing"]]
            if values:
                timing[key] = dict(n=len(values), median=statistics.median(values), maximum=max(values))
        tasks[task] = dict(n=len(records), metrics=metrics, paired_delta=comparisons, timing=timing)
        identified(f"evaluation/{task}/summary.json", **tasks[task])
        for key in keys:
            values = " | ".join(f"{metrics[arm][key]:.2%}" for arm in ARMS)
            lines.append(f"| {task} | {key} | {values} |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    identified("summary.json", status="completed", tasks=tasks,
               scales=[layer.scales.detach().tanh().tolist() for layer in reader.readers])
    status("complete", status="completed")


def launch():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with (ROOT / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / "pipeline.lock").open("a") as running:
            fcntl.flock(running, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (logs / "pipeline.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(__file__), "run"],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=ROOT, start_new_session=True)
        C.save(ROOT / "launch.json", dict(pid=process.pid, host="server-b", time=stamp()))
        print("LAUNCHED", process.pid, flush=True)


def run():
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("prepare", status="running", pid=os.getpid())
            schedule = prepare()
            while True:
                paused()
                free_mib = torch.cuda.mem_get_info()[0] // 2**20
                if free_mib >= CFG["minimum_free_gpu_mib"]:
                    break
                status("waiting_gpu", status="queued", free_mib=free_mib)
                time.sleep(30)
            engine, reader, versions = load()
            smoke(engine, reader, versions, schedule)
            train(engine, reader, versions, schedule)
            evaluate(engine, reader, versions)
        except SystemExit as error:
            if error.code == 75:
                status("paused", status="paused")
            raise
        except Exception as error:
            status("failed", status="failed", error=repr(error))
            raise


if __name__ == "__main__":
    {"prepare": prepare, "launch": launch, "run": run}[sys.argv[1]]()
