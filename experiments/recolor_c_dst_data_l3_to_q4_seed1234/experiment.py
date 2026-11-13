"""Official Recolor + linear R64 attention calibration on the locked DST rows."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
REFERENCE = ROOT.parent / CFG["reference"]
sys.path.insert(0, str(REFERENCE))
import experiment_common as reference
from runtime import batches
from score import score

C, torch = reference.C, reference.torch
BodyReader = reference.Z.DE.BodyReader
from heterofold import recolor
from heterofold.alignment import exact_end_pairs

STAGE2 = CFG["stage2"]
TOTAL = STAGE2["epochs"] * STAGE2["updates_per_epoch"]
MAP = ROOT / "maps/recolor.pt"
LAST = ROOT / "runs/last.pt"
FINAL = ROOT / f"runs/epoch{STAGE2['epochs']}_step{TOTAL}.pt"


def status(phase, **details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), phase=phase, **details))


def pause():
    if (ROOT / "controls/pause_requested.json").exists():
        raise SystemExit(75)


def mark(path, **details):
    C.save(path, dict(manifest_sha256=C.sha(ROOT / "manifest.json"), **details))


def check_record(path, **expected):
    record = C.read(path)
    assert record["manifest_sha256"] == C.sha(ROOT / "manifest.json"), path
    for key, value in expected.items():
        assert record[key] == value, (path, key)
    return record


def prepare():
    assert C.ROOT.name == CFG["historical_c"]
    rank_config = reference.CFG
    for key in ("seed", "train_counts", "validation_counts", "test_counts"):
        assert rank_config[key] == CFG[key]
    for key in ("updates_per_epoch", "effective_batch"):
        assert rank_config[key] == STAGE2[key]
    assert rank_config["stage2"]["epochs"] == STAGE2["epochs"]
    arm = C.CFG["arms"]["C"]
    assert arm == dict(name="HeteroFold-R64", rank=STAGE2["rank"], kind="linear", objective="attention")
    for key in ("learning_rate", "weight_decay", "gradient_clip"):
        assert C.CFG[key] == STAGE2[key]
    sources = {REFERENCE / "manifest.json": reference.identity(),
               C.ROOT / "manifest.json": C.identity()}
    datasets = {}
    for split, expected in CFG["prepared_sha256"].items():
        path = REFERENCE / "prepared" / f"{split}.jsonl"
        assert C.sha(path) == expected, path
        sources[path] = expected
        datasets[split] = reference.rows(split)
        assert Counter(row["task"] for row in datasets[split]) == CFG[f"{split}_counts"]
    assert C.read(REFERENCE / "runs/stage2/completed.json")["checkpoint_sha256"] == CFG["rank64_checkpoint_sha256"]
    prefix_ids = C.read(C.ROOT / "manifest.json")["prefix_ids"]
    assert prefix_ids == [151644, 872, 198]
    manifest = ROOT / "manifest.json"
    if manifest.exists():
        verify()
        return
    sender = C.AutoTokenizer.from_pretrained(C.CFG["models"]["sender"], local_files_only=True)
    receiver = C.AutoTokenizer.from_pretrained(C.CFG["models"]["receiver"], local_files_only=True)
    probes = {}
    minimum_pairs = {}
    for split in ("train", "validation"):
        probes[split] = {}
        for row in datasets[split]:
            text, question = row["prompt"], row["question"].strip()
            start = text.rfind(question)
            assert question and start >= 0, (split, row["id"], question)
            enc = receiver(text, add_special_tokens=False, return_offsets_mapping=True)
            assert enc["input_ids"] == row["content_ids"], row["id"]
            assert row["id"] not in probes[split]
            probes[split][row["id"]] = C.HC.question_token_positions(enc["offset_mapping"], start, start + len(question))
            if split == "train":
                src = sender(text, add_special_tokens=False, return_offsets_mapping=True)
                pairs = len(exact_end_pairs(src["offset_mapping"], enc["offset_mapping"]))
                assert pairs >= recolor.MIN_ALIGNED_TOKENS, (row["id"], pairs)
                task = row["task"]
                minimum_pairs[task] = min(pairs, minimum_pairs.get(task, pairs))
    assert not (probes["train"].keys() & probes["validation"].keys())
    C.save(ROOT / "prepared/probes.json", probes)
    paths = [ROOT / name for name in ("config.json", "experiment.py", "run.py", "prepared/probes.json")]
    C.save(manifest, dict(
        config=CFG, prefix_ids=prefix_ids, sources={str(p): h for p, h in sources.items()},
        local_files={str(p.relative_to(ROOT)): C.sha(p) for p in paths},
        minimum_exact_end_pairs=minimum_pairs,
        stage1="Official RAW whitening/recoloring, span1 stride4, equal task moment weights",
        stage2="Original C question-query attention KL and output calibration, fresh linear Rank64",
    ))
    C.log("PREPARED", minimum_pairs, CFG["train_counts"])


def verify():
    record = C.read(ROOT / "manifest.json")
    for name, expected in record["local_files"].items():
        assert C.sha(ROOT / name) == expected, name
    for name, expected in record["sources"].items():
        assert C.sha(name) == expected, name


def fit():
    verify()
    if (ROOT / "maps/completed.json").exists():
        check_record(ROOT / "maps/completed.json", map_sha256=C.sha(MAP))
        return
    pause()
    sender = C.load_model(C.CFG["models"]["sender"], "cuda", dtype=torch.float16)
    receiver = C.load_model(C.CFG["models"]["receiver"], "cuda", dtype=torch.float16)
    rows = reference.rows("train")
    C.log("RECOLOR STAGE1 START", len(rows), "training prompts")
    corpora = {}
    for task in sorted(CFG["train_counts"]):
        texts = [row["prompt"] for row in rows if row["task"] == task]
        status("recolor_statistics", task=task, samples=len(texts))

        def report(message):
            C.log(message)
            status("recolor_statistics", task=task, detail=message)
            pause()

        corpora[task] = recolor.collect_statistics(
            sender, receiver, texts, span=CFG["recolor"]["span"],
            stride=CFG["recolor"]["stride"], device="cuda", log=report, tag=f" {task}:",
        )
    del sender, receiver
    torch.cuda.empty_cache()
    status("recolor_solve")
    stats = recolor.balance_corpora(corpora)
    del corpora
    mapping = recolor.fit_map(stats, device="cuda", log=C.log)
    mapping.update(sender="llama3", receiver="qwen4")
    mapping["calibration"].update(
        calibration_id=ROOT.name, train_prompts=len(rows),
        prepared_train_sha256=CFG["prepared_sha256"]["train"],
    )
    assert all(torch.isfinite(t).all() for role in "kv" for field in ("A", "b") for t in mapping[f"{field}_{role}"])
    C.atomic_save(mapping, MAP)
    mark(ROOT / "maps/completed.json", map_sha256=C.sha(MAP), calibration=mapping["calibration"])
    C.log("RECOLOR STAGE1 COMPLETE", MAP)


@contextmanager
def using_new_map():
    check_record(ROOT / "maps/completed.json", map_sha256=C.sha(MAP))
    original = C.CFG["recolor"]
    C.CFG["recolor"] = str(MAP)
    try:
        yield
    finally:
        C.CFG["recolor"] = original


def example(engine, row, probes):
    return engine.attention_example(dict(row, probe_positions=probes[row["id"]]), engine.base(row))


def same_cache(left, right):
    return all(torch.equal(a, b) for layer in range(36)
               for a, b in zip(C._cache_layer(left, layer), C._cache_layer(right, layer)))


@torch.no_grad()
def smoke(engine, module, probes):
    rows = reference.rows("train")
    for task in CFG["train_counts"]:
        row = next(row for row in rows if row["task"] == task)
        base = engine.base(row)
        uncorrected = engine.cache(row, base, None)
        offsets, k, v = C.capture_sender(engine.sender_tok, engine.sender, engine.mapping, row["prompt"], device="cpu")
        official, length = C.build_receiver_cache(engine.tok, engine.model, engine.mapping, offsets, k, v, row["prompt"], engine.prefix_ids)
        assert length == len(engine.prefix_ids) + len(row["content_ids"])
        assert same_cache(uncorrected, official)
        del official, k, v
        engine.norm_counts = [0] * 36
        assert same_cache(uncorrected, engine.cache(row, base, module))
        assert engine.norm_counts == [1] * 36
        engine.norm_counts = None
        ex = engine.attention_example(dict(row, probe_positions=probes[row["id"]]), base)
        own = C.attention_objective(engine, module, ex)
        official_loss = C.HC.example_objective(engine.model, module, ex, engine.prefix_keys, engine.prefix_values,
                                                lambda_k_output=C.CFG["lambda_k_output"], backward=False)
        for name in ("routing_kl", "k_post_wo_mse", "post_wo_nmse"):
            assert abs(own[name] - official_loss[name]) < 0.002 + 0.02 * abs(official_loss[name]), (task, name)
        C.log("C SMOKE PASS", task, row["id"])
    engine.unchanged()
    mark(ROOT / "checks/cache.json", status="completed", tasks=list(CFG["train_counts"]))


def train_batch(engine, module, optimizer, scaler, batch, probes):
    for retry in range(9):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for row, _ in batch:  # The DST answer-prefix labels are irrelevant to C.
            ex = example(engine, row, probes)
            values.append(C.attention_objective(engine, module, ex, scaler=scaler, divisor=len(batch)))
            del ex
        scaler.unscale_(optimizer)
        grads = [p.grad for p in module.parameters()]
        assert all(grad is not None for grad in grads)
        if all(torch.isfinite(grad).all() for grad in grads):
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), STAGE2["gradient_clip"])
            assert torch.isfinite(norm) and norm > 0
            scaler.step(optimizer)
            scaler.update()
            return mean(values), float(norm), retry
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() < scale
        C.log("C AMP RETRY", retry + 1, scale, scaler.get_scale())
    raise FloatingPointError("C gradients overflowed after nine same-batch attempts")


def mean(values):
    return {key: sum(value[key] for value in values) / len(values) for key in values[0]}


@torch.no_grad()
def validate(engine, module, epoch, probes):
    module.eval()
    metrics = defaultdict(list)
    for i, row in enumerate(reference.rows("validation"), 1):
        pause()
        metrics[row["task"]].append(C.attention_objective(engine, module, example(engine, row, probes)))
        if i % 32 == 0:
            status("c_validation", epoch=epoch, sample=i, total=sum(CFG["validation_counts"].values()))
    record = {task: mean(values) for task, values in metrics.items()}
    mark(ROOT / f"validation/epoch{epoch}.json", epoch=epoch, tasks=record)
    C.log("C VALIDATION", epoch, record)


def train():
    verify()
    if (ROOT / "runs/completed.json").exists():
        check_record(ROOT / "runs/completed.json", checkpoint_sha256=C.sha(FINAL))
        return
    pause()
    with using_new_map():
        engine = C.Engine()
    module = C.fresh("C")
    optimizer = torch.optim.AdamW(module.parameters(), lr=STAGE2["learning_rate"], weight_decay=STAGE2["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64., growth_interval=1000000)
    probes = C.read(ROOT / "prepared/probes.json")
    manifest_hash = C.sha(ROOT / "manifest.json")
    map_hash = C.sha(MAP)
    epoch0, batch0, step = 1, 0, 0
    if LAST.exists():
        cp = torch.load(LAST, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == manifest_hash and cp["map_sha256"] == map_hash
        module.load_state_dict(cp["state"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"])
        epoch0, batch0, step = cp["epoch"], cp["batch"], cp["step"]
        del cp
    else:
        smoke(engine, module, probes["train"])
    path = ROOT / "runs/steps.jsonl"
    path.parent.mkdir(exist_ok=True)
    history = [row for row in C.read_jsonl(path) if row["step"] <= step] if path.exists() else []
    assert [row["step"] for row in history] == list(range(1, step + 1))
    C.write_jsonl(path, history)

    def save(epoch, batch):
        C.atomic_save(dict(manifest_sha256=manifest_hash, map_sha256=map_hash, epoch=epoch, batch=batch, step=step,
                           state=module.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), LAST)

    with path.open("a") as log:
        for epoch in range(epoch0, STAGE2["epochs"] + 1):
            module.train()
            for bi, batch in enumerate(batches(epoch, "stage2"), 1):
                if epoch == epoch0 and bi <= batch0:
                    continue
                if (ROOT / "controls/pause_requested.json").exists():
                    save(epoch, bi - 1)
                    pause()
                metrics, norm, retries = train_batch(engine, module, optimizer, scaler, batch, probes["train"])
                step += 1
                engine.unchanged()
                entry = dict(epoch=epoch, step=step, total_steps=TOTAL, batch=bi, metrics=metrics, gradient_norm=norm,
                             amp_retries=retries, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                status("c_training", **entry)
                if step == 1 or step % 4 == 0:
                    C.log("C TRAIN", json.dumps(entry))
                if step <= 2 or step % STAGE2["checkpoint_every"] == 0:
                    save(epoch, bi)
            assert step == epoch * STAGE2["updates_per_epoch"]
            save(epoch, STAGE2["updates_per_epoch"])
            validated = ROOT / f"validation/epoch{epoch}.json"
            if not validated.exists():
                validate(engine, module, epoch, probes["validation"])
            else:
                check_record(validated, epoch=epoch)
    assert step == TOTAL
    C.atomic_save(dict(manifest_sha256=manifest_hash, map_sha256=map_hash, state=module.state_dict(), step=step), FINAL)
    mark(ROOT / "runs/completed.json", status="completed", checkpoint_sha256=C.sha(FINAL), updates=step)


@torch.no_grad()
def evaluate():
    verify()
    if (ROOT / "summary.json").exists():
        check_record(ROOT / "summary.json", status="completed")
        return
    pause()
    completion = check_record(ROOT / "runs/completed.json", checkpoint_sha256=C.sha(FINAL), updates=TOTAL)
    with using_new_map():
        engine = C.Engine()
    engine.mode = "test"
    reader = BodyReader(engine)
    module = C.fresh("C")
    cp = torch.load(FINAL, map_location="cpu", weights_only=True)
    assert cp["manifest_sha256"] == completion["manifest_sha256"] and cp["map_sha256"] == C.sha(MAP)
    module.load_state_dict(cp["state"], strict=True)
    module.eval().requires_grad_(False)
    rows = reference.rows("test")
    summaries = {}
    conditions = ("native", "C", "B_rank64", "recolor_dst_data", "recolor_c_dst_data")
    for task, count in CFG["test_counts"].items():
        records = []
        for i, row in enumerate((r for r in rows if r["task"] == task), 1):
            pause()
            rel = f"evaluation/{task}/per_sample/{row['id']}.json"
            path = ROOT / rel
            if not path.exists():
                previous = C.read(REFERENCE / rel)
                assert previous["manifest_sha256"] == reference.identity()
                assert previous["checkpoint_sha256"] == CFG["rank64_checkpoint_sha256"]
                assert all(previous[key] == row[key] for key in ("id", "question", "gold"))
                base = engine.base(row)
                scores = {key: previous["conditions"][key] for key in conditions[:3]}
                scores["recolor_dst_data"] = score(reader, row, engine.cache(row, base, None))
                scores["recolor_c_dst_data"] = score(reader, row, engine.cache(row, base, module))
                assert engine.student_test_native_prefills == 0
                mark(path, id=row["id"], task=task, gold=row["gold"], question=row["question"],
                     checkpoint_sha256=completion["checkpoint_sha256"], conditions=scores)
                engine.unchanged()
            saved = check_record(path, id=row["id"], checkpoint_sha256=completion["checkpoint_sha256"])
            values = saved["conditions"]
            if task == "hellaswag":
                for value in values.values():
                    raw = value["raw_sum_scores"]
                    value["raw_correct"] = int(max(range(len(raw)), key=raw.__getitem__) == row["gold"])
            records.append(values)
            status("evaluation", task=task, sample=i, total=count)
            if i == 1 or i % 8 == 0:
                C.log("EVAL", task, i, count)
        assert len(records) == count
        keys = ["em", "f1"] if task in ("hotpotqa", "2wiki", "musique") else ["correct"]
        if task == "gsm8k":
            keys += ["legacy_correct"]
        if task == "hellaswag":
            keys += ["raw_correct"]
        metrics = {arm: mean([{key: row[arm][key] for key in keys} for row in records]) for arm in conditions}
        paired = {arm: {key: C.paired([row["recolor_c_dst_data"][key] - row[arm][key] for row in records])
                        for key in keys} for arm in conditions[:-1]}
        summaries[task] = dict(n=count, metrics=metrics, recolor_c_minus=paired)
        mark(ROOT / f"evaluation/{task}/summary.json", **summaries[task])
    lines = ["# Recolor+C on the DST dataset", "",
             "| Task | Metric | Native | Historical C | DST+Rank64 | Recolor (DST data) | Recolor+C (DST data) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task, summary in summaries.items():
        metrics = summary["metrics"]
        for key in metrics["recolor_c_dst_data"]:
            values = " | ".join(f"{metrics[arm][key]:.2%}" for arm in conditions)
            lines.append(f"| {task} | {key} | {values} |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    mark(ROOT / "summary.json", status="completed", tasks=summaries)
    C.log("EVALUATION COMPLETE")


if __name__ == "__main__":
    {"prepare": prepare, "fit": fit, "train": train, "evaluate": evaluate}[sys.argv[1]]()
