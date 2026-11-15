"""Offline native KV patching on frozen DST+Rank64, with matched block budgets."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
REFERENCE = ROOT.parent / CFG["reference"]
spec = importlib.util.spec_from_file_location("oracle_rank64_reference", REFERENCE / "experiment_common.py")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
C, K, torch = ref.C, ref.K, ref.torch
Engine = ref.Z.scoped_module("oracle_rank64_engine", ref.STATIC_ROOT / "engine_frozen.py", ref).Engine
BodyReader = ref.Z.DE.BodyReader
GROUP = CFG["layers_per_block"]
SECTORS = CFG["body_sectors"]
BLOCKS = tuple((first, sector) for first in range(0, 36, GROUP) for sector in range(SECTORS))
ARMS = ("rank64", "native", "fixed", "random", "oracle")


def identity():
    return C.sha(ROOT / "manifest.json")


def mark(path, **details):
    C.save(ROOT / path, dict(manifest_sha256=identity(), **details))


def read_mark(path):
    record = C.read(ROOT / path)
    assert record["manifest_sha256"] == identity(), path
    return record


def status(phase, **details):
    C.save(ROOT / "status.json", dict(phase=phase, time=time.strftime("%F %T"), **details))


def pause():
    if (ROOT / "controls/pause_requested.json").exists():
        raise SystemExit(75)


def verify():
    for path, digest in C.read(ROOT / "manifest.json")["pins"].items():
        assert C.sha(path) == digest, path


def prepare():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    ref.verify()
    assert ref.identity() == CFG["reference_manifest_sha256"]
    assert C.sha(ref.RANK64) == CFG["rank64_checkpoint_sha256"]
    pins = [ROOT / "config.json", *ROOT.glob("*.py"), REFERENCE / "manifest.json", ref.RANK64,
            ref.STATIC_ROOT / "engine_frozen.py", C.ROOT / "metrics.py"]
    pins.extend(ref.STATIC_ROOT / f"prepared/{split}.jsonl" for split in ("validation", "test"))
    splits = {}
    for split, count in (("validation", CFG["development_per_task"]), ("test", CFG["test_per_task"])):
        all_rows = ref.rows(split)
        splits[split] = []
        for task in CFG["tasks"]:
            rows = sorted((r for r in all_rows if r["task"] == task), key=lambda r: r["id"])
            random.Random(CFG["seed"]).shuffle(rows)
            selected = rows[:count]
            assert len(selected) == count and all(len(row["content_ids"]) >= SECTORS for row in selected)
            splits[split].extend(selected)
    assert not ({C.norm(r["question"]) for r in splits["validation"]}
                & {C.norm(r["question"]) for r in splits["test"]})
    for name, rows in splits.items():
        path = ROOT / f"prepared/{name}.jsonl"
        C.write_jsonl(path, rows)
        pins.append(path)
    C.save(ROOT / "manifest.json", dict(
        pins={str(p): C.sha(p) for p in pins}, blocks=BLOCKS,
        cohorts={name: dict(Counter(r["task"] for r in rows)) for name, rows in splits.items()},
        protocol="Patch one 6-layer x floor(body length/4) token block in post-norm/post-RoPE coordinates. Select separate fixed blocks per task on validation using gold NLL.",
    ))
    C.log("ORACLE PREPARED", {name: len(rows) for name, rows in splits.items()})


def patch(engine, original, donor, first_layer, layers, start, width):
    assert original.get_seq_length() == donor.get_seq_length()
    end = start + width
    assert len(engine.prefix_ids) <= start < end <= original.get_seq_length()
    data = K.pairs(original)
    target = K.pairs(donor)
    for layer in range(first_layer, first_layer + layers):
        data[layer] = tuple(torch.cat((old[:, :, :start], new[:, :, start:end], old[:, :, end:]), dim=2)
                            for old, new in zip(data[layer], target[layer]))
    return C.DynamicCache(ddp_cache_data=data, config=engine.model.config)


def block_patch(engine, row, body, native, block):
    first, sector = BLOCKS[block]
    tokens = len(row["content_ids"])
    width = tokens // SECTORS
    start = len(engine.prefix_ids) + sector * tokens // SECTORS
    return patch(engine, body, native, first, GROUP, start, width)


def gold_nll(reader, row, cache):
    gold = row["candidates"][row["gold"]] if row["task"] == "arc" else row["gold"]
    tokens = reader.engine.tok.encode(gold, add_special_tokens=False)
    assert tokens, row["id"]
    logits = reader.token_logits(row, cache, tokens, rn=False)
    value = torch.nn.functional.cross_entropy(logits, torch.tensor(tokens, device=logits.device))
    assert torch.isfinite(value)
    return float(value)


def row_caches(engine, row):
    body = engine.cache(row, engine.base(row), engine.correction)
    native = engine.native_cache(row, baseline=True)
    header = len(engine.prefix_ids)
    assert native.get_seq_length() == body.get_seq_length() == header + len(row["content_ids"])
    for pair, prefix in zip(K.pairs(native), K.pairs(body)):
        for value, original in zip(pair, prefix):
            value[:, :, :header].copy_(original[:, :, :header])
    return body, native


def same_pairs(a, b):
    return all(torch.equal(x, y) for x, y in zip(K.flat(a), K.flat(b)))


def smoke(engine, reader, rows):
    if (ROOT / "checks/completed.json").exists():
        read_mark("checks/completed.json")
        return
    for task in CFG["tasks"]:
        row = next(r for r in rows if r["task"] == task)
        body, native = row_caches(engine, row)
        first, sector = BLOCKS[-1]
        start = len(engine.prefix_ids) + sector * len(row["content_ids"]) // SECTORS
        width = len(row["content_ids"]) // SECTORS
        mixed = block_patch(engine, row, body, native, len(BLOCKS) - 1)
        for layer, (base, target, actual) in enumerate(zip(K.pairs(body), K.pairs(native), K.pairs(mixed))):
            for b, t, m in zip(base, target, actual):
                if layer >= first:
                    assert torch.equal(b[:, :, :start], m[:, :, :start])
                    assert torch.equal(t[:, :, start:start+width], m[:, :, start:start+width])
                    assert torch.equal(b[:, :, start+width:], m[:, :, start+width:])
                else:
                    assert torch.equal(b, m)
        same = block_patch(engine, row, body, body, 0)
        assert same_pairs(same, body)
        assert torch.equal(reader.logits(row, body, rn=False), reader.logits(row, same, rn=False))
        endpoint = patch(engine, body, native, 0, 36, len(engine.prefix_ids), len(row["content_ids"]))
        assert same_pairs(endpoint, native)
        assert torch.equal(reader.logits(row, native, rn=False), reader.logits(row, endpoint, rn=False))
        engine.unchanged()
        C.log("ORACLE SMOKE PASS", task, row["id"])
    mark("checks/completed.json", status="completed", tasks=CFG["tasks"])


def scan(engine, reader, row, split):
    key = f'scan/{split}/{row["id"]}.json'
    if (ROOT / key).exists():
        return read_mark(key)
    body, native = row_caches(engine, row)
    base_loss = gold_nll(reader, row, body)
    losses = []
    for i in range(len(BLOCKS)):
        pause()
        losses.append(gold_nll(reader, row, block_patch(engine, row, body, native, i)))
    mark(key, id=row["id"], task=row["task"], base_nll=base_loss,
         native_nll=gold_nll(reader, row, native), patched_nll=losses)
    engine.unchanged()
    return read_mark(key)


def evaluate_choices(engine, reader, row, scan_row, fixed):
    path = f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
    if (ROOT / path).exists():
        return read_mark(path)
    losses = scan_row["patched_nll"]
    seed = int(hashlib.sha256(f'{CFG["seed"]}:{row["id"]}'.encode()).hexdigest()[:16], 16)
    choices = dict(fixed=fixed, random=random.Random(seed).randrange(len(BLOCKS)),
                   oracle=min(range(len(BLOCKS)), key=losses.__getitem__))
    body, native = row_caches(engine, row)
    scores = {name: reader.score(row, cache, rn=False)
              for name, cache in (("rank64", body), ("native", native))}
    for name, block in choices.items():
        pause()
        scores[name] = reader.score(row, block_patch(engine, row, body, native, block), rn=False)
    nll = dict(rank64=scan_row["base_nll"], native=scan_row["native_nll"],
               random_expected=sum(losses) / len(losses), **{arm: losses[i] for arm, i in choices.items()})
    mark(path, id=row["id"], choices=choices, nll=nll, scores=scores)
    engine.unchanged()
    return read_mark(path)


def summarize(results, fixed_blocks):
    lines = ["# Rank64 Oracle KV block scan", "",
             "| Task | Metric | Rank64 | Native | Fixed | Random | Per-question oracle |",
             "|---|---|---:|---:|---:|---:|---:|"]
    tasks = {}
    for task in CFG["tasks"]:
        group = results[task]
        binary = "correct" if task == "arc" else "em"
        keys = ("nll", binary) if task == "arc" else ("nll", "em", "f1")
        metrics, pairs, flips = {}, {}, {}
        for key in keys:
            if key == "nll":
                values = {arm: [r["nll"][arm] for r in group] for arm in (*ARMS, "random_expected")}
            else:
                values = {arm: [r["scores"][arm][key] for r in group] for arm in ARMS}
            metrics[key] = {arm: sum(v) / len(group) for arm, v in values.items()}
            pairs[key] = {f"{a}_minus_{b}": C.paired([x-y for x,y in zip(values[a], values[b])])
                          for a,b in (("fixed","rank64"), ("random","rank64"), ("oracle","rank64"),
                                      ("oracle","fixed"), ("oracle","random"))}
            if key == binary:
                for arm in ("fixed", "random", "oracle"):
                    flips[arm] = dict(
                        rescued=sum(a>b for a,b in zip(values[arm], values["rank64"])),
                        harmed=sum(a<b for a,b in zip(values[arm], values["rank64"])),
                    )
            formatted = " | ".join(f"{metrics[key][arm]:.4f}" if key == "nll" else f"{metrics[key][arm]:.2%}" for arm in ARMS)
            lines.append(f"| {task} | {key} | {formatted} |")
        tasks[task] = dict(n=len(group), fixed_block=fixed_blocks[task], metrics=metrics, paired=pairs, flips=flips,
                           oracle_blocks=dict(Counter(r["choices"]["oracle"] for r in group)))
        mark(f"evaluation/{task}/summary.json", status="completed", **tasks[task])
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    mark("summary.json", status="completed", tasks=tasks)
    status("completed")


@torch.no_grad()
def evaluate():
    verify()
    if (ROOT / "summary.json").exists():
        read_mark("summary.json")
        return
    engine = Engine()
    engine.mode = "test"
    reader = BodyReader(engine)
    dev = C.read_jsonl(ROOT / "prepared/validation.jsonl")
    test = C.read_jsonl(ROOT / "prepared/test.jsonl")
    smoke(engine, reader, dev)
    dev_scans = []
    for i, row in enumerate(dev, 1):
        pause()
        dev_scans.append(scan(engine, reader, row, "validation"))
        status("development", sample=i, total=len(dev), task=row["task"])
        if i == 1 or i % 8 == 0:
            C.log("ORACLE DEVELOPMENT", i, len(dev))
    fixed = {}
    for task in CFG["tasks"]:
        losses = [r["patched_nll"] for r in dev_scans if r["task"] == task]
        fixed[task] = min(range(len(BLOCKS)), key=lambda i: sum(row[i] for row in losses))
    mark("fixed_blocks.json", choices=fixed)
    results = {task: [] for task in CFG["tasks"]}
    for i, row in enumerate(test, 1):
        pause()
        record = scan(engine, reader, row, "test")
        results[row["task"]].append(evaluate_choices(engine, reader, row, record, fixed[row["task"]]))
        status("test", sample=i, total=len(test), task=row["task"])
        if i == 1 or i % 8 == 0:
            C.log("ORACLE TEST", i, len(test))
    summarize(results, fixed)


if __name__ == "__main__":
    {"prepare": prepare, "evaluate": evaluate}[sys.argv[1]]()
