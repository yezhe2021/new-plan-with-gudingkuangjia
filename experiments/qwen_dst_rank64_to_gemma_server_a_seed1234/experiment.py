"""Run the verified DST schedule with a Gemma receiver."""

import sys

import reverse_common as G
import reverse_engine as E
from reverse_common import ARCH, C, CFG, K, N, RESIDUAL, ROOT, TEMPLATE, mark, rows, status, torch


def make(stage, engine=None):
    base = ARCH.DepthStateTranslator(ARCH.DSTConfig(**CFG["model"]), CFG["seed"]).cuda().float()
    if stage == "stage1":
        return base
    record = C.read(ROOT / "runs/stage1/completed.json")
    assert record["checkpoint_sha256"] == C.sha(RUNTIME.checkpoint_path("stage1"))
    RUNTIME.load_stage(base, "stage1")
    base.eval().requires_grad_(False)
    if engine is not None:
        engine.frozen_base, engine.base_versions = base, C.versions(base)
    correction = RESIDUAL.ResidualTranslator(base, layers=34, rank=CFG["rank"], seed=CFG["seed"]).cuda().float()
    assert sum(p.numel() for p in correction.parameters()) == 8912896
    return correction


@torch.no_grad()
def native_prefix(reader, row, native):
    name = f'native_prefix/{row["id"]}.json'
    path = ROOT / name
    if path.exists():
        record = C.read(path)
        assert record["manifest_sha256"] == G.identity() and record["id"] == row["id"]
        return record["tokens"]
    tokens = reader.rollout(row, native, cap=CFG["prefix_max_new_tokens"])
    mark(name, id=row["id"], tokens=tokens)
    return tokens


RUNTIME = G.scoped("reverse_dst_training_runtime", TEMPLATE / "runtime.py",
                   experiment_common=G, engine=E, translator=ARCH, residual=RESIDUAL)
RUNTIME.make = make
RUNTIME.native_prefix = native_prefix


@torch.no_grad()
def native_parity(engine, reader, row):
    key, value, native = engine.teacher(row)
    rebuilt = engine.assemble([engine.finalize(key.cuda(), value.cuda(), len(engine.prefix_ids))])
    header = len(engine.prefix_ids)
    for a, b in zip(rebuilt.layers, native.layers):
        assert a.is_sliding == b.is_sliding and a.get_seq_length() == b.get_seq_length()
        torch.testing.assert_close(a.keys, b.keys, atol=1e-4, rtol=1e-4)
        start = 0 if a.is_sliding else header
        torch.testing.assert_close(a.values[:, :, start:], b.values[:, :, start:], atol=1e-6, rtol=1e-6)
        if not a.is_sliding:
            # The public header was forwarded separately with a different
            # GEMM shape. Body V is still checked with the tighter tolerance.
            torch.testing.assert_close(a.values[:, :, :header], b.values[:, :, :header], atol=2e-4, rtol=1e-4)
    expected = reader.logits(row, native)
    actual = reader.logits(row, rebuilt)
    assert int(expected.argmax()) == int(actual.argmax())
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)
    ids = engine.prefix_ids + row["content_ids"] + row["suffix_ids"]
    direct = engine.forward(ids, None, 0)
    last = engine.model.lm_head(direct.last_hidden_state[0, -1]).float()
    torch.testing.assert_close(last, expected[0], atol=0.015, rtol=0.003)
    length = rebuilt.get_seq_length()
    assert length > engine.model.config.sliding_window
    return dict(absolute_length=length, local_stored=rebuilt.layers[0].keys.shape[2],
                full_stored=next(l.keys.shape[2] for l in rebuilt.layers if not l.is_sliding))


def smoke(stage):
    G.verify()
    name = f"checks/{stage}_gpu_completed.json"
    if G.done(name):
        return
    engine = E.Engine()
    reader = E.BodyReader(engine)
    module = make(stage, engine)
    optimizer = torch.optim.AdamW(module.parameters(), lr=CFG[stage]["learning_rate"], weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64.0, growth_interval=1000000)
    train = rows("train")
    longest = max(train, key=lambda r: len(r["content_ids"]) + len(r["gold_tokens"]))
    if stage == "stage1":
        parity_row = next(r for r in train if 1120 <= len(r["content_ids"]) <= 1250)
        parity = native_parity(engine, reader, parity_row)
        print("Gemma cache parity:", parity, flush=True)
    else:
        parity = None
        with torch.no_grad():
            check_row = next(r for r in train if r["task"] == "arc")
            source = engine.source(check_row)
            baseline = engine.translated(check_row, module.base, source)
            zero = engine.translated(check_row, module, source)
            assert all(torch.equal(a, b) for a, b in zip(K.flat(baseline), K.flat(zero)))
            del source, baseline, zero
    _, long = RUNTIME.backward_batch(engine, reader, module, optimizer, scaler, [(longest, "gold")], stage, 1)
    _, batch = RUNTIME.backward_batch(engine, reader, module, optimizer, scaler, RUNTIME.batches(1, stage)[0], stage, 1)
    engine.unchanged()
    mark(name, status="completed", parity=parity, longest=long, batch=batch,
         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, disposable=True)
    print("Passed", stage, "long and batch-8 updates", flush=True)


def train(stage):
    trainer = G.scoped("reverse_dst_trainer", TEMPLATE / "train.py",
                       experiment_common=G, engine=E, runtime=RUNTIME)
    trainer.main(stage)


def evaluation_metrics():
    hot = G.scoped("reverse_hotpot_metrics", C.ROOT / "metrics.py")
    qa = G.scoped("reverse_qa_metrics", N.ORIGINAL / "cross_dataset5x64_seed1234" / "qa_metrics.py")
    numeric = G.scoped("reverse_gsm_numeric", N.GSM / "numeric.py")
    return hot, qa, numeric


def score(reader, row, cache, metrics):
    task = row["task"]
    if task in ("arc", "sciq", "mmlu_pro", "hellaswag"):
        raw, lengths = [], []
        for candidate in row["candidates"]:
            tokens = reader.engine.tok.encode(candidate, add_special_tokens=False)
            assert tokens
            logits = reader.token_logits(row, cache, tokens, rn=False)
            ids = torch.tensor(tokens, device=logits.device)
            raw.append(float(logits.log_softmax(-1).gather(-1, ids[:, None]).sum()))
            lengths.append(len(tokens))
        scores = [value / n for value, n in zip(raw, lengths)] if row.get("normalize") else raw
        prediction = max(range(len(scores)), key=scores.__getitem__)
        return dict(scores=scores, raw_sum_scores=raw, candidate_token_lengths=lengths,
                    prediction=prediction, correct=int(prediction == row["gold"]))
    hot, qa, numeric = metrics
    cap = CFG["evaluation_generation_caps"][task]
    tokens = reader.rollout(row, cache, cap=cap)
    text = reader.engine.tok.decode(tokens, skip_special_tokens=True)
    stopped = tokens[-1] in reader.engine.stop_ids
    details = dict(generation=text, token_ids=tokens, generated_tokens=len(tokens),
                   reached_cap=int(len(tokens) == cap and not stopped))
    if task == "gsm8k":
        answer = numeric.extract(text)
        legacy = C.SC.extract_numeric_answer(text)
        gold = numeric.normalize(row["gold"])
        return dict(**details, **answer, correct=int(answer["prediction"] == gold),
                    legacy_correct=int(numeric.normalize(legacy) == gold), legacy_prediction=legacy)
    prediction = hot.clean_prediction(text)
    scores = hot.answer_metrics(prediction, row["gold"]) if task == "hotpotqa" else qa.answer_metrics(prediction, row["answers"], task)
    return dict(**details, prediction=prediction, **scores)


@torch.no_grad()
def evaluate():
    G.verify()
    if G.done("completed.json"):
        return
    engine = E.Engine()
    engine.mode = "test"
    reader = E.BodyReader(engine)
    rank64 = make("stage2", engine)
    RUNTIME.load_stage(rank64, "stage2")
    rank64.eval().requires_grad_(False)
    metrics = evaluation_metrics()
    tasks = {}
    all_rows = rows("test")
    for task, count in CFG["test_counts"].items():
        records = []
        for i, row in enumerate([r for r in all_rows if r["task"] == task], 1):
            if G.paused():
                raise SystemExit(75)
            path = f'evaluation/{task}/per_sample/{row["id"]}.json'
            if not (ROOT / path).exists():
                source = engine.source(row)
                values = {}
                for name, module in (("DST_stage1", rank64.base), ("DST_rank64", rank64)):
                    cache = engine.translated(row, module, source, checkpointed=False)
                    values[name] = score(reader, row, cache, metrics)
                    del cache
                del source
                native = engine.native_cache(row, baseline=True)
                values["native_gemma"] = score(reader, row, native, metrics)
                del native
                assert engine.student_test_native_prefills == 0
                mark(path, id=row["id"], task=task, question=row["question"], gold=row["gold"], conditions=values)
            record = C.read(ROOT / path)
            assert record["manifest_sha256"] == G.identity() and record["id"] == row["id"]
            records.append(record)
            engine.unchanged()
            status(status="running", phase="evaluation", task=task, sample=i, total=count)
            if i == 1 or i % 8 == 0:
                print("Gemma receiver eval", task, i, count, flush=True)
        assert len(records) == count
        keys = ("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else (
            ("correct", "legacy_correct") if task == "gsm8k" else ("correct",))
        scores = {name: {k: sum(r["conditions"][name][k] for r in records) / count for k in keys}
                  for name in records[0]["conditions"]}
        pairs = {name: {k: C.paired([r["conditions"]["DST_rank64"][k] - r["conditions"][name][k] for r in records])
                        for k in keys} for name in ("DST_stage1", "native_gemma")}
        tasks[task] = dict(n=count, metrics=scores, rank64_minus=pairs)
        mark(f"evaluation/{task}/summary.json", status="completed", **tasks[task])
    mark("summary.json", status="completed", tasks=tasks)
    mark("completed.json", status="completed")
    status(status="completed")


if __name__ == "__main__":
    {"smoke": smoke, "train": train, "evaluate": evaluate}[sys.argv[1]](*sys.argv[2:])
