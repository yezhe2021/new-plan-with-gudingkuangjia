"""Three-arm frozen evaluation of paired long-context cohorts."""

import argparse
import importlib.util
import json
import re
from pathlib import Path
import sys
import time

import numpy as np

from build_cohort import ROOT, CORE, LENGTHS, COUNT, file_hash, jsonl, save

E4 = ROOT.parent / "dst_e4_quality_latency_seed1234"
sys.path.insert(0, str(E4))
from benchmark import Benchmark, Stopwatch, torch

spec = importlib.util.spec_from_file_location("long_context_hotpot_metrics", CORE / "metrics.py")
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
answer_metrics, normalize = metrics_module.answer_metrics, metrics_module.normalize

ARMS = ("native", "recolor_c", "dst_rank64")
CAP = 64


class LongContextBenchmark(Benchmark):
    def align(self, row, method, sender_offsets):
        if method == "dst_rank64":
            indices = row["sync_source_indices"]
            assert len(indices) == row["actual_length"]
            assert len(sender_offsets) == row["sync_audit"]["source_tokens"]
            assert 0 <= indices[0] <= indices[-1] < len(sender_offsets)
            return indices
        return super().align(row, method, sender_offsets)


def status(task, phase, **details):
    save(ROOT / f"{task}_status.json", dict(time=time.strftime("%F %T"), phase=phase, **details))


def ruler_score(text, gold):
    numbers = re.findall(r"(?<!\d)\d{7}(?!\d)", text)
    return dict(accuracy=int(gold in numbers), strict_em=int(text.strip() == gold))


def helmet_score(text, row):
    prediction = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.I).strip()
    prediction = prediction.splitlines()[0].strip() if prediction else ""
    scores = answer_metrics(prediction, row["gold"])
    pred = normalize(prediction)
    aliases = [normalize(x) for x in row["helmet_answers"]]
    scores["helmet_subem"] = int(any(x and f" {x} " in f" {pred} " for x in aliases))
    return prediction, scores


@torch.no_grad()
def generate(bench, row, cache):
    ids = row["suffix_ids"]
    tokens = []
    for _ in range(CAP):
        output = bench.engine.forward(ids, cache, cache.get_seq_length())
        token = bench.first_token(output)
        tokens.append(token)
        if token in bench.stop_ids:
            break
        cache, ids = output.past_key_values, [token]
    text = bench.engine.tok.decode(tokens, skip_special_tokens=True).strip()
    return text, tokens


@torch.no_grad()
def evaluate_row(bench, row):
    clock = Stopwatch(False)
    values = {}
    source = None
    for arm in ARMS:
        start = time.monotonic()
        if arm == "native":
            cache = bench.engine.native_cache(row, baseline=True)
        else:
            if source is None:
                source = bench.capture(row, "dst_rank64")
            cache = bench.cache_from_source(row, arm, source, clock)
        assert cache.get_seq_length() == bench.header_length + row["actual_length"]
        text, tokens = generate(bench, row, cache)
        del cache
        if row["task"] == "ruler":
            prediction, metrics = text, ruler_score(text, row["gold"])
        else:
            prediction, metrics = helmet_score(text, row)
        values[arm] = dict(prediction=prediction, generation=text, generated_tokens=len(tokens),
                           reached_cap=int(len(tokens) == CAP and tokens[-1] not in bench.stop_ids),
                           elapsed_seconds=time.monotonic() - start, **metrics)
    bench.engine.unchanged()
    assert bench.engine.student_test_native_prefills == 0
    return values


def intervals(native, candidate, seed=1234):
    x, y = np.asarray(native), np.asarray(candidate)
    ix = np.random.default_rng(seed).integers(0, len(x), (4000, len(x)))
    dx, dy = x[ix].mean(1), y[ix].mean(1)
    ratios = (dy[dx > 0] / dx[dx > 0]).tolist()
    return dict(delta=float((y - x).mean()), delta_ci95=np.quantile(dy - dx, [.025, .975]).tolist(),
                retention=float(y.mean() / x.mean()) if x.mean() > 0 else None,
                retention_ci95=np.quantile(ratios, [.025, .975]).tolist() if ratios else None,
                zero_native_bootstrap_draws=int((dx == 0).sum()))


def summarize(task, records, manifest):
    keys = ("accuracy", "strict_em") if task == "ruler" else ("f1", "em", "helmet_subem")
    results = {}
    for n in LENGTHS:
        group = records[n]
        if not group:
            continue
        metrics = {arm: {k: sum(r["conditions"][arm][k] for r in group) / len(group) for k in keys}
                   for arm in ARMS}
        paired = {arm: {k: intervals([r["conditions"]["native"][k] for r in group],
                                     [r["conditions"][arm][k] for r in group]) for k in keys}
                  for arm in ARMS[1:]}
        results[n] = dict(n=len(group), metrics=metrics, versus_native=paired)
    finished = all(len(records[n]) == COUNT for n in LENGTHS)
    value = dict(status="completed" if finished else "running", task=task, manifest_sha256=manifest,
                 primary_metric=keys[0], completed=sum(len(v) for v in records.values()), total=COUNT * len(LENGTHS),
                 lengths=results)
    if finished:
        reference = [r["id"] for r in records[LENGTHS[0]]]
        assert all([r["id"] for r in records[n]] == reference for n in LENGTHS)
        ix = np.random.default_rng(1234).integers(0, COUNT, (4000, COUNT))
        changes = {}
        for arm in ARMS[1:]:
            means, draws = {}, {}
            for n in (LENGTHS[0], LENGTHS[-1]):
                a = np.array([r["conditions"][arm][keys[0]] for r in records[n]])
                b = np.array([r["conditions"]["native"][keys[0]] for r in records[n]])
                da, db = a[ix].mean(1), b[ix].mean(1)
                means[n] = float(a.mean() / b.mean()) if b.mean() > 0 else None
                draws[n] = np.divide(da, db, out=np.full_like(da, np.nan), where=db > 0)
            lo, hi = LENGTHS[0], LENGTHS[-1]
            difference = draws[hi] - draws[lo]
            finite = difference[np.isfinite(difference)]
            changes[arm] = dict(delta_8k_minus_1k=means[hi] - means[lo] if None not in means.values() else None,
                                ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None)
        value["paired_retention_change"] = changes
    save(ROOT / "evaluation" / task / "summary.json", value)


@torch.no_grad()
def main(task):
    cohort = ROOT / "prepared" / task
    data = json.loads((cohort / "manifest.json").read_text())
    assert data["rows_sha256"] == file_hash(cohort / "rows.jsonl")
    previous = json.loads((E4 / "run_manifest.json").read_text())
    assert previous["code_sha256"] == file_hash(E4 / "benchmark.py")
    output = ROOT / "evaluation" / task
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    bench = LongContextBenchmark()
    bench.chunk = 128
    eos = bench.model.generation_config.eos_token_id
    bench.stop_ids = set(eos if isinstance(eos, list) else [eos])
    bench.stop_ids.add(bench.engine.tok.convert_tokens_to_ids("<|im_end|>"))
    assert bench.engine.prefix_ids == data["public_header_ids"]
    version = dict(
        cohort_sha256=file_hash(cohort / "manifest.json"), evaluator_sha256=file_hash(Path(__file__)),
        cache_code_sha256=previous["code_sha256"], dst_checkpoint_sha256=previous["dst_checkpoint_sha256"],
        recolor_checkpoint_sha256=previous["recolor_checkpoint_sha256"], chunk=bench.chunk, cap=CAP,
        decoding="Qwen greedy, same stop IDs and no thinking suffix for every arm",
        device=torch.cuda.get_device_name(),
    )
    if path.exists():
        assert json.loads(path.read_text()) == version
    else:
        save(path, version)
    manifest = file_hash(path)
    all_rows = list(jsonl(cohort / "rows.jsonl"))
    assert len(all_rows) == COUNT * len(LENGTHS)
    records = {n: [] for n in LENGTHS}
    done = 0
    # Probe the longest grid before starting the fixed sweep.
    first_long = next(r for r in all_rows if r["nominal_length"] == LENGTHS[-1])
    ordered = [first_long, *[r for r in all_rows if r is not first_long]]
    for row in ordered:
        n = row["nominal_length"]
        if (ROOT / "controls" / "pause_requested.json").exists():
            status(task, "paused", completed=done, total=len(all_rows))
            raise SystemExit(75)
        result_dir = output / str(n)
        result_dir.mkdir(exist_ok=True)
        result_path = result_dir / (row["id"] + ".json")
        status(task, "running", length=n, sample_id=row["id"], completed=done, total=len(all_rows))
        if result_path.exists():
            record = json.loads(result_path.read_text())
            assert record["manifest_sha256"] == manifest and record["id"] == row["id"]
        else:
            conditions = evaluate_row(bench, row)
            record = dict(id=row["id"], question=row["question"], gold=row["gold"], length=n,
                          actual_length=row["actual_length"], evidence_positions=row["evidence_positions"],
                          manifest_sha256=manifest, conditions=conditions)
            save(result_path, record)
        records[n].append(record)
        done += 1
        if done == 1 or done % 10 == 0 or len(records[n]) == COUNT:
            summarize(task, records, manifest)
            print(task, f"{done}/{len(all_rows)}", "length", n, f"{len(records[n])}/{COUNT}", flush=True)
    summarize(task, records, manifest)
    status(task, "completed", completed=done, total=len(all_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("ruler", "helmet"))
    main(parser.parse_args().task)
