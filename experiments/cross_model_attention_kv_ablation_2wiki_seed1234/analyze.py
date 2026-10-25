"""Calibration-only rectangular head matching and held-out result summaries."""
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from metrics import normalize, compare, bootstrap
from experiment import ROOT, CFG, rows, read, save, identity, verify, load_probe, log, status


def entropy(p):
    return -np.sum(p * np.log2(np.maximum(p, 1e-300)), axis=-1)


def cost_matrix(a, b):
    a, b = normalize(a), normalize(b)
    return np.maximum(0, entropy((a[:, None] + b[None, :]) / 2) - (entropy(a)[:, None] + entropy(b)[None, :]) / 2)


def flatten(metric):
    values = {"spearman": metric["spearman"], "jsd_bits": metric["jsd_bits"]}
    for fraction in CFG["attention_fractions"]:
        for key in ["overlap", "chance_adjusted", "expected_overlap_under_ties"]:
            values[f"overlap_{fraction}_{key}"] = metric[str(fraction)][key]
    return values


def mean_metrics(metrics):
    metrics = list(metrics)
    return {key: (float(np.mean(v)) if (v := [m[key] for m in metrics if m[key] is not None]) else None)
            for key in metrics[0]}


def match():
    verify()
    calibration = [r for r in rows() if r["split"] == "calibration"]
    pairs = []
    for a, b in itertools.combinations(CFG["models"], 2):
        aa = [load_probe(a, r) for r in calibration]
        bb = [load_probe(b, r) for r in calibration]
        la, ha = aa[0].shape[:2]
        lb, hb = bb[0].shape[:2]
        n = min(la, lb)
        depth_pairs = [(int(round(i * (la - 1) / (n - 1))), int(round(i * (lb - 1) / (n - 1)))) for i in range(n)]
        assert len(set(x for x, _ in depth_pairs)) == n and len(set(y for _, y in depth_pairs)) == n
        layer_records = []
        candidates = []
        for depth_index, (l1, l2) in enumerate(depth_pairs):
            cost = np.mean([cost_matrix(x[l1], y[l2]) for x, y in zip(aa, bb)], axis=0)
            mass1 = np.mean([x[l1].sum(-1) for x in aa], axis=0)
            mass2 = np.mean([y[l2].sum(-1) for y in bb], axis=0)
            active1, active2 = np.flatnonzero(mass1 > 1e-8), np.flatnonzero(mass2 > 1e-8)
            assert len(active1) and len(active2), ("No evidence-attending heads", a, b, depth_index)
            left, right = linear_sum_assignment(cost[np.ix_(active1, active2)])
            assignments = []
            for i, j in zip(active1[left], active2[right]):
                item = dict(indices=[[l1, int(i)], [l2, int(j)]], calibration_jsd_bits=float(cost[i, j]),
                            evidence_mass=[float(mass1[i]), float(mass2[j])])
                assignments.append(item)
                if depth_index >= (2 * n // 3):
                    candidates.append(item)
            layer_records.append(dict(layers=[l1, l2], assignments=assignments, total_heads=[ha, hb],
                                      unmatched_heads=[ha - len(assignments), hb - len(assignments)]))
        strong = [x for x in candidates if min(x["evidence_mass"]) >= 1e-4]
        representative = min(strong or candidates, key=lambda x: x["calibration_jsd_bits"])
        pairs.append(dict(models=[a, b], depth_pairs=layer_records, functional_head_pair=representative,
            selection="Minimum calibration JSD among late-third matched head pairs; prefer both evidence masses >=1e-4. No evaluation answers used."))
        log(f"MATCHED {a}/{b}: {n} relative-depth pairs; {sum(len(x['assignments']) for x in layer_records)} heads")
    save(ROOT / "matching.json", dict(manifest_sha256=identity(), calibration_ids=[r["id"] for r in calibration], pairs=pairs))


def compare_attention():
    verify()
    matching = read(ROOT / "matching.json")
    assert matching["manifest_sha256"] == identity()
    evaluation = [r for r in rows() if r["split"] == "evaluation"]
    assert not set(matching["calibration_ids"]) & {r["id"] for r in evaluation}
    summaries = []
    raw_path = ROOT / "comparisons/per_head.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.with_suffix(".jsonl.tmp").open("w") as raw_stream:
        for pair in matching["pairs"]:
            a, b = pair["models"]
            for view in CFG["query_views"]:
                per_question = defaultdict(list)
                for row in evaluation:
                    x, y = normalize(load_probe(a, row, view)), normalize(load_probe(b, row, view))
                    per_question["global"].append(flatten(compare(x.mean((0, 1)), y.mean((0, 1)), CFG["attention_fractions"], CFG["seed"])))
                    for band, (ia, ib) in enumerate(zip(np.array_split(np.arange(len(x)), CFG["layer_bands"]),
                                                       np.array_split(np.arange(len(y)), CFG["layer_bands"]))):
                        per_question[f"layer_band_{band}"].append(flatten(compare(x[ia].mean((0, 1)), y[ib].mean((0, 1)), CFG["attention_fractions"], CFG["seed"] + band)))
                    head_results = []
                    for layer in pair["depth_pairs"]:
                        for assignment in layer["assignments"]:
                            (la, ha), (lb, hb) = assignment["indices"]
                            metric = flatten(compare(x[la, ha], y[lb, hb], CFG["attention_fractions"], CFG["seed"] + ha * 100 + hb))
                            raw_stream.write(json.dumps(dict(id=row["id"], models=[a, b], view=view,
                                indices=assignment["indices"], metrics=metric), allow_nan=False) + "\n")
                            head_results.append(metric)
                    per_question["matched_heads_question_mean"].append(mean_metrics(head_results))
                levels = {level: {key: bootstrap([v[key] for v in values], seed=CFG["seed"], repetitions=CFG["bootstrap_replicates"])
                                 for key in values[0]} for level, values in per_question.items()}
                summaries.append(dict(models=[a, b], view=view, question_count=len(evaluation), levels=levels))
    raw_path.with_suffix(".jsonl.tmp").replace(raw_path)
    save(ROOT / "comparisons/summary.json", dict(manifest_sha256=identity(), comparisons=summaries,
        uncertainty="Question bootstrap; heads/layers within a question are not independent observations."))
    log("HELD-OUT ATTENTION COMPARISONS COMPLETED")


def summarize():
    verify()
    evaluation = [r for r in rows() if r["split"] == "evaluation"]
    models = {}
    for family in CFG["models"]:
        assert read(ROOT / f"ablations/{family}/completed.json")["manifest_sha256"] == identity()
        records = [read(ROOT / f"ablations/{family}/{r['id']}.json") for r in evaluation]
        baseline_correct = [bool(r["baseline"]["exact_match"]) for r in records]
        condition_rows = defaultdict(list)
        for record in records:
            base = record["baseline"]
            for value in record["interventions"]:
                item = dict(value, id=record["id"], baseline_correct=base["exact_match"])
                condition_rows[(value["selector"], value["fraction"])].append(item)
            for fraction in CFG["ablation_fractions"]:
                random_rows = [x for x in record["interventions"] if x["selector"].startswith("random_") and x["fraction"] == fraction]
                assert len(random_rows) == CFG["random_replicates"]
                condition_rows[("random_mean", fraction)].append(dict(id=record["id"], baseline_correct=base["exact_match"],
                    delta_nll_per_token=float(np.mean([x["delta_nll_per_token"] for x in random_rows])),
                    delta_nll_per_character=float(np.mean([x["delta_nll_per_character"] for x in random_rows])),
                    exact_match=float(np.mean([x["exact_match"] for x in random_rows]))))
        conditions = []
        for (name, fraction), values in condition_rows.items():
            paired = dict(selector=name, fraction=fraction, n=len(values),
                delta_nll_per_token=bootstrap([v["delta_nll_per_token"] for v in values], CFG["seed"], CFG["bootstrap_replicates"]),
                delta_nll_per_character=bootstrap([v["delta_nll_per_character"] for v in values], CFG["seed"], CFG["bootstrap_replicates"]))
            if "exact_match" in values[0]:
                paired["greedy_accuracy"] = float(np.mean([v["exact_match"] for v in values]))
                paired["accuracy_drop_pp"] = 100 * (np.mean(baseline_correct) - paired["greedy_accuracy"])
                correct_rows = [v for v in values if v["baseline_correct"]]
                paired["baseline_correct_subset_n"] = len(correct_rows)
                paired["correct_subset_failure_rate"] = float(np.mean([1 - v["exact_match"] for v in correct_rows])) if correct_rows else None
            if fraction < 1 and not name.startswith("random_"):
                random_by_id = {v["id"]: v for v in condition_rows[("random_mean", fraction)]}
                paired["paired_damage_minus_random"] = bootstrap([
                    v["delta_nll_per_token"] - random_by_id[v["id"]]["delta_nll_per_token"] for v in values],
                    CFG["seed"], CFG["bootstrap_replicates"])
            conditions.append(paired)
        models[family] = dict(n=len(records), baseline_accuracy=float(np.mean(baseline_correct)),
            baseline_correct=sum(baseline_correct), baseline_capped=sum(r["baseline"]["reached_cap"] for r in records), conditions=conditions)
    result = dict(experiment=CFG["name"], status="completed", timestamp=__import__("time").strftime("%F %T"),
        manifest_sha256=identity(), config=CFG, attention=read(ROOT / "comparisons/summary.json"),
        functional=models, caveats=CFG["caveats"], raw_head_metrics=str(ROOT / "comparisons/per_head.jsonl"),
        raw_interventions=str(ROOT / "ablations"))
    save(ROOT / "summary.json", result)
    # Follow the project's automatic shared-results policy.
    import fcntl
    with (ROOT.parent / "EXPERIMENT_RESULTS.md").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write("\n\n## " + CFG["name"] + "\n\n```json\n" + json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n```\n")
        stream.flush()
        os.fsync(stream.fileno())
    save(ROOT / "completed.json", dict(status="completed", manifest_sha256=identity()))
    status(status="completed", phase="all")
    log("ALL ATTENTION AND KV-ABLATION EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    import sys
    try:
        {"match": match, "compare": compare_attention, "summarize": summarize}[sys.argv[1]]()
    except Exception as exc:
        import traceback
        save(ROOT / "failure.json", dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        status(status="failed", phase=sys.argv[1], error=str(exc))
        raise
