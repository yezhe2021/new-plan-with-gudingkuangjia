"""Summarize the completed frozen Full-Sync pilot without loading any model."""

import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "runs/study"


def exact_mcnemar(first_only, second_only):
    n = first_only + second_only
    if n == 0:
        return 1.0
    low = min(first_only, second_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(low + 1)) / (2 ** n))


def compare(rows, first, second):
    a = [bool(row["conditions"][first]["correct"]) for row in rows]
    b = [bool(row["conditions"][second]["correct"]) for row in rows]
    first_only = sum(x and not y for x, y in zip(a, b))
    second_only = sum(y and not x for x, y in zip(a, b))
    return {"count": len(rows), "first": first, "second": second,
            "first_correct": sum(a), "second_correct": sum(b),
            "both_correct": sum(x and y for x, y in zip(a, b)),
            "first_only_correct": first_only, "second_only_correct": second_only,
            "both_wrong": sum(not x and not y for x, y in zip(a, b)),
            "second_minus_first": (sum(b) - sum(a)) / len(rows),
            "mcnemar_exact_two_sided_p": exact_mcnemar(first_only, second_only)}


def main():
    rows = [json.loads(line) for line in (ROOT / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        groups[row["dataset"]].append(row)
        fraction = row["unit_counts"].get("copy_target_tokens", 0) / row["target_option_tokens"]
        groups["copy_zero" if fraction == 0 else
               "copy_low_0_to_10pct" if fraction <= 0.1 else "copy_high_over_10pct"].append(row)
    output = {}
    for name, group in groups.items():
        output[name] = {
            "stage_a_pair": compare(group, "rawanchor_stage_a", "fullsync_stage_a"),
            "stage_b_pair": compare(group, "rawanchor_stage_b", "fullsync_stage_b"),
            "stage_a_mean_k_cosine": sum(r["stage_a"]["k_cosine"] for r in group) / len(group),
            "stage_a_mean_v_cosine": sum(r["stage_a"]["v_cosine"] for r in group) / len(group),
            "stage_b_mean_k_cosine": sum(r["stage_b"]["k_cosine"] for r in group) / len(group),
            "stage_b_mean_v_cosine": sum(r["stage_b"]["v_cosine"] for r in group) / len(group),
            "mean_receiver_option_tokens": sum(r["target_option_tokens"] for r in group) / len(group),
        }
    path = ROOT / "paired_analysis.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
