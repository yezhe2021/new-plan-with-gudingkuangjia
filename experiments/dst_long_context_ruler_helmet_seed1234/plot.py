"""Plot paired quality retention after both tasks finish."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, layout="constrained")
    for axis, task, title in zip(axes, ("ruler", "helmet"),
                                ("RULER single-needle recall", "HELMET HotpotQA (paired evidence)")):
        summary = json.loads((ROOT / "evaluation" / task / "summary.json").read_text())
        assert summary["status"] == "completed"
        lengths = sorted(int(n) for n in summary["lengths"])
        metric = summary["primary_metric"]
        for arm, label, color in (("dst_rank64", "DST + Rank64", "#205d9e"), ("recolor_c", "Recolor + C", "#aa632a")):
            comparisons = [summary["lengths"][str(n)]["versus_native"][arm][metric] for n in lengths]
            med = [x["retention"] if x["retention"] is not None else np.nan for x in comparisons]
            lower, upper = zip(*[(x["retention_ci95"] or [np.nan, np.nan]) for x in comparisons])
            axis.plot(lengths, med, "o-", color=color, label=label)
            axis.fill_between(lengths, lower, upper, color=color, alpha=0.14)
        axis.axhline(1, color="#666666", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(lengths, [f"{n // 1024}K" for n in lengths])
        axis.set_title(title)
        axis.set_xlabel("Qwen body tokens")
    axes[0].set_ylabel("Score / Qwen Native score")
    axes[1].legend(frameon=False)
    fig.savefig(ROOT / "quality_retention.png", dpi=200)


if __name__ == "__main__":
    main()
