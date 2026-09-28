"""Full-token Sync Boundary copy/drop pilot; no persistent KV cache.

Reuses the frozen Full28-MLP Stage-A and residual Stage-B checkpoints from the
balanced Llama3.2-3B -> Qwen3-4B experiment. This isolates the memory grid:
RawAnchor32/64 versus every Qwen Options token. No new parameters are trained.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "multidataset_obqa_arc_mmlupro_train1024_full28_mlp_l3_to_q4_seed1234"
SYNC = HERE.parent / "sync_boundary_nativekv_full28_diagonal_translator_llama3_2_3b_to_qwen3_4b_seed1234"
sys.path.insert(0, str(BASE))
sys.path.insert(1, str(SYNC))

from common import configuration, load_model, save_json, seed_all, tokenizer  # noqa: E402
from data import manifest_rows  # noqa: E402
from experiment import load_base, load_branch  # noqa: E402
from offsets import token_spans  # noqa: E402
from protocol import capture_decision, final_logits  # noqa: E402
from translator import component_loss  # noqa: E402
from units import build_synchronized_units  # noqa: E402


def source_rank(m, n, target_rank):
    """Ordered copy/drop: rightmost of each group when compressing."""
    if m < 1 or n < 1 or not 0 <= target_rank < n:
        raise ValueError((m, n, target_rank))
    if m > n:
        return math.ceil((target_rank + 1) * m / n) - 1
    return target_rank * m // n


def fullsync_map(row, llama_tok, qwen_tok):
    text = row["body"]
    fields_l, fields_q = row["encoded"]["llama"], row["encoded"]["qwen"]
    source_spans = token_spans(llama_tok, text, fields_l["body"])
    target_spans = token_spans(qwen_tok, text, fields_q["body"])
    units = build_synchronized_units(text, source_spans, target_spans)
    option_indices = list(fields_q["option_token_indices"])
    option_set = set(option_indices)
    mapping, counts = {}, Counter()
    for unit in units:
        source, target = unit.llama_indices, unit.qwen_indices
        m, n = len(source), len(target)
        touched = [index for index in target if index in option_set]
        if not touched:
            continue
        kind = ("one_to_one" if m == n == 1 else
                "copy" if m == 1 else "drop" if n == 1 else "many_to_many")
        counts[kind + "_units"] += 1
        counts[kind + "_target_tokens"] += len(touched)
        for target_rank, target_index in enumerate(target):
            if target_index in option_set:
                if target_index in mapping:
                    raise RuntimeError(f"Duplicate target token {target_index}: {row['id']}")
                mapping[target_index] = source[source_rank(m, n, target_rank)]
    if set(mapping) != option_set:
        missing = sorted(option_set - set(mapping))
        raise RuntimeError(f"Unmapped Qwen Options tokens in {row['id']}: {missing[:16]}")
    source_indices = [mapping[index] for index in option_indices]
    if any(not 0 <= index < len(fields_l["body"]) for index in source_indices):
        raise RuntimeError(f"Invalid source index in {row['id']}")
    return option_indices, source_indices, dict(counts)


def records_summary(records):
    groups = defaultdict(list)
    groups["overall"] = records
    for record in records:
        groups[record["dataset"]].append(record)
    conditions = ("qwen_full_native", "fullsync_native_oracle", "fullsync_stage_a",
                  "fullsync_stage_b", "rawanchor_stage_a", "rawanchor_stage_b")
    output = {}
    for dataset, rows in groups.items():
        metrics = {}
        for name in conditions:
            present = [r for r in rows if name in r["conditions"]]
            if present:
                correct = sum(r["conditions"][name]["correct"] for r in present)
                metrics[name] = {"correct": correct, "total": len(present),
                                 "accuracy": correct / len(present)}
        output[dataset] = metrics
    return output


def existing_sparse_predictions():
    path = BASE / "runs/study/results/per_sample_metrics.jsonl"
    if not path.exists():
        return {}
    return {record["id"]: record for record in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def audit(cfg, rows):
    llama_tok, qwen_tok = (tokenizer(cfg["models"][family]) for family in ("llama", "qwen"))
    totals, records = Counter(), []
    for number, row in enumerate(rows, 1):
        target, source, counts = fullsync_map(row, llama_tok, qwen_tok)
        totals.update(counts)
        totals["samples"] += 1
        totals["target_option_tokens"] += len(target)
        totals["unique_source_tokens"] += len(set(source))
        records.append({"id": row["id"], "dataset": row["dataset"],
                        "target_option_tokens": len(target),
                        "unique_source_tokens": len(set(source)), "unit_counts": counts})
        if number % 64 == 0 or number == len(rows):
            print(f"Alignment audit {number}/{len(rows)}", flush=True)
    out = HERE / "runs/study"
    save_json(out / "alignment_audit.json", {"status": "completed", "totals": dict(totals),
                                             "records": records})
    print(json.dumps(dict(totals), indent=2), flush=True)


def translated_tokens(base, adapter, key, value, source_indices, chunk=64):
    output_k, output_v = [], []
    for begin in range(0, len(source_indices), chunk):
        selected = source_indices[begin:begin + chunk]
        sk = key[:, selected].unsqueeze(0).cuda()
        sv = value[:, selected].unsqueeze(0).cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = base(sk, sv)
            if adapter is not None:
                pk, pv, _, _ = adapter(pk, pv)
        output_k.append(pk[0])
        output_v.append(pv[0])
    return torch.cat(output_k, dim=1), torch.cat(output_v, dim=1)


def readout(model, row, question_k, question_v, option_k, option_v):
    key = torch.cat((question_k, option_k), dim=1)
    value = torch.cat((question_v, option_v), dim=1)
    logits = final_logits(model, row["encoded"]["qwen"]["receiver_answer"],
                          key, value, positions=torch.arange(key.shape[1], device="cuda"),
                          suffix_start=key.shape[1])
    return logits[row["encoded"]["qwen"]["choice_ids"]].cpu()


@torch.no_grad()
def evaluate(cfg, rows, limit=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; alignment audit can still run on CPU")
    if limit is not None:
        rows = rows[:limit]
    llama_tok, qwen_tok = (tokenizer(cfg["models"][family]) for family in ("llama", "qwen"))
    llama, qwen = load_model(cfg, "llama"), load_model(cfg, "qwen")
    base, base_path, _ = load_base(cfg, frozen=True)
    _, adapter, branch = load_branch(cfg, "residual")
    sparse = existing_sparse_predictions()
    records = []
    try:
        for number, row in enumerate(rows, 1):
            target_indices, source_indices, counts = fullsync_map(row, llama_tok, qwen_tok)
            source_fields, target_fields = row["encoded"]["llama"], row["encoded"]["qwen"]
            source_k, source_v, _, _ = capture_decision(
                llama, source_fields["full"], len(source_fields["body"]), source_fields["choice_ids"])
            native_k, native_v, _, full_logits = capture_decision(
                qwen, target_fields["full"], len(target_fields["body"]), target_fields["choice_ids"])
            question_length = target_fields["question_prefix_length"]
            question_k, question_v = native_k[:, :question_length].cuda(), native_v[:, :question_length].cuda()
            oracle_k, oracle_v = native_k[:, target_indices].cuda(), native_v[:, target_indices].cuda()
            stage_a_k, stage_a_v = translated_tokens(base, None, source_k, source_v, source_indices)
            stage_b_k, stage_b_v = translated_tokens(base, adapter, source_k, source_v, source_indices)
            logits = {
                "qwen_full_native": full_logits,
                "fullsync_native_oracle": readout(qwen, row, question_k, question_v, oracle_k, oracle_v),
                "fullsync_stage_a": readout(qwen, row, question_k, question_v, stage_a_k, stage_a_v),
                "fullsync_stage_b": readout(qwen, row, question_k, question_v, stage_b_k, stage_b_v),
            }
            conditions = {name: {"prediction": int(value.argmax()),
                                 "correct": int(value.argmax()) == row["gold_index"]}
                          for name, value in logits.items()}
            previous = sparse.get(row["id"])
            if previous:
                for old, new in (("stage_a", "rawanchor_stage_a"),
                                 ("residual", "rawanchor_stage_b")):
                    conditions[new] = previous["conditions"][old]
            _, a_kn, a_kc = component_loss(stage_a_k, oracle_k)
            _, a_vn, a_vc = component_loss(stage_a_v, oracle_v)
            _, b_kn, b_kc = component_loss(stage_b_k, oracle_k)
            _, b_vn, b_vc = component_loss(stage_b_v, oracle_v)
            records.append({"id": row["id"], "dataset": row["dataset"],
                            "gold_index": row["gold_index"], "target_option_tokens": len(target_indices),
                            "unique_source_tokens": len(set(source_indices)), "unit_counts": counts,
                            "stage_a": {"k_nmse": a_kn.item(), "v_nmse": a_vn.item(),
                                        "k_cosine": a_kc.item(), "v_cosine": a_vc.item()},
                            "stage_b": {"k_nmse": b_kn.item(), "v_nmse": b_vn.item(),
                                        "k_cosine": b_kc.item(), "v_cosine": b_vc.item()},
                            "conditions": conditions})
            del source_k, source_v, native_k, native_v, question_k, question_v
            del oracle_k, oracle_v, stage_a_k, stage_a_v, stage_b_k, stage_b_v
            if number % 16 == 0 or number == len(rows):
                print(f"Full-Sync frozen evaluation {number}/{len(rows)}", flush=True)
    finally:
        del llama, qwen, base, adapter
        torch.cuda.empty_cache()
    out = HERE / "runs/study"
    summary = {"status": "completed", "type": "frozen_checkpoint_fullsync_pilot",
               "protocol": cfg["protocol"], "split": "test", "count": len(rows),
               "base_checkpoint": str(base_path),
               "residual_checkpoint": branch["best_accuracy"]["path"],
               "note": "No full-token retraining; directly tests whether existing token-wise Writer generalizes to a full Receiver Options grid.",
               "metrics": records_summary(records)}
    save_json(out / ("smoke.json" if limit else "evaluation.json"), summary)
    path = out / ("smoke_per_sample.jsonl" if limit else "per_sample.jsonl")
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "smoke", "evaluate"))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    cfg = configuration("study")
    seed_all(cfg["seed"])
    rows = manifest_rows(cfg, "test")
    if args.action == "audit":
        audit(cfg, rows)
    elif args.action == "smoke":
        evaluate(cfg, rows, limit=args.limit)
    else:
        evaluate(cfg, rows)


if __name__ == "__main__":
    main()
