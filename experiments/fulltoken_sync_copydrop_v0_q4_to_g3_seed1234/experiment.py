"""Fresh Qwen3-4B to Gemma3-4B Full-Sync Stage-A/Stage-B experiment.

No generated KV tensors are persisted. Each sample's Qwen and Gemma native KV
is captured on demand, consumed, and released. The test split is never used
for checkpoint selection. The architecture and objectives match the prior
reverse Full36 HeadMix experiment; receiver Options memory uses its full grid.
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from alignment import HERE, fullsync_map

BASE = HERE.parent / "multidataset_obqa_arc_mmlupro_train1024_full36_headmix_q4_to_g3_seed1234"

sys.path.insert(0, str(BASE))
from common import configuration, load_model, save_json, seed_all, tokenizer  # noqa: E402
from data import manifest_rows  # noqa: E402
from protocol import capture_decision, final_logits  # noqa: E402
from translator import NativeKVTranslator, ResidualKVAdapter, component_loss  # noqa: E402

_ROWS = {}
_CACHE = {}
_RECEIVER = None


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def settings(smoke):
    cfg = configuration("study")
    cfg = dict(cfg)
    cfg["smoke"] = bool(smoke)
    cfg["effective_batch"] = 2 if smoke else cfg["batch_size"]
    cfg["stage_a_epochs"] = 1 if smoke else 4
    cfg["stage_b_epochs"] = 1 if smoke else 2
    cfg["train_limit"] = 2 if smoke else None
    cfg["validation_limit"] = 2 if smoke else None
    cfg["test_limit"] = 2 if smoke else None
    cfg["chunk_tokens"] = 64
    cfg["result_root"] = str(HERE / "runs" / ("retrain_smoke" if smoke else "retrain"))
    cfg["checkpoint_protocol"] = "qwen3_4b_to_gemma3_4b_fullsync_copydrop_v0_seed1234"
    return cfg


def result_root(cfg):
    return Path(cfg["result_root"])


def rows_for(cfg, split):
    rows = _ROWS.get(split)
    if rows is None:
        rows = manifest_rows(cfg, split)
    limit = cfg[f"{split}_limit"]
    if not limit:
        return rows
    # Smoke the longest target memories; short first rows hide peak-memory bugs.
    return sorted(rows, key=lambda row: len(row["encoded"]["gemma"]["option_token_indices"]),
                  reverse=True)[:limit]


@torch.no_grad()
def pair(model_s, model_r, tok_s, tok_r, row, need_readout=True):
    if row["id"] in _CACHE:
        item = dict(_CACHE[row["id"]])
        if not need_readout:
            return item
        # Question KV is intentionally not retained for the full corpus.  A
        # Gemma FP32 question cache for 3,840 rows exceeds the container's
        # 128-GiB cgroup limit.  Recompute only when Reader logits are needed.
        fields_r = row["encoded"]["gemma"]
        native_k, native_v, _, full_logits = capture_decision(
            model_r, fields_r["full"], len(fields_r["body"]), fields_r["choice_ids"])
        qlength = fields_r["question_prefix_length"]
        item.update(question_k=native_k[:, :qlength].contiguous().half(),
                    question_v=native_v[:, :qlength].contiguous().half(),
                    full_choice_logits=full_logits)
        return item
    target_indices, source_indices, counts = fullsync_map(row, tok_s, tok_r)
    fields_s, fields_r = row["encoded"]["qwen"], row["encoded"]["gemma"]
    source_k, source_v, _, _ = capture_decision(
        model_s, fields_s["full"], len(fields_s["body"]), fields_s["choice_ids"])
    native_k, native_v, _, full_logits = capture_decision(
        model_r, fields_r["full"], len(fields_r["body"]), fields_r["choice_ids"])
    qlength = fields_r["question_prefix_length"]
    return {
        "source_k": source_k[:, source_indices].contiguous(),
        "source_v": source_v[:, source_indices].contiguous(),
        "target_k": native_k[:, target_indices].contiguous(),
        "target_v": native_v[:, target_indices].contiguous(),
        "question_k": native_k[:, :qlength].contiguous(),
        "question_v": native_v[:, :qlength].contiguous(),
        "full_choice_logits": full_logits,
        "unit_counts": counts,
        "option_tokens": len(target_indices),
    }


def mapped_base(base, item, chunk):
    out_k, out_v = [], []
    with torch.no_grad():
        for begin in range(0, item["option_tokens"], chunk):
            stop = begin + chunk
            sk = item["source_k"][:, begin:stop].unsqueeze(0).cuda()
            sv = item["source_v"][:, begin:stop].unsqueeze(0).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pk, pv = base(sk, sv)
            out_k.append(pk[0])
            out_v.append(pv[0])
    return torch.cat(out_k, dim=1), torch.cat(out_v, dim=1)


def stage_a_sample_backward(cfg, module, item, scaler, divisor):
    """All target Options tokens contribute once; chunking changes only memory use."""
    count = item["option_tokens"]
    tk_all, tv_all = item["target_k"], item["target_v"]
    kden = tk_all.float().square().sum().clamp_min(1e-8).item()
    vden = tv_all.float().square().sum().clamp_min(1e-8).item()
    aggregate = 0.0
    for begin in range(0, count, cfg["chunk_tokens"]):
        stop = min(begin + cfg["chunk_tokens"], count)
        sk = item["source_k"][:, begin:stop].unsqueeze(0).cuda()
        sv = item["source_v"][:, begin:stop].unsqueeze(0).cuda()
        tk = tk_all[:, begin:stop].unsqueeze(0).cuda().float()
        tv = tv_all[:, begin:stop].unsqueeze(0).cuda().float()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = module(sk, sv)
        pk, pv = pk.float(), pv.float()
        frac = (stop - begin) / count
        kloss = (pk - tk).square().sum() / kden + frac * (1 - F.cosine_similarity(pk, tk, dim=-1).mean())
        vloss = (pv - tv).square().sum() / vden + frac * (1 - F.cosine_similarity(pv, tv, dim=-1).mean())
        loss = (kloss + vloss) / divisor
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite full-token Stage-A loss")
        scaler.scale(loss).backward()
        aggregate += loss.detach().item() * divisor
    return aggregate


def optimizer_step(cfg, module, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg["clip"])
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite gradient norm")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return norm.item()


def save_checkpoint(path, cfg, module, stage, epoch, step, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": stage, "protocol": cfg["checkpoint_protocol"], "epoch": epoch,
                "step": step, "metrics": metrics,
                "state": {name: value.detach().cpu() for name, value in module.state_dict().items()}}, path)


def load_checkpoint(path, cfg, stage, module):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("stage") != stage or payload.get("protocol") != cfg["checkpoint_protocol"]:
        raise RuntimeError(f"Checkpoint/protocol mismatch: {path}")
    module.load_state_dict(payload["state"], strict=True)
    return module


def choice_kl(student, teacher, temperature):
    student = F.log_softmax(student.float() / temperature, -1)
    teacher = F.log_softmax(teacher.float().to(student.device) / temperature, -1)
    return F.kl_div(student, teacher.detach(), reduction="sum", log_target=True) * temperature ** 2


def native_teacher(receiver, row, item):
    with torch.no_grad():
        return student_logits(receiver, row, item, item["target_k"].cuda(), item["target_v"].cuda())


def student_logits(receiver, row, item, key, value):
    key = torch.cat((item["question_k"].cuda(), key), dim=1)
    value = torch.cat((item["question_v"].cuda(), value), dim=1)
    logits = final_logits(receiver, row["encoded"]["gemma"]["receiver_answer"],
                          key, value, positions=torch.arange(key.shape[1], device="cuda"),
                          suffix_start=key.shape[1])
    indices = torch.tensor(row["encoded"]["gemma"]["choice_ids"], device="cuda")
    return logits.index_select(0, indices)


@torch.no_grad()
def validate(cfg, source, receiver, tok_s, tok_r, base, adapter, split):
    base.eval()
    if adapter is not None:
        adapter.eval()
    rows = rows_for(cfg, split)
    totals = {"correct": 0, "oracle_correct": 0, "agreement": 0,
              "choice_kl": 0.0, "k_nmse": 0.0, "v_nmse": 0.0,
              "k_cosine": 0.0, "v_cosine": 0.0}
    records = []
    for index, row in enumerate(rows, 1):
        item = pair(source, receiver, tok_s, tok_r, row)
        key, value = mapped_base(base, item, cfg["chunk_tokens"])
        if adapter is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                key, value, _, _ = adapter(key[None], value[None])
            key, value = key[0], value[0]
        teacher = native_teacher(receiver, row, item)
        student = student_logits(receiver, row, item, key, value)
        prediction, oracle = int(student.argmax()), int(teacher.argmax())
        totals["correct"] += prediction == row["gold_index"]
        totals["oracle_correct"] += oracle == row["gold_index"]
        totals["agreement"] += prediction == oracle
        totals["choice_kl"] += choice_kl(student, teacher, cfg["temperature"]).item()
        _, kn, kc = component_loss(key, item["target_k"].cuda())
        _, vn, vc = component_loss(value, item["target_v"].cuda())
        for name, metric in (("k_nmse", kn), ("v_nmse", vn), ("k_cosine", kc), ("v_cosine", vc)):
            totals[name] += metric.item()
        if split == "test":
            records.append({"id": row["id"], "dataset": row["dataset"],
                            "gold_index": row["gold_index"], "oracle_prediction": oracle,
                            "prediction": prediction, "option_tokens": item["option_tokens"],
                            "choice_logits": student.float().tolist(),
                            "teacher_choice_logits": teacher.float().tolist()})
        if index % 32 == 0 or index == len(rows):
            log(f"{split} evaluation {index}/{len(rows)}")
        del item, key, value, teacher, student
    metrics = {name: value / len(rows) for name, value in totals.items()}
    metrics["count"] = len(rows)
    return metrics, records


def models_and_tokens(cfg):
    if _RECEIVER is not None:
        return None, _RECEIVER, None, None
    source = load_model(cfg, "qwen")
    receiver = load_model(cfg, "gemma")
    return source, receiver, tokenizer(cfg["models"]["qwen"]), tokenizer(cfg["models"]["gemma"])


def audit(cfg):
    tok_s, tok_r = tokenizer(cfg["models"]["qwen"]), tokenizer(cfg["models"]["gemma"])
    reference = HERE.parent / "multidataset_obqa_arc_mmlupro_train1024_full28_mlp_l3_to_q4_seed1234"
    summary = {"protocol": cfg["checkpoint_protocol"], "splits": {}}
    for split in ("train", "validation", "test"):
        rows = rows_for(cfg, split)
        if not cfg["smoke"]:
            old = json.loads((reference / "runs/study/manifests" / f"{split}.json").read_text())["rows"]
            if [r["id"] for r in rows] != [r["id"] for r in old]:
                raise RuntimeError(f"Forward/reverse split ID drift: {split}")
            if any(r["encoded"]["qwen"] != x["encoded"]["qwen"] for r, x in zip(rows, old)):
                raise RuntimeError(f"Qwen source tokenization drift: {split}")
        counts, tokens = {}, 0
        for row in rows:
            target, source, kinds = fullsync_map(row, tok_s, tok_r)
            if len(target) != len(source) or len(set(target)) != len(target):
                raise RuntimeError(f"Incomplete token mapping: {row['id']}")
            tokens += len(target)
            for kind, value in kinds.items():
                counts[kind] = counts.get(kind, 0) + value
        summary["splits"][split] = {"examples": len(rows), "gemma_option_tokens": tokens,
                                      "mapping_counts": counts}
        log(f"AUDIT {split}: {len(rows)} examples, {tokens} mapped Gemma Options tokens")
    save_json(result_root(cfg) / "alignment_audit.json", summary)


def prepare_memory(cfg):
    """Capture only reusable option KV in RAM; recompute Question KV on demand."""
    global _RECEIVER
    for split in ("train", "validation", "test"):
        _ROWS[split] = manifest_rows(cfg, split)
    tok_s, tok_r = tokenizer(cfg["models"]["qwen"]), tokenizer(cfg["models"]["gemma"])
    source_model = load_model(cfg, "qwen")
    for split in ("train", "validation", "test"):
        rows = rows_for(cfg, split)
        for number, row in enumerate(rows, 1):
            target, source, counts = fullsync_map(row, tok_s, tok_r)
            fields = row["encoded"]["qwen"]
            sk, sv, _, _ = capture_decision(source_model, fields["full"], len(fields["body"]),
                                             fields["choice_ids"])
            _CACHE[row["id"]] = {"source_k": sk[:, source].contiguous().half(),
                                 "source_v": sv[:, source].contiguous().half(),
                                 "target_indices": target, "unit_counts": counts,
                                 "option_tokens": len(target)}
            if number % 64 == 0 or number == len(rows):
                log(f"Qwen native capture {split}: {number}/{len(rows)}")
    del source_model
    gc.collect()
    torch.cuda.empty_cache()
    _RECEIVER = load_model(cfg, "gemma")
    for split in ("train", "validation", "test"):
        rows = rows_for(cfg, split)
        for number, row in enumerate(rows, 1):
            fields = row["encoded"]["gemma"]
            nk, nv, _, _ = capture_decision(_RECEIVER, fields["full"],
                                             len(fields["body"]), fields["choice_ids"])
            item = _CACHE[row["id"]]
            indices = item.pop("target_indices")
            item.update(target_k=nk[:, indices].contiguous().half(),
                        target_v=nv[:, indices].contiguous().half())
            if number % 64 == 0 or number == len(rows):
                log(f"Gemma native capture {split}: {number}/{len(rows)}")
    log(f"Option-only in-memory native capture completed: {len(_CACHE)} samples; Question KV is recomputed on demand; no KV files written")


def stage_a(cfg):
    seed_all(cfg["seed"])
    source, receiver, tok_s, tok_r = models_and_tokens(cfg)
    module = NativeKVTranslator("full36_headmix128", hidden_dim=cfg["depth_hidden_dim"],
                                depth_output_dim=cfg["depth_output_dim"],
                                head_mapping=cfg["head_mapping"]).cuda().train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_a_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = result_root(cfg) / "stage_a"
    root.mkdir(parents=True, exist_ok=True)
    save_json(result_root(cfg) / "training_config.json", {
        "protocol": cfg["checkpoint_protocol"], "architecture": cfg["architecture"],
        "split_policy": cfg["split_policy"], "fullsync": "ordered copy/drop on receiver Options grid",
        "train_samples": len(rows_for(cfg, "train")), "validation_samples": len(rows_for(cfg, "validation")),
        "test_samples": len(rows_for(cfg, "test")), "effective_batch": cfg["effective_batch"],
        "stage_a_epochs": cfg["stage_a_epochs"], "stage_b_epochs": cfg["stage_b_epochs"],
        "stage_a_lr": cfg["stage_a_learning_rate"], "stage_b_lr": cfg["learning_rate"],
        "clip": cfg["clip"], "temperature": cfg["temperature"],
        "chunk_tokens": cfg["chunk_tokens"], "no_persistent_kv_cache": True})
    rows = rows_for(cfg, "train")
    step, candidates = 0, []
    optimizer.zero_grad(set_to_none=True)
    with (root / "training_steps.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(1, cfg["stage_a_epochs"] + 1):
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg["seed"] + epoch)).tolist()
            accumulation, batch_loss = 0, 0.0
            for position, row_index in enumerate(order, 1):
                row = rows[row_index]
                item = pair(source, receiver, tok_s, tok_r, row, need_readout=False)
                accumulation += 1
                divisor = min(cfg["effective_batch"], len(order) - (position - accumulation))
                batch_loss += stage_a_sample_backward(cfg, module, item, scaler, divisor)
                del item
                if accumulation == cfg["effective_batch"] or position == len(order):
                    norm = optimizer_step(cfg, module, optimizer, scaler)
                    step += 1
                    stream.write(json.dumps({"epoch": epoch, "step": step,
                                             "loss_per_sample": batch_loss / accumulation,
                                             "pre_clip_grad_norm": norm,
                                             "clipped": norm > cfg["clip"]}) + "\n")
                    stream.flush()
                    accumulation, batch_loss = 0, 0.0
                    if step % 16 == 0 or cfg["smoke"]:
                        log(f"Stage-A epoch={epoch} step={step} train_samples={position}/{len(rows)}")
            metrics, _ = validate(cfg, source, receiver, tok_s, tok_r, module, None, "validation")
            path = root / "checkpoints" / f"epoch_{epoch}.pt"
            save_checkpoint(path, cfg, module, "stage_a", epoch, step, metrics)
            candidates.append({"epoch": epoch, "step": step, "path": str(path), "validation": metrics})
            save_json(root / "candidates.json", candidates)
            log(f"Stage-A epoch={epoch} validation_accuracy={metrics['correct']:.4f} k_nmse={metrics['k_nmse']:.4f} v_nmse={metrics['v_nmse']:.4f}")
            module.train()
    best = max(candidates, key=lambda item: (item["validation"]["correct"],
                                             -item["validation"]["k_nmse"] - item["validation"]["v_nmse"],
                                             -item["epoch"]))
    save_json(root / "selection.json", {"selection": "highest validation accuracy; ties by KV NMSE",
                                        "best": best, "candidates": candidates})
    log(f"Stage-A selected epoch={best['epoch']} step={best['step']}")


def stage_b(cfg):
    seed_all(cfg["seed"] + 1)
    selection = json.loads((result_root(cfg) / "stage_a/selection.json").read_text(encoding="utf-8"))
    base = NativeKVTranslator("full36_headmix128", hidden_dim=cfg["depth_hidden_dim"],
                              depth_output_dim=cfg["depth_output_dim"],
                              head_mapping=cfg["head_mapping"]).cuda().eval()
    load_checkpoint(Path(selection["best"]["path"]), cfg, "stage_a", base)
    base.requires_grad_(False)
    adapter = ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    source, receiver, tok_s, tok_r = models_and_tokens(cfg)
    root = result_root(cfg) / "stage_b"
    root.mkdir(parents=True, exist_ok=True)
    rows = rows_for(cfg, "train")
    step, candidates = 0, []
    optimizer.zero_grad(set_to_none=True)
    with (root / "training_steps.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(1, cfg["stage_b_epochs"] + 1):
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg["seed"] + 100 + epoch)).tolist()
            accumulation, batch_loss = 0, 0.0
            for position, row_index in enumerate(order, 1):
                row = rows[row_index]
                item = pair(source, receiver, tok_s, tok_r, row)
                teacher = native_teacher(receiver, row, item)
                base_k, base_v = mapped_base(base, item, cfg["chunk_tokens"])
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_k, pred_v, _, _ = adapter(base_k[None], base_v[None])
                student = student_logits(receiver, row, item, pred_k[0], pred_v[0])
                accumulation += 1
                divisor = min(cfg["effective_batch"], len(order) - (position - accumulation))
                loss = choice_kl(student, teacher, cfg["temperature"]) / divisor
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite full-token Stage-B choice KL")
                scaler.scale(loss).backward()
                batch_loss += loss.detach().item() * divisor
                del item, teacher, base_k, base_v, pred_k, pred_v, student, loss
                if accumulation == cfg["effective_batch"] or position == len(order):
                    norm = optimizer_step(cfg, adapter, optimizer, scaler)
                    step += 1
                    stream.write(json.dumps({"epoch": epoch, "step": step,
                                             "choice_kl_per_sample": batch_loss / accumulation,
                                             "pre_clip_grad_norm": norm,
                                             "clipped": norm > cfg["clip"]}) + "\n")
                    stream.flush()
                    accumulation, batch_loss = 0, 0.0
                    if step % 16 == 0 or cfg["smoke"]:
                        log(f"Stage-B epoch={epoch} step={step} train_samples={position}/{len(rows)}")
            metrics, _ = validate(cfg, source, receiver, tok_s, tok_r, base, adapter, "validation")
            path = root / "checkpoints" / f"epoch_{epoch}.pt"
            save_checkpoint(path, cfg, adapter, "stage_b", epoch, step, metrics)
            candidates.append({"epoch": epoch, "step": step, "path": str(path), "validation": metrics})
            save_json(root / "candidates.json", candidates)
            log(f"Stage-B epoch={epoch} validation_accuracy={metrics['correct']:.4f} choice_kl={metrics['choice_kl']:.4f}")
            adapter.train()
    best = max(candidates, key=lambda item: (item["validation"]["correct"],
                                             -item["validation"]["choice_kl"], -item["epoch"]))
    save_json(root / "selection.json", {"selection": "highest validation accuracy; ties by choice KL",
                                        "base_checkpoint": selection["best"]["path"],
                                        "best": best, "candidates": candidates})
    log(f"Stage-B selected epoch={best['epoch']} step={best['step']}")


def evaluate(cfg):
    seed_all(cfg["seed"])
    root = result_root(cfg)
    select_a = json.loads((root / "stage_a/selection.json").read_text(encoding="utf-8"))["best"]
    select_b = json.loads((root / "stage_b/selection.json").read_text(encoding="utf-8"))["best"]
    base = NativeKVTranslator("full36_headmix128", hidden_dim=cfg["depth_hidden_dim"],
                              depth_output_dim=cfg["depth_output_dim"],
                              head_mapping=cfg["head_mapping"]).cuda().eval()
    adapter = ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().eval()
    load_checkpoint(Path(select_a["path"]), cfg, "stage_a", base)
    load_checkpoint(Path(select_b["path"]), cfg, "stage_b", adapter)
    base.requires_grad_(False)
    adapter.requires_grad_(False)
    source, receiver, tok_s, tok_r = models_and_tokens(cfg)
    metrics_a, records_a = validate(cfg, source, receiver, tok_s, tok_r, base, None, "test")
    metrics_b, records_b = validate(cfg, source, receiver, tok_s, tok_r, base, adapter, "test")
    if [r["id"] for r in records_a] != [r["id"] for r in records_b]:
        raise RuntimeError("Stage-A/Stage-B test ID drift")
    results = {"status": "completed", "protocol": cfg["checkpoint_protocol"],
               "fullsync": "receiver Options full token grid; ordered copy/drop",
               "stage_a": metrics_a, "stage_b": metrics_b,
               "selected_a_checkpoint": select_a["path"],
               "selected_b_checkpoint": select_b["path"],
               "test_rows": len(records_a)}
    path = root / "results/per_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    with path.open("w", encoding="utf-8") as stream:
        for a, b in zip(records_a, records_b):
            record = {"id": a["id"], "dataset": a["dataset"],
                      "gold_index": a["gold_index"], "stage_a": a, "stage_b": b}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            groups.setdefault(a["dataset"], []).append(record)
    dataset_summary = {}
    for dataset, records in sorted(groups.items()):
        total = len(records)
        correct_a = sum(r["stage_a"]["prediction"] == r["gold_index"] for r in records)
        correct_b = sum(r["stage_b"]["prediction"] == r["gold_index"] for r in records)
        oracle = sum(r["stage_b"]["oracle_prediction"] == r["gold_index"] for r in records)
        both = sum(r["stage_b"]["prediction"] == r["gold_index"] and
                   r["stage_b"]["oracle_prediction"] == r["gold_index"] for r in records)
        dataset_summary[dataset] = {"count": total, "stage_a_correct": correct_a,
                                    "stage_b_correct": correct_b, "oracle_correct": oracle,
                                    "both_stage_b_oracle_correct": both,
                                    "stage_b_only_correct": correct_b - both,
                                    "oracle_only_correct": oracle - both}
        (root / "results/by_dataset").mkdir(parents=True, exist_ok=True)
        with (root / "results/by_dataset" / f"{dataset}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    results["by_dataset"] = dataset_summary
    save_json(root / "results/summary.json", results)
    log(f"FINAL Stage-A accuracy={metrics_a['correct']:.4f} Stage-B accuracy={metrics_b['correct']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "all"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = settings(args.smoke)
    if args.action == "audit":
        audit(cfg)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    prepare_memory(cfg)
    stage_a(cfg)
    gc.collect()
    torch.cuda.empty_cache()
    stage_b(cfg)
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(cfg)


if __name__ == "__main__":
    main()
