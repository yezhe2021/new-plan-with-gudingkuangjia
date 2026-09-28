"""Fresh Stage-A/Stage-B training on the full Sync-aligned Options KV grid.

No generated KV tensors are persisted. Each sample's Llama and Qwen native KV
is captured on demand, consumed, and released. The test split is never used
for checkpoint selection. The architecture and objectives match the prior
balanced RawAnchor experiment; only the token grid changes.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fullsync_v0 import BASE, HERE, fullsync_map

sys.path.insert(0, str(BASE))
from common import configuration, load_model, save_json, seed_all, tokenizer  # noqa: E402
from data import manifest_rows  # noqa: E402
from protocol import capture_decision, final_logits  # noqa: E402
from translator import NativeKVTranslator, ResidualKVAdapter, component_loss  # noqa: E402


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def settings(smoke):
    cfg = configuration("study")
    cfg = dict(cfg)
    cfg["smoke"] = bool(smoke)
    cfg["effective_batch"] = 2 if smoke else 8
    cfg["stage_a_epochs"] = 1 if smoke else 4
    cfg["stage_b_epochs"] = 1 if smoke else 2
    cfg["train_limit"] = 2 if smoke else None
    cfg["validation_limit"] = 2 if smoke else None
    cfg["test_limit"] = 2 if smoke else None
    cfg["chunk_tokens"] = 64
    cfg["result_root"] = str(HERE / "runs" / ("retrain_smoke" if smoke else "retrain"))
    return cfg


def result_root(cfg):
    return Path(cfg["result_root"])


def rows_for(cfg, split):
    rows = manifest_rows(cfg, split)
    limit = cfg[f"{split}_limit"]
    if not limit:
        return rows
    # Smoke the longest target memories; short first rows hide peak-memory bugs.
    return sorted(rows, key=lambda row: len(row["encoded"]["qwen"]["option_token_indices"]),
                  reverse=True)[:limit]


@torch.no_grad()
def pair(model_l, model_q, tok_l, tok_q, row):
    target_indices, source_indices, counts = fullsync_map(row, tok_l, tok_q)
    fields_l, fields_q = row["encoded"]["llama"], row["encoded"]["qwen"]
    source_k, source_v, _, _ = capture_decision(
        model_l, fields_l["full"], len(fields_l["body"]), fields_l["choice_ids"])
    native_k, native_v, _, full_logits = capture_decision(
        model_q, fields_q["full"], len(fields_q["body"]), fields_q["choice_ids"])
    qlength = fields_q["question_prefix_length"]
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
    torch.save({"stage": stage, "protocol": cfg["protocol"], "epoch": epoch,
                "step": step, "metrics": metrics,
                "state": {name: value.detach().cpu() for name, value in module.state_dict().items()}}, path)


def load_checkpoint(path, cfg, stage, module):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("stage") != stage or payload.get("protocol") != cfg["protocol"]:
        raise RuntimeError(f"Checkpoint/protocol mismatch: {path}")
    module.load_state_dict(payload["state"], strict=True)
    return module


def choice_kl(student, teacher, temperature):
    student = F.log_softmax(student.float() / temperature, -1)
    teacher = F.log_softmax(teacher.float().to(student.device) / temperature, -1)
    return F.kl_div(student, teacher.detach(), reduction="sum", log_target=True) * temperature ** 2


def native_teacher(qwen, row, item):
    with torch.no_grad():
        return student_logits(qwen, row, item, item["target_k"].cuda(), item["target_v"].cuda())


def student_logits(qwen, row, item, key, value):
    key = torch.cat((item["question_k"].cuda(), key), dim=1)
    value = torch.cat((item["question_v"].cuda(), value), dim=1)
    logits = final_logits(qwen, row["encoded"]["qwen"]["receiver_answer"],
                          key, value, positions=torch.arange(key.shape[1], device="cuda"),
                          suffix_start=key.shape[1])
    indices = torch.tensor(row["encoded"]["qwen"]["choice_ids"], device="cuda")
    return logits.index_select(0, indices)


@torch.no_grad()
def validate(cfg, llama, qwen, tok_l, tok_q, base, adapter, split):
    base.eval()
    if adapter is not None:
        adapter.eval()
    rows = rows_for(cfg, split)
    totals = {"correct": 0, "oracle_correct": 0, "agreement": 0,
              "choice_kl": 0.0, "k_nmse": 0.0, "v_nmse": 0.0,
              "k_cosine": 0.0, "v_cosine": 0.0}
    records = []
    for index, row in enumerate(rows, 1):
        item = pair(llama, qwen, tok_l, tok_q, row)
        key, value = mapped_base(base, item, cfg["chunk_tokens"])
        if adapter is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                key, value, _, _ = adapter(key[None], value[None])
            key, value = key[0], value[0]
        teacher = native_teacher(qwen, row, item)
        student = student_logits(qwen, row, item, key, value)
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
    llama = load_model(cfg, "llama")
    qwen = load_model(cfg, "qwen")
    return llama, qwen, tokenizer(cfg["models"]["llama"]), tokenizer(cfg["models"]["qwen"])


def stage_a(cfg):
    seed_all(cfg["seed"])
    llama, qwen, tok_l, tok_q = models_and_tokens(cfg)
    module = NativeKVTranslator("full28_mlp", hidden_dim=cfg["mlp_hidden_dim"]).cuda().train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg["stage_a_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = result_root(cfg) / "stage_a"
    root.mkdir(parents=True, exist_ok=True)
    save_json(result_root(cfg) / "training_config.json", {
        "protocol": cfg["protocol"], "architecture": cfg["architecture"],
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
                item = pair(llama, qwen, tok_l, tok_q, row)
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
            metrics, _ = validate(cfg, llama, qwen, tok_l, tok_q, module, None, "validation")
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
    base = NativeKVTranslator("full28_mlp", hidden_dim=cfg["mlp_hidden_dim"]).cuda().eval()
    load_checkpoint(Path(selection["best"]["path"]), cfg, "stage_a", base)
    base.requires_grad_(False)
    adapter = ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    llama, qwen, tok_l, tok_q = models_and_tokens(cfg)
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
                item = pair(llama, qwen, tok_l, tok_q, row)
                teacher = native_teacher(qwen, row, item)
                base_k, base_v = mapped_base(base, item, cfg["chunk_tokens"])
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_k, pred_v, _, _ = adapter(base_k[None], base_v[None])
                student = student_logits(qwen, row, item, pred_k[0], pred_v[0])
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
            metrics, _ = validate(cfg, llama, qwen, tok_l, tok_q, base, adapter, "validation")
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
    base = NativeKVTranslator("full28_mlp", hidden_dim=cfg["mlp_hidden_dim"]).cuda().eval()
    adapter = ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().eval()
    load_checkpoint(Path(select_a["path"]), cfg, "stage_a", base)
    load_checkpoint(Path(select_b["path"]), cfg, "stage_b", adapter)
    base.requires_grad_(False)
    adapter.requires_grad_(False)
    llama, qwen, tok_l, tok_q = models_and_tokens(cfg)
    metrics_a, records_a = validate(cfg, llama, qwen, tok_l, tok_q, base, None, "test")
    metrics_b, records_b = validate(cfg, llama, qwen, tok_l, tok_q, base, adapter, "test")
    if [r["id"] for r in records_a] != [r["id"] for r in records_b]:
        raise RuntimeError("Stage-A/Stage-B test ID drift")
    results = {"status": "completed", "protocol": cfg["protocol"],
               "fullsync": "receiver Options full token grid; ordered copy/drop",
               "stage_a": metrics_a, "stage_b": metrics_b,
               "selected_a_checkpoint": select_a["path"],
               "selected_b_checkpoint": select_b["path"],
               "test_rows": len(records_a)}
    save_json(root / "results/summary.json", results)
    path = root / "results/per_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for a, b in zip(records_a, records_b):
            stream.write(json.dumps({"id": a["id"], "dataset": a["dataset"],
                                     "gold_index": a["gold_index"], "stage_a": a,
                                     "stage_b": b}, ensure_ascii=False) + "\n")
    log(f"FINAL Stage-A accuracy={metrics_a['correct']:.4f} Stage-B accuracy={metrics_b['correct']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage_a", "stage_b", "evaluate"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = settings(args.smoke)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    {"stage_a": stage_a, "stage_b": stage_b, "evaluate": evaluate}[args.action](cfg)


if __name__ == "__main__":
    main()
