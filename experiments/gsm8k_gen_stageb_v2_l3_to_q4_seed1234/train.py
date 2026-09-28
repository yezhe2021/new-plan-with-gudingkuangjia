"""Audit and train CE+EOS, trajectory-KL+EOS, and hybrid Residual64 adapters."""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

import generation_common as C


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def prepare_all():
    train_rows, val_rows, manifest = C.prepare_manifest()
    train_cache, _, tok, stage_a, module = C.prepare_base_cache(
        train_rows, manifest["train_indices"], "train")
    val_cache, _, _, _, _ = C.prepare_base_cache(
        val_rows, manifest["val_indices"], "validation")
    return train_cache, val_cache, tok, stage_a, module


def audits():
    train_rows, _, manifest = C.prepare_manifest()
    rows, indices = train_rows[:3], manifest["train_indices"][:3]
    cache, _, tok, stage_a, module = C.prepare_base_cache(rows, indices, "audit")
    qwen = C.ref.load_model("qwen").requires_grad_(False)
    torch.manual_seed(C.CFG["seed"])
    adapter = C.load_adapter(module).train()
    token0_k, token0_v, token0_id = C.native_token0(qwen, tok, cache)
    records = []
    try:
        eos = tok.eos_token_id
        generation_eos = qwen.generation_config.eos_token_id
        generation_eos = {generation_eos} if isinstance(generation_eos, int) else set(generation_eos)
        if eos not in generation_eos:
            raise RuntimeError("Tokenizer EOS is absent from generation EOS set")
        for entry in cache:
            expected = f"Question: {entry['question']}\nAnswer:"
            if entry["source_text"] + "Answer:" != expected:
                raise RuntimeError("Prompt parity failed")
            native_k, native_v = C.ref.capture(qwen, C.encode_prefix(tok, entry["source_text"]),
                                               entry["qwen_prefix_tokens"], [0])[:2]
            teacher, _, targets = C.native_trajectory(qwen, tok, entry["source_text"], entry["answer"])
            cached, _, cached_targets = C.cached_trajectory(
                qwen, tok, native_k.cuda(), native_v.cuda(), entry["answer"])
            self_kl = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(cached.float(), -1),
                torch.nn.functional.softmax(teacher.float(), -1), reduction="batchmean").item()
            top1 = torch.equal(teacher.argmax(-1), cached.argmax(-1))
            if not torch.equal(targets, cached_targets) or not top1 or self_kl > 1e-3:
                raise RuntimeError(f"Native cache/KL parity failed: kl={self_kl}, top1={top1}")
            records.append({"id": entry["id"], "prompt": expected, "cache_length": native_k.shape[1],
                            "self_kl": self_kl, "top1_equal": top1})
        entry = cache[0]
        key, value = C.adapted_cache(adapter, entry, token0_k, token0_v)
        expected_k = torch.cat((token0_k, entry["base_k"].cuda()), 1)
        expected_v = torch.cat((token0_v, entry["base_v"].cuda()), 1)
        identity = max((key - expected_k).abs().max().item(), (value - expected_v).abs().max().item())
        if identity != 0:
            raise RuntimeError(f"Zero Stage-B parity failed: {identity}")
        teacher, _, targets = C.native_trajectory(qwen, tok, entry["source_text"], entry["answer"])
        student, student_eos, _ = C.cached_trajectory(qwen, tok, key, value, entry["answer"])
        kl, ce, eos_ce = C.loss_components(student, student_eos, teacher, targets, eos)
        loss = C.objective_loss("hybrid", kl, ce, eos_ce)
        loss.backward()
        adapter_grads = sum(p.grad is not None and p.grad.abs().sum().item() > 0 for p in adapter.parameters())
        if adapter_grads == 0 or any(p.grad is not None for p in qwen.parameters()):
            raise RuntimeError("Gradient ownership failed")
        records[0].update(token0_id=token0_id, zero_identity_max_abs=identity,
                          adapter_parameters_with_grad=adapter_grads,
                          qwen_parameters_with_grad=sum(p.grad is not None for p in qwen.parameters()),
                          trajectory_kl=kl.item(), answer_ce=ce.item(), eos_ce=eos_ce.item(),
                          stage_a_checkpoint=stage_a)
        C.save_json(C.HERE / "runs/audits.json", {"status": "passed", "records": records})
        log("ALL PRE-GPU TRAINING AUDITS PASSED")
    finally:
        del qwen, adapter, cache; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def validate(qwen, tok, adapter, cache, token0_k, token0_v, objective, epoch):
    adapter.eval(); rows, kl_sum, ce_sum, eos_sum = [], 0.0, 0.0, 0.0
    eos = tok.eos_token_id
    for position, entry in enumerate(cache, 1):
        key, value = C.adapted_cache(adapter, entry, token0_k, token0_v)
        teacher, _, targets = C.native_trajectory(qwen, tok, entry["source_text"], entry["answer"])
        student, student_eos, _ = C.cached_trajectory(qwen, tok, key, value, entry["answer"])
        kl, ce, eos_ce = C.loss_components(student, student_eos, teacher, targets, eos)
        text, count, reason = C.greedy_generate(qwen, tok, list(tok.encode("Answer:", add_special_tokens=False)),
                                               C.CFG["max_new_tokens"], key, value)
        record = C.generation_record(text, count, reason, entry["answer"])
        record.update(id=entry["id"], trajectory_kl=kl.item(), answer_ce=ce.item(), eos_ce=eos_ce.item())
        rows.append(record); kl_sum += kl.item(); ce_sum += ce.item(); eos_sum += eos_ce.item()
        if position % 16 == 0:
            log(f"{objective} epoch={epoch} validation {position}/{len(cache)}")
        del key, value, teacher, student, student_eos, targets
    summary = C.summarize_generations(rows)
    summary.update(trajectory_kl=kl_sum/len(rows), answer_ce=ce_sum/len(rows), eos_ce=eos_sum/len(rows))
    root = C.HERE / "runs" / objective
    C.write_jsonl(root / f"validation_epoch_{epoch}.jsonl", rows)
    C.save_json(root / f"validation_epoch_{epoch}.json", summary)
    adapter.train(); return summary


def selection_key(summary):
    return (summary["first_hash_accuracy"], summary["strict_accuracy"], summary["flexible_accuracy"],
            -summary["max_token_rate"], -summary["trajectory_kl"])


def train_one(objective, qwen, tok, module, train_cache, val_cache, token0_k, token0_v, stage_a):
    random.seed(C.CFG["seed"]); torch.manual_seed(C.CFG["seed"]); torch.cuda.manual_seed_all(C.CFG["seed"])
    adapter = C.load_adapter(module).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=C.CFG["learning_rate"],
                                  weight_decay=C.CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = C.HERE / "runs" / objective
    root.mkdir(parents=True, exist_ok=True)
    steps_path = root / "training_steps.jsonl"; steps_path.unlink(missing_ok=True)
    history, validations, best, step = [], [], None, 0
    eos = tok.eos_token_id
    try:
        for epoch in range(1, C.CFG["epochs"] + 1):
            order = list(range(len(train_cache))); random.Random(C.CFG["seed"] + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0; sums = {"trajectory_kl": 0.0, "answer_ce": 0.0, "eos_ce": 0.0, "total_loss": 0.0}
            for position, index in enumerate(order, 1):
                entry = train_cache[index]
                key, value = C.adapted_cache(adapter, entry, token0_k, token0_v)
                with torch.no_grad():
                    teacher, _, targets = C.native_trajectory(qwen, tok, entry["source_text"], entry["answer"])
                student, student_eos, _ = C.cached_trajectory(qwen, tok, key, value, entry["answer"])
                kl, ce, eos_ce = C.loss_components(student, student_eos, teacher, targets, eos)
                loss = C.objective_loss(objective, kl, ce, eos_ce)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-B v2 loss")
                scaler.scale(loss / C.CFG["effective_batch"]).backward()
                accumulated += 1
                for name, value_ in (("trajectory_kl", kl), ("answer_ce", ce), ("eos_ce", eos_ce), ("total_loss", loss)):
                    sums[name] += value_.detach().item()
                del key, value, teacher, targets, student, student_eos, kl, ce, eos_ce, loss
                if accumulated == C.CFG["effective_batch"] or position == len(order):
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), C.CFG["gradient_clip"])
                    if not torch.isfinite(norm): raise RuntimeError("Nonfinite gradient norm")
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); step += 1
                    record = {"objective": objective, "epoch": epoch, "step": step,
                              "samples_seen_in_epoch": position,
                              **{name: value_ / accumulated for name, value_ in sums.items()},
                              "pre_clip_grad_norm": norm.item(),
                              "clipped": norm.item() > C.CFG["gradient_clip"]}
                    history.append(record)
                    with steps_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    accumulated = 0; sums = {name: 0.0 for name in sums}
                    if step % 16 == 0:
                        log(f"{objective} epoch={epoch} step={step} samples={position}/{len(order)} "
                            f"loss={record['total_loss']:.4f} kl={record['trajectory_kl']:.4f} "
                            f"ce={record['answer_ce']:.4f} eos={record['eos_ce']:.4f}")
            summary = validate(qwen, tok, adapter, val_cache, token0_k, token0_v, objective, epoch)
            summary.update(epoch=epoch, step=step)
            checkpoint = root / f"checkpoints/epoch_{epoch}.pt"
            C.save_checkpoint(checkpoint, adapter, objective, epoch, step, summary)
            summary["checkpoint"] = str(checkpoint); validations.append(summary)
            if best is None or selection_key(summary) > selection_key(best): best = dict(summary)
            log(f"{objective} epoch={epoch} validation={summary}")
        C.save_json(root / "selection.json", {"objective": objective, "best": best, "epochs": validations,
                    "selection_order": ["first_hash_accuracy", "strict_accuracy", "flexible_accuracy",
                                        "lower_max_token_rate", "lower_trajectory_kl"]})
        C.save_json(root / "training_summary.json", {"status": "completed", "objective": objective,
                    "train_samples": len(train_cache), "val_samples": len(val_cache), "epochs": C.CFG["epochs"],
                    "optimizer_steps": step, "stage_a_checkpoint": stage_a,
                    "first_step": history[0], "last_step": history[-1], "best": best})
    finally:
        del adapter; gc.collect(); torch.cuda.empty_cache()


def train(objectives):
    train_cache, val_cache, tok, stage_a, module = prepare_all()
    qwen = C.ref.load_model("qwen").requires_grad_(False)
    token0_k, token0_v, _ = C.native_token0(qwen, tok, train_cache + val_cache)
    try:
        for objective in objectives:
            train_one(objective, qwen, tok, module, train_cache, val_cache, token0_k, token0_v, stage_a)
    finally:
        del qwen, train_cache, val_cache; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "train"))
    parser.add_argument("--objective", choices=("ce", "kl", "hybrid", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.action == "audit": audits()
    else: train(("ce", "kl", "hybrid") if args.objective == "all" else (args.objective,))


if __name__ == "__main__": main()
