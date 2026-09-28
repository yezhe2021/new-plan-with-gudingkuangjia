"""Train a fresh Residual64 Stage-B with GSM8K answer-token CE only."""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from generation_common import (CFG, HERE, adapted_cache, load_adapter, native_full_teacher_forcing,
    native_prefix_cache, native_token0, prepare_base_cache, prepare_manifest, save_checkpoint,
    save_json, teacher_forcing)
import generation_common as common


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def audits(smoke=False):
    rows, root = prepare_manifest(smoke=True)
    cache, _, qwen_tok, stage_a_path, module = prepare_base_cache(rows, "audit")
    qwen = common.ref.load_model("qwen")
    adapter = load_adapter(module).train()
    token0_k, token0_v, token0_id = native_token0(qwen, qwen_tok)
    records = []
    try:
        # Audit 1: native full text and native prefix-cache teacher forcing.
        for entry in cache:
            full_logits, targets, full_ce = native_full_teacher_forcing(qwen, qwen_tok, entry["question"], entry["answer"])
            key, value = native_prefix_cache(qwen, qwen_tok, entry["question"])
            cached_logits, cached_targets, cached_ce = teacher_forcing(qwen, qwen_tok, key, value, entry["answer"])
            max_abs = (full_logits.float() - cached_logits.float()).abs().max().item()
            ce_abs = abs(full_ce.item() - cached_ce.item())
            top1_equal = torch.equal(full_logits.argmax(-1), cached_logits.argmax(-1))
            if (not torch.equal(targets, cached_targets) or not top1_equal
                    or max_abs > 0.5 or ce_abs > 0.01):
                raise RuntimeError(
                    "Native teacher-forcing equivalence failed: "
                    f"max_logit_abs={max_abs}, ce_abs={ce_abs}, top1_equal={top1_equal}, "
                    f"full_ce={full_ce.item()}, cache_ce={cached_ce.item()}"
                )
            records.append({"id": entry["id"], "native_max_logit_abs": max_abs,
                            "native_ce_abs": ce_abs, "native_top1_equal": top1_equal,
                            "native_full_ce": full_ce.item(), "native_cache_ce": cached_ce.item()})
        # Audit 2: zero residual is exact Stage-A identity.
        entry = cache[0]
        key, value = adapted_cache(adapter, entry, token0_k, token0_v)
        expected_k = torch.cat((token0_k, entry["base_k"].cuda()), 1)
        expected_v = torch.cat((token0_v, entry["base_v"].cuda()), 1)
        identity_error = max((key - expected_k).abs().max().item(), (value - expected_v).abs().max().item())
        if identity_error != 0:
            raise RuntimeError(f"Zero residual is not identity: {identity_error}")
        # Audit 4: gradients pass through frozen Qwen only into Residual64.
        logits, targets, loss = teacher_forcing(qwen, qwen_tok, key, value, entry["answer"])
        loss.backward()
        grads = [parameter.grad for parameter in adapter.parameters()]
        nonzero = sum(int(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0) for grad in grads)
        if nonzero == 0 or any(parameter.grad is not None for parameter in qwen.parameters()):
            raise RuntimeError("Gradient ownership audit failed")
        records[0].update(identity_error=identity_error, generation_ce=loss.item(),
                          adapter_parameters_with_nonzero_grad=nonzero,
                          qwen_parameters_with_grad=sum(p.grad is not None for p in qwen.parameters()),
                          native_token0_id=token0_id, stage_a_checkpoint=stage_a_path)
        save_json(root / "smoke_audits.json", {"status": "passed", "records": records})
        log(f"SMOKE AUDITS PASSED: loss={loss.item():.4f}, nonzero_adapter_grads={nonzero}")
    finally:
        del qwen, adapter, cache; gc.collect(); torch.cuda.empty_cache()


def train(smoke=False):
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    rows, root = prepare_manifest(smoke)
    cache, _, qwen_tok, stage_a_path, module = prepare_base_cache(rows, "train")
    qwen = common.ref.load_model("qwen")
    qwen.requires_grad_(False)
    adapter = load_adapter(module).train()
    token0_k, token0_v, _ = native_token0(qwen, qwen_tok)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=CFG["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    epochs = 1 if smoke else CFG["epochs"]
    effective = 2 if smoke else CFG["effective_batch"]
    history, step = [], 0
    steps_path = root / "training_steps.jsonl"
    steps_path.unlink(missing_ok=True)
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(1, epochs + 1):
            order = list(range(len(cache)))
            random.Random(CFG["seed"] + epoch).shuffle(order)
            accumulated, loss_sum, token_sum = 0, 0.0, 0
            for position, index in enumerate(order, 1):
                entry = cache[index]
                key, value = adapted_cache(adapter, entry, token0_k, token0_v)
                logits, targets, loss = teacher_forcing(qwen, qwen_tok, key, value, entry["answer"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite generation CE")
                scaler.scale(loss / effective).backward()
                accumulated += 1; loss_sum += loss.detach().item(); token_sum += targets.numel()
                del key, value, logits, targets, loss
                if accumulated == effective or position == len(order):
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), CFG["gradient_clip"])
                    if not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite gradient norm")
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    step += 1
                    record = {"epoch": epoch, "step": step, "samples_seen_in_epoch": position,
                              "answer_ce": loss_sum / accumulated, "answer_tokens": token_sum,
                              "pre_clip_grad_norm": norm.item(), "clipped": norm.item() > CFG["gradient_clip"]}
                    history.append(record)
                    with steps_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    accumulated, loss_sum, token_sum = 0, 0.0, 0
                    if step % 8 == 0 or smoke:
                        log(f"epoch={epoch} step={step} samples={position}/{len(order)} ce={record['answer_ce']:.4f}")
        metrics = {"final_step": step, "epochs": epochs, "train_samples": len(cache),
                   "initial_answer_ce": history[0]["answer_ce"], "final_answer_ce": history[-1]["answer_ce"],
                   "stage_a_checkpoint": stage_a_path, "loss": "full answer-token CE only",
                   "effective_batch": effective, "learning_rate": CFG["learning_rate"]}
        checkpoint = root / "checkpoints/final.pt"
        save_checkpoint(checkpoint, adapter, epochs, step, metrics)
        save_json(root / "training_summary.json", {**metrics, "status": "completed", "checkpoint": str(checkpoint)})
        log(f"TRAINING COMPLETED: {metrics}")
    finally:
        del qwen, adapter, cache; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "train"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    audits(args.smoke) if args.action == "audit" else train(args.smoke)


if __name__ == "__main__":
    main()
