"""Frozen original Stage A; fresh Stage B trained with token KL + gold CE."""
import argparse
from collections import defaultdict
import gc
import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from runtime import CONFIG, HERE, PARENT, gpu_check, log, read_rows, save, verify_manifest
from kv_backend import (CFG, FullSyncHFLM, ResidualKVAdapter, assemble, capture, encoding,
                        load_base, load_model, map_base, seed_all, source_map)
from protocol import make_cache


def models():
    seed_all(CONFIG["seed"])
    sender = load_model(CFG, "llama")
    receiver = load_model(CFG, "qwen")
    base = load_base(PARENT / CONFIG["parent_stage_a"])
    tokenizer = AutoTokenizer.from_pretrained(CFG["models"]["llama"], local_files_only=True)
    hf = FullSyncHFLM(receiver, mode="native")
    return sender, receiver, base, tokenizer, hf


@torch.no_grad()
def frozen_prefix(sender, receiver, base, source_tokenizer, hf, row, anchor_cache):
    pairs = [hf._encode_pair(row["context"], continuation) for continuation in row["continuations"]]
    context_ids = pairs[0][0]
    if any(ids != context_ids or not answer for ids, answer in pairs):
        raise RuntimeError("Candidate boundaries differ or answer is empty")
    if encoding(hf.tokenizer, row["context"].rstrip()) != context_ids:
        raise RuntimeError("Harness context tokenization differs; no implicit retokenization")
    if max(len(context_ids) + len(answer) for _, answer in pairs) > CFG["max_length"]:
        raise RuntimeError("Context plus candidate exceeds original Full-Sync budget")
    if len(context_ids) <= 2:
        return None, None, None, pairs
    source_ids, selected, counts = source_map(row["context"].rstrip(), source_tokenizer, hf.tokenizer, context_ids)
    source_k, source_v = capture(sender, source_ids)
    first_id = context_ids[0]
    if first_id not in anchor_cache:
        anchor_cache[first_id] = capture(receiver, context_ids[:1])
    anchor_k, anchor_v = anchor_cache[first_id]
    item = {"source_k": source_k[:, selected], "source_v": source_v[:, selected],
            "anchor_k": anchor_k, "anchor_v": anchor_v, "tokens": len(selected), "counts": counts}
    base_k, base_v = map_base(base, item, CFG["chunk_tokens"])
    # Retain only the native anchor and frozen translated prefix; never cache KV on disk.
    minimal = {"anchor_k": anchor_k, "anchor_v": anchor_v, "tokens": len(selected)}
    return minimal, base_k, base_v, pairs


def suffix_logits(receiver, context_ids, answer_ids, key=None, value=None):
    """Same scored token boundaries as continuation_score, without prompt-wide logits."""
    prefix = 0 if key is None else key.shape[1]
    if key is not None and prefix != len(context_ids) - 1:
        raise RuntimeError("Full-Sync last-readout position mismatch")
    ids = (context_ids if key is None else context_ids[-1:]) + answer_ids[:-1]
    cache = None if key is None else make_cache(receiver, key, value, torch.arange(prefix, device="cuda"))
    tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = receiver.model(input_ids=tensor,
                            attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
                            position_ids=torch.arange(prefix, prefix + len(ids), device="cuda")[None],
                            past_key_values=cache, use_cache=False, return_dict=True)
    return receiver.lm_head(output.last_hidden_state[:, -len(answer_ids):])[0]


def token_terms(student_logits, teacher_logits, answer_ids):
    student_logprob = student_logits.float().log_softmax(-1)
    with torch.no_grad():
        teacher_logprob = teacher_logits.float().log_softmax(-1)
    kd_sum = F.kl_div(student_logprob, teacher_logprob, reduction="sum", log_target=True)
    ids = torch.tensor(answer_ids, device=student_logits.device, dtype=torch.long)
    ce_mean = F.nll_loss(student_logprob, ids, reduction="mean")
    teacher_score = float(teacher_logprob.gather(-1, ids[:, None]).sum())
    student_score = float(student_logprob.detach().gather(-1, ids[:, None]).sum())
    return kd_sum, ce_mean, teacher_score, student_score


def objective(receiver, hf, adapter, row, item, base_k, base_v, pairs, divisor=None, scaler=None):
    if item is None:
        # Original anchor/readout interface leaves no translated token in this case.
        if divisor is not None:
            scaler.scale(next(adapter.parameters()).sum() * 0 / divisor).backward()
        return {"total": 0., "token_kl": 0., "gold_ce": 0., "no_external_tokens": 1.,
                "choice_acc": 0., "teacher_choice_acc": 0., "answer_tokens": 0.}
    all_single_token = all(len(answer) == 1 for _, answer in pairs)
    # Single-token letter candidates share exactly the same causal prefix distribution.
    # Computing it once is algebraically identical to averaging it over all candidates.
    indices = [row["correct_index"]] if all_single_token else list(range(len(pairs)))
    denominator = 1 if all_single_token else sum(len(answer) for _, answer in pairs)
    metrics = defaultdict(float)
    teacher_scores, student_scores = [], []
    for index in indices:
        context_ids, answer_ids = pairs[index]
        with torch.no_grad():
            teacher = suffix_logits(receiver, context_ids, answer_ids)
        key, value = assemble(item, base_k, base_v, adapter)
        student = suffix_logits(receiver, context_ids, answer_ids, key, value)
        kd_sum, ce_mean, teacher_score, student_score = token_terms(student, teacher, answer_ids)
        kd = kd_sum / denominator
        ce = ce_mean if index == row["correct_index"] else ce_mean * 0
        loss = CONFIG["token_kl_weight"] * kd + CONFIG["gold_ce_weight"] * ce
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite token KL / gold CE")
        metrics["token_kl"] += float(kd.detach())
        metrics["gold_ce"] += float(ce.detach())
        metrics["total"] += float(loss.detach())
        if all_single_token:
            with torch.no_grad():
                lp_t, lp_s = teacher.float().log_softmax(-1)[0], student.float().log_softmax(-1)[0]
                teacher_scores = [float(lp_t[answer[0]]) for _, answer in pairs]
                student_scores = [float(lp_s[answer[0]]) for _, answer in pairs]
        else:
            teacher_scores.append(teacher_score)
            student_scores.append(student_score)
        if divisor is not None:
            scaler.scale(loss / divisor).backward()
        del teacher, student, key, value, loss, kd_sum, kd, ce_mean, ce
    metrics["choice_acc"] = float(max(range(len(student_scores)), key=student_scores.__getitem__) == row["correct_index"])
    metrics["teacher_choice_acc"] = float(max(range(len(teacher_scores)), key=teacher_scores.__getitem__) == row["correct_index"])
    metrics["answer_tokens"] = sum(len(answer) for _, answer in pairs)
    metrics["no_external_tokens"] = 0.
    return dict(metrics)


def new_adapter():
    return ResidualKVAdapter(rank=CONFIG["adapter_rank"]).cuda()


def optimizer_for(adapter):
    return torch.optim.AdamW(adapter.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])


def update(adapter, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), CONFIG["gradient_clip"])
    if not torch.isfinite(norm):
        raise RuntimeError("Non-finite Stage B gradients")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def checkpoint(path, adapter, optimizer, scaler, epoch, step, position, complete, validation, manifest):
    payload = {"protocol": CONFIG["protocol"], "parent_protocol": CFG["protocol"],
               "stage_a_sha256": manifest["identities"]["stage_a"]["sha256"],
               "state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
               "epoch": epoch, "step": step, "position": position, "epoch_completed": complete,
               "validation": validation}
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(sender, receiver, base, source_tok, hf, adapter, rows, anchors):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for index, row in enumerate(rows, 1):
        item, bk, bv, pairs = frozen_prefix(sender, receiver, base, source_tok, hf, row, anchors)
        values = objective(receiver, hf, adapter, row, item, bk, bv, pairs)
        for group in ("all", row["view"]):
            counts[group] += 1
            for key, value in values.items():
                sums[(group, key)] += value
        del item, bk, bv
        if index % 64 == 0:
            log(f"Validation {index}/{len(rows)}")
    result = {group: {key: value / counts[group] for (name, key), value in sums.items() if name == group}
              for group in counts}
    result["total"] = result["all"]["total"]
    return result


def smoke():
    manifest = verify_manifest()
    gpu_check()
    sender, receiver, base, source_tok, hf = models()
    adapter = new_adapter()
    optimizer = optimizer_for(adapter)
    scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
    rows = read_rows(HERE / "prepared/train_epoch_1.jsonl")
    chosen = [max((row for row in rows if row["view"] == "text"), key=lambda row: row["qwen_context_tokens"]),
              max((row for row in rows if row["view"] == "letter" and row["fewshot"] == 5),
                  key=lambda row: row["qwen_context_tokens"])]
    anchors, records = {}, []
    first_parameter = next(parameter for name, parameter in adapter.named_parameters() if name.endswith("2.weight"))
    before = first_parameter.detach().clone()
    for row in chosen:
        item, bk, bv, pairs = frozen_prefix(sender, receiver, base, source_tok, hf, row, anchors)
        with torch.no_grad():
            k, v = assemble(item, bk, bv, adapter)
            if not torch.equal(k[:, 1:], bk) or not torch.equal(v[:, 1:], bv):
                raise RuntimeError("Fresh Stage B must exactly preserve Stage A before training")
        values = objective(receiver, hf, adapter, row, item, bk, bv, pairs, divisor=2, scaler=scaler)
        records.append({"id": row["base_id"], "view": row["view"], "fewshot": row["fewshot"],
                        "context_tokens": row["qwen_context_tokens"], **values})
        del item, bk, bv, k, v
        log(f"Backward smoke passed for {row['view']}, shots={row['fewshot']}: {values}")
    if any(parameter.grad is not None for model in (sender, receiver, base) for parameter in model.parameters()):
        raise RuntimeError("Frozen models received gradients")
    norm = update(adapter, optimizer, scaler)
    change = float((first_parameter.detach() - before).abs().max())
    if norm <= 0 or change <= 0:
        raise RuntimeError("Smoke update did not change the fresh adapter")
    save(HERE / "smoke.json", {"SMOKE_ONLY_NOT_BENCHMARK_RESULTS": True, "passed": True,
                               "stage_a_sha256": manifest["identities"]["stage_a"]["sha256"],
                               "records": records, "pre_clip_grad_norm": norm, "max_parameter_change": change,
                               "frozen_parameter_gradients": False,
                               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
    log("Stage B token-KL + CE smoke passed; smoke-updated adapter will not be used for formal training")


def train():
    manifest = verify_manifest()
    gpu_check()
    sender, receiver, base, source_tok, hf = models()
    adapter = new_adapter()
    optimizer = optimizer_for(adapter)
    scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
    output = HERE / "stage_b"
    start_epoch, position_start, step = 1, 0, 0
    history = []
    if (output / "last.pt").exists():
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["protocol"] != CONFIG["protocol"] or state["stage_a_sha256"] != manifest["identities"]["stage_a"]["sha256"]:
            raise RuntimeError("Resume state mismatch")
        adapter.load_state_dict(state["state"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        start_epoch = state["epoch"] + int(state["epoch_completed"])
        position_start = 0 if state["epoch_completed"] else state["position"]
        if (output / "selection.json").exists():
            history = json.loads((output / "selection.json").read_text())["epochs"]
        log(f"Resuming new Stage B at epoch={start_epoch}, position={position_start}, updates={step}")
    validation = read_rows(HERE / "prepared/validation.jsonl")
    anchors = {}
    output.mkdir(exist_ok=True)
    with (output / "training_steps.jsonl").open("a", encoding="utf-8") as stream:
        for epoch in range(start_epoch, CONFIG["epochs"] + 1):
            rows = read_rows(HERE / f"prepared/train_epoch_{epoch}.jsonl")
            order = list(range(len(rows)))
            random.Random(CONFIG["seed"] + epoch).shuffle(order)
            sums = defaultdict(float)
            accumulation = 0
            optimizer.zero_grad(set_to_none=True)
            adapter.train()
            log(f"START new Stage B epoch={epoch}/{CONFIG['epochs']}, view draws={len(rows)}")
            for position, index in enumerate(order, 1):
                if epoch == start_epoch and position <= position_start:
                    continue
                row = rows[index]
                divisor = min(CONFIG["effective_batch"], len(rows) - ((position - 1) // CONFIG["effective_batch"]) * CONFIG["effective_batch"])
                item, bk, bv, pairs = frozen_prefix(sender, receiver, base, source_tok, hf, row, anchors)
                values = objective(receiver, hf, adapter, row, item, bk, bv, pairs, divisor=divisor, scaler=scaler)
                del item, bk, bv
                for key, value in values.items():
                    sums[key] += value
                accumulation += 1
                if accumulation == divisor:
                    norm = update(adapter, optimizer, scaler)
                    step += 1
                    record = {"epoch": epoch, "step": step, "position": position,
                              "pre_clip_grad_norm": norm,
                              **{key: value / accumulation for key, value in sums.items()}}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if step == 1 or step % 8 == 0:
                        log(f"Stage B epoch={epoch}/{CONFIG['epochs']} step={step} views={position}/{len(rows)} "
                            f"loss={record['total']:.6f} KL={record['token_kl']:.6f} CE={record['gold_ce']:.6f}")
                    accumulation, sums = 0, defaultdict(float)
                    if step == 1 or (step % CONFIG["checkpoint_interval_updates"] == 0 and position < len(rows)):
                        checkpoint(output / "last.pt", adapter, optimizer, scaler, epoch, step, position, False, {}, manifest)
                        log(f"Saved resumable Stage B at update={step}")
            adapter.eval()
            metrics = validate(sender, receiver, base, source_tok, hf, adapter, validation, anchors)
            candidate = {"epoch": epoch, "step": step, "validation": metrics}
            history = [entry for entry in history if entry["epoch"] != epoch] + [candidate]
            best = min(history, key=lambda entry: (entry["validation"]["total"], entry["epoch"]))
            if best is candidate:
                checkpoint(output / "best.pt", adapter, None, None, epoch, step, len(rows), True, metrics, manifest)
            save(output / "selection.json", {"selection": "lowest held-out multiview validation total only",
                                              "best": best, "epochs": history})
            checkpoint(output / "last.pt", adapter, optimizer, scaler, epoch, step, len(rows), True, metrics, manifest)
            log(f"EPOCH COMPLETED epoch={epoch}; validation={json.dumps(metrics)}")
            gc.collect()
            torch.cuda.empty_cache()
    save(output / "completed.json", {"epochs": CONFIG["epochs"], "updates": step,
                                     "reused_stage_a": manifest["identities"]["stage_a"],
                                     "old_stage_b_loaded": False})
    log("NEW STAGE B TRAINING COMPLETED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "train"))
    args = parser.parse_args()
    globals()[args.action]()
