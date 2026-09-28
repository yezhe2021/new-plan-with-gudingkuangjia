"""Fresh Gemma Stage-A, then GSM8K Hybrid Stage-B on the fixed Qwen receiver."""

import gc
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import generation_common as C


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def translator_module():
    return C.ref.load_module("gsm8k_gemma_stagea_translator",
                             Path(C.CFG["translator_base"]) / "translator.py")


@torch.no_grad()
def capture_pairs(rows, indices, label):
    gemma_tok, qwen_tok = C.load_tokenizers()
    pairs = []
    gemma = C.ref.load_model("gemma")
    try:
        for position, (row, source_index) in enumerate(zip(rows, indices), 1):
            source_text, _, _ = C.prompt_parts(row["question"])
            aligned = C.alignment_row(f"gsm8k_{source_index}", source_text, gemma_tok, qwen_tok)
            target_indices, source_indices, counts = C.ref.fullsync_map(
                aligned, gemma_tok, qwen_tok, "gemma")
            gf, qf = aligned["encoded"]["gemma"], aligned["encoded"]["qwen"]
            source_k, source_v, _ = C.ref.capture(gemma, gf["body"], len(gf["body"]), [0])
            if len(source_indices) != len(target_indices):
                raise RuntimeError("Full-Sync source/target token count mismatch")
            pairs.append({
                "source_k": source_k[:, source_indices].contiguous().half(),
                "source_v": source_v[:, source_indices].contiguous().half(),
                "qwen_prefix_tokens": len(qf["body"]), "first_qwen_token": qf["body"][0],
                "mapping_counts": counts, "source_text": source_text,
                "question": row["question"], "answer": row["answer"], "id": f"gsm8k_{source_index}",
                "target_indices": target_indices, "target_body": qf["body"],
            })
            if position % 32 == 0 or position == len(rows):
                log(f"{label} Gemma capture {position}/{len(rows)}")
    finally:
        del gemma; gc.collect(); torch.cuda.empty_cache()
    qwen = C.ref.load_model("qwen")
    try:
        for position, item in enumerate(pairs, 1):
            body = item.pop("target_body")
            target_indices = item.pop("target_indices")
            target_k, target_v, _ = C.ref.capture(qwen, body, len(body), [0])
            item["target_k"] = target_k[:, target_indices].contiguous().half()
            item["target_v"] = target_v[:, target_indices].contiguous().half()
            if position % 32 == 0 or position == len(pairs):
                log(f"{label} Qwen capture {position}/{len(pairs)}")
    finally:
        del qwen; gc.collect(); torch.cuda.empty_cache()
    return pairs, gemma_tok, qwen_tok


def stage_a_backward(model, item, scaler, divisor):
    count = item["source_k"].shape[1]
    kden = item["target_k"].float().square().sum().clamp_min(1e-8).item()
    vden = item["target_v"].float().square().sum().clamp_min(1e-8).item()
    total = 0.0
    for begin in range(0, count, C.CFG["chunk_tokens"]):
        stop = min(count, begin + C.CFG["chunk_tokens"])
        sk = item["source_k"][:, begin:stop][None].cuda()
        sv = item["source_v"][:, begin:stop][None].cuda()
        tk = item["target_k"][:, begin:stop][None].cuda().float()
        tv = item["target_v"][:, begin:stop][None].cuda().float()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = model(sk, sv)
        pk, pv = pk.float(), pv.float(); frac = (stop - begin) / count
        k_loss = (pk - tk).square().sum() / kden + frac * (1 - F.cosine_similarity(pk, tk, -1).mean())
        v_loss = (pv - tv).square().sum() / vden + frac * (1 - F.cosine_similarity(pv, tv, -1).mean())
        loss = (k_loss + v_loss) / divisor
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-A loss")
        scaler.scale(loss).backward(); total += loss.detach().item() * divisor
    return total


@torch.no_grad()
def map_stage_a(model, item):
    out_k, out_v = [], []
    for begin in range(0, item["source_k"].shape[1], C.CFG["chunk_tokens"]):
        stop = begin + C.CFG["chunk_tokens"]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            key, value = model(item["source_k"][:, begin:stop][None].cuda(),
                               item["source_v"][:, begin:stop][None].cuda())
        out_k.append(key[0].cpu().half()); out_v.append(value[0].cpu().half())
    return torch.cat(out_k, 1), torch.cat(out_v, 1)


@torch.no_grad()
def representation_metrics(model, pairs, module, label):
    totals = {"k_nmse": 0.0, "v_nmse": 0.0, "k_cosine": 0.0, "v_cosine": 0.0}
    for position, item in enumerate(pairs, 1):
        key, value = map_stage_a(model, item)
        _, kn, kc = module.component_loss(key.cuda(), item["target_k"].cuda())
        _, vn, vc = module.component_loss(value.cuda(), item["target_v"].cuda())
        for name, metric in (("k_nmse", kn), ("v_nmse", vn), ("k_cosine", kc), ("v_cosine", vc)):
            totals[name] += metric.item()
        if position % 32 == 0: log(f"{label} representation validation {position}/{len(pairs)}")
    metrics = {name: value / len(pairs) for name, value in totals.items()}
    metrics["representation_loss"] = (metrics["k_nmse"] + metrics["v_nmse"]
        + 2 - metrics["k_cosine"] - metrics["v_cosine"])
    metrics["count"] = len(pairs); return metrics


def save_checkpoint(path, stage, model, epoch, step, metrics):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": stage, "epoch": epoch, "step": step, "metrics": metrics,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_checkpoint(path, model, expected):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["stage"] != expected: raise RuntimeError("Checkpoint stage mismatch")
    model.load_state_dict(payload["state"], strict=True); return model


def optimizer_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), C.CFG["gradient_clip"])
    if not torch.isfinite(norm): raise RuntimeError("Nonfinite gradient norm")
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return norm.item()


def train_stage_a(train_pairs, val_pairs, module):
    random.seed(C.CFG["seed"]); torch.manual_seed(C.CFG["seed"]); torch.cuda.manual_seed_all(C.CFG["seed"])
    model = module.NativeKVTranslator("full34_headmix256", hidden_dim=1024,
                                      depth_output_dim=256, head_mapping="full_head").cuda().train()
    zeros = torch.zeros((1, 34, 2, 4, 256), device="cuda")
    zk, zv = model(zeros, zeros)
    if zk.abs().max().item() != 0 or zv.abs().max().item() != 0:
        raise RuntimeError("Stage-A bias-free zero identity failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.CFG["stage_a_learning_rate"],
                                  weight_decay=0, foreach=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = C.HERE / "runs/stage_a"; root.mkdir(parents=True, exist_ok=True)
    steps = root / "training_steps.jsonl"; steps.unlink(missing_ok=True)
    candidates, step = [], 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, C.CFG["stage_a_epochs"] + 1):
        order = list(range(len(train_pairs))); random.Random(C.CFG["seed"] + epoch).shuffle(order)
        accumulated, loss_sum = 0, 0.0
        for position, index in enumerate(order, 1):
            accumulated += 1
            loss_sum += stage_a_backward(model, train_pairs[index], scaler, C.CFG["effective_batch"])
            if accumulated == C.CFG["effective_batch"] or position == len(order):
                norm = optimizer_step(model, optimizer, scaler); step += 1
                record = {"epoch": epoch, "step": step, "samples_seen": position,
                          "representation_loss": loss_sum / accumulated,
                          "pre_clip_grad_norm": norm, "clipped": norm > C.CFG["gradient_clip"]}
                with steps.open("a", encoding="utf-8") as stream: stream.write(json.dumps(record) + "\n")
                accumulated, loss_sum = 0, 0.0
                if step % 16 == 0: log(f"Stage-A epoch={epoch} step={step} samples={position}/{len(train_pairs)} loss={record['representation_loss']:.4f}")
        metrics = representation_metrics(model.eval(), val_pairs, module, f"Stage-A epoch={epoch}")
        path = root / f"checkpoints/epoch_{epoch}.pt"; save_checkpoint(path, "gsm8k_gemma_stage_a", model, epoch, step, metrics)
        candidates.append({"epoch": epoch, "step": step, "checkpoint": str(path), "validation": metrics})
        log(f"Stage-A epoch={epoch} validation={metrics}"); model.train()
    best = min(candidates, key=lambda x: (x["validation"]["representation_loss"], x["epoch"]))
    C.save_json(root / "selection.json", {"selection": "lowest validation representation loss", "best": best, "epochs": candidates})
    return load_checkpoint(best["checkpoint"], model.eval().requires_grad_(False), "gsm8k_gemma_stage_a"), best


def convert_pairs(model, pairs, label):
    output = []
    for position, item in enumerate(pairs, 1):
        key, value = map_stage_a(model, item)
        if key.shape[1] + 1 != item["qwen_prefix_tokens"]: raise RuntimeError("Mapped cache length mismatch")
        output.append({"base_k": key, "base_v": value, "qwen_prefix_tokens": item["qwen_prefix_tokens"],
            "first_qwen_token": item["first_qwen_token"], "source_text": item["source_text"],
            "question": item["question"], "answer": item["answer"], "id": item["id"]})
        if position % 32 == 0: log(f"{label} selected Stage-A map {position}/{len(pairs)}")
    return output


@torch.no_grad()
def validate_hybrid(qwen, tok, adapter, cache, token0_k, token0_v, epoch):
    adapter.eval(); rows, sums = [], {"trajectory_kl": 0.0, "answer_ce": 0.0, "eos_ce": 0.0}
    for position, entry in enumerate(cache, 1):
        key, value = C.adapted_cache(adapter, entry, token0_k, token0_v)
        teacher, _, targets = C.native_trajectory(qwen, tok, entry["source_text"], entry["answer"])
        student, eos_logits, _ = C.cached_trajectory(qwen, tok, key, value, entry["answer"])
        kl, ce, eos_ce = C.loss_components(student, eos_logits, teacher, targets, tok.eos_token_id)
        text, count, reason = C.greedy_generate(qwen, tok, list(tok.encode("Answer:", add_special_tokens=False)),
                                               C.CFG["max_new_tokens"], key, value)
        record = C.generation_record(text, count, reason, entry["answer"])
        record.update(id=entry["id"], trajectory_kl=kl.item(), answer_ce=ce.item(), eos_ce=eos_ce.item())
        rows.append(record)
        for name, metric in (("trajectory_kl", kl), ("answer_ce", ce), ("eos_ce", eos_ce)): sums[name] += metric.item()
        if position % 16 == 0: log(f"Hybrid epoch={epoch} validation {position}/{len(cache)}")
    summary = C.summarize_generations(rows); summary.update({k: v/len(rows) for k,v in sums.items()})
    C.write_jsonl(C.HERE / f"runs/hybrid/validation_epoch_{epoch}.jsonl", rows)
    C.save_json(C.HERE / f"runs/hybrid/validation_epoch_{epoch}.json", summary)
    adapter.train(); return summary


def train_hybrid(qwen, tok, module, train_cache, val_cache, token0_k, token0_v, stage_a_best):
    random.seed(C.CFG["seed"]); torch.manual_seed(C.CFG["seed"]); torch.cuda.manual_seed_all(C.CFG["seed"])
    adapter = C.load_adapter(module).train()
    first = train_cache[0]; key, value = C.adapted_cache(adapter, first, token0_k, token0_v)
    identity = max((key[:, 1:] - first["base_k"].cuda()).abs().max().item(),
                   (value[:, 1:] - first["base_v"].cuda()).abs().max().item())
    if identity != 0: raise RuntimeError("Zero residual parity failed")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=C.CFG["stage_b_learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    root = C.HERE / "runs/hybrid"; root.mkdir(parents=True, exist_ok=True)
    steps = root / "training_steps.jsonl"; steps.unlink(missing_ok=True)
    candidates, step = [], 0
    for epoch in range(1, C.CFG["stage_b_epochs"] + 1):
        order = list(range(len(train_cache))); random.Random(C.CFG["seed"] + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); accumulated = 0
        sums = {"trajectory_kl": 0.0, "answer_ce": 0.0, "eos_ce": 0.0, "total_loss": 0.0}
        for position, index in enumerate(order, 1):
            entry = train_cache[index]; key, value = C.adapted_cache(adapter, entry, token0_k, token0_v)
            with torch.no_grad(): teacher, _, targets = C.native_trajectory(qwen, tok, entry["source_text"], entry["answer"])
            student, eos_logits, _ = C.cached_trajectory(qwen, tok, key, value, entry["answer"])
            kl, ce, eos_ce = C.loss_components(student, eos_logits, teacher, targets, tok.eos_token_id)
            loss = C.objective_loss("hybrid", kl, ce, eos_ce)
            scaler.scale(loss / C.CFG["effective_batch"]).backward(); accumulated += 1
            for name, metric in (("trajectory_kl", kl), ("answer_ce", ce), ("eos_ce", eos_ce), ("total_loss", loss)):
                sums[name] += metric.detach().item()
            if accumulated == C.CFG["effective_batch"] or position == len(order):
                norm = optimizer_step(adapter, optimizer, scaler); step += 1
                record = {"epoch": epoch, "step": step, "samples_seen": position,
                          **{k: v/accumulated for k,v in sums.items()},
                          "pre_clip_grad_norm": norm, "clipped": norm > C.CFG["gradient_clip"]}
                with steps.open("a", encoding="utf-8") as stream: stream.write(json.dumps(record) + "\n")
                accumulated = 0; sums = {k: 0.0 for k in sums}
                if step % 16 == 0: log(f"Hybrid epoch={epoch} step={step} samples={position}/{len(train_cache)} loss={record['total_loss']:.4f}")
        log(f"Hybrid epoch={epoch} completed; optimizer_steps={step}")
        if epoch == C.CFG["stage_b_epochs"]:
            summary = validate_hybrid(qwen, tok, adapter, val_cache, token0_k, token0_v, epoch)
            path = root / f"checkpoints/epoch_{epoch}.pt"
            save_checkpoint(path, "gsm8k_hybrid_stage_b", adapter, epoch, step, summary)
            candidates.append({"epoch": epoch, "step": step, "checkpoint": str(path), "validation": summary})
    best=candidates[0]
    C.save_json(root/"selection.json",{"selection":"predeclared final epoch; validation only after epoch 4", "stage_a":stage_a_best,"best":best,"epochs":candidates})
    return load_checkpoint(best["checkpoint"], adapter.eval().requires_grad_(False), "gsm8k_hybrid_stage_b"), best


@torch.no_grad()
def evaluate_test(stage_a, adapter, module, gemma_tok, tok):
    rows=C.read_jsonl(C.CFG["data_test"])[:C.CFG["test_samples"]]; indices=list(range(len(rows)))
    pairs, _, _=capture_pairs(rows,indices,"test")
    stage_a.cuda()
    cache=convert_pairs(stage_a,pairs,"test")
    stage_a.cpu(); del pairs; gc.collect(); torch.cuda.empty_cache()
    qwen=C.ref.load_model("qwen").requires_grad_(False); token0_k,token0_v,_=C.native_token0(qwen,tok,cache)
    records=[]; by={"gemma_stage_a":[],"gemma_stage_a_hybrid2048x4":[]}
    try:
        suffix=list(tok.encode("Answer:",add_special_tokens=False))
        for position,(row,entry) in enumerate(zip(rows,cache),1):
            key=torch.cat((token0_k,entry["base_k"].cuda()),1); value=torch.cat((token0_v,entry["base_v"].cuda()),1)
            text,count,reason=C.greedy_generate(qwen,tok,suffix,C.CFG["max_new_tokens"],key,value)
            a=C.generation_record(text,count,reason,row["answer"])
            bk,bv=C.adapted_cache(adapter,entry,token0_k,token0_v)
            text,count,reason=C.greedy_generate(qwen,tok,suffix,C.CFG["max_new_tokens"],bk,bv)
            b=C.generation_record(text,count,reason,row["answer"])
            by["gemma_stage_a"].append(a); by["gemma_stage_a_hybrid2048x4"].append(b)
            records.append({"id":f"test_{position-1}","question":row["question"],"gold_answer":row["answer"],"conditions":{"gemma_stage_a":a,"gemma_stage_a_hybrid2048x4":b}})
            if position%8==0: log(f"test generation {position}/128")
    finally:
        del qwen,cache; gc.collect(); torch.cuda.empty_cache()
    summary={name:C.summarize_generations(values) for name,values in by.items()}
    reference=C.read_jsonl(C.CFG["llama_reference_per_sample"])
    if len(reference) != len(records) or any(r["id"] != x["id"] or
            r["question"] != x["question"] for r,x in zip(records,reference)):
        raise RuntimeError("Gemma/Llama test ID or question mismatch")
    gemma=[r["conditions"]["gemma_stage_a_hybrid2048x4"]["flexible_correct"] for r in records]
    llama=[r["conditions"]["gsm8k_stage_a_hybrid2048x4"]["flexible_correct"] for r in reference]
    summary.update(status="completed",test_samples=len(rows),
                   llama_reference_summary=C.CFG["llama_reference_summary"],
                   paired_flexible_correct={"both":sum(a and b for a,b in zip(gemma,llama)),
                       "gemma_only":sum(a and not b for a,b in zip(gemma,llama)),
                       "llama_only":sum(b and not a for a,b in zip(gemma,llama)),
                       "neither":sum(not a and not b for a,b in zip(gemma,llama))},
                   official_test_untouched_by_training=True)
    C.write_jsonl(C.HERE/"runs/test128_per_sample.jsonl",records); C.save_json(C.HERE/"runs/test128_summary.json",summary)


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    train_rows,val_rows,manifest=C.prepare_manifest(); module=translator_module()
    train_pairs,gemma_tok,tok=capture_pairs(train_rows,manifest["train_indices"],"train")
    val_pairs,_,_=capture_pairs(val_rows,manifest["val_indices"],"validation")
    stage_a,stage_a_best=train_stage_a(train_pairs[:C.CFG["stage_a_train_samples"]],val_pairs,module)
    train_cache=convert_pairs(stage_a,train_pairs,"train"); val_cache=convert_pairs(stage_a,val_pairs,"validation")
    stage_a.cpu(); del train_pairs,val_pairs; gc.collect(); torch.cuda.empty_cache()
    qwen=C.ref.load_model("qwen").requires_grad_(False); token0_k,token0_v,_=C.native_token0(qwen,tok,train_cache+val_cache)
    adapter,hybrid_best=train_hybrid(qwen,tok,module,train_cache,val_cache,token0_k,token0_v,stage_a_best)
    del qwen,train_cache,val_cache; gc.collect(); torch.cuda.empty_cache()
    evaluate_test(stage_a,adapter,module,gemma_tok,tok)
    C.save_json(C.HERE/"runs/selection_summary.json",{"stage_a":stage_a_best,"hybrid":hybrid_best})
    log("GSM8K GEMMA STAGE-A + FOUR-EPOCH HYBRID STAGE-B COMPLETED")


if __name__=="__main__": main()
