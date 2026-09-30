"""Unified full-body Stage-A and functional Stage-B for Llama->Qwen."""

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
BASE = Path(CFG["base_code"])
sys.path.insert(0, str(BASE))

from common import load_model, save_json, seed_all  # noqa: E402
from data import official_rows, serialize as serialize_mcq  # noqa: E402
from protocol import make_cache  # noqa: E402
from translator import NativeKVTranslator, ResidualKVAdapter  # noqa: E402

LABELS = "ABCDEFGHIJ"


def backbone(model):
    """Return the causal decoder for ordinary and nested HF model wrappers."""
    return getattr(model.model, "language_model", model.model)


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runtime_config(smoke=False):
    cfg = dict(CFG)
    cfg["smoke"] = bool(smoke)
    if smoke:
        cfg["stage_a_epochs"] = cfg["stage_b_epochs"] = 1
        cfg["effective_batch"] = 2
        cfg["stage_b_epoch_exposures"] = 4
    return cfg


def model_config(cfg):
    return {"models": cfg["models"], "attention_implementation": cfg["attention_implementation"]}


def tokenizers(cfg):
    return {name: AutoTokenizer.from_pretrained(path, local_files_only=True)
            for name, path in cfg["models"].items()}


def gsm_split_indices(cfg):
    rows = read_jsonl(cfg["datasets"]["gsm8k_train"])
    order = list(range(len(rows))); random.Random(cfg["seed"]).shuffle(order)
    val = order[1024:1152]
    train = order[:1024] + order[1152:1152 + cfg["train_counts"]["gsm8k"] - 1024]
    test = list(range(cfg["test_counts"]["gsm8k"]))
    if len(train) != cfg["train_counts"]["gsm8k"] or set(train) & set(val):
        raise RuntimeError("GSM8K split construction failed")
    return rows, {"train": train, "validation": val, "test": test}


def raw_mcq_rows(cfg, split):
    proxy = dict(cfg)
    proxy["per_dataset_samples"] = {name: cfg["train_counts" if name == "train" else
        "validation_counts" if name == "validation" else "test_counts"]["openbookqa"]
        for name in ("train", "validation", "test")}
    proxy["train_samples"] = proxy["per_dataset_samples"]["train"] * 3
    proxy["validation_samples"] = proxy["per_dataset_samples"]["validation"] * 3
    proxy["test_samples"] = proxy["per_dataset_samples"]["test"] * 3
    return official_rows(proxy, split)


def raw_hellaswag(cfg):
    items = [json.loads(x) for x in Path(cfg["datasets"]["hellaswag"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [x for x in items if str(x.get("label", "")).isdigit()]
    random.Random(cfg["seed"] + 911).shuffle(items)
    return [{"id": f"hellaswag_{x['ind']}", "dataset": "hellaswag",
             "category": x.get("activity_label", "unknown"),
             "question": "Complete the following passage:\n" + x["ctx"],
             "options": list(x["endings"]), "gold_index": int(x["label"])}
            for x in items[:cfg["test_counts"]["hellaswag"]]]


def serialize_gsm(row):
    question = str(row["question"]).strip()
    body = f"Question:\n{question}\n\n"
    return {"body": body, "answer_prefix": "Answer:", "full_prompt": body + "Answer:",
            "continuation_text": str(row["answer"]), "gold_text": str(row["answer"]),
            "labels": [], "gold_index": None}


def encode_row(tok, serialized, task):
    enc = lambda text: list(tok.encode(text, add_special_tokens=False))
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    body_plain = enc(serialized["body"])
    suffix = enc(serialized["answer_prefix"])
    if body_plain + suffix != enc(serialized["full_prompt"]):
        raise RuntimeError("Noncompositional body/Answer boundary")
    continuation = enc(serialized["continuation_text"])
    if not suffix or not continuation:
        raise RuntimeError("Empty suffix or continuation")
    result = {"body": bos + body_plain, "suffix": suffix, "continuation": continuation}
    if task == "mcq":
        choice_ids = []
        for label in serialized["labels"]:
            joined = enc(serialized["full_prompt"] + " " + label)
            prefix = enc(serialized["full_prompt"])
            if joined[:len(prefix)] != prefix or len(joined) != len(prefix) + 1:
                raise RuntimeError(f"Unstable choice token {label}")
            choice_ids.append(joined[-1])
        result["choice_ids"] = choice_ids
    return result


def prepare_rows(cfg, split, toks):
    output = []
    if split != "ood":
        train_rows, indices = gsm_split_indices(cfg)
        source = train_rows if split != "test" else read_jsonl(cfg["datasets"]["gsm8k_test"])
        for index in indices[split]:
            row = source[index]
            serialized = serialize_gsm(row)
            encoded = {name: encode_row(tok, serialized, "gsm8k") for name, tok in toks.items()}
            output.append({"id": f"gsm8k_{split}_{index}", "dataset": "gsm8k", "task": "generation",
                           "question": row["question"], "gold_answer": row["answer"],
                           **serialized, "encoded": encoded})
        for row in raw_mcq_rows(cfg, split):
            serialized = serialize_mcq(row)
            serialized["continuation_text"] = " " + serialized["gold_label"]
            encoded = {name: encode_row(tok, serialized, "mcq") for name, tok in toks.items()}
            output.append({**row, "task": "mcq", **serialized, "encoded": encoded})
    else:
        for row in raw_hellaswag(cfg):
            serialized = serialize_mcq(row)
            serialized["continuation_text"] = " " + serialized["gold_label"]
            encoded = {name: encode_row(tok, serialized, "mcq") for name, tok in toks.items()}
            output.append({**row, "task": "mcq", **serialized, "encoded": encoded})
    for row in output:
        lengths = {name: len(value["body"]) for name, value in row["encoded"].items()}
        if max(lengths.values()) > cfg["max_body_tokens"]:
            raise RuntimeError(f"Body exceeds safety ceiling: {row['id']} {lengths}")
        row["body_tokens"] = lengths
    if cfg["smoke"]:
        groups = defaultdict(list)
        for row in output: groups[row["dataset"]].append(row)
        output = [max(rows, key=lambda x: x["body_tokens"]["qwen"]) for rows in groups.values()]
    return output


@dataclass(frozen=True)
class Span:
    index: int
    start: int
    end: int


def token_spans(tok, text, expected):
    value = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    if bos + list(value["input_ids"]) != list(expected):
        raise RuntimeError("Offset tokenization mismatch")
    offsets = [(0, 0)] * len(bos) + [tuple(map(int, x)) for x in value["offset_mapping"]]
    return [Span(i, a, b) for i, (a, b) in enumerate(offsets)]


def source_rank(source_count, target_count, target_rank):
    if source_count > target_count:
        return math.ceil((target_rank + 1) * source_count / target_count) - 1
    return target_rank * source_count // target_count


def fullsync_map(row, tok_l, tok_q):
    source = token_spans(tok_l, row["body"], row["encoded"]["llama"]["body"])
    target = token_spans(tok_q, row["body"], row["encoded"]["qwen"]["body"])
    byte_at = [0]
    for character in row["body"]: byte_at.append(byte_at[-1] + len(character.encode("utf-8")))
    def non_bos(spans):
        return [(x.index, byte_at[x.start], byte_at[x.end]) for x in spans if x.index != 0 and x.start != x.end]
    sb, tb = non_bos(source), non_bos(target)
    boundaries = sorted(({end for _, _, end in sb} & {end for _, _, end in tb}) | {0})
    wanted = set(range(1, len(row["encoded"]["qwen"]["body"])))
    mapping, counts = {}, Counter()
    for left, right in zip(boundaries, boundaries[1:]):
        si = [index for index, _, end in sb if left < end <= right]
        ti = [index for index, _, end in tb if left < end <= right]
        if not si or not ti: continue
        counts[("one_to_one" if len(si) == len(ti) == 1 else "copy" if len(si) == 1 else
                "drop" if len(ti) == 1 else "many_to_many") + "_units"] += 1
        for rank, target_index in enumerate(ti):
            if target_index in wanted:
                mapping[target_index] = si[source_rank(len(si), len(ti), rank)]
    if set(mapping) != wanted:
        raise RuntimeError(f"Unmapped full-body tokens: {row['id']} {len(wanted-set(mapping))}")
    target_indices = list(range(1, len(row["encoded"]["qwen"]["body"])))
    return target_indices, [mapping[x] for x in target_indices], dict(counts)


@torch.no_grad()
def capture_body(model, ids):
    captured_k, captured_v, handles = {}, {}, []
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            value = module.v_proj(hidden).view(shape)
            if hasattr(module, "k_norm"): key = module.k_norm(key)
            captured_k[index], captured_v[index] = key[0].cpu(), value[0].cpu()
        return apply
    body = backbone(model)
    for index, layer in enumerate(body.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
        body(input_ids=tensor, attention_mask=torch.ones_like(tensor),
             position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False)
    finally:
        for handle in handles: handle.remove()
    return torch.stack([captured_k[i] for i in range(len(body.layers))]), \
           torch.stack([captured_v[i] for i in range(len(body.layers))])


@torch.no_grad()
def pair(llama, qwen, tok_l, tok_q, row):
    target_indices, source_indices, counts = fullsync_map(row, tok_l, tok_q)
    lk, lv = capture_body(llama, row["encoded"]["llama"]["body"])
    qk, qv = capture_body(qwen, row["encoded"]["qwen"]["body"])
    return {"source_k": lk[:, source_indices].contiguous(), "source_v": lv[:, source_indices].contiguous(),
            "target_k": qk[:, target_indices].contiguous(), "target_v": qv[:, target_indices].contiguous(),
            "token0_k": qk[:, :1].contiguous(), "token0_v": qv[:, :1].contiguous(),
            "tokens": len(target_indices), "mapping_counts": counts}


def mapped_base(base, item, chunk):
    output_k, output_v = [], []
    with torch.no_grad():
        for begin in range(0, item["tokens"], chunk):
            stop = min(begin + chunk, item["tokens"])
            with torch.amp.autocast("cuda", dtype=torch.float16):
                key, value = base(item["source_k"][:, begin:stop][None].cuda(),
                                  item["source_v"][:, begin:stop][None].cuda())
            output_k.append(key[0]); output_v.append(value[0])
    return torch.cat(output_k, 1), torch.cat(output_v, 1)


def stage_a_backward(cfg, model, item, scaler, divisor):
    count = item["tokens"]
    kden = item["target_k"].float().square().sum().clamp_min(1e-8).item()
    vden = item["target_v"].float().square().sum().clamp_min(1e-8).item()
    aggregate = 0.0
    for begin in range(0, count, cfg["chunk_tokens"]):
        stop = min(begin + cfg["chunk_tokens"], count)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = model(item["source_k"][:, begin:stop][None].cuda(),
                           item["source_v"][:, begin:stop][None].cuda())
        tk = item["target_k"][:, begin:stop][None].cuda().float()
        tv = item["target_v"][:, begin:stop][None].cuda().float()
        fraction = (stop - begin) / count
        loss = ((pk.float()-tk).square().sum()/kden + fraction*(1-F.cosine_similarity(pk.float(),tk,-1).mean()) +
                (pv.float()-tv).square().sum()/vden + fraction*(1-F.cosine_similarity(pv.float(),tv,-1).mean())) / divisor
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite Stage-A loss")
        scaler.scale(loss).backward(); aggregate += loss.detach().item() * divisor
    return aggregate


def optimizer_step(cfg, model, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
    if not torch.isfinite(norm): raise RuntimeError("Nonfinite gradient norm")
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return norm.item()


def save_checkpoint(path, cfg, model, stage, epoch, step, metrics):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": stage, "protocol": cfg["protocol"], "epoch": epoch, "step": step,
                "metrics": metrics, "state": {k: v.detach().cpu() for k,v in model.state_dict().items()}}, path)


def load_checkpoint(path, cfg, stage, model):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["stage"] != stage or payload["protocol"] != cfg["protocol"]:
        raise RuntimeError("Checkpoint/protocol mismatch")
    model.load_state_dict(payload["state"], strict=True)
    return model


def trajectory_logits(qwen, row, key=None, value=None):
    fields = row["encoded"]["qwen"]
    suffix, continuation = fields["suffix"], fields["continuation"]
    if key is None:
        ids = fields["body"] + suffix + continuation
        prefix, cache = 0, None
        first = len(fields["body"]) + len(suffix) - 1
    else:
        ids = suffix + continuation
        prefix = key.shape[1]
        cache = make_cache(qwen, key, value, torch.arange(prefix, device="cuda"))
        first = len(suffix) - 1
    tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = backbone(qwen)(input_ids=tensor,
        attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
        position_ids=torch.arange(prefix, prefix + len(ids), device="cuda")[None],
        past_key_values=cache, use_cache=False, return_dict=True)
    logits = qwen.lm_head(output.last_hidden_state)[0]
    answer_logits = logits[first:first+len(continuation)]
    eos_logits = logits[first+len(continuation)]
    return answer_logits, eos_logits, torch.tensor(continuation, device="cuda")


@torch.no_grad()
def next_token_logits(qwen, ids=None, key=None, value=None):
    """Return logits after a native prompt or a supplied past plus native suffix."""
    if ids is None:
        raise ValueError("ids are required")
    if key is None:
        prefix, cache = 0, None
    else:
        prefix = key.shape[1]
        cache = make_cache(qwen, key, value, torch.arange(prefix, device="cuda"))
    tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = backbone(qwen)(
        input_ids=tensor,
        attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
        position_ids=torch.arange(prefix, prefix + len(ids), device="cuda")[None],
        past_key_values=cache, use_cache=False, return_dict=True)
    return qwen.lm_head(output.last_hidden_state[:, -1])[0]


def losses(qwen, row, key, value, cfg, objective):
    with torch.no_grad(): teacher, _, _ = trajectory_logits(qwen, row)
    student, eos_logits, targets = trajectory_logits(qwen, row, key, value)
    teacher_prob = F.softmax(teacher.float()/cfg["temperature"], -1)
    kl = F.kl_div(F.log_softmax(student.float()/cfg["temperature"], -1), teacher_prob,
                  reduction="batchmean") * cfg["temperature"]**2
    ce = F.cross_entropy(student.float(), targets)
    eos_id = qwen.generation_config.eos_token_id
    eos_id = int(eos_id if isinstance(eos_id, int) else eos_id[0])
    eos = F.cross_entropy(eos_logits.float()[None], torch.tensor([eos_id], device="cuda"))
    total = kl if objective == "universal_kl" else kl + cfg["ce_weight"]*ce + cfg["eos_weight"]*eos
    return total, kl, ce, eos


def representation_metrics(cfg, llama, qwen, tok_l, tok_q, base, rows, label):
    totals = defaultdict(float)
    base.eval()
    with torch.no_grad():
        for number, row in enumerate(rows, 1):
            item = pair(llama,qwen,tok_l,tok_q,row); pk,pv = mapped_base(base,item,cfg["chunk_tokens"])
            for name,pred,target in (("k",pk,item["target_k"]),("v",pv,item["target_v"])):
                pred,target=pred.float(),target.cuda().float()
                totals[name+"_nmse"] += ((pred-target).square().mean()/target.square().mean().clamp_min(1e-8)).item()
                totals[name+"_cosine"] += F.cosine_similarity(pred,target,-1).mean().item()
            if number % 32 == 0 or number == len(rows): log(f"{label} {number}/{len(rows)}")
            del item,pk,pv
    result={k:v/len(rows) for k,v in totals.items()}; result["count"]=len(rows)
    result["representation_loss"] = result["k_nmse"]+result["v_nmse"]+2-result["k_cosine"]-result["v_cosine"]
    return result


def mixed_stage_b_order(cfg, rows, epoch):
    generation=[i for i,r in enumerate(rows) if r["task"]=="generation"]
    mcq=[i for i,r in enumerate(rows) if r["task"]=="mcq"]
    each=cfg["stage_b_epoch_exposures"]//2
    def take(values, salt):
        rng=random.Random(cfg["seed"]+salt); pool=[]
        while len(pool)<each:
            current=list(values); rng.shuffle(current); pool.extend(current)
        return pool[:each]
    left,right=take(generation,1000+epoch),take(mcq,2000+epoch)
    order=[]
    for a,b in zip(left,right): order.extend((a,b))
    return order


def stage_a(cfg):
    seed_all(cfg["seed"]); toks=tokenizers(cfg)
    train,validation=prepare_rows(cfg,"train",toks),prepare_rows(cfg,"validation",toks)
    llama,qwen=load_model(model_config(cfg),"llama"),load_model(model_config(cfg),"qwen")
    model=NativeKVTranslator("full28_mlp",hidden_dim=cfg["mlp_hidden_dim"]).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["stage_a_learning_rate"],weight_decay=0)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.0,growth_interval=1000000)
    root=HERE/("runs_smoke" if cfg["smoke"] else "runs")/"stage_a"; root.mkdir(parents=True,exist_ok=True)
    candidates=[]; step=0
    with (root/"training_steps.jsonl").open("w",encoding="utf-8") as stream:
        for epoch in range(1,cfg["stage_a_epochs"]+1):
            order=list(range(len(train))); random.Random(cfg["seed"]+epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True); accumulation=batch_loss=0
            for position,index in enumerate(order,1):
                item=pair(llama,qwen,toks["llama"],toks["qwen"],train[index]); accumulation+=1
                divisor=min(cfg["effective_batch"],len(order)-(position-accumulation))
                batch_loss+=stage_a_backward(cfg,model,item,scaler,divisor); del item
                if accumulation==cfg["effective_batch"] or position==len(order):
                    norm=optimizer_step(cfg,model,optimizer,scaler); step+=1
                    stream.write(json.dumps({"epoch":epoch,"step":step,"loss_per_sample":batch_loss/accumulation,
                        "pre_clip_grad_norm":norm,"clipped":norm>cfg["gradient_clip"]})+"\n"); stream.flush()
                    accumulation=batch_loss=0
                    if step%16==0 or cfg["smoke"]: log(f"Stage-A epoch={epoch} step={step} samples={position}/{len(order)}")
            metrics=representation_metrics(cfg,llama,qwen,toks["llama"],toks["qwen"],model,validation,f"Stage-A val epoch={epoch}")
            candidate={"epoch":epoch,"step":step,"validation":metrics}; candidates.append(candidate)
            best=min(candidates,key=lambda x:(x["validation"]["representation_loss"],x["epoch"]))
            if best is candidate:
                save_checkpoint(root/"best.pt",cfg,model,"stage_a",epoch,step,metrics)
            save_json(root/"selection.json",{"selection":"lowest validation representation loss","best":best,"epochs":candidates})
            log(f"Stage-A epoch={epoch} representation_loss={metrics['representation_loss']:.6f}"); model.train()
    del llama,qwen,model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def functional_validation(cfg,llama,qwen,tok_l,tok_q,base,adapter,rows,objective):
    sums=defaultdict(float); counts=defaultdict(int)
    adapter.eval()
    for number,row in enumerate(rows,1):
        item=pair(llama,qwen,tok_l,tok_q,row); bk,bv=mapped_base(base,item,cfg["chunk_tokens"])
        with torch.amp.autocast("cuda",dtype=torch.float16): pk,pv,_,_=adapter(bk[None],bv[None])
        key=torch.cat((item["token0_k"].cuda(),pk[0]),1); value=torch.cat((item["token0_v"].cuda(),pv[0]),1)
        total,kl,ce,eos=losses(qwen,row,key,value,cfg,objective)
        for name,x in (("total",total),("kl",kl),("ce",ce),("eos",eos)): sums[name]+=x.item()
        counts[row["task"]]+=1; sums[row["task"]+"_kl"]+=kl.item()
        if number%32==0 or number==len(rows): log(f"{objective} validation {number}/{len(rows)}")
    result={name:value/len(rows) for name,value in sums.items() if not name.endswith("_kl") or name=="kl"}
    for task in ("generation","mcq"): result[task+"_kl"]=sums[task+"_kl"]/counts[task]
    result["count"]=len(rows); adapter.train(); return result


def stage_b(cfg,objective):
    seed_all(cfg["seed"]+(1 if objective=="universal_kl" else 2)); toks=tokenizers(cfg)
    train,validation=prepare_rows(cfg,"train",toks),prepare_rows(cfg,"validation",toks)
    root_base=HERE/("runs_smoke" if cfg["smoke"] else "runs")
    selection=json.loads((root_base/"stage_a/selection.json").read_text(encoding="utf-8"))
    base=NativeKVTranslator("full28_mlp",hidden_dim=cfg["mlp_hidden_dim"]).cuda().eval().requires_grad_(False)
    load_checkpoint(root_base/"stage_a/best.pt",cfg,"stage_a",base)
    adapter=ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().train()
    llama,qwen=load_model(model_config(cfg),"llama"),load_model(model_config(cfg),"qwen")
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["stage_b_learning_rate"],weight_decay=0)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.0,growth_interval=1000000)
    root=root_base/objective; root.mkdir(parents=True,exist_ok=True); candidates=[]; step=0
    with (root/"training_steps.jsonl").open("w",encoding="utf-8") as stream:
        for epoch in range(1,cfg["stage_b_epochs"]+1):
            order=mixed_stage_b_order(cfg,train,epoch); optimizer.zero_grad(set_to_none=True); accumulation=0; sums=defaultdict(float)
            for position,index in enumerate(order,1):
                row=train[index]; item=pair(llama,qwen,toks["llama"],toks["qwen"],row); bk,bv=mapped_base(base,item,cfg["chunk_tokens"])
                with torch.amp.autocast("cuda",dtype=torch.float16): pk,pv,_,_=adapter(bk[None],bv[None])
                key=torch.cat((item["token0_k"].cuda(),pk[0]),1); value=torch.cat((item["token0_v"].cuda(),pv[0]),1)
                total,kl,ce,eos=losses(qwen,row,key,value,cfg,objective)
                scaler.scale(total/cfg["effective_batch"]).backward(); accumulation+=1
                for name,x in (("total",total),("kl",kl),("ce",ce),("eos",eos)): sums[name]+=x.detach().item()
                del item,bk,bv,pk,pv,key,value,total,kl,ce,eos
                if accumulation==cfg["effective_batch"] or position==len(order):
                    norm=optimizer_step(cfg,adapter,optimizer,scaler); step+=1
                    record={"epoch":epoch,"step":step,"samples_seen":position,
                            **{name:value/accumulation for name,value in sums.items()},
                            "pre_clip_grad_norm":norm,"clipped":norm>cfg["gradient_clip"]}
                    stream.write(json.dumps(record)+"\n"); stream.flush(); accumulation=0; sums=defaultdict(float)
                    if step%16==0 or cfg["smoke"]: log(f"{objective} epoch={epoch} step={step} samples={position}/{len(order)} loss={record['total']:.4f}")
            metrics=functional_validation(cfg,llama,qwen,toks["llama"],toks["qwen"],base,adapter,validation,objective)
            candidate={"epoch":epoch,"step":step,"validation":metrics}; candidates.append(candidate)
            best=min(candidates,key=lambda x:(x["validation"]["total"],x["epoch"]))
            if best is candidate: save_checkpoint(root/"best.pt",cfg,adapter,objective,epoch,step,metrics)
            save_json(root/"selection.json",{"selection":"lowest mixed validation objective","stage_a":selection["best"],"best":best,"epochs":candidates})
            log(f"{objective} epoch={epoch} validation={metrics}")
    del llama,qwen,base,adapter; gc.collect(); torch.cuda.empty_cache()


def prediction_record(logits,gold):
    pred=int(logits.argmax()); return {"prediction":pred,"correct":pred==gold,"choice_logits":logits.float().tolist()}


@torch.no_grad()
def greedy_generate(qwen,tok,prompt,max_tokens,key=None,value=None):
    past = None if key is None else make_cache(qwen,key,value,torch.arange(key.shape[1],device="cuda"))
    current=torch.tensor([prompt],device="cuda")
    eos=qwen.generation_config.eos_token_id; eos={int(eos)} if isinstance(eos,int) else {int(x) for x in eos}
    generated=[]; reason="max_tokens"
    for _ in range(max_tokens):
        prefix=0 if past is None else int(past.get_seq_length()); output=backbone(qwen)(input_ids=current,
            attention_mask=torch.ones((1,prefix+current.shape[1]),device="cuda",dtype=torch.long),
            position_ids=torch.arange(prefix,prefix+current.shape[1],device="cuda")[None],
            past_key_values=past,use_cache=True,return_dict=True)
        token=int(qwen.lm_head(output.last_hidden_state[:,-1])[0].argmax()); past=output.past_key_values
        if token in eos: reason="eos"; break
        generated.append(token); current=torch.tensor([[token]],device="cuda")
    return tok.decode(generated,skip_special_tokens=True),len(generated),reason


def gold_number(answer):
    found=re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)",answer)
    return found[-1].replace(",","") if found else ""


def flexible_number(text):
    values=re.findall(r"-?[0-9][0-9,]*(?:\.[0-9]+)?",text)
    return values[-1].replace(",","").rstrip(".") if values else "[invalid]"


def evaluate(cfg):
    seed_all(cfg["seed"]); toks=tokenizers(cfg); root=HERE/("runs_smoke" if cfg["smoke"] else "runs")
    base=NativeKVTranslator("full28_mlp",hidden_dim=cfg["mlp_hidden_dim"]).cuda().eval().requires_grad_(False)
    load_checkpoint(root/"stage_a/best.pt",cfg,"stage_a",base)
    adapters={}
    for objective in ("universal_kl",):
        adapters[objective]=ResidualKVAdapter(rank=cfg["adapter_rank"]).cuda().eval().requires_grad_(False)
        load_checkpoint(root/objective/"best.pt",cfg,objective,adapters[objective])
    llama,qwen=load_model(model_config(cfg),"llama"),load_model(model_config(cfg),"qwen")
    conditions=("qwen_full_native","native_oracle","stage_a","universal_kl")
    summary={"status":"running","datasets":{}}
    for split in ("test","ood"):
        rows=prepare_rows(cfg,split,toks)
        for dataset in sorted({row["dataset"] for row in rows}):
            chosen=[row for row in rows if row["dataset"]==dataset]; records=[]; correct=Counter()
            for number,row in enumerate(chosen,1):
                item=pair(llama,qwen,toks["llama"],toks["qwen"],row); bk,bv=mapped_base(base,item,cfg["chunk_tokens"])
                caches={
                    "native_oracle": (item["target_k"].cuda(), item["target_v"].cuda()),
                    "stage_a":(bk,bv),
                }
                for objective,adapter in adapters.items():
                    with torch.amp.autocast("cuda",dtype=torch.float16): pk,pv,_,_=adapter(bk[None],bv[None])
                    caches[objective]=(pk[0],pv[0])
                condition={}
                for name in conditions:
                    if name == "qwen_full_native":
                        key = value = None
                    else:
                        pk,pv = caches[name]
                        key=torch.cat((item["token0_k"].cuda(),pk),1)
                        value=torch.cat((item["token0_v"].cuda(),pv),1)
                    if row["task"]=="mcq":
                        fields=row["encoded"]["qwen"]
                        prompt=fields["body"]+fields["suffix"] if name=="qwen_full_native" else fields["suffix"]
                        full=next_token_logits(qwen,prompt,key,value)
                        choice=full[torch.tensor(row["encoded"]["qwen"]["choice_ids"],device="cuda")]
                        condition[name]=prediction_record(choice,row["gold_index"]); correct[name]+=condition[name]["correct"]
                    else:
                        fields=row["encoded"]["qwen"]
                        prompt=fields["body"]+fields["suffix"] if name=="qwen_full_native" else fields["suffix"]
                        text,count,reason=greedy_generate(qwen,toks["qwen"],prompt,cfg["gsm8k_max_new_tokens"],key,value)
                        pred=flexible_number(text); gold=gold_number(row["gold_answer"])
                        condition[name]={"generated_text":text,"generated_tokens":count,"stop_reason":reason,
                                         "prediction":pred,"gold":gold,"correct":pred==gold}; correct[name]+=pred==gold
                records.append({"id":row["id"],"dataset":dataset,"task":row["task"],"conditions":condition})
                if number%16==0 or number==len(chosen): log(f"evaluate {dataset} {number}/{len(chosen)}")
            metrics={name:{"correct":correct[name],"accuracy":correct[name]/len(chosen)} for name in conditions}
            for name in conditions:
                if name in ("qwen_full_native","native_oracle"): continue
                both=sum(r["conditions"][name]["correct"] and r["conditions"]["native_oracle"]["correct"] for r in records)
                method_only=sum(r["conditions"][name]["correct"] and not r["conditions"]["native_oracle"]["correct"] for r in records)
                oracle_only=sum(not r["conditions"][name]["correct"] and r["conditions"]["native_oracle"]["correct"] for r in records)
                metrics[name]["vs_native_oracle"]={"both_correct":both,"method_only_correct":method_only,
                    "oracle_only_correct":oracle_only,"both_wrong":len(chosen)-both-method_only-oracle_only}
            full_oracle_disagreement=sum(
                r["conditions"]["qwen_full_native"]["prediction"] != r["conditions"]["native_oracle"]["prediction"]
                for r in records)
            metrics["native_path_parity"]={"prediction_disagreements":full_oracle_disagreement,
                "passed":full_oracle_disagreement==0}
            metrics["count"]=len(chosen); summary["datasets"][dataset]=metrics
            write_jsonl(root/"evaluation"/f"{dataset}_per_sample.jsonl",records)
            save_json(root/"evaluation"/f"{dataset}_summary.json",metrics)
    summary["status"]="completed"; save_json(root/"evaluation/summary.json",summary)


def audit(cfg):
    toks=tokenizers(cfg); result={}
    split_ids={}
    for split in ("train","validation","test","ood"):
        rows=prepare_rows(cfg,split,toks); split_ids[split]={row["id"] for row in rows}
        for row in rows:
            target,source,_=fullsync_map(row,toks["llama"],toks["qwen"])
            if not target or len(target) != len(source): raise RuntimeError("Alignment length mismatch")
        result[split]={"count":len(rows),"by_dataset":dict(Counter(row["dataset"] for row in rows)),
                       "ids_sha256":digest(sorted(split_ids[split]))}
    if split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"]:
        raise RuntimeError("Split ID overlap")
    result["hellaswag_ood_only"]=all(x.startswith("hellaswag_") for x in split_ids["ood"])
    result["protocol"]=cfg["protocol"]; save_json(HERE/"protocol_audit.json",result); log(result)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("audit","stage_a","stage_b_kl","stage_b_hybrid","evaluate")); parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); cfg=runtime_config(args.smoke)
    if args.action!="audit" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    {"audit":audit,"stage_a":stage_a,"stage_b_kl":lambda x:stage_b(x,"universal_kl"),
     "stage_b_hybrid":lambda x:stage_b(x,"universal_hybrid"),"evaluate":evaluate}[args.action](cfg)


if __name__=="__main__": main()
