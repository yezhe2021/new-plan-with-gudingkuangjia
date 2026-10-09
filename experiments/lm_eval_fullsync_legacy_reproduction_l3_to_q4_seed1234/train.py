"""Old method, independent MCQ/GSM training, true epoch sampling and resumable state."""
import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from kv_backend import (CFG, ROOT, FullSyncHFLM, NativeKVTranslator, ResidualKVAdapter,
                        assemble, continuation_score, forward_continuation, load_model,
                        map_base, pair, save_json, seed_all)
from tasks_native import save_prepared, jsonl


def log(text):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}", flush=True)


def root(group, smoke=False):
    return ROOT / ("runs_smoke" if smoke else "runs") / group


def atomic_checkpoint(path, model, stage, epoch, step, metric, optimizer=None, scaler=None,
                      sample_position=0, epoch_completed=True):
    data = {"protocol": CFG["protocol"], "stage": stage, "epoch": epoch,
            "step": step, "metrics": metric,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "sample_position":sample_position,"epoch_completed":epoch_completed}
    if optimizer is not None: data["optimizer"] = optimizer.state_dict()
    if scaler is not None: data["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp); tmp.replace(path)


def models():
    cfg = {"models": CFG["models"], "attention_implementation": CFG["attention_implementation"]}
    sender, receiver = load_model(cfg, "llama"), load_model(cfg, "qwen")
    hf = FullSyncHFLM(receiver)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["models"]["llama"], local_files_only=True)
    return sender, receiver, tok, hf


def item_for(sender, receiver, source_tok, hf, row):
    return pair(sender, receiver, source_tok, hf.tokenizer, row["context"])


def stage_a_backward(model, item, divisor):
    count = item["tokens"]
    if count==0:
        yield next(model.parameters()).sum()*0/divisor,0.
        return
    den_k = item["target_k"].float().square().sum().clamp_min(1e-8).item()
    den_v = item["target_v"].float().square().sum().clamp_min(1e-8).item()
    aggregate = 0.
    for begin in range(0, count, CFG["chunk_tokens"]):
        end = min(count, begin + CFG["chunk_tokens"])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pk, pv = model(item["source_k"][:, begin:end][None].cuda(),
                           item["source_v"][:, begin:end][None].cuda())
        tk, tv = item["target_k"][:, begin:end][None].cuda().float(), item["target_v"][:, begin:end][None].cuda().float()
        frac = (end - begin) / count
        loss = ((pk.float()-tk).square().sum()/den_k + (pv.float()-tv).square().sum()/den_v
                + frac*(2 - F.cosine_similarity(pk.float(), tk, -1).mean()
                          - F.cosine_similarity(pv.float(), tv, -1).mean()))
        if not torch.isfinite(loss): raise RuntimeError("Non-finite Stage A")
        aggregate += float(loss.detach())
        yield loss/divisor, aggregate


def update(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip"])
    if not torch.isfinite(norm): raise RuntimeError("Non-finite gradients")
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(norm)


def scores(receiver, hf, row, key=None, value=None):
    result = []
    for ctx, continuation in row["requests"]:
        ci, ai = hf._encode_pair(ctx, continuation)
        score, _ = continuation_score(receiver, ci, ai, key, value)
        result.append(score)
    return torch.stack(result)


def mcq_loss(receiver, hf, row, item, bk, bv, adapter, divisor=None, scaler=None):
    if item["tokens"]==0:
        if divisor is not None: scaler.scale(next(adapter.parameters()).sum()*0/divisor).backward()
        return {"total":0.,"choice_kl":0.,"no_external_tokens":1.}
    # Exact candidate-KL gradient, sequential continuation forwards avoid holding 4 decoder graphs.
    with torch.no_grad():
        teacher = scores(receiver, hf, row)
        key, value = assemble(item, bk, bv, adapter)
        student = scores(receiver, hf, row, key, value)
        qn, qw = teacher.softmax(-1), student.softmax(-1)
        total = F.kl_div(student.log_softmax(-1), qn, reduction="sum")
        weights = (qw-qn).detach()
    if divisor is not None:
        for index, (ctx, continuation) in enumerate(row["requests"]):
            key, value = assemble(item, bk, bv, adapter)
            ci, ai = hf._encode_pair(ctx, continuation)
            score, _ = continuation_score(receiver, ci, ai, key, value)
            scaler.scale(weights[index]*score/divisor).backward()
    return {"total": float(total), "choice_kl": float(total)}


def gsm_loss(receiver, hf, row, item, bk, bv, adapter, divisor=None, scaler=None):
    ci, ai = hf._encode_pair(row["context"], row["target_delimiter"]+row["target"])
    with torch.no_grad(): teacher, _ = forward_continuation(receiver, ci, ai)
    key, value = assemble(item, bk, bv, adapter)
    student, eos_logits = forward_continuation(receiver, ci, ai, key, value)
    prob = teacher.float().softmax(-1)
    kl = F.kl_div(student.float().log_softmax(-1), prob, reduction="batchmean")
    gold = torch.tensor(ai, device="cuda", dtype=torch.long)
    ce = F.cross_entropy(student.float(), gold)
    eos_ids = receiver.generation_config.eos_token_id
    eos_id = eos_ids if isinstance(eos_ids, int) else eos_ids[0]
    eos = F.cross_entropy(eos_logits.float()[None], torch.tensor([eos_id], device="cuda"))
    total = kl + CFG["ce_weight"]*ce + CFG["eos_weight"]*eos
    if not torch.isfinite(total): raise RuntimeError("Non-finite Hybrid Stage B")
    if divisor is not None: scaler.scale(total/divisor).backward()
    return {"total": float(total.detach()), "trajectory_kl": float(kl.detach()),
            "gold_ce": float(ce.detach()), "eos_ce": float(eos.detach())}


@torch.no_grad()
def validate_a(base, sender, receiver, source_tok, hf, rows):
    sums = defaultdict(float)
    for number, row in enumerate(rows, 1):
        item = item_for(sender, receiver, source_tok, hf, row)
        if item["tokens"]==0:
            sums["k_cosine"]+=1.; sums["v_cosine"]+=1.
            continue
        k, v = map_base(base, item, CFG["chunk_tokens"])
        for name, prediction, target in (("k",k,item["target_k"]), ("v",v,item["target_v"])):
            p, t = prediction.float(), target.cuda().float()
            sums[name+"_nmse"] += float((p-t).square().mean()/t.square().mean().clamp_min(1e-8))
            sums[name+"_cosine"] += float(F.cosine_similarity(p,t,-1).mean())
        if number%32==0: log(f"Stage A validation {number}/{len(rows)}")
    sums = {k:v/len(rows) for k,v in sums.items()}
    sums["total"] = sums["k_nmse"]+sums["v_nmse"]+2-sums["k_cosine"]-sums["v_cosine"]
    return sums


@torch.no_grad()
def validate_b(base, adapter, sender, receiver, source_tok, hf, rows):
    sums = defaultdict(float)
    for number, row in enumerate(rows,1):
        item = item_for(sender, receiver, source_tok, hf, row)
        bk,bv = map_base(base,item,CFG["chunk_tokens"])
        values = (mcq_loss if row["output_type"]=="multiple_choice" else gsm_loss)(receiver,hf,row,item,bk,bv,adapter)
        for name, value in values.items(): sums[name] += value
        if number%32==0: log(f"Stage B validation {number}/{len(rows)}")
    return {k:v/len(rows) for k,v in sums.items()}


def train(group, stage, smoke=False):
    seed_all(CFG["seed"])
    folder = root(group,smoke); folder.mkdir(parents=True,exist_ok=True)
    if not (folder/"train.jsonl").exists(): save_prepared(group,smoke)
    train_rows, val_rows = jsonl(folder/"train.jsonl"), jsonl(folder/"validation.jsonl")
    sender, receiver, source_tok, hf = models()
    base = NativeKVTranslator("full28_mlp",hidden_dim=CFG["mlp_hidden_dim"]).cuda()
    epochs = 1 if smoke else CFG[f"stage_{stage}_epochs"]
    batch = 2 if smoke else CFG["effective_batch"]
    if stage=="b":
        payload=torch.load(folder/"stage_a/best.pt",map_location="cpu",weights_only=True)
        base.load_state_dict(payload["state"],strict=True)
        base.eval().requires_grad_(False)
        model=ResidualKVAdapter(rank=CFG["adapter_rank"]).cuda()
    else: model=base
    output=folder/f"stage_{stage}"; output.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG[f"stage_{stage}_learning_rate"],weight_decay=0)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.,growth_interval=1000000)
    candidates=[]; start_epoch=1; step=0; resume_position=0
    if (output/"last.pt").exists():
        payload=torch.load(output/"last.pt",map_location="cpu",weights_only=True)
        if payload["protocol"]!=CFG["protocol"]: raise RuntimeError("Resume protocol mismatch")
        model.load_state_dict(payload["state"],strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scaler.load_state_dict(payload["scaler"])
        step=payload["step"]
        complete=payload.get("epoch_completed",True)
        start_epoch=payload["epoch"]+1 if complete else payload["epoch"]
        resume_position=0 if complete else payload["sample_position"]
        if (output/"selection.json").exists():
            candidates=json.loads((output/"selection.json").read_text())["epochs"]
        log(f"Resume {group} Stage {stage} at epoch={start_epoch}, step={step}")
    with (output/"training_steps.jsonl").open("a",encoding="utf-8") as stream:
        for epoch in range(start_epoch,epochs+1):
            model.train(); order=list(range(len(train_rows)))
            random.Random(CFG["seed"]+epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True); totals=defaultdict(float); accumulation=0
            for position,index in enumerate(order,1):
                if epoch==start_epoch and position<=resume_position: continue
                row=train_rows[index]
                divisor=min(batch,len(order)-((position-1)//batch)*batch)
                item=item_for(sender,receiver,source_tok,hf,row)
                if stage=="a":
                    total=0.
                    for loss,total in stage_a_backward(model,item,divisor): scaler.scale(loss).backward()
                    values={"total":total}
                else:
                    bk,bv=map_base(base,item,CFG["chunk_tokens"])
                    fn=mcq_loss if row["output_type"]=="multiple_choice" else gsm_loss
                    values=fn(receiver,hf,row,item,bk,bv,model,divisor,scaler)
                    del bk,bv
                del item
                for name,value in values.items(): totals[name]+=value
                accumulation+=1
                if accumulation==batch or position==len(order):
                    norm=update(model,optimizer,scaler); step+=1
                    record={"group":group,"stage":stage,"epoch":epoch,"step":step,"samples_seen":position,
                            "pre_clip_grad_norm":norm,"clipped":norm>CFG["gradient_clip"],
                            **{k:v/accumulation for k,v in totals.items()}}
                    stream.write(json.dumps(record)+"\n"); stream.flush()
                    if step%8==0 or smoke: log(f"{group} Stage-{stage.upper()} epoch={epoch}/{epochs} step={step} samples={position}/{len(order)} loss={record['total']:.6f}")
                    accumulation=0; totals=defaultdict(float)
                    if not smoke and step%128==0 and position<len(order):
                        atomic_checkpoint(output/"last.pt",model,stage,epoch,step,{},optimizer,scaler,
                                          sample_position=position,epoch_completed=False)
                        log(f"{group} Stage-{stage.upper()} partial state saved at step={step}")
            model.eval()
            metrics=(validate_a(base,sender,receiver,source_tok,hf,val_rows) if stage=="a" else
                     validate_b(base,model,sender,receiver,source_tok,hf,val_rows))
            candidate={"epoch":epoch,"step":step,"validation":metrics}; candidates.append(candidate)
            best=min(candidates,key=lambda x:(x["validation"]["total"],x["epoch"]))
            if best is candidate: atomic_checkpoint(output/"best.pt",model,stage,epoch,step,metrics)
            # Selection then state: recoverable if interrupted before either atomic replacement.
            save_json(output/"selection.json",{"selection":"lowest validation loss only","best":best,"epochs":candidates})
            atomic_checkpoint(output/"last.pt",model,stage,epoch,step,metrics,optimizer,scaler)
            log(f"{group} Stage-{stage.upper()} EPOCH COMPLETED {epoch}/{epochs}: {metrics}")
    save_json(output/"completed.json",{"epochs":epochs,"steps":step,"group":group,"stage":stage})
    log(f"{group} Stage-{stage.upper()} COMPLETED")


def smoke_parity(group):
    seed_all(CFG["seed"]); rows=save_prepared(group,True)["train"]
    sender,receiver,source_tok,native=models()
    oracle=FullSyncHFLM(receiver,mode="oracle")
    from lm_eval.api.instance import Instance
    records=[]
    for row in rows:
        item=item_for(sender,receiver,source_tok,native,row)
        if row["output_type"]=="multiple_choice":
            requests=[Instance("loglikelihood",row["doc"],tuple(args),i,metadata=(row["task"],i,1))
                      for i,args in enumerate(row["requests"])]
            expected=native.loglikelihood(requests); actual=oracle.loglikelihood(requests)
            direct=[]
            for ctx,cont in row["requests"]:
                ci,ai=native._encode_pair(ctx,cont)
                with torch.no_grad(): score,_=continuation_score(receiver,ci,ai,softmax_dtype=native.softmax_dtype)
                direct.append(float(score))
            max_error=max(abs(x[0]-y[0]) for x,y in zip(expected,actual))
            direct_error=max(abs(x[0]-y) for x,y in zip(expected,direct))
            # Cached and full prefill change FP16 GEMM shapes; test per-token error, not length-dependent sum.
            lengths=[len(native._encode_pair(ctx,cont)[1]) for ctx,cont in row["requests"]]
            per_token_error=max(abs(x[0]-y[0])/n for x,y,n in zip(expected,actual,lengths))
            same_prediction=max(range(len(expected)),key=lambda i:expected[i][0])==max(range(len(actual)),key=lambda i:actual[i][0])
            if direct_error>1e-5 or per_token_error>0.02 or not same_prediction:
                raise RuntimeError(f"Native likelihood parity failed: direct={direct_error}, per_token={per_token_error}, same_prediction={same_prediction}; hf={expected}, oracle={actual}")
            records.append({"id":row["id"],"max_ll_error":max_error,"direct_ll_error":direct_error,
                            "max_per_token_ll_error":per_token_error,"prediction_agreement":same_prediction,"passed":True})
        else:
            kwargs=dict(row["requests"][0][1]); kwargs["max_gen_toks"]=16
            request=Instance("generate_until",row["doc"],(row["context"],kwargs),0,metadata=(row["task"],0,1))
            a=native.generate_until([request]); b=oracle.generate_until([request])
            if a!=b: raise RuntimeError(f"Native generation parity failed: {a!r} != {b!r}")
            records.append({"id":row["id"],"native":a,"oracle":b,"passed":True})
        log(f"Native parity passed: {row['id']}")
    save_json(root(group,True)/"native_parity.json",records)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("prepare","parity","stage_a","stage_b"))
    parser.add_argument("--group",choices=("mcq","gsm8k"),required=True); parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.action=="prepare": save_prepared(args.group,args.smoke)
    elif args.action=="parity": smoke_parity(args.group)
    else: train(args.group,args.action[-1],args.smoke)
