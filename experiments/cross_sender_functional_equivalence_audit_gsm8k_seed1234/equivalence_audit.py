"""Frozen Llama/Gemma -> Qwen predictive-equivalence audit on GSM8K."""

import argparse
import gc
import importlib.util
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
REF = Path(CFG["reference_eval"])
sys.path.insert(0, str(REF))
import evaluate as ref  # noqa: E402


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def encode_prefix(tok, text):
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return bos + list(tok.encode(text, add_special_tokens=False))


def aligned_row(sample_id, source_text, source_tok, qwen_tok, family):
    encoded = {}
    for name, tok in ((family, source_tok), ("qwen", qwen_tok)):
        body = encode_prefix(tok, source_text)
        suffix = list(tok.encode("Answer:", add_special_tokens=False))
        if body + suffix != encode_prefix(tok, source_text + "Answer:"):
            raise RuntimeError(f"Noncompositional boundary: {sample_id}/{name}")
        encoded[name] = {"body": body, "option_token_indices": list(range(1, len(body)))}
    return {"id": sample_id, "body": source_text, "encoded": encoded}


def checkpoint_state(path, expected):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("stage") != expected:
        raise RuntimeError(f"Checkpoint stage mismatch: {path} {payload.get('stage')} != {expected}")
    return payload["state"]


def make_models(family):
    module = load_module(f"equiv_{family}_translator", CFG[family]["translator"])
    if family == "llama":
        base = module.NativeKVTranslator("full28_mlp", hidden_dim=1024)
    else:
        base = module.NativeKVTranslator("full34_headmix256", hidden_dim=1024,
            depth_output_dim=256, head_mapping="full_head")
    base.load_state_dict(checkpoint_state(CFG[family]["stage_a"],
        "gsm8k_stage_a" if family == "llama" else "gsm8k_gemma_stage_a"), strict=True)
    adapter = module.ResidualKVAdapter(rank=64)
    adapter.load_state_dict(checkpoint_state(CFG[family]["adapter"], "gsm8k_hybrid_stage_b"), strict=True)
    return base.cuda().eval().requires_grad_(False), adapter.cuda().eval().requires_grad_(False)


def rows(smoke=False):
    count = 2 if smoke else CFG["sample_count"]
    return read_jsonl(CFG["gsm8k_test"])[:count]


def temp_root(smoke=False):
    return ROOT / ("temp_states_smoke" if smoke else "temp_states")


@torch.no_grad()
def capture_sender(family, smoke=False):
    selected = rows(smoke); root = temp_root(smoke) / family
    source_root = root / "source"; state_root = root / "translated"
    source_root.mkdir(parents=True, exist_ok=True); state_root.mkdir(parents=True, exist_ok=True)
    source_tok = AutoTokenizer.from_pretrained(ref.CFG["models"][family], local_files_only=True)
    qwen_tok = AutoTokenizer.from_pretrained(ref.CFG["models"]["qwen"], local_files_only=True)
    missing = [i for i in range(len(selected)) if not (source_root / f"{i:04d}.pt").exists()]
    if missing:
        sender = ref.load_model(family)
        try:
            for done, index in enumerate(missing, 1):
                row = selected[index]; source_text = f"Question: {row['question']}\n"
                aligned = aligned_row(f"test_{index}", source_text, source_tok, qwen_tok, family)
                target_indices, source_indices, counts = ref.fullsync_map(aligned, source_tok, qwen_tok, family)
                source_fields, qwen_fields = aligned["encoded"][family], aligned["encoded"]["qwen"]
                key, value, _ = ref.capture(sender, source_fields["body"], len(source_fields["body"]), [0])
                torch.save({"id":f"test_{index}","question":row["question"],"answer":row["answer"],
                    "source_text":source_text,"source_k":key[:,source_indices].half(),
                    "source_v":value[:,source_indices].half(),"target_indices":target_indices,
                    "qwen_body":qwen_fields["body"],"mapping_counts":counts}, source_root/f"{index:04d}.pt")
                if done%16==0 or done==len(missing): log(f"{family} source capture {done}/{len(missing)}")
        finally:
            del sender; gc.collect(); torch.cuda.empty_cache()
    base, adapter = make_models(family)
    try:
        for index in range(len(selected)):
            destination=state_root/f"{index:04d}.pt"
            if destination.exists(): continue
            item=torch.load(source_root/f"{index:04d}.pt",map_location="cpu",weights_only=True)
            out_k,out_v=[],[]
            for begin in range(0,item["source_k"].shape[1],CFG["chunk_tokens"]):
                stop=begin+CFG["chunk_tokens"]
                with torch.amp.autocast("cuda",dtype=torch.float16):
                    bk,bv=base(item["source_k"][:,begin:stop][None].cuda(),item["source_v"][:,begin:stop][None].cuda())
                    pk,pv,_,_=adapter(bk,bv)
                out_k.append(pk[0].cpu().half()); out_v.append(pv[0].cpu().half())
            item.pop("source_k"); item.pop("source_v")
            item.update(key=torch.cat(out_k,1),value=torch.cat(out_v,1),family=family)
            torch.save(item,destination); (source_root/f"{index:04d}.pt").unlink()
            if (index+1)%16==0 or index+1==len(selected): log(f"{family} translated state {index+1}/{len(selected)}")
    finally:
        del base,adapter; gc.collect(); torch.cuda.empty_cache()
    if source_root.exists() and not any(source_root.iterdir()): source_root.rmdir()


def symmetric_nmse(a,b):
    a,b=a.float(),b.float()
    return (2*(a-b).square().sum()/(a.square().sum()+b.square().sum()).clamp_min(1e-12)).item()


def representation(a,b):
    af,bf=a.float(),b.float()
    per_layer_cos=F.cosine_similarity(af,bf,-1).mean(dim=(1,2))
    numerator=2*(af-bf).square().sum(dim=(1,2,3))
    denominator=(af.square().sum(dim=(1,2,3))+bf.square().sum(dim=(1,2,3))).clamp_min(1e-12)
    return {"cosine":F.cosine_similarity(af,bf,-1).mean().item(),"symmetric_nmse":symmetric_nmse(af,bf),
            "per_layer_cosine":per_layer_cos.cpu().tolist(),"per_layer_symmetric_nmse":(numerator/denominator).cpu().tolist()}


def distribution_distance(logits_a,logits_b):
    la,lb=F.log_softmax(logits_a.float(),-1),F.log_softmax(logits_b.float(),-1)
    pa,pb=la.exp(),lb.exp(); m=(pa+pb)/2; lm=m.clamp_min(1e-30).log()
    return {"kl_a_b":(pa*(la-lb)).sum().item(),"kl_b_a":(pb*(lb-la)).sum().item(),
            "js":(0.5*(pa*(la-lm)).sum()+0.5*(pb*(lb-lm)).sum()).item(),
            "top1_agree":int(logits_a.argmax()==logits_b.argmax())}


@torch.no_grad()
def logits_for(qwen, state, prefix_length, suffix, continuation):
    ids=suffix+continuation
    key,value=state
    cache=ref.make_cache(qwen,key,value,torch.arange(prefix_length,device="cuda"))
    tensor=torch.tensor([ids],device="cuda",dtype=torch.long)
    output=ref.backbone(qwen)(input_ids=tensor,
        attention_mask=torch.ones((1,prefix_length+len(ids)),device="cuda",dtype=torch.long),
        position_ids=torch.arange(prefix_length,prefix_length+len(ids),device="cuda")[None],
        past_key_values=cache,use_cache=False,return_dict=True)
    return qwen.lm_head(output.last_hidden_state[:,-1])[0]


@torch.no_grad()
def rollout(qwen,state,prefix_length,suffix,eos,steps):
    generated=[]
    for _ in range(steps):
        token=int(logits_for(qwen,state,prefix_length,suffix,generated).argmax())
        generated.append(token)
        if token in eos: break
    return generated


def mean_dict(values):
    if not values: return {}
    return {key:sum(float(x[key]) for x in values)/len(values) for key in values[0]}


def quantiles(values):
    tensor=torch.tensor(values,dtype=torch.float64)
    return {"mean":tensor.mean().item(),"p10":torch.quantile(tensor,.1).item(),
            "median":torch.quantile(tensor,.5).item(),"p90":torch.quantile(tensor,.9).item(),"max":tensor.max().item()}


def rank(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); result=[0.0]*len(values)
    for position,index in enumerate(order): result[index]=float(position)
    return result


def correlation(a,b):
    if len(a)<2:return float("nan")
    ta,tb=torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)
    return torch.corrcoef(torch.stack((ta,tb)))[0,1].item()


def weak_correctness(count):
    outputs={}
    for family in ("llama","gemma"):
        records=read_jsonl(CFG[family]["per_sample"]); condition=CFG[family]["condition"]
        outputs[family]={row["id"]:bool(row["conditions"][condition]["flexible_correct"]) for row in records[:count]}
    ids=[f"test_{i}" for i in range(count)]
    return {"both":sum(outputs["llama"][x] and outputs["gemma"][x] for x in ids),
            "llama_only":sum(outputs["llama"][x] and not outputs["gemma"][x] for x in ids),
            "gemma_only":sum(not outputs["llama"][x] and outputs["gemma"][x] for x in ids),
            "neither":sum(not outputs["llama"][x] and not outputs["gemma"][x] for x in ids)}


@torch.no_grad()
def audit(smoke=False):
    selected=rows(smoke); root=temp_root(smoke); out=ROOT/("results_smoke" if smoke else "results")
    qwen_tok=AutoTokenizer.from_pretrained(ref.CFG["models"]["qwen"],local_files_only=True)
    qwen=ref.load_model("qwen"); suffix=list(qwen_tok.encode("Answer:",add_special_tokens=False))
    eos=qwen.generation_config.eos_token_id; eos={int(eos)} if isinstance(eos,int) else {int(x) for x in eos}
    records=[]; layer_sums=defaultdict(lambda:torch.zeros(36,dtype=torch.float64))
    pair_names=("llama_native","gemma_native","llama_gemma")
    try:
        for index,row in enumerate(selected):
            ls=torch.load(root/"llama/translated"/f"{index:04d}.pt",map_location="cpu",weights_only=True)
            gs=torch.load(root/"gemma/translated"/f"{index:04d}.pt",map_location="cpu",weights_only=True)
            if ls["qwen_body"]!=gs["qwen_body"] or ls["id"]!=gs["id"]: raise RuntimeError("Cross-sender state mismatch")
            qk,qv,_=ref.capture(qwen,ls["qwen_body"],len(ls["qwen_body"]),[0])
            target=ls["target_indices"]
            native=(qk[:,target].half(),qv[:,target].half()); llama=(ls["key"],ls["value"]); gemma=(gs["key"],gs["value"])
            states={"native":native,"llama":llama,"gemma":gemma}
            rep={}
            for pair_name,left,right in (("llama_native","llama","native"),("gemma_native","gemma","native"),("llama_gemma","llama","gemma")):
                rep[pair_name]={"k":representation(states[left][0],states[right][0]),"v":representation(states[left][1],states[right][1])}
                for component in ("k","v"):
                    for metric in ("per_layer_cosine","per_layer_symmetric_nmse"):
                        layer_sums[f"{pair_name}.{component}.{metric}"]+=torch.tensor(rep[pair_name][component][metric],dtype=torch.float64)
            token0=(qk[:,:1].cuda().half(),qv[:,:1].cuda().half()); full_states={}
            for name,(key,value) in states.items():
                full_k=torch.cat((token0[0],key.cuda()),1); full_v=torch.cat((token0[1],value.cuda()),1)
                full_states[name]=(full_k,full_v)
            length=len(ls["qwen_body"])
            trajectories={name:rollout(qwen,full_states[name],length,suffix,eos,CFG["probe_steps"] if not smoke else 2) for name in states}
            functional={}
            all_lg=[]
            for trajectory_name,tokens in trajectories.items():
                distances=defaultdict(list)
                for step in range(len(tokens)):
                    logits={name:logits_for(qwen,full_states[name],length,suffix,tokens[:step]) for name in states}
                    for pair_name,left,right in (("llama_native","llama","native"),("gemma_native","gemma","native"),("llama_gemma","llama","gemma")):
                        value=distribution_distance(logits[left],logits[right]); distances[pair_name].append(value)
                        if pair_name=="llama_gemma": all_lg.append(value["js"])
                functional[trajectory_name]={name:mean_dict(values) for name,values in distances.items()}
            interpolation={}
            native_tokens=trajectories["native"]
            for alpha in CFG["interpolation_alphas"]:
                ik=(1-alpha)*gemma[0].cuda()+alpha*llama[0].cuda(); iv=(1-alpha)*gemma[1].cuda()+alpha*llama[1].cuda()
                ik=torch.cat((token0[0],ik),1); iv=torch.cat((token0[1],iv),1)
                interpolated=(ik,iv); values=[]
                for step in range(len(native_tokens)):
                    li=logits_for(qwen,interpolated,length,suffix,native_tokens[:step]); ln=logits_for(qwen,full_states["native"],length,suffix,native_tokens[:step])
                    values.append(distribution_distance(li,ln))
                interpolation[str(alpha)]=mean_dict(values)
            records.append({"id":f"test_{index}","question":row["question"],"tokens":length,
                "representation":rep,"functional":functional,"interpolation_native_trajectory":interpolation,
                "rollout_token_ids":trajectories,"llama_gemma_mean_js":sum(all_lg)/len(all_lg)})
            if (index+1)%8==0 or index+1==len(selected): log(f"functional audit {index+1}/{len(selected)}")
            del qk,qv,full_states; torch.cuda.empty_cache()
    finally:
        del qwen; gc.collect(); torch.cuda.empty_cache()
    write_jsonl(out/"per_sample.jsonl",records)
    summary={"status":"completed","count":len(records),"probe_steps":CFG["probe_steps"] if not smoke else 2,
             "joint_answer_correctness":weak_correctness(len(records)),"representation":{},"functional":{},"interpolation":{}}
    for pair_name in pair_names:
        summary["representation"][pair_name]={}
        for component in ("k","v"):
            summary["representation"][pair_name][component]={
                "cosine":quantiles([r["representation"][pair_name][component]["cosine"] for r in records]),
                "symmetric_nmse":quantiles([r["representation"][pair_name][component]["symmetric_nmse"] for r in records]),
                "per_layer_cosine_mean":(layer_sums[f"{pair_name}.{component}.per_layer_cosine"]/len(records)).tolist(),
                "per_layer_symmetric_nmse_mean":(layer_sums[f"{pair_name}.{component}.per_layer_symmetric_nmse"]/len(records)).tolist()}
        for trajectory in ("native","llama","gemma"):
            summary["functional"].setdefault(trajectory,{})[pair_name]={metric:quantiles([
                r["functional"][trajectory][pair_name][metric] for r in records]) for metric in ("js","kl_a_b","kl_b_a","top1_agree")}
    for alpha in CFG["interpolation_alphas"]:
        summary["interpolation"][str(alpha)]={metric:quantiles([r["interpolation_native_trajectory"][str(alpha)][metric] for r in records]) for metric in ("js","kl_a_b","kl_b_a","top1_agree")}
    rep_distance=[1-(r["representation"]["llama_gemma"]["k"]["cosine"]+r["representation"]["llama_gemma"]["v"]["cosine"])/2 for r in records]
    func_distance=[r["llama_gemma_mean_js"] for r in records]
    summary["representation_function_correlation"]={"pearson":correlation(rep_distance,func_distance),
        "spearman":correlation(rank(rep_distance),rank(func_distance))}
    save_json(out/"summary.json",summary); log(f"audit completed: {out/'summary.json'}")


def preflight(smoke=False):
    required=[CFG["gsm8k_test"],CFG["reference_eval"]]
    for family in ("llama","gemma"):
        required.extend((CFG[family]["translator"],CFG[family]["stage_a"],CFG[family]["adapter"],CFG[family]["per_sample"]))
    missing=[str(x) for x in required if not Path(x).exists()]
    if missing: raise RuntimeError(f"Missing assets: {missing}")
    weak=weak_correctness(2 if smoke else CFG["sample_count"])
    save_json(ROOT/("preflight_smoke.json" if smoke else "preflight.json"),{"required_assets":len(required),"weak_correctness":weak})
    log(f"preflight passed: {weak}")


def cleanup(smoke=False):
    path=temp_root(smoke)
    if path.exists(): shutil.rmtree(path)
    log(f"removed temporary states: {path}")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("preflight","capture_llama","capture_gemma","audit","cleanup")); parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); torch.manual_seed(CFG["seed"])
    if args.action not in ("preflight","cleanup") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    {"preflight":preflight,"capture_llama":lambda x:capture_sender("llama",x),
     "capture_gemma":lambda x:capture_sender("gemma",x),"audit":audit,"cleanup":cleanup}[args.action](args.smoke)


if __name__=="__main__": main()
