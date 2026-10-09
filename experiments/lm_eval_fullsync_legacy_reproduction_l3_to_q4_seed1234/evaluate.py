"""Official lm-eval evaluator; no private parser or metric implementation."""
import argparse
import gc
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from kv_backend import CFG, ROOT, FullSyncHFLM, load_model, load_base, load_adapter, save_json, seed_all
from tasks_native import install_local_data, tasks


def benchmark(group, condition, ood=False):
    seed_all(CFG["seed"]); install_local_data()
    folder=ROOT/"runs"/group
    names=["hellaswag","mmlu_pro"] if ood else CFG["groups"][group]
    cfg={"models":CFG["models"],"attention_implementation":CFG["attention_implementation"]}
    if condition=="llama_native":
        model=load_model(cfg,"llama")
        lm=HFLM(pretrained=model,tokenizer=CFG["models"]["llama"],batch_size=1,
                backend="causal",max_length=CFG["max_length"],logits_cache=False)
    else:
        receiver=load_model(cfg,"qwen")
        mode={"qwen_native":"native","native_oracle":"oracle","stage_a":"translated","stage_b":"translated"}[condition]
        sender=base=adapter=None
        if mode=="translated":
            sender=load_model(cfg,"llama"); base=load_base(folder/"stage_a/best.pt")
            if condition=="stage_b": adapter=load_adapter(folder/"stage_b/best.pt")
        lm=FullSyncHFLM(receiver,mode,sender,base,adapter)
    output=folder/("evaluation_ood" if ood else "evaluation")
    output.mkdir(parents=True,exist_ok=True)
    samples=None; limit=CFG["test_count"]
    if ood:
        # Limit128 is PER TASK in harness. Allocate128 across14 MMLU subjects instead.
        obj=tasks(["mmlu_pro"])
        def flatten(value):
            found={}
            for k,v in value.items():
                if isinstance(v,dict): found.update(flatten(v))
                elif hasattr(v,"get_config"): found[k]=v
                elif isinstance(v,tuple) and hasattr(v[-1],"get_config"): found[k]=v[-1]
            return found
        leaves=flatten(obj)
        if not leaves: raise RuntimeError("Cannot allocate MMLU pilot sample IDs")
        samples={"hellaswag":list(range(CFG["test_count"]))}
        keys=sorted(leaves); n=CFG["test_count"]
        for i,key in enumerate(keys): samples[key]=list(range(n//len(keys)+(i<n%len(keys))))
        limit=None
    result=evaluator.simple_evaluate(model=lm,tasks=names,num_fewshot=None,limit=limit,samples=samples,
                                    bootstrap_iters=1000,log_samples=True,apply_chat_template=False,
                                    random_seed=CFG["seed"],numpy_random_seed=CFG["seed"],
                                    torch_random_seed=CFG["seed"],fewshot_random_seed=CFG["seed"])
    def plain(x):
        if hasattr(x,"item"): return x.item()
        if isinstance(x,Path): return str(x)
        return str(x)
    payload=json.loads(json.dumps(result,default=plain,ensure_ascii=False))
    payload["experiment"]={"condition":condition,"group":group,"protocol":CFG["protocol"],
                           "pilot_subset":True,"full_official_test":False,
                           "stage_a":"stage_a/best.pt" if condition.startswith("stage") else None,
                           "stage_b":"stage_b/best.pt" if condition=="stage_b" else None}
    save_json(output/f"{condition}.json",payload)
    for task_name,rows in payload.get("samples",{}).items():
        with (output/f"{condition}__{task_name}__samples.jsonl").open("w",encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps(payload.get("results",{}),ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--group",choices=("mcq","gsm8k"),required=True)
    p.add_argument("--condition",choices=CFG["evaluation_conditions"],required=True); p.add_argument("--ood",action="store_true")
    a=p.parse_args(); benchmark(a.group,a.condition,a.ood)
