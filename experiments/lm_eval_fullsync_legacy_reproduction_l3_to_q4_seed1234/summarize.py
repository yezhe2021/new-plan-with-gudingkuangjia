"""Automatically assemble all official metrics; never replace them with manual parsing."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
summary={"protocol":"official_lm_eval_0.4.12_pilot","groups":{}}
lines=["# Standard lm-eval legacy-method reproduction", "", "Pilot subsets, not full-benchmark scores. All metrics below are returned by the official evaluator.", ""]
for group in ("mcq","gsm8k"):
    folder=ROOT/"runs"/group
    metrics={}
    overlap={}; clipping={}
    for name in ("evaluation","evaluation_ood"):
        if not (folder/name).exists(): continue
        payloads={}
        for path in sorted((folder/name).glob("*.json")):
            payload=json.loads(path.read_text(encoding="utf-8"))
            if "results" not in payload: continue
            payloads[path.stem]=payload
            metrics[name+"/"+path.stem]=payload["results"]
            lines.extend([f"## {group} / {name} / {path.stem}", "", "```json",json.dumps(payload["results"],ensure_ascii=False,indent=2),"```",""])
        for condition,payload in payloads.items():
            for reference in ("qwen_native","native_oracle","stage_a"):
                if reference==condition or reference not in payloads: continue
                for task_name,rows in payload.get("samples",{}).items():
                    refs=payloads[reference].get("samples",{}).get(task_name,[])
                    index={(r["doc_hash"],r["filter"]):r for r in refs}
                    counts={}
                    for row in rows:
                        other=index.get((row["doc_hash"],row["filter"]))
                        if other is None: continue
                        for metric in ("acc","acc_norm","exact_match"):
                            if metric not in row or metric not in other: continue
                            key=f"{name}/{task_name}/{row['filter']}/{metric}/{condition}_vs_{reference}"
                            record=counts.setdefault(key,{"both_correct":0,"method_only_correct":0,"reference_only_correct":0,"both_wrong":0})
                            a,b=bool(float(row[metric])),bool(float(other[metric]))
                            label="both_correct" if a and b else "method_only_correct" if a else "reference_only_correct" if b else "both_wrong"
                            record[label]+=1
                    overlap.update(counts)
    for stage in ("stage_a","stage_b"):
        path=folder/stage/"training_steps.jsonl"
        if not path.exists(): continue
        latest={}
        for line in path.read_text().splitlines():
            if line.strip():
                row=json.loads(line); latest[(row["epoch"],row["step"])]=row
        rows=list(latest.values()); norms=[r["pre_clip_grad_norm"] for r in rows]
        if not norms: continue
        labels=("min","p10","p25","median","p75","p90","p95","max")
        values=np.percentile(norms,[0,10,25,50,75,90,95,100]).tolist()
        clipping[stage]={"optimizer_steps":len(rows),"clip_rate":sum(r["clipped"] for r in rows)/len(rows),
                         "pre_clip_grad_norm":dict(zip(labels,values))}
    summary["groups"][group]={"official_metrics":metrics,"correctness_overlap":overlap,"gradient_clipping":clipping}
    if overlap: lines.extend([f"## {group} official-sample correctness overlap","","```json",json.dumps(overlap,indent=2),"```",""])
    if clipping: lines.extend([f"## {group} gradient diagnostics","","```json",json.dumps(clipping,indent=2),"```",""])
(ROOT/"RESULTS.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(ROOT/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
print("RESULTS.md and RESULTS.json refreshed",flush=True)
