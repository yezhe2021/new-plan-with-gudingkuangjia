"""Supplement: remove donor label-vocabulary mismatch; no rerun or sample tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    read=lambda p:json.loads(p.read_text())
    assert read(ROOT/'completed.json')['status']=='completed'
    records=[json.loads(line) for line in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
    prepared={r['id']:r for r in records}
    result_rows=[read(p) for p in (ROOT/'evaluation').glob('*/*.json')]
    results={r['id']:r for r in result_rows}
    assert set(results)==set(prepared)
    output=dict(manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
                criterion='Same task and exactly the same candidate label vocabulary. Average matched donor repeats within question; no outcome-based selection.',tasks={})
    for task in sorted({r['task'] for r in records}):
        values=[];pairs=0;included=[]
        for record in records:
            if record['task']!=task:continue
            result=results[record['id']];trial=[]
            for rep,donor in enumerate(record['donors']):
                if record['original']['candidates']!=prepared[donor['id']]['original']['candidates']:continue
                prediction=result['arms'][f'wrong_{rep}']['prediction']
                trial.append([result['arms'][f'wrong_{rep}']['correct'],
                              prediction==results[donor['id']]['arms']['true']['prediction'],
                              prediction==result['arms']['true']['prediction']])
            if trial:
                values.append([result['arms']['true']['correct']]+np.mean(trial,axis=0).tolist())
                pairs+=len(trial);included.append(record['id'])
        matrix=np.array(values);drop=matrix[:,0]-matrix[:,1]
        rng=np.random.default_rng(1234);boot=drop[rng.integers(0,len(drop),(10000,len(drop)))].mean(1)
        output['tasks'][task]=dict(question_count=len(values),donor_pair_count=pairs,
            true_accuracy=float(matrix[:,0].mean()),wrong_accuracy=float(matrix[:,1].mean()),
            accuracy_drop_ci95=np.quantile(boot,[.025,.975]).tolist(),
            wrong_matches_donor_true_prediction=float(matrix[:,2].mean()),
            wrong_matches_recipient_true_prediction=float(matrix[:,3].mean()),included_ids=included)
    target=ROOT/'diagnostics/donor_label_subset.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({task:{k:v for k,v in value.items() if k!='included_ids'} for task,value in output['tasks'].items()},indent=2))


if __name__=='__main__':main()
