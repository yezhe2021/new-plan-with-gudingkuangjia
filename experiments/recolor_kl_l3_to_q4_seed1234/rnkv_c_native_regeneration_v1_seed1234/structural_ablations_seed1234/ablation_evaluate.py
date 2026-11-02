"""Reuse original seven-task scorers and fixed per-question baselines."""
import importlib.util
from ablation_common import *
from ablation_runtime import load
sys.path.append(str(PARENT/CFG['cross_experiment']))
spec=importlib.util.spec_from_file_location('frozen_cross_task_scorer',PARENT/CFG['cross_experiment']/'experiment.py')
X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X)

def score(module,row,cache):
    return module.score(row,cache,rn=True) if row['task'] in ['arc','hotpotqa'] else X.score(module,row,cache,rn=True)

@torch.no_grad()
def evaluate(arm):
    verify()
    if CFG['arms'][arm]['train']:assert done(f'arms/{arm}/runs/completed.json')
    engine,correction,module=load(arm,final=True);module.eval().requires_grad_(False);engine.mode='test'
    snapshots=(C.versions(correction),C.versions(module))
    for task in CFG['test_counts']:
        if done(f'arms/{arm}/evaluation/{task}/completed.json'):continue
        rr=[r for r in rows('test') if r['task']==task];records=[]
        for i,row in enumerate(rr,1):
            if paused():status(status='paused',arm=arm,task=task,sample=i);sys.exit(75)
            path=ROOT/f'arms/{arm}/evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);translated=engine.cache(row,base,correction);memory=module.construct(row,translated)
            result=score(module,row,memory)
            values=dict(baseline(row),**{arm:result})
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=values)
            records.append(C.read(path));del base,translated,memory
            engine.unchanged();assert (C.versions(correction),C.versions(module))==snapshots
            assert engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',arm=arm,task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('ABLATION EVAL',arm,task,i,64)
        keys=['correct'] if task in ['arc','sciq','mmlu_pro','hellaswag'] else ['em','f1']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in ['C','rnkv',arm,'native']}
        diffs={b:{k:C.paired([r['conditions'][arm][k]-r['conditions'][b][k] for r in records]) for k in keys} for b in ['rnkv','C']}
        changes={b:{name:[r['id'] for r in records if (r['conditions'][arm][keys[0]]-r['conditions'][b][keys[0]])*sign>0]
                    for name,sign in [('rescued_ids',1),('harmed_ids',-1)]} for b in ['rnkv','C']}
        mark(f'arms/{arm}/evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,arm_minus=diffs,rescued_harmed=changes,
             cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in ['C','rnkv',arm,'native']})
        mark(f'arms/{arm}/evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))
    mark(f'arms/{arm}/evaluation/completed.json',status='completed',n=448,finished_at=time.strftime('%F %T'))

def summarize():
    verify();arms=list(CFG['arms']);tasks={}
    for task in CFG['test_counts']:
        tasks[task]={a:C.read(ROOT/f'arms/{a}/evaluation/{task}/summary.json') for a in arms}
        assert all(x['status']=='completed' and x['manifest_sha256']==identity() for x in tasks[task].values())
    mark('summary.json',status='completed',tasks=tasks,config=CFG)
    names=['C','rnkv',*arms,'native']
    lines=['# RN-KV structural ablations: same budgets and seven fixed task cohorts','',
           '| Task | Metric | C | Full RN-KV | Memory-only from scratch | Keep body+Memory | Native |',
           '|---|---|---:|---:|---:|---:|---:|']
    for task,armresults in tasks.items():
        keys=['correct'] if task in ['arc','sciq','mmlu_pro','hellaswag'] else ['em','f1']
        for k in keys:
            values={a:v for a,v in armresults[arms[0]]['metrics'].items()}
            for a in arms:values[a]=armresults[a]['metrics'][a]
            label='accuracy' if k=='correct' else k
            lines.append('| '+task+' | '+label+' | '+' | '.join(f'{values[n][k]*100:.2f}%' for n in names)+' |')
    lines+=['',*['- '+v for v in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

if __name__=='__main__':
    if sys.argv[1]=='summarize':summarize()
    else:evaluate(sys.argv[1])
