"""Eight fixed64-question cohorts, immutable baselines, two explicit GSM scores."""
import importlib.util
from experiment_common import *
from runtime_new import load
sys.path.append(str(ORIGINAL/'cross_dataset5x64_seed1234'))
spec=importlib.util.spec_from_file_location('three_task_cross_scorer',ORIGINAL/'cross_dataset5x64_seed1234/experiment.py')
X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X)
spec=importlib.util.spec_from_file_location('unchanged_gsm_numeric',GSM/'numeric.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)

@torch.no_grad()
def score(module,row,cache):
    if row['task']!='gsm8k':
        return module.score(row,cache,rn=True) if row['task'] in ['arc','hotpotqa'] else X.score(module,row,cache,rn=True)
    cap=CFG['evaluation_generation_caps']['gsm8k'];started=time.time();tokens=module.rollout(row,cache,rn=True,cap=cap)
    eos=module.engine.model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(module.engine.tok.convert_tokens_to_ids('<|im_end|>'))
    text=module.engine.tok.decode(tokens,skip_special_tokens=True);answer=N.extract(text);legacy=C.SC.extract_numeric_answer(text)
    gold=N.normalize(row['gold']);stopped=bool(tokens and tokens[-1] in stops)
    return dict(**answer,correct=int(answer['prediction']==gold),legacy_prediction=legacy,legacy_correct=int(N.normalize(legacy)==gold),generation=text,
        token_ids=tokens,generated_tokens=len(tokens),stopped_on_eos=stopped,reached_cap=int(len(tokens)==cap and not stopped),max_new_tokens=cap,seconds=time.time()-started)

@torch.no_grad()
def main():
    verify();assert done('runs/completed.json');engine,correction,module=load(final=True);module.eval().requires_grad_(False);engine.mode='test'
    before=[C.versions(correction),C.versions(module)];tasks={};names=['C',DYNAMIC,ARM,'native']
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():status(status='paused',phase='evaluation',task=task,sample=i);raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);translated=engine.cache(row,base,correction);memory=module.construct(row,translated)
            result=score(module,row,memory);conditions={**baseline(row),ARM:result}
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=conditions)
            records.append(C.read(path));del base,translated,memory
            engine.unchanged();assert before==[C.versions(correction),C.versions(module)] and engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('ALL36 THREE-TASK EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={b:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][b][k] for r in records]) for k in keys} for b in ['C',DYNAMIC,'native']}
        changes={b:{k:{name:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions'][b][k])*sign>0] for name,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys} for b in ['C',DYNAMIC]}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,new_minus=differences,rescued_harmed=changes,
            cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json');mark(f'evaluation/{task}/completed.json',status='completed',n=64)
    verify();mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# Static versus dynamic Memory: Llama to Qwen','','| Task | Metric | C | Dynamic Memory | Static Memory | Native |','|---|---|---:|---:|---:|---:|']
    for task,s in tasks.items():
        for k in s['metrics'][ARM]:
            label=('strict final numeric' if k=='correct' else 'legacy numeric') if task=='gsm8k' else ('accuracy' if k=='correct' else k)
            lines.append('| '+task+' | '+label+' | '+' | '.join(f'{s["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed');C.log('ALL36 THREE-TASK ALL COMPLETED')

if __name__=='__main__':main()
