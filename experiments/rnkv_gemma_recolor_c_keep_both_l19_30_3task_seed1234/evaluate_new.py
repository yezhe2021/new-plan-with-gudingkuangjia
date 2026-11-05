"""Eight fixed64-question cohorts, immutable baselines, two explicit GSM scores."""
import importlib.util
from experiment_common import *
from runtime_new import load
sys.path.append(str(ORIGINAL/'cross_dataset5x64_seed1234'))
spec=importlib.util.spec_from_file_location('three_task_cross_scorer',ORIGINAL/'cross_dataset5x64_seed1234/experiment.py')
X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X)
spec=importlib.util.spec_from_file_location('unchanged_gsm_numeric',GSM/'numeric.py')
NUM=importlib.util.module_from_spec(spec);spec.loader.exec_module(NUM)

@torch.no_grad()
def score(module,row,cache,*,rn=True):
    if row['task']!='gsm8k':
        return module.score(row,cache,rn=rn) if row['task'] in ['arc','hotpotqa'] else X.score(module,row,cache,rn=rn)
    cap=CFG['evaluation_generation_caps']['gsm8k'];started=time.time();tokens=module.rollout(row,cache,rn=rn,cap=cap)
    eos=module.engine.model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(module.engine.tok.convert_tokens_to_ids('<|im_end|>'))
    text=module.engine.tok.decode(tokens,skip_special_tokens=True);answer=NUM.extract(text);legacy=C.SC.extract_numeric_answer(text)
    gold=NUM.normalize(row['gold']);stopped=bool(tokens and tokens[-1] in stops)
    return dict(**answer,correct=int(answer['prediction']==gold),legacy_prediction=legacy,legacy_correct=int(NUM.normalize(legacy)==gold),generation=text,
        token_ids=tokens,generated_tokens=len(tokens),stopped_on_eos=stopped,reached_cap=int(len(tokens)==cap and not stopped),max_new_tokens=cap,seconds=time.time()-started)

@torch.no_grad()
def main():
    verify();assert done('runs/completed.json');assert done('evaluation_native_parity.json')
    engine,correction,module=load(final=True);module.eval().requires_grad_(False);engine.mode='test'
    before=[C.versions(correction),C.versions(module)];tasks={};names=['recolor_c',ARM,'native']
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);translated=engine.cache(row,base,correction)
            plain=score(module,row,translated,rn=False);memory=module.construct(row,translated);result=score(module,row,memory,rn=True)
            conditions=dict(recolor_c=plain,native=N.baseline(row)['native'],**{ARM:result})
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=conditions,
                native_source=str(N.baseline_path(row)),recolor_sha256=map_sha(),c_checkpoint_sha256=C.sha(ROOT/'runs/C/epoch4_step800.pt'))
            records.append(C.read(path));del base,translated,memory
            engine.unchanged();assert before==[C.versions(correction),C.versions(module)] and engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('GEMMA RECOLOR C MEMORY EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={k:C.paired([r['conditions'][ARM][k]-r['conditions']['recolor_c'][k] for r in records]) for k in keys}
        changes={k:{label:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions']['recolor_c'][k])*sign>0] for label,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,memory_minus_recolor_c=differences,rescued_harmed=changes,
            cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json');mark(f'evaluation/{task}/completed.json',status='completed',n=64)
    verify();mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# Gemma Recolor+C, three-task Keep-both, Memory readable19-30','','| Task | Metric | Gemma Recolor+C | +Memory19-30 | Qwen Native |','|---|---|---:|---:|---:|']
    for task,s in tasks.items():
        for k in s['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{s["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

@torch.no_grad()
def parity():
    verify()
    if done('evaluation_native_parity.json'):return
    engine,correction,module=load();module.eval().requires_grad_(False);checks=[]
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task);native=engine.native_cache(row,baseline=True)
        current=score(module,row,native,rn=False);reference=N.baseline(row)['native']
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        assert all(current[k]==reference[k] for k in keys),(task,current,reference)
        if 'generation' in reference:assert current['generation']==reference['generation'],('Native generation mismatch',task)
        else:assert current['prediction']==reference['prediction'],('Native prediction mismatch',task)
        checks.append(dict(task=task,id=row['id'],scores_and_generation_parity=True));del native
    engine.unchanged();mark('evaluation_native_parity.json',status='completed',checks=checks,canonical_native_reuse=True)

if __name__=='__main__':parity() if len(sys.argv)>1 and sys.argv[1]=='parity' else main()
