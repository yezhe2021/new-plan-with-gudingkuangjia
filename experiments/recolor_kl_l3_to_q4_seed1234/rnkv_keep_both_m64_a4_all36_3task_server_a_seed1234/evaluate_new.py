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
def parity(engine=None):
    if done('evaluation_parity.json'):return
    from runtime_new import Memory64
    from engine_offload import Engine,to_cache
    if engine is None:engine=Engine()
    correction=T.load_final('C').requires_grad_(False)
    cfg=dict(CFG,memory_tokens=32);module=Memory64(engine,cfg);assert module.M==32
    cp=torch.load(REFERENCE/'runs/epoch4_step1536.pt',map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==CFG['reference_manifest_sha256'] and cp['step']==1536
    module.load_state_dict(cp['state'],strict=True);module.eval().requires_grad_(False);checks=[]
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task);base=engine.base(row);translated=engine.cache(row,base,correction);old=baseline(row)
        native=engine.native_cache(row,baseline=True);cuda=to_cache(native,engine.model,'cuda')
        results={'native':score(module,row,cuda,rn=False),'C':score(module,row,translated,rn=False)};del cuda,native
        memory=module.construct(row,translated);results[N.ARM]=score(module,row,memory,rn=True)
        for name,current in results.items():
            reference=old[name];keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
            assert all(current[k]==reference[k] for k in keys),(task,name,'score mismatch')
            if 'generation' in reference:assert current['generation']==reference['generation'],(task,name,'generation mismatch')
            else:assert current['prediction']==reference['prediction'],(task,name,'prediction mismatch')
        checks.append(dict(task=task,id=row['id'],native_C_and_M32_parity=True));del base,translated,memory
        C.log('M64 HOST PARITY PASSED',task)
    module.routing.close();engine.unchanged();mark('evaluation_parity.json',status='completed',checks=checks)

@torch.no_grad()
def main():
    verify();assert done('runs/completed.json') and done('evaluation_parity.json')
    engine,correction,module=load(final=True);module.eval().requires_grad_(False);engine.mode='test';before=[C.versions(correction),C.versions(module)]
    tasks={};names=['C',N.ARM,ARM,'native']
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():raise SystemExit(75)
            p=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if p.exists():
                v=C.read(p);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);translated=engine.cache(row,base,correction);memory=module.construct(row,translated)
            conditions={**baseline(row),ARM:score(module,row,memory,rn=True)}
            mark(str(p.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=conditions)
            records.append(C.read(p));del base,translated,memory;engine.unchanged()
            assert before==[C.versions(correction),C.versions(module)] and engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('M64 EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={k:C.paired([r['conditions'][ARM][k]-r['conditions'][N.ARM][k] for r in records]) for k in keys}
        changes={k:{label:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions'][N.ARM][k])*sign>0] for label,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,m64_minus_m32=differences,rescued_harmed=changes,
            cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json');mark(f'evaluation/{task}/completed.json',status='completed',n=64)
    verify();mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# M64 vs M32, four Rank16 adapters, all-layer Keep-both, three tasks','','| Task | Metric | C | M32+A4 | M64+A4 | Native |','|---|---|---:|---:|---:|---:|']
    for task,s in tasks.items():
        for k in s['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{s["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')
if __name__=='__main__':parity() if len(sys.argv)>1 and sys.argv[1]=='parity' else main()
