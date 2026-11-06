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
def parity():
    verify()
    if done('evaluation_reference_parity.json'):return
    from engine_frozen import Engine
    from runtime_new import GemmaMemory
    engine=Engine();correction=load_c();PR=parent_module('gemma_partial_reference_runtime_parity','runtime_new.py')
    module=PR.GemmaMemory(engine);cp=torch.load(PARENT/'runs/epoch4_step1536.pt',map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==CFG['reference_manifest_sha256'] and C.sha(PARENT/'runs/epoch4_step1536.pt')==CFG['partial_checkpoint_sha256']
    module.load_state_dict(cp['state'],strict=True);module.eval().requires_grad_(False)
    all_module=GemmaMemory(engine);all_module.load_state_dict(cp['state'],strict=True);all_module.eval().requires_grad_(False)
    checks=[]
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task);old=baseline(row);base=engine.base(row);translated=engine.cache(row,base,correction)
        native=engine.native_cache(row,baseline=True);memory=module.construct(row,translated)
        values={'recolor_c':score(module,row,translated,rn=False),'native':score(module,row,native,rn=False),PARTIAL:score(module,row,memory,rn=True)}
        all_memory=all_module.construct(row,translated)
        assert all(torch.equal(a,b) for pa,pb in zip(K.pairs(memory),K.pairs(all_memory)) for a,b in zip(pa,pb)),'Mask scope must not change Memory construction at equal weights'
        for name,v in values.items():
            reference=old[name];keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
            assert all(v[k]==reference[k] for k in keys),(task,name,'score mismatch')
            if 'generation' in reference:assert v['generation']==reference['generation'],(task,name,'generation mismatch')
            else:assert v['prediction']==reference['prediction'],(task,name,'prediction mismatch')
        checks.append(dict(task=task,id=row['id'],native_C_partial_prediction_parity=True,memory_construction_equal_at_same_weights=True))
        del base,translated,native,memory,all_memory;engine.unchanged();C.log('GEMMA ALL36 REFERENCE PARITY PASSED',task)
    module.routing.close();all_module.routing.close();mark('evaluation_reference_parity.json',status='completed',checks=checks)

@torch.no_grad()
def main():
    verify();assert done('runs/completed.json') and done('evaluation_reference_parity.json')
    engine,correction,module=load(final=True);module.eval().requires_grad_(False);engine.mode='test';before=[C.versions(correction),C.versions(module)]
    tasks={};names=['recolor_c',PARTIAL,ARM,'native']
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():raise SystemExit(75)
            p=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if p.exists():
                v=C.read(p);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);translated=engine.cache(row,base,correction);memory=module.construct(row,translated)
            conditions={**baseline(row),ARM:score(module,row,memory,rn=True)}
            mark(str(p.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=conditions,
                reference_source=str(reference_path(row)),recolor_sha256=map_sha(),c_checkpoint_sha256=CFG['c_checkpoint_sha256'])
            records.append(C.read(p));del base,translated,memory;engine.unchanged()
            assert before==[C.versions(correction),C.versions(module)] and engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('GEMMA ALL36 EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={a:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in ['recolor_c',PARTIAL]}
        changes={a:{k:{label:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions'][a][k])*sign>0] for label,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys} for a in ['recolor_c',PARTIAL]}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,all36_minus=differences,rescued_harmed=changes,
            cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json');mark(f'evaluation/{task}/completed.json',status='completed',n=64)
    verify();mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# Gemma Recolor+C: fresh all36 vs fresh19-30 Keep-both, identical three-task budget','','| Task | Metric | Recolor+C | Memory19-30 | Memory1-36 | Qwen Native |','|---|---|---:|---:|---:|---:|']
    for task,s in tasks.items():
        for k in s['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{s["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')
if __name__=='__main__':parity() if len(sys.argv)>1 and sys.argv[1]=='parity' else main()
