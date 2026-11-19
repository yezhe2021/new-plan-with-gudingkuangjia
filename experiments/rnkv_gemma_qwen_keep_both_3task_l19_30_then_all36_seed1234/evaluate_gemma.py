"""Shared frozen Gemma A+B/Native baselines and two independently trained memories."""
from gemma_common import *
from runtime_gemma import load
E=module_at('gemma_task_scorers',REFERENCE/'evaluate_new.py')

@torch.no_grad()
def score(module,row,cache,*,rn=True):
    if row['task']=='gsm8k':
        if rn:return E.score(module,row,cache)
        tokens=module.rollout(row,cache,rn=False,cap=512);text=module.engine.tok.decode(tokens,skip_special_tokens=True)
        ans=E.N.extract(text);legacy=C.SC.extract_numeric_answer(text);gold=E.N.normalize(row['gold'])
        eos=module.engine.model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(module.engine.tok.convert_tokens_to_ids('<|im_end|>'))
        return dict(**ans,correct=int(ans['prediction']==gold),legacy_prediction=legacy,legacy_correct=int(E.N.normalize(legacy)==gold),generation=text,generated_tokens=len(tokens),reached_cap=int(len(tokens)==512 and tokens[-1] not in stops))
    if row['task'] in ['arc','hotpotqa']:return module.score(row,cache,rn=rn)
    return E.X.score(module,row,cache,rn=rn)

def baseline_path(row):return TOP/f'baselines/{row["task"]}/{row["id"]}.json'

@torch.no_grad()
def baselines(engine,module,row,translated):
    path=baseline_path(row)
    if path.exists():
        v=C.read(path);assert v['manifest_sha256']==identity() and v['row_sha256']==C.digest(row);return v['conditions']
    transferred=score(module,row,translated,rn=False)
    prior=REFERENCE/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json';native=None;reuse=None
    if prior.exists():
        old=next(r for r in N.rows('test') if r['id']==row['id'])
        for k in ['prompt','content_ids','suffix_ids','gold','candidates','normalize']:
            assert row.get(k)==old.get(k),('Native baseline input mismatch',row['id'],k)
        old=C.read(prior);assert old['manifest_sha256']==CFG['data_reference_manifest_sha256'];native=old['conditions']['native'];reuse=str(prior)
    if native is None:
        cache=engine.native_cache(row,baseline=True);native=score(module,row,cache,rn=False);del cache
    values=dict(gemma_stage_ab=transferred,native=native)
    C.save(path,dict(manifest_sha256=identity(),row_sha256=C.digest(row),id=row['id'],conditions=values,native_reuse=reuse))
    return values

@torch.no_grad()
def main():
    verify();assert done('runs/completed.json');engine,translator,module=load(final=True);module.eval().requires_grad_(False);engine.mode='test'
    tv=C.versions(translator);mv=C.versions(module);tasks={}
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():status(status='paused',phase='evaluation',task=task,sample=i);raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity() and v['arm']==ARM;records.append(v);continue
            base=engine.base(row);translated=engine.cache(row,base,translator);memory=module.construct(row,translated)
            own=score(module,row,memory);values={**baselines(engine,module,row,translated),ARM:own}
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,gold=row['gold'],conditions=values);records.append(C.read(path));del base,translated,memory
            engine.unchanged();assert C.versions(translator)==tv and C.versions(module)==mv and engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('GEMMA EVAL',ARM,task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in ['gemma_stage_ab',ARM,'native']}
        diffs={b:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][b][k] for r in records]) for k in keys} for b in ['gemma_stage_ab','native']}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,arm_minus=diffs,
            cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in metrics})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json')
    verify();mark('summary.json',status='completed',tasks=tasks);mark('completed.json',status='completed',finished_at=time.strftime('%F %T'))

def summarize():
    verify();tasks={};names=['gemma_stage_ab',*CFG['arm_order'],'native'];lines=['# Gemma to Qwen: matched three-task Keep-both training','','| Task | Metric | Frozen Gemma A+B | Memory19-30 | Memory1-36 | Native |','|---|---|---:|---:|---:|---:|']
    for task in CFG['test_counts']:
        records={a:[C.read(TOP/f'arms/{a}/evaluation/{task}/per_sample/{r["id"]}.json') for r in rows('test') if r['task']==task] for a in CFG['arm_order']}
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        first=records[CFG['arm_order'][0]];last=records[CFG['arm_order'][1]]
        metrics={a:{k:sum(r['conditions'][a][k] for r in (first if a in ['gemma_stage_ab','native',CFG['arm_order'][0]] else last))/64 for k in keys} for a in names}
        difference={k:C.paired([b['conditions']['all_36'][k]-a['conditions']['restricted_19_30'][k] for a,b in zip(first,last)]) for k in keys}
        tasks[task]=dict(metrics=metrics,all36_minus_19_30=difference)
        for k in keys:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{metrics[a][k]*100:.2f}%' for a in names)+' |')
    C.save(TOP/'summary.json',dict(manifest_sha256=identity(),status='completed',tasks=tasks))
    (TOP/'RESULTS.md').write_text('\n'.join(lines+['',*['- '+s for s in CFG['caveats']]])+'\n')
    C.save(TOP/'completed.json',dict(manifest_sha256=identity(),status='completed',finished_at=time.strftime('%F %T')));status(status='completed')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='summarize':summarize()
    else:main()
