"""Original official scoring; no native body access in student cache construction."""
from core import *
from train import load_final
from metrics import answer_metrics,clean_prediction


@torch.no_grad()
def result(engine,row,module,base,*,native=False):
    suffix=row['continuation_suffix'];rt=engine.runtime
    if row['task'] in ['arc','hellaswag']:
        if native:scores,_=RT.native_candidate_scores(rt,row['prompt'],row['candidates'],row['normalize'],suffix)
        else:
            cache=engine.cache(row,base,module);length=cache.get_seq_length()
            scores,_=RT.transfer_candidate_scores(rt,cache,length,row['candidates'],row['normalize'],suffix)
        assert all(np.isfinite(x) for x in scores)
        prediction=max(range(len(scores)),key=scores.__getitem__)
        return dict(scores=list(map(float,scores)),prediction=prediction,correct=int(prediction==row['gold']))
    cap=CFG['max_new_tokens'][row['task']]
    if native:
        ids=rt.native_ids(row['prompt'],suffix)
        text,count=RT.greedy_decode(rt,ids,cap) if row['task']=='gsm8k' else RT.generate(rt,ids,cap,seed=CFG['seed'])
    else:
        cache=engine.cache(row,base,module)
        text,count=RT.generate(rt,rt.suffix_ids(suffix),cap,past_key_values=cache,past_length=cache.get_seq_length(),seed=CFG['seed'])
    if row['task']=='gsm8k':
        prediction=SC.extract_numeric_answer(text)
        return dict(prediction=prediction,correct=int(prediction==SC.normalize_number(row['gold'])),generation=text,generated_tokens=count,reached_cap=int(count>=cap))
    prediction=clean_prediction(text)
    return dict(prediction=prediction,generation=text,generated_tokens=count,reached_cap=int(count>=cap),**answer_metrics(prediction,row['gold']))


def smoke():
    verify();engine=Engine();engine.mode='test'
    # Sample IDs were fixed before training. This gate compares implementation paths,
    # without inspecting accuracy for hyperparameter or checkpoint selection.
    checks=[]
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task)
        base=engine.base(row)
        if task in ['arc','hellaswag']:
            native=result(engine,row,None,None,native=True)
            cache=engine.native_cache(row,baseline=True)
            scores,_=RT.transfer_candidate_scores(engine.runtime,cache,cache.get_seq_length(),row['candidates'],row['normalize'],row['continuation_suffix'])
            pred=max(range(len(scores)),key=scores.__getitem__)
            p=torch.tensor(native['scores'],dtype=torch.float64).log_softmax(-1)
            q=torch.tensor(scores,dtype=torch.float64).log_softmax(-1)
            kl=float((p.exp()*(p-q)).sum());assert pred==native['prediction'] and abs(kl)<.001
        else:
            cache=engine.native_cache(row,baseline=True)
            a,_=RT.generate(engine.runtime,engine.runtime.native_ids(row['prompt'],row['continuation_suffix']),16,seed=CFG['seed'])
            b,_=RT.generate(engine.runtime,engine.runtime.suffix_ids(row['continuation_suffix']),16,past_key_values=cache,past_length=cache.get_seq_length(),seed=CFG['seed'])
            assert a==b,('Native cached generation parity',task);kl=None
        # Every trained arm was already checked to equal A at zero initialization.
        # Here additionally ensure plain raw transfer leaves no nonfinite cache/logits.
        cache=engine.cache(row,base,None)
        logits=engine.logits(row,cache);assert torch.isfinite(logits).all()
        checks.append(dict(task=task,id=row['id'],native_cached_interface_parity=True,choice_kl=kl,
                           raw_student_finite=True,student_native_prefills=engine.student_test_native_prefills))
        del base,cache,logits
    assert engine.student_test_native_prefills==0;engine.unchanged()
    mark('smoke/evaluation.json',status='completed',checks=checks)


@torch.no_grad()
def evaluate(task):
    verify();assert read(ROOT/'smoke/evaluation.json')['manifest_sha256']==identity()
    engine=Engine();engine.mode='test';modules={a:load_final(a) for a in ['A','B','C','D','E']}
    for m in modules.values():
        if m is not None:m.requires_grad_(False)
    records=[];rr=[r for r in rows('test') if r['task']==task]
    assert len(rr)==CFG['test_counts'][task]
    for i,row in enumerate(rr,1):
        folder=ROOT/'evaluation'/task/'per_sample';path=folder/(row['id']+'.json')
        if path.exists():
            item=read(path);assert item['manifest_sha256']==identity();records.append(item);continue
        base=None;conditions={}
        for a in ['native','A','B','C','D','E']:
            arm_path=ROOT/'evaluation'/task/'per_arm'/a/(row['id']+'.json')
            if arm_path.exists():
                item=read(arm_path);assert item['manifest_sha256']==identity();value=item['result']
            else:
                if a!='native' and base is None:base=engine.base(row)
                value=result(engine,row,modules.get(a),base,native=a=='native')
                mark(str(arm_path.relative_to(ROOT)),id=row['id'],task=task,arm=a,result=value)
            conditions[a]=value
            status(status='running',phase='evaluation',task=task,sample=i,total=len(rr),arm=a)
            if task=='gsm8k':log('GSM8K',f'{i}/{len(rr)}',a,value['correct'],value['generated_tokens'])
        item=dict(id=row['id'],task=task,gold=row['gold'],conditions=conditions,student_native_body_prefills=engine.student_test_native_prefills)
        mark(str(path.relative_to(ROOT)),**item);records.append(item);del base
        engine.unchanged()
        if i%8==0 or task=='gsm8k':log('EVALUATION',task,f'{i}/{len(rr)}')
    assert engine.student_test_native_prefills==0;verify()
    write_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl',records)
    keys=['em','f1'] if task=='hotpotqa' else ['correct']
    metrics={a:{k:sum(r['conditions'][a][k] for r in records)/len(rr) for k in keys} for a in ['native','A','B','C','D','E']}
    for a in metrics:
        metrics[a]['n']=len(rr)
        if keys==['correct']:metrics[a]['accuracy']=metrics[a].pop('correct')
        metrics[a]['cap_count']=sum(r['conditions'][a].get('reached_cap',0) for r in records)
    comparisons={a+'_minus_'+b:{k:paired([r['conditions'][a][k]-r['conditions'][b][k] for r in records]) for k in keys}
                 for a,b in [('D','C'),('E','D'),('D','A'),('D','B'),('C','B')]}
    mark(f'evaluation/{task}/summary.json',status='completed',task=task,metrics=metrics,comparisons=comparisons,
         student_native_body_prefills=0,primary_checkpoint='fixed800')
    mark(f'evaluation/{task}/completed.json',status='completed',n=len(rr),finished_at=time.strftime('%F %T'))


def summarize():
    verify();tasks={t:read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    assert all(x['manifest_sha256']==identity() and x['status']=='completed' for x in tasks.values())
    diagnostic=read(ROOT/'diagnostics/all_arms/completed.json')
    mark('summary.json',status='completed',tasks=tasks,validation_diagnostics=diagnostic['metrics'],config=CFG,caveats=CFG['caveats'])
    names=['native','A','B','C','D','E']
    lines=['# Recolor-KL fixed-Recolor Stage-2 comparison','',
           '| Group | ARC accuracy | HellaSwag accuracy | GSM8K accuracy | HotpotQA EM | HotpotQA F1 |',
           '|---|---:|---:|---:|---:|---:|']
    for a in names:
        values=[tasks[t]['metrics'][a]['accuracy'] for t in ['arc','hellaswag','gsm8k']]+[tasks['hotpotqa']['metrics'][a][k] for k in ['em','f1']]
        lines.append('| '+a+' | '+' | '.join(f'{v*100:.2f}%' for v in values)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    if sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='evaluate':evaluate(sys.argv[2])
    elif sys.argv[1]=='summarize':summarize()
    else:raise ValueError(sys.argv[1])
