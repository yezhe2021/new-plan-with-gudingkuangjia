"""Frozen C/RN-KV/Native on five previously untrained RN tasks,64 each."""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
import common as R
import model as M
import cache as K
from metrics import clean_prediction
from qa_metrics import answer_metrics
C=R.C;torch=R.torch
ARMS=CFG['arms']

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def paused():return (ROOT/'controls/pause_requested.json').exists()

def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    R.verify();assert R.identity()==CFG['parent_manifest_sha256']
    assert C.read(PARENT/'completed.json')['status']=='completed'
    assert C.sha(PARENT/'runs/epoch4_step1024.pt')==CFG['checkpoint_sha256']
    excluded=[]
    for split in ['train','validation','test']:excluded+=R.rows(split)+C.rows(split)
    dependency=Path(R.CFG['dependency']);excluded+=C.read_jsonl(dependency/'prepared/test.jsonl')
    used={C.norm(r['question']) for r in excluded}
    blocked_ids={(r.get('task','hotpotqa' if k=='hotpot_id' else 'calibration'),str(r[k])) for r in excluded for k in ['source_id','hotpot_id'] if r.get(k)}
    import pyarrow.parquet as pq
    raw={t:(pq.read_table(p).to_pylist() if str(p).endswith('.parquet') else C.read(p) if t=='2wiki' else C.read_jsonl(p)) for t,p in CFG['sources'].items()}
    assert {t:len(v) for t,v in raw.items()}==CFG['expected_counts']
    tok=C.AutoTokenizer.from_pretrained(C.CFG['models']['receiver'],local_files_only=True)
    affix=C.read(C.ROOT/'manifest.json');assistant=affix['assistant']
    sender=C.AutoTokenizer.from_pretrained(C.CFG['models']['sender'],local_files_only=True)
    prepared=[];selection={};overlaps={}
    for task,n in CFG['test_counts'].items():
        order=list(range(len(raw[task])));rng=random.Random(CFG['seed']);rng.shuffle(order)
        if task=='mmlu_pro':
            cats=sorted({d['category'] for d in raw[task]});rng.shuffle(cats)
            quotas={c:n//len(cats)+int(i<n%len(cats)) for i,c in enumerate(cats)};counts=Counter()
        selected=[];removed=0
        for index in order:
            d=raw[task][index];question=d['ctx'] if task=='hellaswag' else d['question'];key=C.norm(question)
            source_id=str(d.get('_id',d.get('id',d.get('question_id',d.get('ind',index)))))
            if key in used or (task,source_id) in blocked_ids:removed+=1;continue
            if task=='mmlu_pro' and counts[d['category']]>=quotas[d['category']]:continue
            row=dict(task=task,source_id=source_id,source_index=index,question=question,source_doc_sha256=C.digest(d))
            if task in ['2wiki','musique']:
                if task=='2wiki':context=d['context'];answers=[d['answer']]
                else:
                    assert d['answerable'],'MuSiQue-Ans source must be fully answerable; no outcome-dependent filtering'
                    context=[(p['title'],[p['paragraph_text']]) for p in d['paragraphs']]
                    answers=list(dict.fromkeys([d['answer']]+d['answer_aliases']))
                passages=['Passage '+str(i)+':\n'+title+'\n'+''.join(sentences).strip() for i,(title,sentences) in enumerate(context,1)]
                prompt=('Answer the question using the passages below. Output only the short answer, without an explanation.\n\n'
                        +'\n\n'.join(passages)+'\n\nQuestion: '+question+'\nGive only the answer.')
                row.update(kind='generation',gold=d['answer'],answers=answers,paragraph_count=len(context),context_sha256=C.digest(context),continuation_suffix=assistant)
            elif task=='hellaswag':
                prompt,candidates,gold,kind,_=C.SC.format_item('hellaswag',d)
                row.update(kind='continuation',candidates=candidates,gold=gold,normalize=True,continuation_suffix=assistant+C.SC.ANSWER_CUE[kind])
            else:
                if task=='sciq':
                    options=[d['correct_answer'],d['distractor1'],d['distractor2'],d['distractor3']]
                    permutation=list(range(4));random.Random(int(C.digest(dict(seed=CFG['seed'],question=key))[:16],16)).shuffle(permutation)
                    options=[options[i] for i in permutation];gold=permutation.index(0)
                    prefix='Answer the following multiple-choice question using the passage. Select the single correct option.\n\nPassage:\n'+d['support']
                    row.update(has_support=bool(d['support'].strip()),option_permutation=permutation,support_sha256=C.digest(d['support']))
                else:
                    options=d['options'];gold=int(d['answer_index']);assert d['answer']==chr(65+gold)
                    prefix='Answer the following multiple-choice question about '+d['category'].replace('_',' ')+'. Select the single correct option.'
                    row['category']=d['category'];counts[d['category']]+=1
                assert 2<=len(options)<=26 and 0<=gold<len(options)
                labels=[chr(65+i) for i in range(len(options))]
                prompt=prefix+'\n\nQuestion: '+question+'\n\nChoices:\n'+'\n'.join(label+'. '+str(text).strip() for label,text in zip(labels,options))+'\n\nAnswer with only the option label.'
                row.update(kind='label',candidates=labels,gold=gold,normalize=False,option_count=len(options),continuation_suffix=assistant+C.SC.ANSWER_CUE['arc_label'])
            row.update(id=task+'_'+(d['category']+'_' if task=='mmlu_pro' else '')+source_id,prompt=prompt)
            row['content_ids']=tok.encode(prompt,add_special_tokens=False);row['suffix_ids']=tok.encode(row['continuation_suffix'],add_special_tokens=False)
            assert tok.encode(affix['prefix']+prompt+row['continuation_suffix'],add_special_tokens=False)==affix['prefix_ids']+row['content_ids']+row['suffix_ids']
            row['sender_tokens']=len(sender.encode(prompt,add_special_tokens=False))
            assert len(affix['prefix_ids'])+len(row['content_ids'])+32+len(row['suffix_ids'])+64<CFG['max_context_tokens'],('Selected row too long; no truncation/resampling',row['id'])
            used.add(key);selected.append(row)
            if len(selected)==n:break
        assert len(selected)==n,task
        if task=='mmlu_pro':assert dict(counts)==quotas,(counts,quotas)
        prepared+=selected;overlaps[task]=removed
        selection[task]=dict(source_indices=[r['source_index'] for r in selected],source_ids=[r['source_id'] for r in selected],
            max_body_tokens=max(len(r['content_ids']) for r in selected),max_sender_tokens=max(r['sender_tokens'] for r in selected))
        if task=='mmlu_pro':selection[task].update(category_counts=dict(counts),option_counts=dict(Counter(r['option_count'] for r in selected)))
    assert len({r['id'] for r in prepared})==len(prepared)==320
    C.write_jsonl(ROOT/'prepared/test.jsonl',prepared)
    refs=[PARENT/n for n in ['manifest.json','completed.json','config.json','prepared/train.jsonl','prepared/validation.jsonl','prepared/test.jsonl','runs/epoch4_step1024.pt']]
    refs+=list(PARENT.glob('*.py'))+[Path(p) for p in CFG['sources'].values()]+[ROOT/'prepared/test.jsonl',dependency/'prepared/test.jsonl']
    refs+=[C.ROOT/'prepared'/(s+'.jsonl') for s in ['train','validation','test']]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},
        sources={str(f):C.sha(f) for f in refs},selection=selection,question_overlap_after_filter=0,overlap_candidates_excluded=overlaps,
        counts=CFG['test_counts'],prediction_configurations=960,training=False))
    mark('prepared/completed.json',status='completed',counts=CFG['test_counts'],selection=selection)
    C.log('FIVE-TASK PREPARED',json.dumps({t:s['max_body_tokens'] for t,s in selection.items()}))

def verify():
    R.verify();m=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in m['sources'].items():assert C.sha(n)==d,n

@torch.no_grad()
def score(module,row,cache,*,rn):
    if row['kind']!='generation':
        scores=[];raw=[];lengths=[]
        for candidate in row['candidates']:
            tokens=module.engine.tok.encode(candidate,add_special_tokens=False);assert tokens
            logits=module.token_logits(row,cache,tokens,rn=rn)
            targets=torch.tensor(tokens,device=logits.device)
            value=float(logits.log_softmax(-1).gather(-1,targets[:,None]).sum());raw.append(value);lengths.append(len(tokens))
            scores.append(value/len(tokens) if row['normalize'] else value)
        assert all(C.np.isfinite(s) for s in scores)
        prediction=max(range(len(scores)),key=scores.__getitem__)
        return dict(prediction=prediction,correct=int(prediction==row['gold']),scores=scores,raw_sum_scores=raw,candidate_token_lengths=lengths)
    tokens=module.rollout(row,cache,rn=rn,cap=CFG['max_new_tokens'])
    text=module.engine.tok.decode(tokens,skip_special_tokens=False);prediction=clean_prediction(text)
    return dict(prediction=prediction,generation=text,generated_tokens=len(tokens),reached_cap=int(len(tokens)>=64),
                **answer_metrics(prediction,row['answers'],row['task']))

@torch.no_grad()
def conditions(engine,correction,module,row):
    base=engine.base(row);translated=engine.cache(row,base,correction)
    values={'C':score(module,row,translated,rn=False)}
    regenerated=module.construct(row,translated);values['rnkv']=score(module,row,regenerated,rn=True)
    del regenerated,translated,base
    native=engine.native_cache(row,baseline=True);values['native']=score(module,row,native,rn=False)
    del native
    assert engine.student_test_native_prefills==0
    return values

def unchanged(engine,correction,module,snapshots):
    engine.unchanged();assert C.versions(correction)==snapshots[0] and C.versions(module)==snapshots[1]
    assert all(not p.requires_grad and p.grad is None for m in [correction,module] for p in m.parameters())

@torch.no_grad()
def smoke(engine,correction,module,snapshots):
    checks=[]
    for task in CFG['test_counts']:
        rr=[r for r in rows() if r['task']==task]
        selected={r['id']:r for r in [rr[0],max(rr,key=lambda r:len(r['content_ids']))]}
        for row in selected.values():
            values=conditions(engine,correction,module,row);parity=None
            if row['kind']!='generation' and row['id']==rr[0]['id']:
                # Compare scoring formula with the existing official scorer, then
                # diagnose canonical-cache vs full native prefill separately.
                cache=engine.native_cache(row,baseline=True)
                cached,_=C.RT.transfer_candidate_scores(engine.runtime,cache,cache.get_seq_length(),row['candidates'],row['normalize'],row['continuation_suffix'])
                direct,_=C.RT.native_candidate_scores(engine.runtime,row['prompt'],row['candidates'],row['normalize'],row['continuation_suffix'])
                own=torch.tensor(values['native']['scores'],dtype=torch.float64).log_softmax(-1)
                ck=torch.tensor(cached,dtype=torch.float64).log_softmax(-1);di=torch.tensor(direct,dtype=torch.float64).log_softmax(-1)
                formula_kl=float((own.exp()*(own-ck)).sum());direct_kl=float((own.exp()*(own-di)).sum())
                assert abs(formula_kl)<.001 and abs(direct_kl)<.001,('Native scoring parity',task,formula_kl,direct_kl)
                parity=dict(own_vs_official_cached_kl=formula_kl,canonical_cache_vs_direct_kl=direct_kl,
                    official_cached_prediction_matches=int(own.argmax())==int(ck.argmax()),direct_prediction_matches=int(own.argmax())==int(di.argmax()))
                del cache
            unchanged(engine,correction,module,snapshots)
            mark(f'smoke/per_sample/{row["id"]}.json',status='completed',id=row['id'],task=task,gold=row['gold'],conditions=values,parity=parity)
            checks.append(dict(id=row['id'],task=task,body_tokens=len(row['content_ids']),native_scoring_parity=parity))
            C.log('FIVE-TASK SMOKE PASSED',task,row['id'],len(row['content_ids']))
    mark('smoke/completed.json',status='completed',checks=checks,all_weights_frozen=True,student_native_body_prefills=0,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

def summarize_task(task,records):
    keys=['em','f1'] if task in ['2wiki','musique'] else ['correct']
    metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in ARMS}
    differences={b:{k:C.paired([r['conditions']['rnkv'][k]-r['conditions'][b][k] for r in records]) for k in keys} for b in ['C','native']}
    rescues={b:{name:[r['id'] for r in records if (r['conditions']['rnkv'][keys[0]]-r['conditions'][b][keys[0]])*sign>0]
                for name,sign in [('rescued_ids',1),('harmed_ids',-1)]} for b in ['C','native']}
    mark(f'evaluation/{task}/summary.json',status='completed',task=task,n=64,metrics=metrics,rnkv_minus=differences,rescued_harmed=rescues,
         cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in ARMS})
    mark(f'evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))

@torch.no_grad()
def evaluate(engine,correction,module,snapshots,task):
    records=[];started=time.time();data=[r for r in rows() if r['task']==task]
    for i,row in enumerate(data,1):
        if paused():status(status='paused',task=task,sample=i);sys.exit(75)
        path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
        if path.exists():
            value=C.read(path);assert value['manifest_sha256']==identity();records.append(value);continue
        reuse=ROOT/f'smoke/per_sample/{row["id"]}.json'
        if reuse.exists():
            value=C.read(reuse);assert value['manifest_sha256']==identity();values=value['conditions']
        else:values=conditions(engine,correction,module,row)
        mark(str(path.relative_to(ROOT)),id=row['id'],source_id=row['source_id'],task=task,question=row['question'],gold=row['gold'],
             category=row.get('category'),conditions=values,reused_smoke=str(reuse) if reuse.exists() else None)
        records.append(C.read(path));unchanged(engine,correction,module,snapshots)
        status(status='running',phase='evaluation',task=task,sample=i,total=64,elapsed_seconds=time.time()-started)
        if i==1 or i%8==0:C.log('FIVE-TASK EVAL',task,i,64)
    summarize_task(task,records)

def summarize():
    tasks={t:C.read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    assert all(x['status']=='completed' and x['manifest_sha256']==identity() for x in tasks.values())
    mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# Frozen RN-KV: five new tasks,64 each','','| Dataset | Metric | C | RN-KV | Native | RN-KV minus C (pp) |','|---|---|---:|---:|---:|---:|']
    for task in tasks:
        for k in (['em','f1'] if task in ['2wiki','musique'] else ['correct']):
            metric='accuracy' if k=='correct' else k
            vv=[tasks[task]['metrics'][a][k] for a in ARMS]
            lines.append('| '+task+' | '+metric+' | '+' | '.join(f'{v*100:.2f}%' for v in vv)+' | '+f'{(vv[1]-vv[0])*100:+.2f}'+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    verify();mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

def run():
    verify();engine,correction,module=M.load('final');engine.mode='test';module.eval().requires_grad_(False)
    snapshots=(C.versions(correction),C.versions(module));smoke(engine,correction,module,snapshots)
    for task in CFG['test_counts']:evaluate(engine,correction,module,snapshots,task)
    summarize()

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError(sys.argv[1])
