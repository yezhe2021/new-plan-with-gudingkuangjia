"""Frozen HotpotQA evidence-only KV pilot; generated answers, no Reader training."""
import backend as B
from backend import ROOT,CFG,E,torch,np,read,save,sha,ident,rows,status,temp
from pilot_metrics import seed_for,answer_metrics,bootstrap
import copy
import fcntl
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from collections import Counter
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


def write_rows(path,records):
    path.parent.mkdir(exist_ok=True)
    assert not path.exists(),str(path)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))


def prepare():
    E.verify()
    tokenizers={f:E.tok(f) for f in ['llama','qwen']};qt=tokenizers['qwen']
    previous=Path(CFG['previous_hotpot']);excluded_ids=set();excluded_questions=set();old_files=[]
    for p in sorted((previous/'prepared').glob('*.jsonl')):
        old_files.append(p)
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if 'id' in r:excluded_ids.add(r['id'])
            if 'question' in r:excluded_questions.add(B.norm(r['question']))
    def collect_ids(x):
        if isinstance(x,str) and re.fullmatch('[0-9a-f]{24}',x):excluded_ids.add(x)
        elif isinstance(x,dict):
            for v in x.values():collect_ids(v)
        elif isinstance(x,list):
            for v in x:collect_ids(v)
    for name in read(previous/'config.json')['old_pilot_manifests']:
        p=Path(name)
        if p.exists():collect_ids(read(p));old_files.append(p)
    for split in ['train','validation']:
        excluded_questions.update(r['question_key'] for r in E.U.rows(split))
    raw=read(CFG['source_json']);random.Random(CFG['seed']).shuffle(raw)
    chosen=[];skips=Counter();seen_ids=set(excluded_ids);seen_q=set(excluded_questions)
    calibration_titles=set();test_titles=set()
    for d in raw:
        if len(chosen)==CFG['pilot_count']+CFG['reserved_test_count']:break
        rid=d['_id'];q=d['question'].strip()
        if d['type']!='bridge':skips['not_bridge']+=1;continue
        if rid in seen_ids or B.norm(q) in seen_q:skips['previous_or_duplicate_question']+=1;continue
        titles={v[0] for v in d['supporting_facts']}
        docs=[(title,sentences) for title,sentences in d['context'] if title in titles]
        if len(titles)!=2 or len(docs)!=2:skips['not_two_gold_documents']+=1;continue
        split='calibration' if len(chosen)<CFG['pilot_count'] else 'reserved_test'
        if split=='reserved_test' and titles&calibration_titles:skips['cross_split_document']+=1;continue
        support='\n\n'.join(f'Document {i+1}: {title}\n'+''.join(sentences).strip() for i,(title,sentences) in enumerate(docs))
        evidence='Reference information:\n'+support+'\nEnd of reference'
        try:
            row=E.U.make_row('hotpot_'+rid,'gsm8k',dict(question=q),tokenizers['llama'],qt,
                prompt=evidence,candidates=[],gold=d['answer'],normalize=False,cue='')
            frame=row['frames']['qwen'];body=frame['body_ids']
            assert not row['native_tail']
            assert len(body)<=CFG['max_evidence_tokens'] and len(row['frames']['llama']['body_ids'])<=CFG['max_sender_tokens']
            before='\n\nAnswer the question. Use the reference only when relevant. Give only the short answer, without explanation.\n\nQuestion: '
            query=before+q
            rendered=frame['rendered_prefix']+evidence+query+frame['rendered_suffix']
            encoded=qt(rendered,add_special_tokens=False,return_offsets_mapping=True);allids=list(encoded['input_ids'])
            assert allids[:len(body)]==body and len(allids)+CFG['max_new_tokens']<=CFG['max_total_tokens']
            qstart=len(frame['rendered_prefix'])+len(evidence)+len(before);qend=qstart+len(q)
            qpos=[i-len(body) for i,(a,b) in enumerate(encoded['offset_mapping']) if qstart<=a<b<=qend]
            assert qpos and min(qpos)>0
            # Gold is used only in separate teacher-forced scoring, never generation or selection features.
            goldfull=qt.encode(rendered+d['answer'],add_special_tokens=False)
            assert goldfull[:len(allids)]==allids and len(goldfull)>len(allids)
            compact=qt.encode(frame['rendered_prefix']+query.lstrip()+frame['rendered_suffix'],add_special_tokens=False)
            row.update(task='hotpotqa_controlled',question=q,answer=d['answer'],support=support,titles=sorted(titles),
                source_id=rid,split=split,continuation_ids=allids[len(body):],question_positions=qpos,
                gold_ids=goldfull[len(allids):],compact_ids=compact,raw_sha256=E.C.digest(d))
            chosen.append(row);seen_ids.add(rid);seen_q.add(B.norm(q))
            (calibration_titles if split=='calibration' else test_titles).update(titles)
        except (AssertionError,ValueError):skips['length_or_boundary']+=1
    assert len(chosen)==CFG['pilot_count']+CFG['reserved_test_count'],(len(chosen),skips)
    assert not calibration_titles&test_titles
    for split in ['calibration','reserved_test']:
        group=[r for r in chosen if r['split']==split];n=len(group)
        lengths=np.array([len(r['source_indices']) for r in group])
        cost=np.abs(lengths[:,None]-lengths[None,:]).astype(float)
        cost+=np.random.default_rng(seed_for(CFG['seed'],split,'donor')).uniform(0,.5,cost.shape)
        for i in range(n):
            for j in range(n):
                if i==j or set(group[i]['titles'])&set(group[j]['titles']):cost[i,j]=1e9
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            assert cost[i,j]<1e9
            group[i]['donor_id']=group[j]['id']
    pilot=[r for r in chosen if r['split']=='calibration'];reserved=[r for r in chosen if r['split']=='reserved_test']
    write_rows(ROOT/'prepared/questions.jsonl',pilot);write_rows(ROOT/'prepared/reserved_test.jsonl',reserved)
    anchors={tuple(r['frames']['qwen']['body_ids'][:r['frames']['qwen']['native_prefix_length']]) for r in chosen}
    assert len(anchors)==1
    checkpoints={k:v for k,v in read(E.ROOT/'manifest.json')['checkpoints'].items() if k in ['stage_a','stage_b']}
    m=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),prepared_sha256=sha(ROOT/'prepared/questions.jsonl'),
        reserved_test_sha256=sha(ROOT/'prepared/reserved_test.jsonl'),source_sha256=sha(CFG['source_json']),
        reference_manifest_sha256=E.identity(),code={p.name:sha(p) for p in ROOT.glob('*.py')},
        checkpoints=checkpoints,anchor_ids=list(next(iter(anchors))),anchor_text=qt.decode(next(iter(anchors))),
        counts=dict(calibration=len(pilot),reserved_test=len(reserved)),skips=dict(skips),
        excluded_question_ids=sorted(excluded_ids),exclusion_sources={str(p):sha(p) for p in old_files},
        no_reader_installed=True,all_parameters_frozen=True,selection_uses_predictions=False,
        reserved_test_not_evaluated=True,environment=dict(torch=torch.__version__,python=sys.version))
    save(ROOT/'manifest.json',m);save(ROOT/'prepared/completed.json',dict(manifest_sha256=ident(),counts=m['counts']))
    status(status='prepared',phase='prepare');E.log(f'PREPARED {m["counts"]}; skips={dict(skips)}')


@torch.inference_mode()
def forward(model,ids,cache=None,base_length=0,blocked=None,use_cache=True):
    t=torch.tensor([ids],device='cuda');mask=torch.ones(1,base_length+len(ids),device='cuda',dtype=torch.long)
    if blocked is not None:mask[:,blocked[0]:blocked[1]]=0
    return model.model(t,past_key_values=cache,attention_mask=mask,
        position_ids=torch.arange(base_length,base_length+len(ids),device='cuda')[None],use_cache=use_cache)


@torch.inference_mode()
def prompt_forward(model,row,cache,blocked=False):
    frame=row['frames']['qwen'];n=len(frame['body_ids'])
    span=(frame['native_prefix_length'],n) if blocked else None
    out=forward(model,row['continuation_ids'],copy.deepcopy(cache),n,span)
    return out,span,n+len(row['continuation_ids'])


@torch.inference_mode()
def answer_nll(model,row,cache,blocked=False,compact=False):
    gold=row['gold_ids']
    if compact:prefix=row['compact_ids'];b=0;span=None;cache=None
    else:
        prefix=row['continuation_ids'];b=len(row['frames']['qwen']['body_ids'])
        span=(row['frames']['qwen']['native_prefix_length'],b) if blocked else None
    ids=prefix+gold[:-1]
    out=forward(model,ids,copy.deepcopy(cache),b,span,use_cache=False)
    logits=model.lm_head(out.last_hidden_state[0,len(prefix)-1:]).float()
    assert logits.shape[0]==len(gold)
    return float(F.cross_entropy(logits,torch.tensor(gold,device='cuda')))


@torch.inference_mode()
def generate(model,qt,row,cache=None,blocked=False,compact=False):
    began=time.time()
    if compact:
        out=forward(model,row['compact_ids']);span=None;total=len(row['compact_ids']);qlp=None
    else:
        out,span,total=prompt_forward(model,row,cache,blocked)
        positions=row['question_positions'];ix=torch.tensor(positions,device='cuda')
        qlogits=model.lm_head(out.last_hidden_state[0,ix-1]).float()
        qlp=float(-F.cross_entropy(qlogits,torch.tensor([row['continuation_ids'][p] for p in positions],device='cuda')))
    logits=model.lm_head(out.last_hidden_state[0,-1]).float();lp=logits.log_softmax(-1)
    first_entropy=float(-(lp.exp()*lp).sum());memory=out.past_key_values
    eos=model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos])
    stops.update([qt.eos_token_id,qt.convert_tokens_to_ids('<|im_end|>')]);stops.discard(None)
    tokens=[];logps=[];stopped=False
    for _ in range(CFG['max_new_tokens']):
        token=int(logits.argmax());tokens.append(token);logps.append(float(logits.log_softmax(-1)[token]))
        if token in stops:stopped=True;break
        if len(tokens)<CFG['max_new_tokens']:
            out=forward(model,[token],memory,total,span);total+=1;memory=out.past_key_values
            logits=model.lm_head(out.last_hidden_state[0,-1]).float()
    assert all(np.isfinite(logps))
    prediction=qt.decode(tokens,skip_special_tokens=True).strip()
    # Teacher forcing happens after greedy completion and starts from an untouched cache.
    nll=answer_nll(model,row,cache,blocked,compact)
    return dict(prediction=prediction,raw_output=qt.decode(tokens,skip_special_tokens=False),generated_ids=tokens,
        generated_tokens=len(tokens),eos=stopped,**answer_metrics(prediction,row['answer']),answer_token_nll=nll,
        question_logp=qlp,first_token_entropy=first_entropy,mean_generation_logp=float(np.mean(logps)),seconds=time.time()-began)


@torch.inference_mode()
def preflight():
    B.verify();assert read(ROOT/'translate_completed.json')['manifest_sha256']==ident()
    model=B.load_model('qwen');anchor=B.native_cache(model,read(ROOT/'manifest.json')['anchor_ids']);checks=[]
    lookup={r['id']:r for r in rows()}
    for row in rows()[:4]:
        k,v=B.memory(row);variants,_=B.corrupted(row,k,v,B.memory(lookup[row['donor_id']]))
        cache=E.U.translated_cache(model,row,anchor,k.cuda(),v.cuda())
        out,_,_=prompt_forward(model,row,cache,True);masked=model.lm_head(out.last_hidden_state[0,-1]).float()
        nk,nv=variants['gaussian'];noise=E.U.translated_cache(model,row,anchor,nk.cuda(),nv.cuda())
        out,_,_=prompt_forward(model,row,noise,True);masked_noise=model.lm_head(out.last_hidden_state[0,-1]).float()
        error=float((masked-masked_noise).abs().max());assert error==0,error
        native=B.native_cache(model,row['frames']['qwen']['body_ids'])
        out,_,_=prompt_forward(model,row,native);cl=model.lm_head(out.last_hidden_state[0,-1]).float()
        full=forward(model,row['frames']['qwen']['body_ids']+row['continuation_ids'],use_cache=False)
        fl=model.lm_head(full.last_hidden_state[0,-1]).float();diff=float((cl-fl).abs().max())
        assert diff<.3,(row['id'],diff)
        assert int(cl.argmax())==int(fl.argmax()),'Native cache/full prefill greedy first token differs'
        checks.append(dict(id=row['id'],masked_content_logit_error=error,native_cache_full_max_logit_error=diff,first_token_equal=True))
    assert not any(p.requires_grad for p in model.parameters())
    save(ROOT/'preflight/completed.json',dict(manifest_sha256=ident(),checks=checks));E.log('PREFLIGHT PASSED')


@torch.inference_mode()
def evaluate():
    B.verify();assert read(ROOT/'preflight/completed.json')['manifest_sha256']==ident()
    model=B.load_model('qwen');qt=E.tok('qwen');anchor=B.native_cache(model,read(ROOT/'manifest.json')['anchor_ids'])
    records=rows();lookup={r['id']:r for r in records}
    for i,row in enumerate(records,1):
        path=B.result_path(row)
        result=read(path) if path.exists() else dict(id=row['id'],answer=row['answer'],question=row['question'],
            manifest_sha256=ident(),donor_id=row['donor_id'],conditions={},complete=False)
        assert result['manifest_sha256']==ident()
        key,value=B.memory(row);variants,indices=B.corrupted(row,key,value,B.memory(lookup[row['donor_id']]))
        result['partial_replaced_indices']=indices
        for name in CFG['conditions']:
            if name in result['conditions']:continue
            status(status='running',phase='evaluate',sample=i,total=len(records),condition=name)
            if name=='question_only_compact':cache=None
            elif name=='native_evidence':cache=B.native_cache(model,row['frames']['qwen']['body_ids'])
            else:
                k,v=variants['clean' if name=='reject_all' else name]
                cache=E.U.translated_cache(model,row,anchor,k.cuda(),v.cuda())
            result['conditions'][name]=generate(model,qt,row,cache,blocked=name=='reject_all',compact=name=='question_only_compact')
            save(path,result)
            m=result['conditions'][name];E.log(f'PILOT {i}/{len(records)} {name} EM={m["em"]:.0f} F1={m["f1"]:.3f} tokens={m["generated_tokens"]} sec={m["seconds"]:.1f}')
        result['complete']=True;save(path,result)
    save(ROOT/'evaluation/completed.json',dict(n=len(records),manifest_sha256=ident()))


def summarize():
    B.verify();records=[read(B.result_path(r)) for r in rows()]
    assert all(r['complete'] and r['manifest_sha256']==ident() for r in records)
    n=len(records);summary=dict(experiment=CFG['name'],status='completed_pilot',timestamp=time.strftime('%F %T'),n=n,
        manifest_sha256=ident(),conditions={},comparisons={},reserved_test_evaluated=False)
    for name in CFG['conditions']:
        group=[r['conditions'][name] for r in records]
        summary['conditions'][name]=dict(n=n,correct=sum(x['em'] for x in group),
            **{k:float(np.mean([x[k] for x in group])) for k in ['em','f1','answer_token_nll','eos','generated_tokens','seconds']})
    for a,b in [('native_evidence','question_only_compact'),('native_evidence','reject_all'),('clean','reject_all'),('clean','question_only_compact'),('clean','wrong'),('clean','partial_wrong')]:
        summary['comparisons'][a+'_minus_'+b]={metric:bootstrap([r['conditions'][a][metric]-r['conditions'][b][metric] for r in records]) for metric in ['em','f1']}
    clean=summary['comparisons']['clean_minus_reject_all']['f1'];native=summary['comparisons']['native_evidence_minus_question_only_compact']['f1']
    checks=dict(native_evidence_useful=native['mean']>=CFG['pilot_gate']['native_f1_gain_min'],
        translated_gain_large_enough=clean['mean']>=CFG['pilot_gate']['translated_f1_gain_min'],
        translated_paired_ci_positive=clean['ci95'][0]>0,
        clean_better_than_wrong=summary['comparisons']['clean_minus_wrong']['f1']['mean']>0)
    summary['pilot_gate']=dict(checks=checks,passed=all(checks.values()),
        next_step='Ready to lock selector on pilot and evaluate reserved128' if all(checks.values()) else 'Do not expand automatically; diagnose native evidence versus translated KV interface')
    summary['helpful_clean_em']=sum(r['conditions']['clean']['em']>r['conditions']['reject_all']['em'] for r in records)
    summary['harmful_clean_em']=sum(r['conditions']['clean']['em']<r['conditions']['reject_all']['em'] for r in records)
    summary['caveats']=['Controlled bridge questions with two gold paragraphs, not full HotpotQA distractor benchmark.',
        'Only calibration32 evaluated; no selector fitted and reserved128 untouched.',
        'Wrong/partial KV corruption is synthetic; length matching and resizing do not model all real transmission errors.',
        'Masks hide evidence keys from all question and answer tokens, retaining common anchor and position slots.',
        'No Reader, training or quantization; original grouped Stage-A and residual Stage-B frozen.',
        'Short-answer generation scored with normalized whole-output EM/F1; no answer-dependent parsing.']
    save(ROOT/'summary.json',summary)
    lines=['# HotpotQA frozen evidence-KV pilot','',f'Completed {n} calibration questions. Reserved128 has not been evaluated.','',
        '|Condition|EM|F1|Gold answer token NLL|','|---|---:|---:|---:|']
    for name,x in summary['conditions'].items():lines.append(f'|{name}|{x["em"]:.4f}|{x["f1"]:.4f}|{x["answer_token_nll"]:.4f}|')
    lines+=['','Pilot gate: '+str(summary['pilot_gate']), '',*summary['caveats']]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (ROOT.parent/'EXPERIMENT_RESULTS.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    for row in rows():temp(row,'mapped').unlink()
    save(ROOT/'completed.json',dict(status='completed_pilot',manifest_sha256=ident(),temporary_tensors_cleaned=True))
    status(status='completed_pilot',phase='all');E.log('HOTPOT PILOT COMPLETE '+json.dumps(summary['pilot_gate']))


if __name__=='__main__':
    action=sys.argv[1]
    try:
        dict(prepare=prepare,capture=B.capture,translate=B.translate,preflight=preflight,evaluate=evaluate,summarize=summarize)[action]()
    except Exception as exc:
        import traceback
        save(ROOT/'failure.json',dict(action=action,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed',phase=action,error=str(exc));raise
