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


from selection_evaluation import prepare, summarize
from selector_rules import features,decide


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
    locked=read(ROOT/'selector.json');assert locked['manifest_sha256']==ident()
    assert read(ROOT/'prepared/completed.json')['selector_sha256']==sha(ROOT/'selector.json')
    records=rows();lookup={r['id']:r for r in records}
    for i,row in enumerate(records,1):
        path=B.result_path(row)
        result=read(path) if path.exists() else dict(id=row['id'],answer=row['answer'],question=row['question'],
            manifest_sha256=ident(),selector_sha256=sha(ROOT/'selector.json'),donor_id=row['donor_id'],conditions={},complete=False)
        assert result['manifest_sha256']==ident() and result['selector_sha256']==sha(ROOT/'selector.json')
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
        result['selection']={}
        for c in CFG['corruption_conditions']:
            x=features(result['conditions'][c],result['conditions']['reject_all'])
            result['selection'][c]=dict(features=x,decisions={name:decide(x,rule) for name,rule in locked['rules'].items()})
        result['complete']=True;save(path,result)
    save(ROOT/'evaluation/completed.json',dict(n=len(records),manifest_sha256=ident()))


if __name__=='__main__':
    action=sys.argv[1]
    try:
        dict(prepare=prepare,capture=B.capture,translate=B.translate,preflight=preflight,evaluate=evaluate,summarize=summarize)[action]()
    except Exception as exc:
        import traceback
        save(ROOT/'failure.json',dict(action=action,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed',phase=action,error=str(exc));raise
