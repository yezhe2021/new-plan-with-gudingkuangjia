"""Frozen MCQ Writer visibility diagnostic; 32 OBQA + 32 ARC test items."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234'
sys.path.insert(0, str(LEGACY))
import kv_backend as kb
from tasks_native import tasks, stable_id
from protocol import make_cache


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def body_map(st, rt, source, body):
    s = st(source, add_special_tokens=False, return_offsets_mapping=True)
    t = rt(body, add_special_tokens=False, return_offsets_mapping=True)
    offset = len(source)-len(body)
    sb = [(i+1,a-offset,b-offset) for i,(a,b) in enumerate(s['offset_mapping'])
          if offset <= a < b <= len(source)]
    tb = [(i,a,b) for i,(a,b) in enumerate(t['offset_mapping']) if a < b]
    bounds = sorted({0} | ({b for _,_,b in sb} & {b for _,_,b in tb}))
    mapping = {}
    # Receiver token0 remains native. Every subsequent option token is transmitted.
    for left,right in zip(bounds,bounds[1:]):
        si = [i for i,_,b in sb if left < b <= right]
        ti = [i for i,_,b in tb if left < b <= right]
        if not si or not ti: continue
        for rank,index in enumerate(ti):
            sr = math.ceil((rank+1)*len(si)/len(ti))-1 if len(si)>len(ti) else rank*len(si)//len(ti)
            if index > 0: mapping[index] = si[sr]
    assert set(mapping) == set(range(1,len(t['input_ids']))), 'Incomplete options mapping'
    assert st.bos_token_id is not None
    return [st.bos_token_id]+s['input_ids'], [mapping[i] for i in sorted(mapping)]


def scores(model, tok, context_ids, choices, prefix=None):
    result=[]
    n = 0 if prefix is None else prefix[0].shape[1]
    for choice in choices:
        cid = tok.encode(' '+choice, add_special_tokens=False)
        assert cid
        ids = context_ids[n:] + cid[:-1]
        cache = None if prefix is None else make_cache(model,*prefix,torch.arange(n,device='cuda'))
        out = model.model(input_ids=torch.tensor([ids],device='cuda'),
            attention_mask=torch.ones((1,n+len(ids)),device='cuda',dtype=torch.long),
            position_ids=torch.arange(n,n+len(ids),device='cuda')[None],
            past_key_values=cache,use_cache=False)
        logits = model.lm_head(out.last_hidden_state[0,-len(cid):]).float()
        lp=F.log_softmax(logits,-1)
        score=float(lp.gather(-1,torch.tensor(cid,device='cuda')[:,None]).sum())
        result.append(score)
    return result


def prediction(values, choices):
    norm=[v/len(c) for v,c in zip(values,choices)]
    return {'scores':values,'pred':max(range(len(values)),key=values.__getitem__),
            'pred_norm':max(range(len(norm)),key=norm.__getitem__)}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    kb.seed_all(1234)
    output=ROOT/('smoke' if a.smoke else 'results'); output.mkdir(parents=True,exist_ok=True)
    ck_a=LEGACY/'runs/mcq/stage_a/best.pt'; ck_b=LEGACY/'runs/mcq/stage_b/best.pt'
    cfg={'models':kb.CFG['models'],'attention_implementation':kb.CFG['attention_implementation']}
    rt=AutoTokenizer.from_pretrained(cfg['models']['qwen'],local_files_only=True)
    st=AutoTokenizer.from_pretrained(cfg['models']['llama'],local_files_only=True)
    receiver=kb.load_model(cfg,'qwen'); sender=kb.load_model(cfg,'llama')
    base=kb.load_base(ck_a); adapter=kb.load_adapter(ck_b)
    objects=tasks(['openbookqa','arc_challenge'])
    kb.save_json(output/'config.json',{'seed':1234,'train':False,'count':2 if a.smoke else 64,
        'selection':'first32 official test rows per dataset; smoke first1 each',
        'weights':{str(ck_a):digest(ck_a),str(ck_b):digest(ck_b)},
        'memory':'Full-Sync all option tokens except receiver-native token0; no Top32 budget',
        'reader':'Options memory -> native Question -> Answer; no chat template; 0shot',
        'scoring':'candidate answer-text summed likelihood and character-length normalized likelihood',
        'primary_comparison':'query_aware_memory vs evidence_only_memory; identical reader/mapping/gold',
        'old_protocol':'original official task context + legacy full-context frozen Writer',
        'scope':'changed prompt mechanism diagnostic, not an official benchmark score'})
    records=[]
    for name,task in objects.items():
        split=task.get_config('test_split') or task.get_config('validation_split')
        for index,doc in enumerate(list(task.dataset[split])[:1 if a.smoke else 32]):
            q=doc.get('question_stem',doc.get('question'))
            choices=task.doc_to_choice(doc); gold=task.doc_to_target(doc)
            if not isinstance(gold,int): gold=doc['choices']['label'].index(str(doc['answerKey']))
            body='Options:\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(choices))
            readout='\n\nQuestion: '+q+'\nAnswer:'
            # Split exactly at an actual target token boundary, leaving crossings native.
            enc=rt(body+readout,add_special_tokens=False,return_offsets_mapping=True)
            n=sum(b<=len(body) for _,b in enc['offset_mapping'])
            bids=rt.encode(body,add_special_tokens=False)
            assert enc['input_ids'][:n]==bids[:n] and n>1
            nk,nv=kb.capture(receiver,bids[:n]); ctx=enc['input_ids']
            full=scores(receiver,rt,ctx,choices)
            oracle=scores(receiver,rt,ctx,choices,(nk.cuda(),nv.cuda()))
            assert max(range(len(full)),key=full.__getitem__)==max(range(len(oracle)),key=oracle.__getitem__), 'Native cache prediction mismatch'
            row={'task':name,'index':index,'id':stable_id(name,doc),'question':q,
                 'choices':choices,'gold':gold,'receiver_prompt':body+readout,'memory_tokens':n,
                 'conditions':{'qwen_full_optionsfirst':prediction(full,choices),
                               'native_oracle_memory':prediction(oracle,choices)}}
            for mode,source in [('evidence_only_memory',body),('query_aware_memory','Question: '+q+'\n\n'+body)]:
                sid,selected=body_map(st,rt,source,body)
                selected=selected[:n-1]
                assert len(selected)==n-1
                sk,sv=kb.capture(sender,sid)
                item={'source_k':sk[:,selected],'source_v':sv[:,selected],'tokens':len(selected)}
                bk,bv=kb.map_base(base,item)
                with torch.amp.autocast('cuda',dtype=torch.float16): k,v,_,_=adapter(bk[None],bv[None])
                fk=torch.cat((nk[:,:1].cuda(),k[0]),1); fv=torch.cat((nv[:,:1].cuda(),v[0]),1)
                row['conditions'][mode]=prediction(scores(receiver,rt,ctx,choices,(fk,fv)),choices)
                row[mode+'_sender_prompt']=source
                del sk,sv,bk,bv,k,v,fk,fv
            original=task.fewshot_context(doc,num_fewshot=0).rstrip()
            item=kb.pair(sender,receiver,st,rt,original)
            k,v=kb.map_base(base,item); k,v=kb.assemble(item,k,v,adapter)
            # Preserve official continuation handling, including shifted whitespace.
            vals=[]
            for request in task.construct_requests(doc,original,metadata=(name,index,1)):
                c,continuation=request.args
                shift=len(c)-len(c.rstrip()); continuation=c[len(c.rstrip()):]+continuation; c=c.rstrip()
                cids=rt.encode(c,add_special_tokens=False)
                allids=rt.encode(c+continuation,add_special_tokens=False)
                value,_=kb.continuation_score(receiver,cids,allids[len(cids):],k,v)
                vals.append(float(value))
            assert len(vals)==len(choices)
            row['conditions']['old_original_protocol']=prediction(vals,choices)
            records.append(row); kb.save_json(output/'per_sample.json',records)
            print(f"DONE {len(records)}/{'2' if a.smoke else '64'} {name}:{index}",flush=True)
            del item,k,v,nk,nv
    summary={}
    for name in ['all','openbookqa','arc_challenge']:
        rr=records if name=='all' else [r for r in records if r['task']==name]
        metrics={c:{'correct':sum(r['conditions'][c]['pred']==r['gold'] for r in rr),
                    'correct_norm':sum(r['conditions'][c]['pred_norm']==r['gold'] for r in rr),'n':len(rr)}
                 for c in records[0]['conditions']}
        u='query_aware_memory'; v='evidence_only_memory'
        paired={'both_correct':0,'query_aware_only':0,'evidence_only_only':0,'both_wrong':0}
        for r in rr:
            x=r['conditions'][u]['pred_norm']==r['gold']; y=r['conditions'][v]['pred_norm']==r['gold']
            paired['both_correct' if x and y else 'query_aware_only' if x else 'evidence_only_only' if y else 'both_wrong']+=1
        summary[name]={'metrics':metrics,'paired_norm':paired}
    kb.save_json(output/'summary.json',summary)
    print(json.dumps(summary),flush=True); print('COMPLETED',flush=True)


if __name__=='__main__': main()
