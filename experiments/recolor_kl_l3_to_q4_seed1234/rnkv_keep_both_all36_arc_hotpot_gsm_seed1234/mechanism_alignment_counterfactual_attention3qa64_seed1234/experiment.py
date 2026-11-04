"""Three frozen mechanism diagnostics and controlled fact-pair addendum."""
from common_diag import *
from interventions import *
from attention_analysis import local

def memory_file(row):return ROOT/f'memory/{row["id"]}.pt'
def saved_memory(row,device):
    v=torch.load(memory_file(row),map_location='cpu',weights_only=True)
    assert v['manifest_sha256']==identity() and v['row_sha256']==C.digest(row)
    return [(a.to(device),b.to(device)) for a,b in v['memory']],v['L']

@torch.no_grad()
def precompute():
    verify();engine,correction,module=load();snapshots=[C.versions(correction),C.versions(module)]
    data=rows()+C.read_jsonl(ROOT/'prepared/counterfactual.jsonl')
    for i,row in enumerate(data,1):
        paused();path=memory_file(row)
        if path.exists():saved_memory(row,'cpu');continue
        base=engine.base(row);translated=engine.cache(row,base,correction);cache=module.construct(row,translated);L=translated.get_seq_length()
        memories=[tuple(x.cpu() for x in p) for p in memory_pairs(cache,L)]
        C.atomic_save(dict(manifest_sha256=identity(),id=row['id'],row_sha256=C.digest(row),L=L,memory=memories),path)
        del base,translated,cache;unchanged(engine,correction,module,snapshots)
        status(status='running',phase='memory_precompute',sample=i,total=len(data))
        if i==1 or i%16==0:C.log('MECHANISM MEMORY',i,len(data))
    mark('memory/completed.json',status='completed',n=len(data))

def baseline_parity(actual,old):
    assert actual['prediction']==old['prediction']
    if 'generation' in old:assert actual['generation']==old['generation']
    for k in ['em','f1']:assert actual[k]==old[k]

@torch.no_grad()
def analyze(engine,correction,module,row,check=False):
    paused();base=engine.base(row);translated=engine.cache(row,base,correction);L=translated.get_seq_length();tokens=gold_tokens(engine,row)
    memory,storedL=saved_memory(row,'cuda');assert storedL==L
    keep=assemble(engine,translated,memory);native=engine.native_cache(row,baseline=True);native_shift=padded(engine,native)
    with access(module,header_only=True):null=module.construct(row,translated)
    donor_id=C.read(ROOT/'prepared/donors.json')[row['id']];donor=next(r for r in rows() if r['id']==donor_id)
    dm,donorL=saved_memory(donor,'cuda');moved=donor_memory(engine,dm,donorL,L);wrong=assemble(engine,translated,moved)
    for candidate in [keep,null,wrong]:
        assert candidate.get_seq_length()==L+32
        for l in range(36):
            for a,b in zip(C._cache_layer(candidate,l),C._cache_layer(translated,l)):assert torch.equal(a[:,:,:L],b)
    conditions={'native':(native,False,'keep'),'C':(translated,False,'keep'),'translated_shifted_masked':(keep,True,'masked'),
        'keep':(keep,True,'keep'),'native_shifted_masked':(native_shift,True,'masked'),'wrong_memory':(wrong,True,'keep'),'header_only_memory':(null,True,'keep')}
    pred={name:logits(module,row,c,tokens,rn,mode) for name,(c,rn,mode) in conditions.items()}
    functional_values={name:functional(v,pred['native'],pred['native_shifted_masked'],tokens) for name,v in pred.items()}
    assert functional_values['native']['kl_to_native']<1e-7 and functional_values['native']['top1_match_native']==1
    old=prior(row);free={'native':old['native'],'C':old['C'],'keep':old[N.ARM]}
    if check:
        for name in ['native','C','keep']:
            c,rn,mode=conditions[name];baseline_parity(score(module,row,c,rn,mode),free[name])
        # Masked Memory values cannot affect an answer at the same positions.
        changed=K.clone(keep,engine.model,detach=True)
        for l in range(36):
            for tensor in C._cache_layer(changed,l):tensor[:,:,L:L+32]=torch.randn_like(tensor[:,:,L:L+32])
        assert torch.equal(logits(module,row,changed,tokens,True,'masked'),pred['translated_shifted_masked'])
        del changed
    for name in ['translated_shifted_masked','wrong_memory','header_only_memory']:
        c,rn,mode=conditions[name];free[name]=score(module,row,c,rn,mode)
    group_effects={}
    for group,layers in CFG['layer_groups'].items():
        paused();value=logits(module,row,keep,tokens,True,blocked=layers)
        v=functional(value,pred['native'],pred['native_shifted_masked'],tokens)
        group_effects[group]=dict(layers=layers,blocked_metrics=v,
            gold_nll_gain_from_group_access=v['answer_nll']-functional_values['keep']['answer_nll'],
            native_kl_gain_from_group_access=v['kl_to_native']-functional_values['keep']['kl_to_native'])
    nt=trace(engine,module,row,native_shift,tokens,'masked');kt=trace(engine,module,row,keep,tokens)
    assert torch.allclose(nt.logits,pred['native_shifted_masked'],atol=.02,rtol=.002)
    assert torch.allclose(kt.logits,pred['keep'],atol=.02,rtol=.002)
    attention=local(engine,row,nt,kt,translated,native,memory)
    return dict(id=row['id'],task=row['task'],question=row['question'],gold=row['gold'],gold_tokens=tokens,stratum=C.read(ROOT/'prepared/strata.json')[row['id']],
        donor=dict(id=donor_id,source_L=donorL,recipient_L=L,post_rope_K_rephased=True,V_unchanged=all(torch.equal(a[1],b[1]) for a,b in zip(dm,moved))),
        experiment1_functional=functional_values,experiment2_generation=free,experiment3_fixedq=attention,experiment3_group_effects=group_effects)

@torch.no_grad()
def smoke():
    verify();engine,correction,module=load();snapshots=[C.versions(correction),C.versions(module)]
    data=rows();chosen={r['id']:r for r in [next(r for r in data if r['task']==t) for t in CFG['test_counts']]}
    longest=max(data,key=lambda r:len(r['content_ids'])*len(engine.tok.encode(r['gold'],add_special_tokens=False)));chosen[longest['id']]=longest
    for row in chosen.values():
        result=analyze(engine,correction,module,row,check=True);unchanged(engine,correction,module,snapshots)
        mark(f'smoke/per_sample/{row["id"]}.json',**result);C.log('MECHANISM SMOKE PASSED',row['id'])
    mark('smoke/completed.json',status='completed',ids=list(chosen),frozen=True,all36_layers=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)

@torch.no_grad()
def run():
    verify();assert C.read(ROOT/'smoke/completed.json')['status']=='completed'
    engine,correction,module=load();snapshots=[C.versions(correction),C.versions(module)]
    for i,row in enumerate(rows(),1):
        paused();path=ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
        if path.exists():assert C.read(path)['manifest_sha256']==identity();continue
        status(status='running',phase='mechanism_experiments_1_2_3',task=row['task'],sample=i,total=192)
        reuse=ROOT/f'smoke/per_sample/{row["id"]}.json'
        if reuse.exists():result=C.read(reuse);assert result.pop('manifest_sha256')==identity()
        else:result=analyze(engine,correction,module,row)
        mark(str(path.relative_to(ROOT)),**result);unchanged(engine,correction,module,snapshots)
        if i%8==0:C.log('MECHANISM EVAL',i,192)
    mark('evaluation/completed.json',status='completed',n=192)

@torch.no_grad()
def counterfactual():
    verify();engine,correction,module=load();snapshots=[C.versions(correction),C.versions(module)]
    data={r['id']:r for r in C.read_jsonl(ROOT/'prepared/counterfactual.jsonl')}
    for i,pair in enumerate(C.read(ROOT/'prepared/counterfactual_pairs.json'),1):
        paused();path=ROOT/f'counterfactual/per_pair/{pair["id"]}.json'
        if path.exists():assert C.read(path)['manifest_sha256']==identity();continue
        memories={};caches={};native={};base_scores={};native_scores={}
        for tag in ['a','b']:
            row=data[pair[tag]];mem,L=saved_memory(row,'cuda');memories[tag]=mem
            caches[tag]=engine.cache(row,engine.base(row),correction);assert L==caches[tag].get_seq_length()
            native[tag]=engine.native_cache(row,baseline=True)
            base_scores[tag]=score(module,row,caches[tag],False);native_scores[tag]=score(module,row,native[tag],False)
        assert caches['a'].get_seq_length()==caches['b'].get_seq_length()
        values={}
        for body in ['a','b']:
            for source in ['a','b']:
                row=data[pair[body]];cache=assemble(engine,caches[body],memories[source]);result=score(module,row,cache,True)
                memrow=dict(row,gold=data[pair[source]]['gold'])
                from metrics import answer_metrics
                result.update(matches_body_answer=int(result['em']),matches_memory_answer=int(answer_metrics(result['prediction'],memrow['gold'])['em']))
                values['body_'+body+'_memory_'+source]=result;del cache
        mark(str(path.relative_to(ROOT)),id=pair['id'],question=data[pair['a']]['question'],answers={t:data[pair[t]]['gold'] for t in ['a','b']},
            native_both_correct=bool(native_scores['a']['em']==native_scores['b']['em']==1),native=native_scores,translated=base_scores,conditions=values,
            receiver_body_length=caches['a'].get_seq_length(),same_sender_token_length=True)
        del memories,caches,native;unchanged(engine,correction,module,snapshots)
        status(status='running',phase='synthetic_counterfactual',sample=i,total=16);C.log('COUNTERFACTUAL PAIR',i,16)
    mark('counterfactual/completed.json',status='completed',n=16)

if __name__=='__main__':globals()[sys.argv[1]]()
