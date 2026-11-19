"""Real Gemma FP32 capture, postnorm cache convention, and contiguous FP16 backward checks."""
from gemma_common import *
from runtime_gemma import load,GemmaMemory,backward_batch,batches
from evaluate_gemma import score

def main():
    verify();assert done('cpu_checks.json')
    if done('smoke/completed.json'):return
    engine,translator,module=load();tv=C.versions(translator);initial=module.memory.prompt.detach().clone()
    oldrow=min(C.read_jsonl(GP/'prepared/test.jsonl'),key=lambda r:len(r['frames']['gemma']['body_ids']))
    source_path=GP/'source'/(oldrow['id'].replace(':','_')+'.pt')
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    assert source['manifest_sha256']==CFG['gemma_manifest_sha256'] and source['row_sha256']==C.digest(oldrow)
    with torch.no_grad():
        ids=oldrow['frames']['gemma']['body_ids'];key,value,_=L.capture(engine.sender,ids,len(ids),[0]);source_errors={}
        for name,tensor in [('key',key),('value',value)]:
            actual=tensor[:,oldrow['source_indices']].half();old=source[name]
            error=float((actual.float()-old.float()).norm()/old.float().norm().clamp_min(1e-8))
            assert error<.005,('Historical Gemma capture mismatch',name,error);source_errors[name]=error
    del source,key,value,actual,old
    reference=M.RNKV(engine);assert torch.equal(reference.memory.prompt,initial);reference.routing.close();del reference
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();longchecks=[]
    for task in CFG['train_counts']:
        row=max([r for r in rows('train') if r['task']==task],key=lambda r:len(r['gemma_body_ids'])+len(r['gold_tokens']))
        detail,outcome,retries=backward_batch(engine,translator,module,opt,scaler,[(row,'gold')],1,0,scope='smoke_long_'+task)
        longchecks.append(dict(task=task,id=row['id'],gemma_tokens=len(row['gemma_body_ids']),gold_tokens=len(row['gold_tokens']),amp_retries=retries,**outcome))
        engine.unchanged();assert C.versions(translator)==tv;C.log('GEMMA LONG BACKWARD PASSED',ARM,task)
    module.routing.close();del module,opt,scaler
    module=GemmaMemory(engine);assert torch.equal(module.memory.prompt,initial)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();checks=[]
    for bi,batch in enumerate(batches(1)[:4],1):
        _,outcome,retries=backward_batch(engine,translator,module,opt,scaler,batch,1,bi,scope='smoke')
        checks.append(dict(batch=bi,amp_retries=retries,**outcome));engine.unchanged();assert C.versions(translator)==tv
        C.log('GEMMA SMOKE BATCH PASSED',ARM,bi)
    assert not torch.equal(initial,module.memory.prompt)
    with torch.no_grad():
        row=rows('train')[0];base=engine.base(row);cache=engine.cache(row,base,translator);memory=module.construct(row,cache)
        L0=cache.get_seq_length();q=torch.tensor([L0+32,L0+33],device='cuda')
        for l in range(1,37):
            allowed=module.routing.allowed('answer',l,len(engine.prefix_ids),L0,32,q,L0+34)
            assert allowed[:,:L0].all() and not allowed[0,L0+33] and allowed[1,L0+33]
            if l in CFG['arms'][ARM]:assert allowed[:,L0:L0+32].all()
            else:assert not allowed[:,L0:L0+32].any()
            for a,b in zip(C._cache_layer(cache,l-1),C._cache_layer(memory,l-1)):assert torch.equal(a,b[:,:,:L0])
        del base,cache,memory
    # Verify canonical-cache Native scoring against the pinned same-Qwen outputs
    # on the first untouched test example of every task.
    engine.mode='test';parity=[]
    with torch.no_grad():
        for task in CFG['test_counts']:
            row=next(r for r in rows('test') if r['task']==task and (REFERENCE/f'evaluation/{task}/per_sample/{r["id"]}.json').exists())
            cache=engine.native_cache(row,baseline=True);actual=score(module,row,cache,rn=False);old=C.read(REFERENCE/f'evaluation/{task}/per_sample/{row["id"]}.json')['conditions']['native']
            assert actual['prediction']==old['prediction'],('Native prediction parity',task)
            if 'generation' in old:assert actual['generation']==old['generation'],('Native generation parity',task)
            parity.append(dict(task=task,id=row['id']));del cache
    engine.unchanged();assert engine.student_test_native_prefills==0
    mark('smoke/completed.json',status='completed',same_initial_memory=True,long_backward=longchecks,consecutive_batches=checks,
        all36_masks_checked=True,source_dtype='float32',translator_key_space='post_k_norm/pre_RoPE; RoPE once',native_parity=parity,
        historical_source_capture_relative_errors=source_errors,
        disposable_updates=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__':main()
