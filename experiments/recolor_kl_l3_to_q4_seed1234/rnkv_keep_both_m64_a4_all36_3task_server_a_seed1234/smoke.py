"""Actual shared-GPU feasibility test; full samples, same fixed64 Memory architecture."""
from experiment_common import *
from runtime_new import load,Memory64,backward_batch,batches,offload_context

def main():
    verify();assert done('base_cache/probe_completed.json') and done('cpu_checks.json')
    if done('smoke/completed.json'):return
    engine,correction,module=load();cv=C.versions(correction);initial=module.memory.prompt.detach().clone()
    master=M.Memory(2560,32,16,[6,12,18,24],1234)
    assert torch.equal(initial[:32].cpu(),master.prompt) and all(torch.equal(a.cpu(),b) for a,b in zip(module.memory.adapters.state_dict().values(),master.adapters.state_dict().values()));del master
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();checks=[]
    for split in ['train','validation','test']:
        for task in CFG[split+'_counts']:
            rr=[r for r in rows(split) if r['task']==task]
            row=max(rr,key=lambda r:len(r['content_ids'])+len(r.get('gold_tokens',[])))
            if split=='train':
                details,outcome,retries=backward_batch(engine,correction,module,opt,scaler,[(row,'gold')],1,0,scope='smoke_long_'+task)
                checks.append(dict(split=split,task=task,id=row['id'],body_tokens=len(row['content_ids']),gold_tokens=len(row['gold_tokens']),gradient=outcome,amp_retries=retries))
            else:
                # Full longest held-out body and answer-time suffix, no held-out training.
                with torch.no_grad():
                    translated=engine.cache(row,engine.base(row),correction);memory=module.construct(row,translated)
                    logits=module.logits(row,memory,(),[0],rn=True);assert torch.isfinite(logits).all();del translated,memory,logits
                checks.append(dict(split=split,task=task,id=row['id'],body_tokens=len(row['content_ids']),forward_only=True))
            engine.unchanged();assert C.versions(correction)==cv;C.log('M64 LONG PROBE PASSED',split,task,row['id'])
    module.routing.close();del module,opt,scaler;torch.cuda.empty_cache()
    module=Memory64(engine);assert torch.equal(module.memory.prompt,initial)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();batchchecks=[]
    for bi,batch in enumerate(batches(1)[:4],1):
        details,outcome,retries=backward_batch(engine,correction,module,opt,scaler,batch,1,bi,scope='smoke_consecutive')
        batchchecks.append(dict(batch=bi,amp_retries=retries,**outcome));engine.unchanged();assert C.versions(correction)==cv
        C.log('M64 CONSECUTIVE PROBE PASSED',bi)
    assert not torch.equal(module.memory.prompt,initial)
    row=next(r for r in rows('train') if r['task']=='gsm8k')
    with torch.no_grad():
        translated=engine.cache(row,engine.base(row),correction);memory=module.construct(row,translated);L=translated.get_seq_length()
        q=torch.tensor([L+64,L+65],device='cuda')
        for l in range(1,37):
            allowed=module.routing.allowed('answer',l,len(engine.prefix_ids),L,64,q,L+66)
            assert allowed[:,:L+64].all() and allowed[0,L+64] and not allowed[0,L+65] and allowed[1,L+65]
            for a,b in zip(C._cache_layer(memory,l-1),C._cache_layer(translated,l-1)):assert torch.equal(a[:,:,:L],b)
    assert all(not p.requires_grad and p.grad is None for p in correction.parameters())
    module.routing.close();del module,opt,scaler,memory,translated;torch.cuda.empty_cache()
    from evaluate_new import parity
    parity(engine)
    mark('smoke/completed.json',status='completed',memory_tokens=64,trainable_parameters=491520,shared_initialization_matches=True,checks=checks,consecutive_batches=batchchecks,
        all36_masks_checked=True,body_prefix_unchanged=True,disposable_updates=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**engine.budget)
if __name__=='__main__':main()
