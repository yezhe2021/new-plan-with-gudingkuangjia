"""Disposable full-length real-GPU backwards before matched formal initialization."""
from experiment_common import *
from runtime_new import load,GemmaMemory,backward_batch,batches

def main():
    verify();assert done('cpu_checks.json')
    if done('smoke/completed.json'):return
    engine,correction,module=load();cv=C.versions(correction);initial=module.memory.prompt.detach().clone()
    ref=M.RNKV(engine);assert torch.equal(ref.memory.prompt,initial);ref.routing.close();del ref
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();longchecks=[]
    for task in CFG['train_counts']:
        row=max([r for r in rows('train') if r['task']==task],key=lambda r:len(r['content_ids'])+len(r['gold_tokens']))
        detail,outcome,retries=backward_batch(engine,correction,module,opt,scaler,[(row,'gold')],1,0,scope='smoke_long_'+task)
        longchecks.append(dict(task=task,id=row['id'],body_tokens=len(row['content_ids']),gold_tokens=len(row['gold_tokens']),amp_retries=retries,**outcome))
        engine.unchanged();assert C.versions(correction)==cv;C.log('ALL36 LONG BACKWARD PASSED',task,row['id'])
    module.routing.close();del module,opt,scaler
    module=GemmaMemory(engine);assert torch.equal(module.memory.prompt,initial)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();batchchecks=[]
    for bi,batch in enumerate(batches(1)[:4],1):
        detail,outcome,retries=backward_batch(engine,correction,module,opt,scaler,batch,1,bi,scope='smoke')
        batchchecks.append(dict(batch=bi,amp_retries=retries,**outcome));engine.unchanged();assert C.versions(correction)==cv
        C.log('ALL36 THREE-TASK SMOKE BATCH PASSED',bi,outcome['gradient_norm'])
    assert not torch.equal(module.memory.prompt,initial)
    row=next(r for r in rows('train') if r['task']=='gsm8k')
    with torch.no_grad():
        translated=engine.cache(row,engine.base(row),correction);memory=module.construct(row,translated);L=translated.get_seq_length();q=torch.tensor([L+32,L+33],device='cuda')
        for l in range(1,37):
            allowed=module.routing.allowed('answer',l,len(engine.prefix_ids),L,32,q,L+34)
            assert allowed[:,:L].all() and bool(allowed[:,L:L+32].all())==(l in CFG['replace_layers'])
            assert allowed[0,L+32] and not allowed[0,L+33] and allowed[1,L+33]
            for a,b in zip(C._cache_layer(memory,l-1),C._cache_layer(translated,l-1)):assert torch.equal(a[:,:,:L],b)
    assert all(not p.requires_grad and p.grad is None for p in correction.parameters())
    mark('smoke/completed.json',status='completed',trainable_parameters=409600,same_original_initial_prompt=True,long_backward=longchecks,
        consecutive_batches=batchchecks,all36_masks_checked=True,memory_visible_layers=CFG['replace_layers'],body_prefix_unchanged=True,disposable_updates=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__':main()
