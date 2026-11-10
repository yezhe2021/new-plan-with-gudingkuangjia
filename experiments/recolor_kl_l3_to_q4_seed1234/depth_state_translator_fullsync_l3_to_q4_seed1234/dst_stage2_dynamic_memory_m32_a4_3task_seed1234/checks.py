"""Prove actual sample conditioning, matched interfaces and connected gradients."""
from types import SimpleNamespace
from experiment_common import *
from runtime_new import DynamicMemory,load,losses,batches,backward_batch

@torch.no_grad()
def conditioning(module,body_a,body_b):
    L=body_a.get_seq_length();assert body_b.get_seq_length()==L
    module.eval();a=module.construct(None,body_a);b=module.construct(None,body_b)
    different=[]
    for l in range(module.engine.model.config.num_hidden_layers):
        for result,body in [(a,body_a),(b,body_b)]:
            for x,y in zip(C._cache_layer(result,l),C._cache_layer(body,l)):assert torch.equal(x[:,:,:L],y)
        aa=tuple(x[:,:,L:L+module.M] for x in C._cache_layer(a,l));bb=tuple(x[:,:,L:L+module.M] for x in C._cache_layer(b,l))
        changed=any(not torch.equal(x,y) for x,y in zip(aa,bb))
        if l==0:assert not changed,'First-layer memory projections precede body reading'
        if changed:different.append(l+1)
    assert different,'Memory did not respond to same-length body intervention'
    reference=SM.StaticMemory(module.engine,module.cfg);reference.load_state_dict(module.state_dict(),strict=True);reference.eval()
    sa=reference.construct(None,body_a);sb=reference.construct(None,body_b)
    for pa,pb in zip(K.pairs(sa),K.pairs(sb)):
        for x,y in zip(pa,pb):assert torch.equal(x[:,:,L:L+module.M],y[:,:,L:L+module.M])
    reference.routing.close()
    return dict(same_recipient_length=L,first_layer_memory_equal=True,body_prefix_unchanged=True,dynamic_changed_layers=different,static_control_memory_equal=True)

def cpu():
    verify();torch.set_num_threads(1);torch.manual_seed(1234)
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    cfg._attn_implementation='sdpa';llm=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    with torch.no_grad():
        header=llm.model(torch.tensor([[2,3]]),use_cache=True).past_key_values
        body=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
        changed=llm.model(torch.tensor([[2,3,8,7,6,5,4]]),use_cache=True).past_key_values
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3],header=header)
    engine.forward=lambda ids,cache,length,use_cache=True:llm.model(torch.tensor([ids]),past_key_values=cache,position_ids=torch.arange(length,length+len(ids))[None],attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),use_cache=use_cache)
    config=dict(seed=1234,memory_tokens=4,adapter_rank=2,adapter_after_layers=[6,12,18,24])
    module=DynamicMemory(engine,config);dependence=conditioning(module,body,changed)
    assert DynamicMemory.construct is M.RNKV.construct
    for layer in range(1,37):
        answer=module.routing.allowed('answer',layer,2,7,4,torch.tensor([11,12]),13)
        assert answer[:,:11].all() and not answer[0,12] and answer[1,12]
        build=module.routing.allowed('construct',layer,2,7,4,torch.arange(7,11),11)
        assert build[:,:7].all() and torch.equal(build[:,7:],torch.ones(4,4,dtype=torch.bool).tril())
    row=dict(id='dynamic_cpu_probe',task='gsm8k',content_ids=[4,5,6,7,8],suffix_ids=[9,10],gold_tokens=[11,12,13,14]*20+[1])
    module.train();grads=[];outputs=[]
    for enabled in [False,True]:
        module.activation_checkpointing=enabled;module.zero_grad(set_to_none=True)
        loss,detail=losses(engine,module,row,body,body,'gold',1);loss.backward()
        assert detail['gold_tokens']==81 and detail['positions']==8
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()) and module.memory.prompt.grad.norm()>0
        grads.append(torch.cat([p.grad.flatten() for p in module.parameters()]).clone());outputs.append(loss.detach())
    error=float((grads[0]-grads[1]).norm()/grads[0].norm());assert error<1e-5 and torch.allclose(*outputs)
    assert all(p.grad is None for p in llm.parameters()) and not torch.cuda.is_initialized()
    mark('cpu_checks.json',status='completed',conditioning=dependence,original_full_body_construct_used=True,all36_construct_and_answer_masks=True,full_gold_tokens=81,checkpoint_gradient_relative_error=error,frozen_receiver_gradients_none=True)
    module.routing.close();C.log('DST DYNAMIC CPU CHECKS PASSED',error)

@torch.no_grad()
def parity():
    verify()
    if done('parity/completed.json'):return
    from score import score
    from engine_frozen import Engine
    engine=Engine();reader=Z.DE.BodyReader(engine);static=SM.StaticMemory(engine,F.CFG)
    path=STATIC_ROOT/'runs/epoch4_step1536.pt';assert C.sha(path)==CFG['static_reference_checkpoint_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['manifest_sha256']==F.identity()
    static.load_state_dict(cp['state'],strict=True);static.eval().requires_grad_(False);checks=[]
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task);old=baseline(row)
        body=engine.cache(row,engine.base(row),engine.dst);native=engine.native_cache(row,baseline=True)
        values={BASE:score(reader,row,body,rn=False),'native':score(reader,row,native,rn=False),STATIC:score(static,row,static.construct(row,body),rn=True)}
        for name,result in values.items():same_prediction(result,old[name],task,name)
        checks.append(dict(task=task,id=row['id'],DST_native_trained_static_prediction_parity=True));del body,native
        engine.unchanged();C.log('DST DYNAMIC REFERENCE PARITY',task)
    static.routing.close();mark('parity/completed.json',status='completed',n=8,checks=checks)

def smoke():
    verify();assert done('cpu_checks.json') and done('parity/completed.json')
    if done('smoke/completed.json'):return
    engine,correction,module=load();cv=C.versions(correction)
    fresh=SM.StaticMemory(engine,F.CFG)
    assert module.state_dict().keys()==fresh.state_dict().keys() and all(torch.equal(v,fresh.state_dict()[n]) for n,v in module.state_dict().items())
    fresh.routing.close();del fresh;initial=module.memory.prompt.detach().clone()
    selected=[max([r for r in rows('train') if r['task']==task],key=lambda r:len(r['content_ids'])+len(r['gold_tokens'])) for task in CFG['train_counts']]
    with torch.no_grad():
        row=selected[0];body=engine.cache(row,engine.base(row),correction);changed=K.clone(body,engine.model,detach=True)
        H=len(engine.prefix_ids)
        for k,v in K.pairs(changed):v[:,:,H:].mul_(-1)
        dependence=conditioning(module,body,changed)
        original=M.RNKV(engine,CFG);original.load_state_dict(module.state_dict(),strict=True);original.eval()
        a=module.construct(row,body);b=original.construct(row,body)
        assert all(torch.equal(x,y) for pa,pb in zip(K.pairs(a),K.pairs(b)) for x,y in zip(pa,pb))
        original.routing.close();del original,a,b,body,changed
    module.train();opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();long=[]
    for row in selected:
        _,outcome,retries=backward_batch(engine,correction,module,opt,scaler,[(row,'gold')],1,0,scope='smoke_long_'+row['task'])
        long.append(dict(task=row['task'],id=row['id'],amp_retries=retries,**outcome));engine.unchanged();assert C.versions(correction)==cv
        C.log('DST DYNAMIC LONG BACKWARD PASSED',row['task'])
    module.routing.close();del module,opt,scaler
    module=DynamicMemory(engine,CFG);assert torch.equal(module.memory.prompt,initial);module.train()
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();bb=[]
    for bi,batch in enumerate(batches(1)[:4],1):
        _,outcome,retries=backward_batch(engine,correction,module,opt,scaler,batch,1,bi,scope='smoke')
        bb.append(dict(batch=bi,amp_retries=retries,**outcome));engine.unchanged();assert C.versions(correction)==cv
        C.log('DST DYNAMIC CONSECUTIVE BATCH PASSED',bi)
    assert not torch.equal(module.memory.prompt,initial)
    mark('smoke/completed.json',status='completed',fresh_initialization_matches_static=True,trainable_parameters=409600,conditioning=dependence,original_dynamic_construct_bitwise_parity=True,long_backward=long,consecutive_batches=bb,disposable_updates=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__':{'cpu':cpu,'parity':parity,'smoke':smoke}[sys.argv[1]]()
