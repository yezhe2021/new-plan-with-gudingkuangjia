"""Actual Qwen checks for input independence, positions, gradients and matched endpoints."""
from types import SimpleNamespace
from experiment_common import *
from runtime_new import StaticMemory,load,losses,batches,backward_batch

def independence(module,body_a,body_b):
    H=len(module.engine.prefix_ids);L=body_a.get_seq_length();other=body_b.get_seq_length()
    module.eval()
    with torch.no_grad():
        canon=module.canonical();assert all(not x.requires_grad for x in canon)
        assert module.canonical() is canon
        a=module.construct({'deliberately_unused':'A'},body_a)
        b=module.construct({'deliberately_unused':'B'},body_b)
        moved=module.relocate(canon,L)
        error=0.
        for layer in range(module.engine.model.config.num_hidden_layers):
            ak,av=C._cache_layer(a,layer);bk,bv=C._cache_layer(b,layer)
            assert torch.equal(av[:,:,L:L+module.M],bv[:,:,other:other+module.M])
            assert torch.equal(av[:,:,L:L+module.M],canon[2*layer+1])
            for result,original,length in [(a,body_a,L),(b,body_b,other)]:
                for x,y in zip(C._cache_layer(result,layer),C._cache_layer(original,layer)):assert torch.equal(x[:,:,:length],y)
            assert torch.equal(ak[:,:,L:L+module.M],moved[2*layer])
            if L==other:assert torch.equal(ak[:,:,L:L+module.M],bk[:,:,other:other+module.M])
        # At the canonical endpoint attachment is exact; other positions obey
        # the explicitly tested inverse+forward rotation, without renormalizing.
        from heterofold.attention import rotary_cos_sin
        from static_memory import rephase
        for layer in range(module.engine.model.config.num_hidden_layers):
            key=C._cache_layer(a,layer)[0][:,:,L:L+module.M]
            dev=key.device;model=module.engine.model
            sc,ss=rotary_cos_sin(model,torch.arange(H,H+module.M,device=dev),layer,model.dtype)
            tc,ts=rotary_cos_sin(model,torch.arange(L,L+module.M,device=dev),layer,model.dtype)
            back=rephase(key,tc,ts,sc,ss)
            relative=float((back.float()-canon[2*layer].float()).norm()/canon[2*layer].float().norm().clamp_min(1e-12))
            assert relative<.003,(layer,relative);error=max(error,relative)
        return dict(values_identical_across_inputs=True,body_prefix_unchanged=True,canonical_input_header_tokens=H,recipient_lengths=[L,other],rope_roundtrip_max_relative_error=error)

def cpu():
    verify();torch.set_num_threads(1);torch.manual_seed(1234)
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    cfg._attn_implementation='sdpa';llm=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    with torch.no_grad():
        header=llm.model(torch.tensor([[2,3]]),use_cache=True).past_key_values
        body=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
        changed=llm.model(torch.tensor([[2,3,8,7,6,5,4]]),use_cache=True).past_key_values
        longer=llm.model(torch.tensor([[2,3,4,5,6,7,8,4,5]]),use_cache=True).past_key_values
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3],header=header)
    engine.forward=lambda ids,cache,length,use_cache=True:llm.model(torch.tensor([ids]),past_key_values=cache,position_ids=torch.arange(length,length+len(ids))[None],attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),use_cache=use_cache)
    module=StaticMemory(engine,dict(seed=1234,memory_tokens=4,adapter_rank=2,adapter_after_layers=[6,12,18,24]))
    same=independence(module,body,changed);different=independence(module,body,longer)
    row=dict(id='static_cpu_probe',task='gsm8k',content_ids=[4,5,6,7,8],suffix_ids=[9,10],gold_tokens=[11,12,13,14]*20+[1])
    gradients=[];outputs=[];module.train()
    for enabled in [False,True]:
        module.activation_checkpointing=enabled;module.zero_grad(set_to_none=True)
        loss,detail=losses(engine,module,row,body,body,'gold',1);loss.backward()
        assert detail['gold_tokens']==81 and detail['positions']==8
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert module.memory.prompt.grad.norm()>0
        gradients.append(torch.cat([p.grad.flatten() for p in module.parameters()]).clone());outputs.append(loss.detach())
    error=float((gradients[0]-gradients[1]).norm()/gradients[0].norm());assert error<1e-5 and torch.allclose(*outputs)
    assert all(p.grad is None for p in llm.parameters())
    module.eval()
    with torch.no_grad():
        old=module.canonical();module.memory.prompt.add_(.001);new=module.canonical();assert old is not new
    for layer in range(1,37):
        allowed=module.routing.allowed('answer',layer,2,7,4,torch.tensor([11,12]),13)
        assert allowed[:,:11].all() and not allowed[0,12] and allowed[1,12]
    assert not torch.cuda.is_initialized();module.routing.close()
    mark('cpu_checks.json',status='completed',same_length=same,different_length=different,checkpoint_gradient_relative_error=error,full_gold_tokens=81,all36_answer_masks=True,inference_cache_invalidated_on_update=True)
    C.log('STATIC CPU CHECKS PASSED',error)

@torch.no_grad()
def parity():
    verify()
    if done('parity/completed.json'):return
    from evaluate_new import score
    dynamic=reference_module('static_pinned_dynamic_runtime','runtime_new.py')
    engine=R.Engine();correction=T.load_final('C').requires_grad_(False)
    module=dynamic.AllLayerMemory(engine)
    cp=torch.load(PARENT/'runs/epoch4_step1536.pt',map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==N.identity() and cp['step']==1536
    module.load_state_dict(cp['state'],strict=True);module.eval().requires_grad_(False)
    checks=[]
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task)
        translated=engine.cache(row,engine.base(row),correction)
        result=score(module,row,module.construct(row,translated));ref=baseline(row)[DYNAMIC]
        for key in ['prediction','correct','legacy_prediction','legacy_correct','em','f1','generation','generated_tokens','reached_cap','token_ids','scores']:
            if key not in ref:continue
            if key=='scores':assert max(abs(a-b) for a,b in zip(result[key],ref[key]))<1e-4,(task,key)
            else:assert result[key]==ref[key],(task,key,result[key],ref[key])
        checks.append(dict(task=task,id=row['id'],exact_predictions=True));C.log('STATIC DYNAMIC ENDPOINT PARITY',task)
    engine.unchanged();module.routing.close()
    mark('parity/completed.json',status='completed',n=8,checks=checks,reference_checkpoint_sha256=C.sha(PARENT/'runs/epoch4_step1536.pt'))

def smoke():
    verify();assert done('cpu_checks.json') and done('parity/completed.json')
    if done('smoke/completed.json'):return
    engine,correction,module=load();cv=C.versions(correction)
    fresh=M.RNKV(engine)
    assert all(torch.equal(a,b) for a,b in zip(module.parameters(),fresh.parameters()))
    fresh.routing.close();del fresh
    selected=[max([r for r in rows('train') if r['task']==task],key=lambda r:len(r['content_ids'])+len(r['gold_tokens'])) for task in CFG['train_counts']]
    with torch.no_grad():
        aa=engine.cache(selected[0],engine.base(selected[0]),correction)
        bb=engine.cache(selected[1],engine.base(selected[1]),correction)
        check=independence(module,aa,bb);del aa,bb
    module.train();initial=module.memory.prompt.detach().clone()
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();long=[]
    for row in selected:
        _,outcome,retries=backward_batch(engine,correction,module,opt,scaler,[(row,'gold')],1,0,scope='smoke_long_'+row['task'])
        long.append(dict(id=row['id'],task=row['task'],amp_retries=retries,**outcome));C.log('STATIC LONG BACKWARD PASSED',row['task'])
    module.routing.close();del module,opt,scaler
    module=StaticMemory(engine,CFG);assert torch.equal(module.memory.prompt,initial)
    module.train();opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();bb=[]
    for bi,batch in enumerate(batches(1)[:4],1):
        _,outcome,retries=backward_batch(engine,correction,module,opt,scaler,batch,1,bi,scope='smoke')
        bb.append(dict(batch=bi,amp_retries=retries,**outcome));C.log('STATIC CONSECUTIVE BATCH PASSED',bi)
    assert not torch.equal(module.memory.prompt,initial)
    engine.unchanged();assert C.versions(correction)==cv and all(p.grad is None for p in correction.parameters())
    mark('smoke/completed.json',status='completed',fresh_initialization_identical=True,trainable_parameters=409600,input_independence=check,long_backward=long,consecutive_batches=bb,disposable_updates=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__':{'cpu':cpu,'parity':parity,'smoke':smoke}[sys.argv[1]]()
