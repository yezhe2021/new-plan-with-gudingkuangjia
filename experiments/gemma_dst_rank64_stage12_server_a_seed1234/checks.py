"""Meaningful causal, chunking, gradient and real Qwen cache-coordinate checks."""
from types import SimpleNamespace
from dataclasses import replace
from experiment_common import *
from translator import DepthStateTranslator,DSTConfig
from alignment import fullsync,spans,synchronized_units,build_synchronized_units,source_rank
from engine import Engine,BodyReader,raw_cache

def cpu():
    verify();torch.set_num_threads(1)
    for m in range(1,9):
        for n in range(1,9):
            mapped=[source_rank(m,n,i) for i in range(n)]
            assert mapped==sorted(mapped) and mapped[-1]==m-1
    st=C.AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True)
    rt=C.AutoTokenizer.from_pretrained(CFG['receiver'],local_files_only=True)
    for text in ['Hello, heterogeneous KV!','中文、café，🙂 a\n b 12345.','A = 17; B = 24.\nQuestion: A+B?']:
        a=fullsync(text,st,rt)
        se=st(text,add_special_tokens=False,return_offsets_mapping=True);re=rt(text,add_special_tokens=False,return_offsets_mapping=True)
        ss,rs=spans(text,se),spans(text,re)
        assert synchronized_units(text,ss,rs)==build_synchronized_units(text,ss,rs)
        assert len(a.source_indices)==len(a.receiver_ids) and min(a.source_indices)>=0
    cfg=DSTConfig(sender_layers=3,receiver_layers=6,source_width=8,target_width=8,state_width=4)
    torch.manual_seed(1234);k=tuple(torch.randn(23,8) for _ in range(3));v=tuple(torch.randn(23,8) for _ in range(3));module=DepthStateTranslator(cfg)
    a,b=module(k,v,range(23));pieces=[module(k,v,range(i,min(i+5,23))) for i in range(0,23,5)]
    assert torch.allclose(a,torch.cat([x[0] for x in pieces],1),atol=2e-6) and torch.allclose(b,torch.cat([x[1] for x in pieces],1),atol=2e-6)
    changed=[x.clone() for x in k];changed[0][7]+=3;c,d=module(changed,v,range(23));mask=[i for i in range(23) if i!=7]
    assert torch.equal(a[:,mask],c[:,mask]) and torch.equal(b[:,mask],d[:,mask]),'Token interaction introduced'
    (a[-1].sum()+b[-1].sum()).backward();carry_gradient=float(module.layers[0].input.weight.grad.norm());assert carry_gradient>1e-10
    independent=DepthStateTranslator(replace(cfg,carry_state=False));independent.load_state_dict(module.state_dict());c,d=independent(k,v,range(23));(c[-1].sum()+d[-1].sum()).backward()
    g=independent.layers[0].input.weight.grad;assert g is None or float(g.norm())==0
    # Actual tiny Qwen, same36-layer caching path and normalized+rotated keys.
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfgq=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    cfgq._attn_implementation='sdpa';qwen=Qwen3ForCausalLM(cfgq).eval().requires_grad_(False);prefix=[2,3];body=[4,5,6,7,8,9];H=2
    with torch.no_grad():header=qwen.model(torch.tensor([prefix]),use_cache=True).past_key_values
    keys={};values={};handles=[]
    for l,block in enumerate(qwen.model.layers):
        for name,store in [('k_proj',keys),('v_proj',values)]:
            def capture(m,args,out,store=store,l=l):store[l]=out[0,H:].detach()
            handles.append(getattr(block.self_attn,name).register_forward_hook(capture))
    with torch.no_grad():native=qwen.model(torch.tensor([prefix+body]),use_cache=True).past_key_values
    for handle in handles:handle.remove()
    for l in range(36):
        for x,y in zip(C._cache_layer(native,l),C._cache_layer(header,l)):x[:,:,:H].copy_(y)
    oracle=raw_cache(qwen,header,torch.stack(list(keys.values())),torch.stack(list(values.values())))
    error=max(float((x-y).abs().max()) for pa,pb in zip(K.pairs(native),K.pairs(oracle)) for x,y in zip(pa,pb));assert error<1e-5,error
    engine=SimpleNamespace(model=qwen,prefix_ids=prefix,header=header)
    engine.forward=lambda ids,cache,length,use_cache=True:qwen.model(torch.tensor([ids]),past_key_values=cache,attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),position_ids=torch.arange(length,length+len(ids))[None],use_cache=use_cache)
    reader=BodyReader(engine);small=DepthStateTranslator(DSTConfig(sender_layers=3,receiver_layers=36,source_width=16,target_width=16,state_width=8))
    source=(tuple(torch.randn(6,16) for _ in range(3)),tuple(torch.randn(6,16) for _ in range(3)))
    row=dict(id='tiny_dst',content_ids=body,suffix_ids=[10,11],sync_source_indices=list(range(6)),gold_tokens=[12,13,1])
    old=CFG['token_chunk'];CFG['token_chunk']=3;grads=[];outputs=[]
    try:
        for enabled in [False,True]:
            small.zero_grad(set_to_none=True);cache=Engine.translated(engine,row,small,source,checkpointed=enabled)
            logits=reader.token_logits(row,cache,row['gold_tokens'],rn=False)
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor(row['gold_tokens']));loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
            grads.append(torch.cat([p.grad.flatten() for p in small.parameters()]));outputs.append(loss.detach())
    finally:CFG['token_chunk']=old
    grad_error=float((grads[0]-grads[1]).norm()/grads[0].norm());assert grad_error<1e-5 and torch.allclose(*outputs)
    assert all(p.grad is None for p in qwen.parameters()) and not torch.cuda.is_initialized()
    rank_cpu()
    mark('checks/cpu_completed.json',status='completed',legacy_sync_unit_parity=True,receiver_first_token_mapped=True,token_chunk_output_parity=True,no_cross_token_interaction=True,early_to_late_state_gradient=carry_gradient,reset_state_blocks_dependency=True,native_cache_oracle_max_abs_error=error,frozen_qwen_checkpoint_gradient_relative_error=grad_error)
    C.log('DST CPU CHECKS PASSED',grad_error,error)

def rank_cpu():
    from residual import ResidualTranslator
    base=DepthStateTranslator(DSTConfig(sender_layers=3,receiver_layers=6,source_width=8,target_width=8,state_width=4))
    module=ResidualTranslator(base,layers=6,width=8,rank=2)
    source=(tuple(torch.randn(9,8) for _ in range(3)),tuple(torch.randn(9,8) for _ in range(3)))
    with torch.no_grad():a,b=base(*source,range(9))
    x,y=module(*source,range(9));assert torch.equal(a,x) and torch.equal(b,y)
    opt=torch.optim.AdamW(module.parameters(),lr=.01)
    for _ in range(2):
        opt.zero_grad();x,y=module(*source,range(9));(x.square().mean()+y.square().mean()).backward();opt.step()
    assert all(p.grad is None for p in base.parameters())
    assert module.keys[0][0].weight.grad.norm()>0 and module.values[0][0].weight.grad.norm()>0
    assert not any(k.startswith('base.') for k in module.state_dict())

def gpu(stage):
    verify();assert done('checks/cpu_completed.json')
    if done(f'checks/{stage}_gpu_completed.json'):return
    from runtime import make,backward_batch,batches
    from score import score
    engine=Engine();reader=BodyReader(engine);module=make(stage,engine);outcomes=[];parity=[]
    row=next(r for r in rows('train') if r['task']=='arc')
    with torch.no_grad():
        tk,tv,native=engine.teacher(row);engine.norm_counts=[0]*36
        oracle=raw_cache(engine.model,engine.header,tk.cuda(),tv.cuda());assert engine.norm_counts==[1]*36
        engine.norm_counts=None;target=row['gold_tokens'];ref=reader.logits(row,native,target[:1],(0,1),rn=False);got=reader.logits(row,oracle,target[:1],(0,1),rn=False)
        oracle_kl=float(C.vocab_kl(ref,got));assert oracle_kl<1e-4,oracle_kl
        del tk,tv,native,oracle,ref,got
        if stage=='stage2':
            for task in CFG['train_counts']:
                row=next(r for r in rows('train') if r['task']==task);source=engine.source(row)
                a,b=module.base(*source,row['sync_source_indices'][:16]);x,y=module(*source,row['sync_source_indices'][:16])
                assert torch.equal(a,x) and torch.equal(b,y)
                original=engine.translated(row,module.base,source,checkpointed=False);corrected=engine.translated(row,module,source,checkpointed=False)
                assert all(torch.equal(a,b) for aa,bb in zip(K.pairs(original),K.pairs(corrected)) for a,b in zip(aa,bb))
                parity.append(task);del original,corrected,source,a,b,x,y
    opt=torch.optim.AdamW(module.parameters(),lr=CFG[stage]['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)
    initial=next(module.parameters()).detach().clone()
    for task in CFG['train_counts']:
        row=max([r for r in rows('train') if r['task']==task],key=lambda r:len(r['content_ids'])+len(r['gold_tokens']))
        details,out=backward_batch(engine,reader,module,opt,scaler,[(row,'gold')],stage,1)
        outcomes.append(dict(task=task,id=row['id'],metrics=details,**out));engine.unchanged();C.log('GEMMA DST LONG BACKWARD',stage,task,out)
    for i,batch in enumerate(batches(1,stage)[:2]):
        _,out=backward_batch(engine,reader,module,opt,scaler,batch,stage,1)
        outcomes.append(dict(batch=i,effective_batch=8,**out));engine.unchanged();C.log('GEMMA DST CONSECUTIVE BATCH',stage,i,out)
    assert not torch.equal(initial,next(module.parameters()))
    engine.mode='test'
    try:engine.teacher(row)
    except RuntimeError:pass
    else:raise AssertionError('Native teacher allowed in student test')
    mark(f'checks/{stage}_gpu_completed.json',status='completed',stage=stage,disposable_updates=True,memory_used=False,native_raw_cache_kl=oracle_kl,zero_residual_parity=parity,exactly_once_k_norm_checked=True,trainable_parameters=sum(p.numel() for p in module.parameters()),outcomes=outcomes,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    C.log('GEMMA DST GPU CHECKS PASSED',stage,'DISPOSABLE UPDATES ONLY')

if __name__=='__main__':cpu() if sys.argv[1]=='cpu' else gpu(sys.argv[1])
