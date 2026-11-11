from experiment_common import *
from residual import ResidualTranslator
from engine import Engine,BodyReader
def cpu():
    verify();torch.set_num_threads(1)
    base=Z.DT.DepthStateTranslator(Z.DT.DSTConfig(sender_layers=3,receiver_layers=6,source_width=8,target_width=8,state_width=4))
    model=ResidualTranslator(base,layers=6,width=8,rank=2)
    source=(tuple(torch.randn(9,8) for _ in range(3)),tuple(torch.randn(9,8) for _ in range(3)))
    with torch.no_grad():a,b=base(*source,range(9))
    x,y=model(*source,range(9));assert torch.equal(a,x) and torch.equal(b,y)
    assert sum(p.numel() for p in model.parameters())==6*2*2*8*2
    assert not any(k.startswith('base.') for k in model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    for _ in range(2):
        opt.zero_grad();x,y=model(*source,range(9));(x.square().mean()+y.square().mean()).backward();opt.step()
    assert model.keys[0][0].weight.grad.norm()>0 and model.values[0][0].weight.grad.norm()>0
    assert all(p.grad is None for p in base.parameters())
    x,y=model(*source,range(9));parts=[model(*source,range(i,min(9,i+3))) for i in range(0,9,3)]
    assert torch.allclose(x,torch.cat([r[0] for r in parts],1),atol=1e-6)
    mark('checks/cpu_completed.json',status='completed',zero_residual_exact_identity=True,down_up_gradients=True,frozen_base_no_grad=True,chunk_parity=True)
    C.log('B CPU CHECKS PASSED')
def gpu():
    verify();assert done('checks/cpu_completed.json')
    if done('checks/gpu_completed.json'):return
    from runtime import backward_batch,batches
    from score import score
    engine=Engine();module=engine.correction;reader=BodyReader(engine);parity=[]
    with torch.no_grad():
        for task in CFG['test_counts']:
            row=next(r for r in rows('test') if r['task']==task);old=baseline(row)
            source=engine.source(row)
            native=engine.native_cache(row,baseline=True)
            cache=engine.translated(row,module,source,checkpointed=False)
            original=engine.translated(row,engine.frozen_dst,source,checkpointed=False)
            assert all(torch.equal(a,b) for aa,bb in zip(K.pairs(cache),K.pairs(original)) for a,b in zip(aa,bb))
            F.same_prediction(score(reader,row,cache),old['DST_stage1'],task,'zero residual DST Stage1')
            F.same_prediction(score(reader,row,native),old['native'],task,'native')
            parity.append(task);del native,cache,original,source;C.log('B ZERO RESIDUAL PARITY',task)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['stage2']['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000);outcomes=[]
    for task in CFG['train_counts']:
        row=max([r for r in rows('train') if r['task']==task],key=lambda r:len(r['content_ids'])+len(r['gold_tokens']))
        _,out=backward_batch(engine,reader,module,opt,scaler,[(row,'gold')],'stage2',1);outcomes.append(dict(task=task,**out));engine.unchanged();C.log('B LONG BACKWARD',task,out)
    for i,batch in enumerate(batches(1,'stage2')[:2]):
        _,out=backward_batch(engine,reader,module,opt,scaler,batch,'stage2',1);outcomes.append(dict(batch=i,effective_batch=8,**out));engine.unchanged();C.log('B CONSECUTIVE BATCH',i,out)
    assert module.keys[0][0].weight.grad.norm()>0
    mark('checks/gpu_completed.json',status='completed',parity=parity,outcomes=outcomes,trainable_parameters=9437184,disposable_updates=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
if __name__=='__main__':{'cpu':cpu,'gpu':gpu}[sys.argv[1]]()
