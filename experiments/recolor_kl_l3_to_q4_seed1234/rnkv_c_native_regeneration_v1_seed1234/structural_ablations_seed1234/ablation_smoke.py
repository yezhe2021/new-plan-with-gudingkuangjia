"""Real parameter scope, routing, initialization, scorer and contiguous-batch checks."""
from ablation_common import *
from ablation_runtime import load,backward_batch,batches
from ablation_evaluate import score

def main(arm):
    verify()
    if done(f'smoke/{arm}/completed.json'):return
    assert C.read(ROOT/'cpu_checks.json')['status']=='completed'
    engine,correction,module=load(arm);versions=C.versions(correction)
    initial=module.memory.prompt.detach().clone();reference=M.RNKV(engine)
    assert torch.equal(reference.memory.prompt,initial)
    # Memory-only is an identity-adapter model from initialization, not a
    # jointly trained prompt with adapters removed after learning.
    row=min(rows('train'),key=lambda r:len(r['content_ids']))
    with torch.no_grad():
        base=engine.base(row);translated=engine.cache(row,base,correction)
        if arm=='memory_only':
            a=module.construct(row,translated);b=reference.construct(row,translated)
            assert all(torch.equal(x,y) for x,y in zip(K.flat(a),K.flat(b)))
            assert not module.memory.adapters and list(dict(module.named_parameters()))==['memory.prompt']
            del a,b
        else:
            L=translated.get_seq_length();q=torch.tensor([L+32],device='cuda')
            for layer in [1,18,19,30,31,36]:
                a=module.routing.allowed('answer',layer,len(engine.prefix_ids),L,32,q,L+33)
                assert a[:,:L].all() and bool(a[:,L:L+32].all())==(19<=layer<=30)
        del base,translated
    reference.routing.close();del reference
    # The largest training example must permit a complete memory/Qwen backward.
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for()
    longrow=max(rows('train'),key=lambda r:len(r['content_ids']))
    detail,outcome,retries=backward_batch(engine,correction,module,opt,scaler,[(longrow,'gold')],1,0,scope='smoke_long')
    engine.unchanged();assert C.versions(correction)==versions
    longcheck=dict(id=longrow['id'],body_tokens=len(longrow['content_ids']),amp_retries=retries,**outcome)
    # Recreate original initialization before consecutive exact training batches.
    module.routing.close();del module,opt,scaler
    from ablation_runtime import AblationModel
    module=AblationModel(engine,arm);assert torch.equal(module.memory.prompt,initial)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for();batchchecks=[]
    for bi,batch in enumerate(batches(1)[:4],1):
        detail,outcome,retries=backward_batch(engine,correction,module,opt,scaler,batch,1,bi,scope='smoke')
        batchchecks.append(dict(batch=bi,mean_loss=sum(d['loss'] for d in detail)/8,amp_retries=retries,**outcome))
        engine.unchanged();assert C.versions(correction)==versions
        C.log('STRUCTURAL SMOKE BATCH PASSED',arm,bi,outcome['gradient_norm'])
    assert not torch.equal(module.memory.prompt,initial)
    if arm=='memory_only':assert sum(p.numel() for p in module.parameters())==81920
    module.routing.close();del module,opt,scaler
    parity=[]
    if arm=='memory_only':
        reference=M.RNKV(engine);cp=torch.load(PARENT/'runs/epoch4_step1024.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==R.identity();reference.load_state_dict(cp['state']);reference.eval().requires_grad_(False);del cp
        engine.mode='test'
        with torch.no_grad():
            for task in CFG['test_counts']:
                row=next(r for r in rows('test') if r['task']==task)
                base=engine.base(row);translated=engine.cache(row,base,correction);memory=reference.construct(row,translated)
                value=score(reference,row,memory);old=baseline(row)['rnkv']
                assert value['prediction']==old['prediction'],('Full-RN baseline parity',task)
                if 'generation' in value:assert value['generation']==old['generation']
                error=max([abs(a-b) for a,b in zip(value.get('scores',[]),old.get('scores',[]))] or [0.]);assert error<.001
                parity.append(dict(task=task,id=row['id'],score_error=error));del base,translated,memory
        reference.routing.close();del reference
    engine.unchanged();assert engine.student_test_native_prefills==0
    mark(f'smoke/{arm}/completed.json',status='completed',arm=arm,trainable_parameters=CFG['arms'][arm]['parameters'],
        same_initial_prompt=True,long_backward=longcheck,consecutive_batches=batchchecks,baseline_parity=parity,disposable_updates=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__':main(sys.argv[1])
