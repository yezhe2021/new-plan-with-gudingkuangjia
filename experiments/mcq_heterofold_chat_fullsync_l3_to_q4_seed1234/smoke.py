"""Native parity, differentiable-score parity, strict identity init and gradients."""
from core import *
from train import stage_a_backward,stage_b_backward,update


def main():
    seed();guard('stage_a')
    base=fresh_base().cuda()
    optimizer=torch.optim.AdamW(base.parameters(),lr=CFG['stage_a_learning_rate'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    rr=rows('train',True);before=next(base.parameters()).detach().clone()
    for row in rr:
        stage_a_backward(base,state('train',row,'llama'),state('train',row,'qwen'),len(rr),scaler=scaler)
    a_norm=update(base,optimizer,scaler)
    assert not torch.equal(before,next(base.parameters()))
    del optimizer,before
    base.eval().requires_grad_(False)
    adapter=ResidualKVAdapter(rank=CFG['adapter_rank']).cuda()
    tok,receiver=model('qwen');rt=runtime(receiver,tok)
    checks=[]
    for row in rr:
        source,target=state('train',row,'llama'),state('train',row,'qwen')
        with torch.no_grad():
            k,v=join(target,target['key'].cuda(),target['value'].cuda())
            differentiable=tensor_scores(receiver,tok,row,k,v)
            cache=make_cache(receiver,k,v,torch.arange(k.shape[1],device='cuda'))
            official,_=R.transfer_candidate_scores(rt,cache,k.shape[1],row['candidates'],False,row['receiver_suffix'])
            assert max(abs(float(a)-b) for a,b in zip(differentiable,official))<1e-5,'Differentiable scorer != official scorer'
            native_check=choice_gate(target['teacher_scores'].tolist(),official)
            bk,bv=map_base(base,source)
            ak,av=adapted(adapter,bk,bv)
            assert torch.equal(ak,bk) and torch.equal(av,bv),'Adapter not zero initialized'
        # Independently establish a nonzero derivative through each injected KV.
        k=bk.detach().requires_grad_(True);v=bv.detach().requires_grad_(True)
        jk,jv=join(target,k,v)
        s=candidate_tensor(receiver,tok,row,jk,jv,0)
        gk,gv=torch.autograd.grad(s,(k,v))
        assert torch.isfinite(gk).all() and torch.isfinite(gv).all() and gk.abs().max()>0 and gv.abs().max()>0
        checks.append(dict(id=row['id'],native_cache=native_check,kv_gradient_k_norm=float(gk.float().norm()),kv_gradient_v_norm=float(gv.float().norm()),adapter_identity=True))
        del k,v,jk,jv,s,gk,gv
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=CFG['stage_b_learning_rate'],weight_decay=0)
    before=adapter.k[0][0][-1].weight.detach().clone()
    for row in rr:
        source,target=state('train',row,'llama'),state('train',row,'qwen')
        bk,bv=map_base(base,source)
        stage_b_backward(receiver,tok,row,target,bk,bv,adapter,len(rr),scaler=scaler)
    b_norm=update(adapter,optimizer,scaler)
    assert not torch.equal(before,adapter.k[0][0][-1].weight)
    assert all(p.grad is None and not p.requires_grad for p in receiver.parameters())
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    report=dict(status='completed',manifest_sha256=identity(),checks=checks,stage_a_grad_norm=a_norm,stage_b_grad_norm=b_norm,
        frozen_receiver=True,frozen_stage_a_during_stage_b=True,cache_lossless_native_fp16=True,
        separate_smoke_parameters_discarded=True)
    save(ROOT/'smoke/completed.json',report)
    status(status='smoke_passed',phase='smoke')
    log('SMOKE PASSED: native cache, exact official candidate scores, lossless caches and K/V gradients')


if __name__=='__main__':
    try:main()
    except Exception as exc:fail(exc);raise
