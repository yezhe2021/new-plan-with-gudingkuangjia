"""GPU checks run only after dependency success; disposable memory updates."""
from common import *
from model import load,losses
import cache as K

def main():
    verify()
    if done('smoke/completed.json'):return
    assert C.read(ROOT/'cpu_checks.json')['status']=='completed'
    dependency=Path(CFG['dependency']);assert C.read(dependency/'completed.json')['status']=='completed'
    assert C.read(dependency/'runner_status.json')['status']=='completed'
    engine,correction,module=load();correction_versions=C.versions(correction)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=T.scaler_for()
    shortest=[min([r for r in rows('train') if r['task']==task],key=lambda r:len(r['content_ids'])) for task in CFG['train_counts']]
    selected=shortest+[max(rows('train'),key=lambda r:len(r['content_ids']))];checks=[]
    for i,row in enumerate(selected):
        with torch.no_grad():
            base=engine.base(row);translated=engine.cache(row,base,correction);native=engine.native_cache(row)
            teacher_before=module.logits(row,native,rn=False).clone()
            memory=module.construct(row,translated);L=translated.get_seq_length();header=len(engine.prefix_ids)
            assert memory.get_seq_length()==L+32
            assert all(torch.equal(a[:,:,:L],b) for ap,bp in zip(K.pairs(memory),K.pairs(translated)) for a,b in zip(ap,bp))
            original=module.logits(row,memory,rn=True);changed=K.clone(memory,engine.model)
            for l,pair in enumerate(K.pairs(changed),1):
                start,end=(header,L) if l in CFG['replace_layers'] else (L,L+32)
                for value in pair:value[:,:,start:end,:]=torch.randn_like(value[:,:,start:end,:])*10
            blocked=module.logits(row,changed,rn=True)
            assert torch.equal(original,blocked),'Forbidden cache source leaks into answers'
            del memory,changed,original,blocked
        loss,detail=losses(engine,module,row,translated,native,['gold','native','student'][i],1)
        scaler.scale(loss).backward();del loss
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert module.memory.prompt.grad.norm()>0
        assert any(p.grad.norm()>0 for p in module.memory.adapters.parameters())
        scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(module.parameters(),1.);assert torch.isfinite(grad) and grad>0
        scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        with torch.no_grad():teacher_after=module.logits(row,native,rn=False)
        assert torch.equal(teacher_before,teacher_after)
        engine.unchanged();assert C.versions(correction)==correction_versions
        checks.append(dict(id=row['id'],body_tokens=len(row['content_ids']),grad=float(grad),**detail))
        del base,translated,native,teacher_before,teacher_after
        C.log('RNKV GPU SMOKE PASSED',row['id'],len(row['content_ids']))
    # Verify shared original-C baseline scoring against the existing official candidate scorer.
    row=next(r for r in rows('test') if r['task']=='arc');engine.mode='test'
    with torch.no_grad():
        base=engine.base(row);translated=engine.cache(row,base,correction)
        own=module.score(row,translated,rn=False)
        scores,_=C.RT.transfer_candidate_scores(engine.runtime,K.clone(translated,engine.model),translated.get_seq_length(),row['candidates'],False,row['continuation_suffix'])
        assert own['prediction']==max(range(len(scores)),key=scores.__getitem__)
        a=torch.tensor(own['scores'],dtype=torch.float64).log_softmax(-1);b=torch.tensor(scores,dtype=torch.float64).log_softmax(-1)
        choice_kl=float((a.exp()*(a-b)).sum());assert abs(choice_kl)<.001
        rn=module.construct(row,translated);value=module.score(row,rn,rn=True)
        assert all(C.np.isfinite(x) for x in value['scores'])
    assert engine.student_test_native_prefills==0;engine.unchanged()
    mark('smoke/completed.json',status='completed',checks=checks,original_C_candidate_parity_kl=choice_kl,disposable_optimizer_updates=True,
        trainable_parameters=409600,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

if __name__=='__main__':main()
