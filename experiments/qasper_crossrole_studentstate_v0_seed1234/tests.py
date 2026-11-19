import copy
import json
import torch
import torch.nn.functional as F
from common_exp import ROOT, CONFIG, CFG, BASELINE, save, setup, old_base, adapter, collect, mapped, seed
from cross_role import RoleResidual, AugmentedMapper
from student_probe import collect_student, local_loss


def cpu_tests():
    torch.manual_seed(CONFIG['seed'])
    for mode in ('self','cross'):
        m=RoleResidual(mode); count=sum(p.numel() for p in m.parameters())
        assert count==9437184
        k=torch.randn(1,28,2,8,128); v=torch.randn_like(k)
        k0=torch.randn(1,36,2,8,128); v0=torch.randn_like(k0)
        pk,pv=m(k,v,k0,v0)
        assert torch.equal(pk,k0) and torch.equal(pv,v0)
        with torch.no_grad():
            for modules in (m.k,m.v):
                for layer in modules:
                    for branch in layer: torch.nn.init.normal_(branch[-1].weight,std=0.01)
        if mode=='cross':
            nk,_=m(k+1,v,k0,v0); _,nv=m(k,v+1,k0,v0)
            assert torch.equal(nk,m(k,v,k0,v0)[0]) and torch.equal(nv,m(k,v,k0,v0)[1])
            ck,cv=m(k,v,k0,v0); changed_k,changed_v=m(k+1,v+1,k0,v0)
            assert not torch.equal(ck,changed_k) and not torch.equal(cv,changed_v)
        assert m.neighbours[0]==(0,0,1) and m.neighbours[-1]==(26,27,27)
    print('CPU IDENTITY/ROLE/PARAMETER TESTS PASSED',flush=True)


def main():
    cpu_tests(); seed()
    sender,receiver,rt,cache=setup()
    data=json.loads((ROOT/'prepared.json').read_text())
    rows=[min(data['train'],key=lambda r:r['body_tokens']),max(data['train'],key=lambda r:r['body_tokens'])]
    frozen=old_base(); calibrator=adapter(); records=[]
    for number,row in enumerate(rows):
        item=cache.get(row); query=item['protocol']['queries'][0]; bk,bv=mapped(frozen,item)
        if number==0:
            tiny={**item,'source_k':item['source_k'][:,:16],'source_v':item['source_v'][:,:16],
                'target_k':item['target_k'][:,:16],'target_v':item['target_v'][:,:16],'tokens':16}
            for mode in ('self','cross'):
                residual=RoleResidual(mode).cuda(); model=AugmentedMapper(frozen,residual)
                with torch.no_grad():
                    x,y=mapped(model,tiny); x0,y0=mapped(frozen,tiny)
                    assert torch.equal(x,x0) and torch.equal(y,y0), 'Zero residual must be bit-identical'
                opt=torch.optim.AdamW(residual.parameters(),lr=1e-4,weight_decay=0)
                scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
                for update in range(2):
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda',dtype=torch.float16):
                        k,v=model(tiny['source_k'][None].cuda(),tiny['source_v'][None].cuda())
                    loss=sum((p.float()-tiny[t][None].cuda().float()).square().mean() for p,t in ((k,'target_k'),(v,'target_v')))
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in residual.parameters())
                    if update==1:
                        assert any(branch[0].weight.grad.abs().sum()>0 for layer in residual.k for branch in layer)
                        assert any(branch[0].weight.grad.abs().sum()>0 for layer in residual.v for branch in layer)
                    norm=torch.nn.utils.clip_grad_norm_(residual.parameters(),30.); assert torch.isfinite(norm)
                    scaler.step(opt); scaler.update()
                assert all(p.grad is None for p in frozen.parameters())
                del model,residual,opt,k,v,x,y,x0,y0
            torch.cuda.empty_cache()
        # Identity diagnostic uses the exact teacher wrapper on BOTH prefix paths;
        # normal training still uses separately computed wrapper, never native evidence.
        exact=dict(item); n=len(item['protocol']['head_ids'])
        exact['wrapper_k']=item['native_k'][:,:n]; exact['wrapper_v']=item['native_v'][:,:n]
        native=collect(receiver,item,query)
        identity=collect_student(receiver,exact,query,item['target_k'].cuda(),item['target_v'].cuda(),calibrator)
        assert all(torch.equal(native['layers'][l]['q'],identity['layers'][l]['q']) for l in range(36))
        student=collect_student(receiver,item,query,bk,bv,calibrator)
        max_replay=0.; max_identity_loss=0.
        with torch.no_grad():
            for l in range(36):
                loss,_=local_loss(receiver,item,identity,l,item['target_k'][l].cuda(),item['target_v'][l].cuda(),calibrator)
                max_identity_loss=max(max_identity_loss,abs(float(loss)))
                loss,stats=local_loss(receiver,item,student,l,bk[l],bv[l],calibrator)
                assert torch.isfinite(loss)
                max_replay=max(max_replay,stats['actual_student_replay_nmse'])
        assert max_identity_loss<1e-4
        assert max_replay<1e-3, f'Actual student attention replay: {max_replay}'
        assert all(not tensor.requires_grad for s in student['layers'].values() for tensor in s.values())
        if number==0:
            calibrator.zero_grad(set_to_none=True)
            loss,stats=local_loss(receiver,item,student,0,bk[0],bv[0],calibrator)
            loss.backward()
            for modules in (calibrator.k,calibrator.v):
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in modules.parameters())
            opt=torch.optim.SGD(calibrator.parameters(),lr=1e-4)
            norm=torch.nn.utils.clip_grad_norm_(calibrator.parameters(),1.)
            assert torch.isfinite(norm) and norm>0; opt.step(); opt.zero_grad(set_to_none=True)
            # Refresh audit under deliberate small V perturbation at an early layer.
            weight=calibrator.v[0][0][-1].weight
            with torch.no_grad():
                before=weight.clone(); weight.add_(torch.randn_like(weight)*0.01)
            refreshed=collect_student(receiver,item,query,bk,bv,calibrator)
            assert not torch.equal(refreshed['layers'][1]['q'],student['layers'][1]['q'])
            with torch.no_grad(): weight.copy_(before)
            del opt,refreshed
        assert all(p.grad is None for p in sender.parameters()) and all(p.grad is None for p in receiver.parameters())
        assert all(p.grad is None for p in frozen.parameters())
        records.append({'paper_id':row['paper_id'],'tokens':item['tokens'],
            'identity_native_student_query_exact':True,'identity_loss_max':max_identity_loss,
            'actual_student_replay_nmse_max':max_replay,'base_model_gradients_absent':True})
        save(ROOT/'smoke/tests.json',records)
        print('GPU SMOKE PASSED',number+1,row['paper_id'],item['tokens'],max_replay,flush=True)
        calibrator=adapter()
    save(ROOT/'smoke/completed.json',{'papers':len(rows),'A_identity_role_gradient_pass':True,
        'B_real_query_detach_and_finite_KV_gradient_pass':True,'refresh_pass':True,'no_test_used':True})
    print('ALL NEW MODULE IDENTITY/REPLAY/GRADIENT TESTS PASSED',flush=True)


if __name__=='__main__': main()
