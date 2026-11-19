"""Small-training-set fitting, real backward paths, and HeteroFold folding."""
import json
import torch
from common import ROOT, CFG, rows, save, seed, log, release
from models import Pool, activate, offload, generate
from algorithms import (Engine, our_model, our_loss, our_map, residual, token_kl,
                        LayerCorrection, hf_objective, hf_features, fold,
                        receiver_to_sender_indices, stencil_taps)
from vendor.heterofold.recolor import fit_affine


def tiny_recolor():
    torch.manual_seed(CFG['seed'])
    x=torch.randn(256,32,dtype=torch.float64);w=torch.randn(32,16,dtype=torch.float64)*.1
    y=x@w+.2
    a,b,r=fit_affine(x.sum(0),y.sum(0),x.T@x,x.T@y,y.T@y,len(x),device='cpu')
    error=float((x/r@a+b-y).square().mean())
    assert error<float(y.square().mean())*.01
    return {'passed':True,'synthetic_fit_mse':error,'not_a_benchmark_result':True}


def family_smoke(family):
    seed();pool=Pool();engine=Engine(pool)
    selected=rows(ROOT/'prepared/train.jsonl')[:16]
    pairs=[]
    for start in range(0,16,CFG['capture_batch_questions']):
        pairs+=engine.pairs(selected[start:start+CFG['capture_batch_questions']],family,('joint',))
    pool.cpu();learner=our_model(family).cuda()
    opt=torch.optim.AdamW(learner.parameters(),lr=CFG['our_stage_a']['learning_rate'],weight_decay=0.)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():before=sum(our_loss(learner,p) for p in pairs)/len(pairs)
    for epoch in range(2):
        for pair in pairs:
            opt.zero_grad(set_to_none=True);our_loss(learner,pair,True)
            norm=torch.nn.utils.clip_grad_norm_(learner.parameters(),30.)
            assert torch.isfinite(norm) and norm>0
            opt.step()
        log(f'FIT16 SMOKE {family} pass={epoch+1}/2')
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():after=sum(our_loss(learner,p) for p in pairs)/len(pairs)
    assert after<before,'Small-training-set Writer loss failed to decrease'
    del opt
    learner.eval().requires_grad_(False)
    pair=max(pairs,key=lambda p:len(p['proto']['target']['body_ids']))
    k,v=our_map(learner,pair);offload(learner)
    receiver=pool.gpu('qwen')
    adapter=residual().cuda();adapter_opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=0.)
    ids,_=generate(receiver,pair['proto']['target']['ids'],limit=16)
    loss=token_kl(receiver,adapter,pair,k,v,ids);loss.backward()
    bnorm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),30.)
    assert bnorm>0 and torch.isfinite(bnorm)
    assert all(p.grad is None for p in receiver.parameters())
    assert all(p.grad is None for p in learner.parameters())
    adapter_opt.step();bvalue=float(loss.detach())
    del loss,adapter,adapter_opt,k,v
    # An identity central-layer affine is ONLY an interface smoke input; formal
    # HeteroFold weights are refit by Recolor on the complete fixed train split.
    matrix=torch.zeros(3072,1024);matrix[1024:2048]=torch.eye(1024)
    base={'A_k':[matrix]*36,'A_v':[matrix]*36,'b_k':[torch.zeros(1024)]*36,'b_v':[torch.zeros(1024)]*36,
          'rms_k':[1.]*36,'rms_v':[1.]*36,'Lr':36,'family':family}
    corrections=torch.nn.ModuleList([LayerCorrection(1024,CFG['heterofold']['rank']) for _ in range(36)]).cuda()
    opt=torch.optim.AdamW(corrections.parameters(),lr=CFG['heterofold']['learning_rate'])
    metrics=hf_objective(receiver,base,corrections,pair,True)
    role_norms={role:sum(float(p.grad.square().sum()) for n,p in corrections.named_parameters() if n.endswith('right_'+role) and p.grad is not None)**.5 for role in 'kv'}
    assert all(v>0 for v in role_norms.values()),role_norms
    norm=torch.nn.utils.clip_grad_norm_(corrections.parameters(),1.);assert torch.isfinite(norm)
    opt.step();assert all(p.grad is None for p in receiver.parameters())
    folded=fold(base,corrections,{'rank':CFG['heterofold']['rank'],'smoke_only':True})
    indices=receiver_to_sender_indices(pair['proto']['source'][family]['spans'],pair['proto']['target']['spans'])
    drift=0.
    with torch.no_grad():
        for layer in (0,18,35):
            for role in 'kv':
                x=hf_features(pair,layer,role,indices[:16])
                y=x@base['A_'+role][layer]+base['b_'+role][layer]
                direct=getattr(corrections[layer],'key' if role=='k' else 'value')(y.cuda()).cpu()
                actual=x@folded['A_'+role][layer]+folded['b_'+role][layer]
                drift=max(drift,float((direct-actual).abs().max()))
                assert torch.allclose(direct,actual,atol=3e-5,rtol=3e-5)
    result={'family':family,'passed':True,'fit16_initial_loss':before,'fit16_after_32_updates_loss':after,
            'writer_b_kl':bvalue,'writer_b_gradient_norm':float(bnorm),'heterofold_objective':metrics,
            'heterofold_role_gradient_norms':role_norms,'folding_max_abs_error':drift,
            'frozen_model_parameter_gradients':False,'peak_allocated_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'smoke_weights_discarded':True,'small_fit_is_not_generalization_evidence':True}
    save(ROOT/f'reports/smoke_{family}.json',result)
    log('ALGORITHM SMOKE PASSED '+json.dumps(result))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=['llama','gemma','recolor']);args=ap.parse_args()
    if args.family=='recolor':save(ROOT/'reports/smoke_recolor.json',tiny_recolor())
    else:family_smoke(args.family)
