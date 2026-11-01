"""Streaming error marginals and signed, GQA-correct logit contributions."""
import torch


def rotate_half(x):return torch.cat((-x[...,64:],x[...,:64]),-1)


def undo_rope(x,cos,sin):
    c,s=cos.float()[None],sin.float()[None]
    return (x.float()*c-rotate_half(x.float())*s)/(c.square()+s.square()).clamp_min(1e-20)


def error_stats(delta,native,floor_fraction=.05):
    delta=delta.float();native=native.float()
    dim_scale=native.square().mean(1).sqrt()
    head_scale=native.square().mean((1,2)).sqrt()[:,None]
    scale=torch.maximum(dim_scale,(head_scale*floor_fraction).clamp_min(1e-6))
    z=delta/scale[:,None]
    channel=dict(bias=delta.mean(1),mae=delta.abs().mean(1),rmse=delta.square().mean(1).sqrt(),
        native_rms=dim_scale,normalizer=scale,normalized_bias=z.mean(1),normalized_mae=z.abs().mean(1),normalized_rmse=z.square().mean(1).sqrt())
    token=dict(bias=delta.mean(-1),mae=delta.abs().mean(-1),rmse=delta.square().mean(-1).sqrt(),normalized_rmse=z.square().mean(-1).sqrt())
    energy=delta.square().sum(-1);total=energy.sum(-1).clamp_min(1e-20)
    head=dict(relative_l2=(delta.square().sum((1,2))/native.square().sum((1,2)).clamp_min(1e-20)).sqrt(),
        raw_rmse=delta.square().mean((1,2)).sqrt(),normalized_rmse=z.square().mean((1,2)).sqrt(),
        floored_dimension_fraction=(dim_scale<scale).float().mean(-1))
    for fraction,name in [(.01,'top1pct_token_energy'),(.1,'top10pct_token_energy')]:
        k=max(1,int(energy.shape[1]*fraction));head[name]=energy.topk(k,dim=1).values.sum(-1)/total
    return channel,token,head


def rankdata(x):
    n=x.shape[-1];order=x.argsort(dim=-1,stable=True);values=x.gather(-1,order)
    p=torch.arange(n,device=x.device)
    starts=torch.cat((torch.ones_like(values[...,:1],dtype=torch.bool),values[...,1:]!=values[...,:-1]),-1)
    ends=torch.cat((values[...,:-1]!=values[...,1:],torch.ones_like(values[...,:1],dtype=torch.bool)),-1)
    lo=torch.where(starts,p,0).cummax(-1).values
    hi=torch.where(ends,p,n-1).flip(-1).cummin(-1).values.flip(-1)
    rank=torch.empty_like(x,dtype=torch.float32)
    return rank.scatter(-1,order,(lo+hi).float()/2+1)


def spearman(x,y):
    a,b=rankdata(x),rankdata(y);a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
    denominator=(a.square().sum(-1)*b.square().sum(-1)).sqrt()
    return torch.where(denominator>1e-20,(a*b).sum(-1)/denominator.clamp_min(1e-20),torch.full_like(denominator,float('nan')))


def overlap(x,y,k):
    a=x.topk(k,dim=-1).indices;b=y.topk(k,dim=-1).indices
    return (a[..., :,None]==b[...,None,:]).any(-1).float().mean(-1)


@torch.no_grad()
def contributions(delta,q,scale,p_native,body_start,body_end,token_chunk=128,probe_chunk=4,check=False):
    """q32xPx128, delta8xTx128; never allocate the full HxPxTxD tensor."""
    delta=delta.float();q=q.float().reshape(8,4,q.shape[1],128);probes=q.shape[2];tokens=delta.shape[1]
    ds=torch.einsum('hgpd,htd->hgpt',q,delta)*scale
    l1=torch.einsum('hgpd,htd->hgpt',q.abs(),delta.abs())*scale
    # Exact factorization of mean absolute coordinate contribution; this statistic
    # alone is not causal sensitivity and cannot account for cancellation.
    abs_q=q.abs().mean((1,2));mean_q=q.mean((1,2));rms_q=q.square().mean((1,2)).sqrt()
    coordinate=dict(mean_abs=abs_q*delta.abs().mean(1)*scale,
        signed_mean=mean_q*delta.mean(1)*scale,rms=rms_q*delta.square().mean(1).sqrt()*scale)
    pair_token=torch.zeros(8,tokens,64,device=delta.device)
    for t in range(0,tokens,token_chunk):
        chunk=delta[:,t:t+token_chunk]
        for p in range(0,probes,probe_chunk):
            query=q[:,:,p:p+probe_chunk]
            signed=(query[:,:,:,None,:64]*chunk[:,None,None,:,:64]+
                    query[:,:,:,None,64:]*chunk[:,None,None,:,64:])*scale
            pair_token[:,t:t+token_chunk]+=signed.abs().sum((1,2))
    pair_token/=4*probes
    pair_importance=pair_token.mean(1)
    pair_error=(delta[...,:64].square()+delta[...,64:].square()).mean(1).sqrt()
    native_body=p_native[:,:,body_start:body_end].reshape(8,4,probes,tokens)
    mu=(native_body*ds).sum(-1)
    weighted_variance=((native_body*ds.square()).sum(-1)-mu.square()).clamp_min(0)
    cancellation=ds.abs()/l1.clamp_min(1e-20)
    tok_abs_contribution=delta.abs()*abs_q[:,None]*scale
    token_stats=dict(logit_rmse=ds.square().mean((1,2)).sqrt(),
        cancellation=cancellation.mean((1,2)),dimension_rank_rho=spearman(delta.abs(),tok_abs_contribution),
        top8_dimension_overlap=overlap(delta.abs(),tok_abs_contribution,8))
    head=dict(logit_rmse=ds.square().mean((1,2,3)).sqrt(),
        native_weighted_logit_variance=weighted_variance.mean((1,2)),
        mean_abs_coordinate_contribution=coordinate['mean_abs'].mean(-1),
        cancellation_ratio=cancellation.mean((1,2,3)),
        dimension_rank_rho=spearman(delta.abs().mean(1),coordinate['mean_abs']),
        pair_rank_rho=spearman(pair_error,pair_importance),
        top8_dimension_overlap=overlap(delta.abs().mean(1),coordinate['mean_abs'],8),
        top16_dimension_overlap=overlap(delta.abs().mean(1),coordinate['mean_abs'],16),
        top4_pair_overlap=overlap(pair_error,pair_importance,4),top8_pair_overlap=overlap(pair_error,pair_importance,8))
    verification=None
    if check:
        tiny=delta[:,:min(5,tokens)]
        c=q[:,:,:,None,:]*tiny[:,None,None,:,:]*scale
        explicit=c.sum(-1);direct=ds[...,:tiny.shape[1]]
        err=float((explicit-direct).abs().max());bound=2e-5*(1+float(direct.abs().max()))
        assert err<bound,(err,bound)
        direct_logits=(torch.einsum('hgpd,htd->hgpt',q,tiny)*scale)
        assert torch.allclose(direct_logits,direct,atol=2e-5,rtol=2e-5)
        verification=dict(max_contribution_sum_error=err,bound=bound)
    return dict(coordinate=coordinate,pair_mean_abs=pair_importance,pair_error_rms=pair_error,
        head=head,token=token_stats,delta_logits=ds,mean_abs_token_dimension=tok_abs_contribution,
        verification=verification)


@torch.no_grad()
def attention_metrics(C,D,engine,layer,trace,translated,native):
    start=len(engine.prefix_ids);end=engine._diagnostic_body_end
    tail=C._cache_layer(trace.cache,layer);kt,vt=C._cache_layer(translated,layer);kn,vn=C._cache_layer(native,layer)
    keys={};values={}
    for name,k,v in [('T',kt,vt),('N',kn,vn)]:
        keys[name]=tail[0].clone();values[name]=tail[1].clone()
        keys[name][:,:,start:end]=k[:,:,start:end];values[name][:,:,start:end]=v[:,:,start:end]
    lp={}
    for name in ['T','N']:lp[name],allowed=D.distribution(C,engine,layer,trace.q[layer],keys[name],trace.qpos)
    pt,pn=lp['T'].exp(),lp['N'].exp();mix=(pt+pn)/2;lm=mix.clamp_min(1e-30).log()
    kl=(pn*(lp['N']-lp['T']).masked_fill(~allowed[None],0)).sum(-1).clamp_min(0)
    js=.5*((pn*(lp['N']-lm).masked_fill(~allowed[None],0)).sum(-1)+(pt*(lp['T']-lm).masked_fill(~allowed[None],0)).sum(-1))
    js=js.clamp_min(0)
    att=C.layers_of(engine.model)[layer].self_attn;assert att.o_proj.bias is None
    wo=att.o_proj.weight.float().reshape(2560,8,512);outputs={}
    for case,k,v in [('TT','T','T'),('NT','N','T'),('TN','T','N'),('NN','N','N')]:
        context=torch.einsum('hpt,htd->phd',lp[k].exp(),values[v][0].repeat_interleave(4,0).float()).reshape(trace.qpos.numel(),8,512)
        outputs[case]=torch.einsum('phd,mhd->hpm',context,wo)
    denominator=outputs['NN'].square().sum(-1).clamp_min(1e-20)
    errors={case:((o-outputs['NN']).square().sum(-1)/denominator).mean(-1) for case,o in outputs.items()}
    total_native=outputs['NN'].sum(0);global_errors={case:float(((o.sum(0)-total_native).square().sum(-1)/total_native.square().sum(-1).clamp_min(1e-20)).mean()) for case,o in outputs.items()}
    cross=outputs['NN']-outputs['NT']-outputs['TN']+outputs['TT']
    head=dict(route_kl=kl.reshape(8,4,-1).mean((1,2)),route_js=js.reshape(8,4,-1).mean((1,2)),
        **{'wo_'+case.lower()+'_nmse':v for case,v in errors.items()},
        wo_interaction_nmse=(cross.square().sum(-1)/denominator).mean(-1))
    return dict(head=head,global_output_nmse=global_errors,p_native=pn,
        route_kl=float(kl.mean()),route_js=float(js.mean()),
        reconstruction_relative_l2=D.relative_l2(total_native,D.readout(C,engine,layer,lp['N'],values['N'])))
