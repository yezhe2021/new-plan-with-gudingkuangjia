"""Read-only Qwen query capture, fixed-query counterfactuals and attention checks."""
import torch
from heterofold.attention import apply_rope,causal_mask,rotary_cos_sin,softcap_logits,output_projection


def relative_l2(value,target):
    return float(((value.float()-target.float()).square().sum()/target.float().square().sum().clamp_min(1e-20)).sqrt())


class Trace:
    pass


@torch.no_grad()
def forward_trace(C,engine,body,tokens,positions,layers,clamp=None,clamp_layers=()):
    cache=C.clone_cache(body,engine.model);past=int(cache.get_seq_length())
    # Suffix already contains the literal unchanged official assistant/answer cue.
    ids=engine._diagnostic_suffix+tokens[:max(positions)]
    ix=torch.tensor([len(engine._diagnostic_suffix)-1+p for p in positions],device='cuda')
    qpos=ix+past;captures={};handles=[];clamp_layers=set(clamp_layers)
    for layer in layers:
        att=C.layers_of(engine.model)[layer].self_attn;captures[layer]={}
        def query_hook(module,args,output,l=layer):
            assert output.ndim==4 and output.shape[0]==1 and output.shape[2:]==(32,128),output.shape
            value=output
            if clamp is not None and l in clamp_layers:
                value=output.clone();value[0,ix]=clamp.qpre[l]
            captures[l]['qpre']=value[0].index_select(0,ix).detach().clone()
            return value
        def output_hook(module,args,output,l=layer):
            captures[l]['o']=output[0].index_select(0,ix).detach().clone()
        handles.append(att.q_norm.register_forward_hook(query_hook))
        handles.append(output_projection(att).register_forward_hook(output_hook))
    try:out=engine.forward(ids,cache,past,use_cache=True)
    finally:
        for h in handles:h.remove()
    assert cache.get_seq_length()==past+len(ids)
    trace=Trace();trace.cache=cache;trace.qpos=qpos;trace.positions=positions
    trace.candidate_ids=engine._diagnostic_candidate_ids;trace.gold_index=engine._diagnostic_gold_index
    trace.qpre={l:captures[l]['qpre'] for l in layers}
    trace.q={};trace.o={l:captures[l]['o'] for l in layers}
    trace.logits=engine.model.lm_head(out.last_hidden_state[0].index_select(0,ix)).float()
    for layer in layers:
        cos,sin=rotary_cos_sin(engine.model,qpos,layer,engine.model.dtype)
        trace.q[layer]=apply_rope(trace.qpre[layer],cos,sin).transpose(0,1)
        if clamp is not None and layer in clamp_layers:
            assert torch.equal(trace.qpre[layer],clamp.qpre[layer]),('Q clamp not exact',layer)
    return trace


@torch.no_grad()
def distribution(C,engine,layer,q,keys,qpos):
    att=C.layers_of(engine.model)[layer].self_attn
    assert q.shape[0]==32 and keys.shape[:2]==(1,8)
    k=keys[0].repeat_interleave(4,dim=0)
    allowed=causal_mask(att,qpos,k.shape[1])
    logits=torch.einsum('hpd,htd->hpt',q.float(),k.float())*float(att.scaling)
    logits=softcap_logits(att,logits).masked_fill(~allowed[None],float('-inf'))
    return logits.log_softmax(-1),allowed


@torch.no_grad()
def readout(C,engine,layer,logp,values):
    v=values[0].repeat_interleave(4,dim=0).float()
    context=torch.einsum('hpt,htd->phd',logp.exp(),v).reshape(logp.shape[1],4096)
    output=output_projection(C.layers_of(engine.model)[layer].self_attn)(context.to(engine.model.dtype)).float()
    return output


@torch.no_grad()
def compare_attention(C,engine,layer,q,keys,values,qpos,ref_q,ref_keys,ref_values):
    lp,mask=distribution(C,engine,layer,q,keys,qpos)
    lr,_=distribution(C,engine,layer,ref_q,ref_keys,qpos)
    p,r=lp.exp(),lr.exp()
    difference=(lr-lp).masked_fill(~mask[None],0.)
    head_kl=(r*difference).sum(-1).mean(-1).clamp_min(0)
    out=readout(C,engine,layer,lp,values);target=readout(C,engine,layer,lr,ref_values)
    per_probe_nmse=(out-target).square().sum(-1)/target.square().sum(-1).clamp_min(1e-20)
    start=len(engine.prefix_ids);end=engine._diagnostic_body_end
    return dict(route_kl=float(head_kl.mean()),route_tv=float((p-r).abs().sum(-1).mean()/2),
        top1_agreement=float((p.argmax(-1)==r.argmax(-1)).float().mean()),
        wo_nmse=float(per_probe_nmse.mean()),
        body_attention_mass=float(p[:,:,start:end].sum(-1).mean()),
        head_route_kl=head_kl.cpu().tolist(),probe_wo_nmse=per_probe_nmse.cpu().tolist()),out


@torch.no_grad()
def fixed_grid(C,engine,layer,trace,translated,native):
    """Hold Q and all nonbody keys/values fixed while toggling body K/V."""
    start=len(engine.prefix_ids);end=engine._diagnostic_body_end
    tail=C._cache_layer(trace.cache,layer)
    translated=C._cache_layer(translated,layer);native=C._cache_layer(native,layer)
    keys={};values={}
    for name,cache in [('T',translated),('N',native)]:
        keys[name]=tail[0].clone();values[name]=tail[1].clone()
        keys[name][:,:,start:end]=cache[0][:,:,start:end]
        values[name][:,:,start:end]=cache[1][:,:,start:end]
    result={};outputs={}
    for case,k,v in [('TT','T','T'),('NT','N','T'),('TN','T','N'),('NN','N','N')]:
        result[case],outputs[case]=compare_attention(C,engine,layer,trace.q[layer],keys[k],values[v],trace.qpos,
            trace.q[layer],keys['N'],values['N'])
    assert result['NT']['route_kl']<1e-10 and result['NN']['wo_nmse']<1e-10
    assert abs(result['TT']['route_kl']-result['TN']['route_kl'])<1e-10
    # Functional K/V interaction in output space, before any answer metric.
    cross=outputs['NN']-outputs['NT']-outputs['TN']+outputs['TT']
    denom=outputs['NN'].square().sum(-1).clamp_min(1e-20)
    interaction=float((cross.square().sum(-1)/denom).mean())
    synergy=result['NT']['wo_nmse']+result['TN']['wo_nmse']-result['TT']['wo_nmse']-result['NN']['wo_nmse']
    return dict(cases=result,output_interaction_nmse=interaction,output_damage_synergy=synergy)


@torch.no_grad()
def actual_metrics(C,engine,layer,current,native,translated):
    pair=C._cache_layer(current.cache,layer);reference=C._cache_layer(native.cache,layer)
    result,reconstructed=compare_attention(C,engine,layer,current.q[layer],*pair,current.qpos,
        native.q[layer],*reference)
    # Compare actual model o_proj output too; this includes real Q/tail changes.
    result['actual_wo_nmse']=relative_l2(current.o[layer],native.o[layer])**2
    result['reconstruction_relative_l2']=relative_l2(reconstructed,current.o[layer])
    result['query_native_relative_l2']=relative_l2(current.q[layer],native.q[layer])
    result['query_translated_relative_l2']=relative_l2(current.q[layer],translated.q[layer])
    return result


def functional(C,trace,native,tokens,positions):
    lp=trace.logits.log_softmax(-1);target=torch.tensor([tokens[i] for i in positions],device=lp.device)
    nll=-lp[torch.arange(len(positions),device=lp.device),target]
    ref=native.logits.log_softmax(-1)
    kl=(ref.exp()*(ref-lp)).sum(-1).clamp_min(0)
    result=dict(next_token_kl=float(kl.mean()),target_token_nll=float(nll.mean()),
        per_probe_kl=kl.cpu().tolist(),per_probe_nll=nll.cpu().tolist(),
        native_next_token_agreement=float((lp.argmax(-1)==ref.argmax(-1)).float().mean()),
        probes=len(positions))
    if trace.candidate_ids is not None:
        scores=lp[positions.index(0),trace.candidate_ids]
        result.update(correct_option_log_score=float(scores[trace.gold_index]),
            correct_option_margin=float(scores[trace.gold_index]-max(v for i,v in enumerate(scores) if i!=trace.gold_index)))
    return result


def mean_scalars(values):
    keys=[k for k,v in values[0].items() if isinstance(v,(int,float))]
    return {k:sum(v[k] for v in values)/len(values) for k in keys}
