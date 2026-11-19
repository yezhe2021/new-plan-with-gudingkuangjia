import math
import torch
import torch.nn.functional as F
from build_cache import rope,normalized_key
from common import CFG


def probability(q,k,positions,scale):
    repeat=q.shape[1]//k.shape[1]
    scores=torch.einsum('phd,thd->hpt',q.float(),k.float().repeat_interleave(repeat,1))*scale
    masked=torch.arange(len(k),device=k.device)[None]>positions[:,None]
    scores=scores.masked_fill(masked[None],-torch.inf)
    return scores.softmax(-1),scores.log_softmax(-1).masked_fill(masked[None],0)


def outputs(p,v,weight):
    perhead=torch.einsum('hpt,thd->phd',p.float(),v.float().repeat_interleave(p.shape[0]//v.shape[1],1))
    return F.linear(perhead.flatten(1),weight.float()),perhead


def per_head_key_error(delta,weight):
    # weight=[d_model,Hq*d_head], delta=[probes,Hq,d_head].
    # DO NOT sum heads before square: cross-head cancellation changes Eq8.
    h,d=delta.shape[1:]
    blocks=weight.float().reshape(weight.shape[0],h,d).permute(1,0,2)
    projected=torch.einsum('phd,hmd->phm',delta.float(),blocks)
    return projected.square().mean()  # divide by d_model, average p and h


def objectives(model,layer,native,proto,mapped_k,mapped_v,k_corr,v_corr,epsilon=1e-8):
    attn=model.model.layers[layer].self_attn; device=next(model.parameters()).device
    pos=torch.tensor(proto['probe_positions'],device=device); allpos=torch.arange(len(native['k'][layer]),device=device)
    q=rope(model,native['q'][layer].to(device),pos).detach()
    nk=rope(model,normalized_key(model,layer,native['k'][layer].to(device)),allpos).detach()
    nv=native['v'][layer].to(device).float().detach()
    n=proto['head']; end=len(proto['ids'])
    # Each correction mixes all target KV heads in raw receiver space.
    # Quantize only final corrected features: folded inference has the same boundary.
    mk=k_corr(mapped_k.float()).to(getattr(torch,CFG['mapped_feature_dtype'])).reshape(-1,8,128)
    mv=v_corr(mapped_v.float()).to(getattr(torch,CFG['mapped_feature_dtype'])).reshape(-1,8,128)
    sk=torch.cat((nk[:n],rope(model,normalized_key(model,layer,mk),allpos[n:end])),0)
    sv=torch.cat((nv[:n],mv),0)
    scale=getattr(attn,'scaling',attn.head_dim**-.5)
    tp,_=probability(q,nk,pos,scale); p,logp=probability(q,sk,pos,scale)
    weight=attn.o_proj.weight.detach(); assert attn.o_proj.bias is None
    target,_=outputs(tp,nv,weight)
    _,delta=outputs(p-tp,nv,weight)
    kl=F.kl_div(logp,tp,reduction='sum')/(p.shape[0]*p.shape[1])
    ke=per_head_key_error(delta,weight)
    actual,_=outputs(p.detach(),sv,weight)
    ve=(actual-target.detach()).square().sum()/target.detach().square().sum().clamp_min(epsilon)
    return kl+ke,ve,{'route_kl':float(kl.detach()),'key_head_output_mse':float(ke.detach()),'value_output_nmse':float(ve.detach())}
