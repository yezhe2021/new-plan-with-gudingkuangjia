import torch
from transformers import DynamicCache


def rope(model,tensor,positions):
    # FP32 arithmetic even if stored/mapped features are BF16.
    x=tensor.float(); positions=positions.to(x.device)[None]
    cos,sin=model.model.rotary_emb(x[None],positions)
    a,b=x.chunk(2,-1)
    return x*cos[0,:,None].float()+torch.cat((-b,a),-1)*sin[0,:,None].float()


def normalized_key(model,layer,raw):
    attn=model.model.layers[layer].self_attn
    x=raw.to(next(model.parameters()).dtype)
    return attn.k_norm(x) if hasattr(attn,'k_norm') else x


def build_cache(model,raw_k,value,positions):
    """Only accepts raw mapped K/V, never reads native evidence."""
    dtype=next(model.parameters()).dtype; pairs=[]
    for layer in range(len(raw_k)):
        k=rope(model,normalized_key(model,layer,raw_k[layer]),positions).to(dtype)
        pairs.append((k.permute(1,0,2)[None],value[layer].to(dtype).permute(1,0,2)[None]))
    return DynamicCache(ddp_cache_data=pairs,config=model.config)
