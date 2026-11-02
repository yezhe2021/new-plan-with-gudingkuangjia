"""Differentiable cache copies; never relocate/renormalize memory keys."""
from common import C,torch

def pairs(cache):return [C._cache_layer(cache,l) for l in range(len(cache.layers))]
def clone(cache,model,detach=False):
    return C.DynamicCache(ddp_cache_data=[tuple((x.detach() if detach else x).clone() for x in p) for p in pairs(cache)],config=model.config)
def from_values(values,model):
    return C.DynamicCache(ddp_cache_data=[(values[i].clone(),values[i+1].clone()) for i in range(0,len(values),2)],config=model.config)
def flat(cache):return tuple(x for p in pairs(cache) for x in p)
def append(original,memory,model):
    pp=pairs(original);assert len(memory)==2*len(pp)
    return C.DynamicCache(ddp_cache_data=[(torch.cat([k.detach(),memory[2*l]],2),torch.cat([v.detach(),memory[2*l+1]],2)) for l,(k,v) in enumerate(pp)],config=model.config)
