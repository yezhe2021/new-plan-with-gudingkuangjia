"""Separate self/cross softmax paths; frozen backbone modules are never changed."""
from contextlib import contextmanager
import math
import torch
from torch import nn
import torch.nn.functional as F


class LowRank(nn.Module):
    def __init__(self, source, target, rank, zero_up):
        super().__init__()
        self.down=nn.Linear(source,rank,bias=False)
        self.up=nn.Linear(rank,target,bias=False)
        nn.init.normal_(self.down.weight,std=1/math.sqrt(source))
        if zero_up:nn.init.zeros_(self.up.weight)
        else:nn.init.normal_(self.up.weight,std=1/math.sqrt(rank))
    def forward(self,x):return self.up(self.down(x))


class CrossLayer(nn.Module):
    def __init__(self,attn,rank,initialization):
        super().__init__()
        self.initialization=initialization
        # References remain frozen; checkpoint only the independently trainable parameters.
        self.q_base=attn.q_proj;self.o_base=attn.o_proj;self.q_norm=attn.q_norm
        self.norm=nn.Parameter(torch.ones(attn.q_proj.in_features))
        self.query=LowRank(attn.q_proj.in_features,attn.q_proj.out_features,rank,initialization=='native_query')
        self.output=LowRank(attn.o_proj.in_features,attn.o_proj.out_features,rank,True)
        self.gate=nn.Parameter(torch.zeros(()))
    def forward(self,hidden,key,value):
        # K is canonical post-k_norm/pre-RoPE. Neither external Q nor K receives
        # autoregressive RoPE; the external bank occupies no self-cache positions.
        h=(hidden.float()*torch.rsqrt(hidden.float().square().mean(-1,keepdim=True)+1e-6)*self.norm).to(hidden.dtype)
        q=self.query(h)
        if self.initialization=='native_query':q=q+self.q_base(h)
        q=self.q_norm(q.view(*h.shape[:-1],32,128)).transpose(1,2)
        k=key.transpose(0,1)[None].repeat_interleave(4,dim=1)
        v=value.transpose(0,1)[None].repeat_interleave(4,dim=1)
        z=F.scaled_dot_product_attention(q,k.to(q.dtype),v.to(q.dtype),dropout_p=0.0,is_causal=False)
        z=z.transpose(1,2).reshape(*h.shape[:-1],4096)
        delta=self.o_base(z)+self.output(z)
        return hidden+(self.gate.tanh()*delta).to(hidden.dtype)


class DecoupledReader(nn.Module):
    def __init__(self,model,layers,rank,initialization):
        super().__init__();self.memory=None;self.handles=[]
        self.branches=nn.ModuleDict({str(i):CrossLayer(model.model.layers[i].self_attn,rank,initialization) for i in layers})
        for i in layers:
            def before_norm(module,args,index=i):
                if self.memory is None:return None
                k,v=self.memory
                return (self.branches[str(index)](args[0],k[index],v[index]),)
            # H after the frozen self-attention residual, before frozen MLP.
            self.handles.append(model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(before_norm))
    @contextmanager
    def scope(self,memory):
        previous=self.memory;self.memory=memory
        try:yield
        finally:self.memory=previous
    def detach(self):
        for h in self.handles:h.remove()
        self.handles=[]


def trainable(module):return {n:p for n,p in module.named_parameters() if p.requires_grad}
def state(module):return {n:p.detach().cpu().clone() for n,p in trainable(module).items()}
def restore(module,values):
    pp=trainable(module);assert set(pp)==set(values)
    with torch.no_grad():
        for n,p in pp.items():p.copy_(values[n].to(p))
