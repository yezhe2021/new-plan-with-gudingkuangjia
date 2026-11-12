"""Only the bias-free 9,437,184 adapter weights are registered/trainable/saved."""
import torch
from torch import nn
class ResidualTranslator(nn.Module):
    def __init__(self,base,layers=36,width=1024,rank=64,seed=1234):
        super().__init__();object.__setattr__(self,'base',base)
        self.base.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.keys=nn.ModuleList(nn.Sequential(nn.Linear(width,rank,bias=False),nn.GELU(),nn.Linear(rank,width,bias=False)) for _ in range(layers))
            self.values=nn.ModuleList(nn.Sequential(nn.Linear(width,rank,bias=False),nn.GELU(),nn.Linear(rank,width,bias=False)) for _ in range(layers))
            for net in [*self.keys,*self.values]:nn.init.zeros_(net[-1].weight)
    def forward(self,source_k,source_v,indices):
        with torch.no_grad():k,v=self.base(source_k,source_v,indices)
        return (torch.stack([x+self.keys[l](x) for l,x in enumerate(k)]),torch.stack([x+self.values[l](x) for l,x in enumerate(v)]))
