"""Only prompt embeddings and construction-only post-block residuals train."""
from common import torch
from torch import nn

class Memory(nn.Module):
    def __init__(self,hidden,tokens,rank,after,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.prompt=nn.Parameter(torch.randn(tokens,hidden)*.02)
            self.adapters=nn.ModuleDict({str(l):nn.Sequential(nn.Linear(hidden,rank,bias=False),nn.GELU(),nn.Linear(rank,hidden,bias=False)) for l in after})
            for block in self.adapters.values():nn.init.zeros_(block[-1].weight)

    def adapt(self,layer,hidden):
        adapter=self.adapters[str(layer)]
        return (hidden.float()+adapter(hidden.float())).to(hidden.dtype)
