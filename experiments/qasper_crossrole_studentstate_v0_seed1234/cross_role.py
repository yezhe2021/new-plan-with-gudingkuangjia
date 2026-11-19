import torch
from torch import nn


class RoleResidual(nn.Module):
    """Per-token/head three-neighbour source residual. No token mixing or K renorm."""
    def __init__(self, mode='cross', rank=32, source_layers=28, layers=36, heads=8, dim=128):
        super().__init__()
        assert mode in ('self', 'cross')
        self.mode, self.layers, self.heads = mode, layers, heads
        self.neighbours = [tuple(max(0, min(source_layers-1, round(l*(source_layers-1)/(layers-1))+d))
            for d in (-1,0,1)) for l in range(layers)]
        def branch():
            m = nn.Sequential(nn.Linear(3*dim, rank, bias=False), nn.GELU(), nn.Linear(rank, dim, bias=False))
            nn.init.zeros_(m[-1].weight)
            return m
        self.k = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)]) for _ in range(layers)])
        self.v = nn.ModuleList([nn.ModuleList([branch() for _ in range(heads)]) for _ in range(layers)])

    def _delta(self, x, modules):
        output = []
        for l in range(self.layers):
            selected = x[:, self.neighbours[l]].permute(0,2,3,1,4).flatten(-2)
            output.append(torch.stack([modules[l][h](selected[:,:,h]) for h in range(self.heads)], dim=2))
        return torch.stack(output, dim=1)

    def forward(self, sk, sv, k0, v0):
        xk,xv = (sk,sv) if self.mode == 'self' else (sv,sk)
        return k0+self._delta(xk,self.k), v0+self._delta(xv,self.v)


class AugmentedMapper(nn.Module):
    def __init__(self, baseline, residual):
        super().__init__()
        self.baseline, self.residual = baseline, residual

    def forward(self, k, v):
        with torch.no_grad(): k0,v0 = self.baseline(k,v)
        return self.residual(k,v,k0,v0)
