"""Frozen grouped translator plus per-layer cross-head residuals."""
import torch
from torch import nn


class CrossHeadResidual(nn.Module):
    def __init__(self, layers=36, heads=8, dim=128, rank=64):
        super().__init__()
        self.layers, self.heads, self.dim, self.rank = layers, heads, dim, rank
        width = heads * dim
        def branch():
            network = nn.Sequential(nn.Linear(width, rank, bias=False), nn.GELU(),
                                    nn.Linear(rank, width, bias=False))
            nn.init.zeros_(network[-1].weight)
            return network
        self.k = nn.ModuleList([branch() for _ in range(layers)])
        self.v = nn.ModuleList([branch() for _ in range(layers)])

    def _map(self, tensor, networks):
        return torch.stack([tensor[:, layer] + network(tensor[:, layer].flatten(-2)).reshape_as(tensor[:, layer])
                            for layer, network in enumerate(networks)], dim=1)

    def forward(self, key, value):
        if (key.ndim != 5 or key.shape != value.shape or key.shape[1] != self.layers
                or key.shape[3:] != (self.heads, self.dim)):
            raise ValueError('Expected matching [B,target_layers,T,heads,dim] tensors')
        return self._map(key, self.k), self._map(value, self.v)


class FrozenTranslatorWithResidual(nn.Module):
    def __init__(self, base, rank=64):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.residual = CrossHeadResidual(base.target_layers, base.target_heads, base.target_dim, rank)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def assert_frozen(self):
        assert not self.base.training
        assert all(not p.requires_grad and p.grad is None for p in self.base.parameters())

    def forward(self, key, value):
        self.assert_frozen()
        with torch.no_grad():
            key, value = self.base(key, value)
        return self.residual(key, value)
