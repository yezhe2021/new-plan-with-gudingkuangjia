"""Token-local, group-local KV translator with dense cross-head connections."""
import torch
from torch import nn


class GroupedKVTranslator(nn.Module):
    def __init__(self, source_layers=28, target_layers=36, source_heads=8,
                 target_heads=8, source_dim=128, target_dim=128, groups=4,
                 hidden_dim=1024):
        super().__init__()
        sizes = (source_layers, target_layers, source_heads, target_heads,
                 source_dim, target_dim, groups, hidden_dim)
        if any(not isinstance(v, int) or v <= 0 for v in sizes):
            raise ValueError("All dimensions must be positive integers")
        if source_layers % groups or target_layers % groups:
            raise ValueError("Version 1 requires equal, nonempty contiguous groups")
        self.source_layers, self.target_layers = source_layers, target_layers
        self.source_heads, self.target_heads = source_heads, target_heads
        self.source_dim, self.target_dim = source_dim, target_dim
        self.groups, self.hidden_dim = groups, hidden_dim
        self.source_per_group = source_layers // groups
        self.target_per_group = target_layers // groups
        input_width = self.source_per_group * source_heads * source_dim
        output_width = target_heads * target_dim
        # Each role has its own group encoder and independent target-layer heads.
        self.k_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(input_width, hidden_dim, bias=False), nn.GELU())
            for _ in range(groups)])
        self.v_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(input_width, hidden_dim, bias=False), nn.GELU())
            for _ in range(groups)])
        self.k_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, output_width, bias=False) for _ in range(target_layers)])
        self.v_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, output_width, bias=False) for _ in range(target_layers)])

    def _map(self, tensor, encoders, outputs):
        batch, _, tokens, _, _ = tensor.shape
        result = []
        for group in range(self.groups):
            start = group * self.source_per_group
            # [B,Lg,T,H,D] -> [B,T,Lg*H*D]: layers AND heads enter the same Linear.
            x = tensor[:, start:start + self.source_per_group].permute(0, 2, 1, 3, 4).flatten(2)
            hidden = encoders[group](x)
            for target in range(group * self.target_per_group,
                                (group + 1) * self.target_per_group):
                result.append(outputs[target](hidden).reshape(
                    batch, tokens, self.target_heads, self.target_dim))
        return torch.stack(result, dim=1)

    def forward(self, key, value):
        if (key.ndim != 5 or key.shape != value.shape or key.shape[2] == 0
                or key.shape[1] != self.source_layers
                or key.shape[3:] != (self.source_heads, self.source_dim)):
            raise ValueError("Expected matching [batch,source_layers,tokens,source_heads,source_dim]")
        return (self._map(key, self.k_encoders, self.k_outputs),
                self._map(value, self.v_encoders, self.v_outputs))
