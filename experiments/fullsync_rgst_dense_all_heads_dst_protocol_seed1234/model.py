"""Independent dense head MLPs with one shared input tensor per layer and KV type."""

import math

import torch
from torch import nn
from torch.nn import functional as F

LAYERS, HEADS, WIDTH = 36, 8, 128
TAPS, HIDDEN = 4, 128
GROUPS = 2 * LAYERS
BLOCKS = GROUPS * HEADS
INPUT = TAPS * HEADS * WIDTH


class DenseRidge:
    def __init__(self, ridge):
        self.ridge = ridge

    def features(self, key, value):
        ridge = self.ridge
        full = torch.stack([
            tensor.cuda()[ridge.taps].permute(0, 2, 1, 3, 4).contiguous().flatten(-3)
            for tensor in (key, value)
        ])
        prediction = full.float() @ ridge.weights + ridge.bias[:, :, None, :]
        packed = prediction.half().unflatten(-1, (HEADS, WIDTH)).transpose(2, 3).flatten(0, 2)
        return full.flatten(0, 1), packed


def input_statistics(stats):
    """Convert head-local [head, tap × width] statistics to Ridge's tap-major layout."""
    return {
        **stats,
        **{
            name: stats[name].reshape(GROUPS, HEADS, TAPS, WIDTH).transpose(1, 2).reshape(GROUPS, 1, INPUT)
            for name in ("center_x", "scale_x")
        },
    }


class DenseAllHeads(nn.Module):
    def __init__(self, rank):
        super().__init__()
        if rank not in (16, 32):
            raise ValueError("Expected rank 16 or 32")
        self.rank = rank
        self.down = nn.Parameter(torch.empty(GROUPS, INPUT, HEADS * HIDDEN, device="cuda"))
        self.bias_down = nn.Parameter(torch.empty(GROUPS, 1, HEADS * HIDDEN, device="cuda"))
        self.up = nn.Parameter(torch.empty(BLOCKS, HIDDEN, rank, device="cuda"))
        self.bias_up = nn.Parameter(torch.empty(BLOCKS, 1, rank, device="cuda"))
        with torch.no_grad():
            for tensor, fan_in in ((self.down, INPUT), (self.bias_down, INPUT),
                                   (self.up, HIDDEN), (self.bias_up, HIDDEN)):
                bound = 1 / math.sqrt(fan_in)
                tensor.uniform_(-bound, bound)

    def coordinates(self, features, stats):
        x = (features.float() - stats["center_x"]) / stats["scale_x"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = F.gelu(x @ self.down + self.bias_down)
            hidden = hidden.unflatten(-1, (HEADS, HIDDEN)).transpose(1, 2).flatten(0, 1)
            return hidden @ self.up + self.bias_up

    def forward(self, features, ridge, stats):
        coordinates = self.coordinates(features, stats)
        basis = stats["axes"][..., :self.rank]
        ridge = ridge.float()
        original = (ridge - stats["center_y"]) @ basis
        return ridge + (coordinates.float() - original) @ basis.transpose(-1, -2)
