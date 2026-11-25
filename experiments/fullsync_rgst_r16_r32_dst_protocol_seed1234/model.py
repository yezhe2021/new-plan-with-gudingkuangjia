"""Frozen cross-head Full-Sync Ridge and trainable nested-subspace neural writers."""

import math

import torch
from torch import nn

LAYERS, HEADS, WIDTH = 36, 8, 128
BLOCKS = 2 * LAYERS * HEADS
INPUT, HIDDEN = 4 * WIDTH, 128
CHUNK = 128


def pack_native(source):
    return torch.stack([v.transpose(1, 2) for v in source]).flatten(0, 2)


def unpack_body(body):
    return tuple(v.transpose(1, 2).flatten(-2) for v in body.unflatten(0, (2, LAYERS, HEADS)))


class FrozenRidge:
    def __init__(self, payload):
        self.taps = torch.tensor(payload["taps"], dtype=torch.long, device="cuda")
        self.weights = payload["weight"].cuda()
        self.bias = payload["bias"].cuda()
        assert self.weights.shape == (2, LAYERS, 4096, 1024)
        assert self.bias.shape == (2, LAYERS, 1024) and self.taps.shape == (LAYERS, 4)

    def features(self, key, value):
        gathered = [t.cuda()[self.taps].permute(0, 2, 1, 3, 4).contiguous() for t in (key, value)]
        neural = torch.stack([z.permute(0, 3, 1, 2, 4).flatten(-2) for z in gathered]).flatten(0, 2)
        full = torch.stack([z.flatten(-3) for z in gathered]).float()
        prediction = full @ self.weights + self.bias[:, :, None, :]
        prediction = prediction.half().unflatten(-1, (HEADS, WIDTH)).transpose(2, 3).flatten(0, 2)
        return neural, prediction


class RGST(nn.Module):
    def __init__(self, rank):
        super().__init__()
        assert rank in (16, 32)
        self.rank = rank
        self.down = nn.Parameter(torch.empty(BLOCKS, INPUT, HIDDEN, device="cuda"))
        self.up = nn.Parameter(torch.empty(BLOCKS, HIDDEN, rank, device="cuda"))
        self.bias_down = nn.Parameter(torch.empty(BLOCKS, 1, HIDDEN, device="cuda"))
        self.bias_up = nn.Parameter(torch.empty(BLOCKS, 1, rank, device="cuda"))
        with torch.no_grad():
            for p in (self.down, self.bias_down):
                p.uniform_(-1 / math.sqrt(INPUT), 1 / math.sqrt(INPUT))
            for p in (self.up, self.bias_up):
                p.uniform_(-1 / math.sqrt(HIDDEN), 1 / math.sqrt(HIDDEN))

    def forward(self, features, ridge, stats):
        x = (features.float() - stats["center_x"]) / stats["scale_x"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = torch.nn.functional.gelu(x @ self.down + self.bias_down)
            z = hidden @ self.up + self.bias_up
        basis = stats["axes"][..., :self.rank]
        current = (ridge.float() - stats["center_y"]) @ basis
        return ridge.float() + (z.float() - current) @ basis.transpose(-1, -2)


def oracle(ridge, native, stats, rank):
    basis = stats["axes"][..., :rank]
    return ridge.float() + ((native.float() - ridge.float()) @ basis) @ basis.transpose(-1, -2)
