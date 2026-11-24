"""Independent per-layer and per-head Full-Sync translator branches."""

import math

import torch
from torch import nn

HEADS = 8
WIDTH = 128
LAYERS = 36
BLOCKS = 2 * LAYERS * HEADS
RANK = 16
SPECS = {
    "E1": (512, 106, 128, None),
    "E2": (512, 128, 16, "random"),
    "E3": (512, 128, 16, "residual"),
    "E4": (1024, 128, 16, "residual"),
    "E5": (512, 128, 16, "native"),
}


def pack_source(source, taps):
    features = [x[taps].permute(0, 3, 2, 1, 4).contiguous().flatten(-2) for x in source]
    return torch.stack(features).flatten(0, 2)


def pack_target(target):
    return torch.stack([x.transpose(1, 2) for x in target]).flatten(0, 2)


def unpack_body(value):
    return tuple(x.transpose(1, 2).flatten(-2) for x in value.unflatten(0, (2, LAYERS, HEADS)))


class HeadMLP(nn.Module):
    """One small MLP per layer, KV head and role, with independent parameters."""

    def __init__(self, name, stats):
        super().__init__()
        in_dim, hidden, out_dim, basis = SPECS[name]
        self.basis = stats[basis][..., :RANK] if basis else None
        self.center = stats["center_y"]
        self.wide = in_dim == 1024
        self.down = nn.Parameter(torch.empty(BLOCKS, in_dim, hidden, device="cuda"))
        self.up = nn.Parameter(torch.empty(BLOCKS, hidden, out_dim, device="cuda"))
        self.bias_down = nn.Parameter(torch.empty(BLOCKS, 1, hidden, device="cuda"))
        self.bias_up = nn.Parameter(torch.empty(BLOCKS, 1, out_dim, device="cuda"))
        with torch.no_grad():
            for tensor in (self.down, self.bias_down):
                tensor.uniform_(-1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
            for tensor in (self.up, self.bias_up):
                tensor.uniform_(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden))

    def coordinates(self, source):
        if self.wide:
            opposite = source.roll(BLOCKS // 2, dims=0)
            source = torch.cat((source, opposite), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = torch.nn.functional.gelu(source @ self.down + self.bias_down)
            output = hidden @ self.up + self.bias_up
        return output.float()

    def target_coordinates(self, native):
        centered = native - self.center
        return centered if self.basis is None else centered @ self.basis

    def reconstruct(self, source, ridge):
        coordinates = self.coordinates(source)
        if self.basis is None:
            return coordinates + self.center
        # The fixed easy branch is exactly the Ridge projection onto U_s.
        current_h = (ridge - self.center) @ self.basis
        easy = ridge - current_h @ self.basis.transpose(-1, -2)
        return easy + coordinates @ self.basis.transpose(-1, -2)


def normalized(source, stats):
    return (source.float() - stats["center_x"]) / stats["scale_x"]


def ridge_prediction(source, stats):
    return source @ stats["ridge_w"] + stats["ridge_b"]


def repair(native, ridge, stats, rank):
    basis = stats["residual"][..., :rank]
    return ridge + ((native - ridge) @ basis) @ basis.transpose(-1, -2)
