"""Per-head Gaussian shrinkage with fixed or prediction-dependent error variance."""

import torch


class TargetMoments:
    def __init__(self):
        self.count = 0
        self.total = 0
        self.cross = 0

    def add(self, native):
        self.count += native.shape[1]
        self.total = self.total + native.sum(dim=1)
        self.cross = self.cross + native.transpose(1, 2) @ native

    def finish(self, rank):
        center = self.total / self.count
        cov = self.cross / self.count - center.unsqueeze(1) * center.unsqueeze(2)
        cov = (cov + cov.transpose(-1, -2)) / 2
        values, vectors = torch.linalg.eigh(cov.double())
        basis = (vectors[..., -rank:] * values[..., None, -rank:].clamp_min(0).sqrt()).float()
        low = 0.9 * (basis @ basis.transpose(-1, -2))
        variance = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(0)
        variance = variance + 0.001 * variance.mean(-1, keepdim=True).clamp_min(1e-6)
        diagonal = variance - torch.diagonal(low, dim1=-2, dim2=-1)
        return center, variance, low + torch.diag_embed(diagonal)


def gate_score(values, center, variance):
    return ((values - center[:, None, :]).square() / variance[:, None, :]).mean(-1)


def bins(values, edges):
    return (values.unsqueeze(-1) > edges.unsqueeze(1)).sum(-1)


class ErrorMoments:
    def __init__(self, center, variance, edges):
        self.center, self.variance, self.edges = center, variance, edges
        shape = (center.shape[0], edges.shape[1] + 1)
        self.counts = center.new_zeros(shape)
        self.sums = center.new_zeros((*shape, center.shape[-1]))

    def add(self, prediction, native):
        group = bins(gate_score(prediction, self.center, self.variance), self.edges)
        mask = torch.nn.functional.one_hot(group, self.counts.shape[-1]).transpose(1, 2).float()
        self.counts += mask.sum(-1)
        self.sums += mask @ (prediction - native).square()

    def finish(self):
        fixed = self.sums.sum(1) / self.counts.sum(1, keepdim=True)
        fixed = fixed.maximum(1e-4 * self.variance)
        # Regularize each bucket with the equivalent of 256 pooled residuals.
        adaptive = (self.sums + 256 * fixed[:, None]) / (self.counts[..., None] + 256)
        return fixed, adaptive.maximum(1e-4 * self.variance[:, None])


def prepare_maps(parameters, strengths):
    center, variance, cov, edges, fixed, adaptive = parameters
    maps = {}
    designs = (("fixed_lowrank", cov, fixed[:, None]),
               ("adaptive_diagonal", torch.diag_embed(variance), adaptive),
               ("adaptive_lowrank", cov, adaptive))
    for name, prior, noise in designs:
        prior = prior[:, None].double()
        noise = torch.diag_embed(noise.double())
        for strength in strengths:
            # Vectors are rows. solve(A, B) computes (Sigma + lambda R)^-1 Sigma.
            maps[name, strength] = torch.linalg.solve(prior + strength * noise, prior).float()
    return maps


def apply_map(values, center, weights, group=None):
    centered = values - center.unsqueeze(1)
    if weights.shape[1] == 1:
        result = centered @ weights[:, 0]
    else:
        assert group is not None
        result = torch.zeros_like(values)
        for index in range(weights.shape[1]):
            result += (centered @ weights[:, index]) * (group == index).unsqueeze(-1)
    return result + center.unsqueeze(1)
