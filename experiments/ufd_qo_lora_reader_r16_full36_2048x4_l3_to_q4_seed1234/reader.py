"""Continuation-only Q/O LoRA; fixed physical KV is never rewritten."""
import math
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F


class LoRAProjection(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = False
        self.enabled_calls = 0
        self.enabled_tokens = 0

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        self.enabled_calls += 1
        self.enabled_tokens += x.numel() // x.shape[-1]
        # Generation does not use autocast. Keep small LoRA matmuls in FP32,
        # with FP32 master weights and a final cast to the existing model dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return original + delta.to(original.dtype)


class QOReader:
    """Plain controller: state/optimizer contain ONLY LoRA A/B, never base weights."""
    def __init__(self, model, rank=16, alpha=16, layers=36):
        assert len(model.model.layers) == layers
        self.branches = {}
        for i, layer in enumerate(model.model.layers):
            for target in ['q_proj', 'o_proj']:
                wrapped = LoRAProjection(getattr(layer.self_attn, target), rank, alpha)
                setattr(layer.self_attn, target, wrapped)
                self.branches[f'layers.{i}.{target}'] = wrapped
        self.disable()

    def parameters(self):
        return [p for module in self.branches.values() for p in [module.a, module.b]]

    def disable(self):
        for module in self.branches.values():
            module.enabled = False

    @contextmanager
    def mode(self, enabled):
        previous = [m.enabled for m in self.branches.values()]
        for module in self.branches.values():
            module.enabled = enabled
        try:
            yield
        finally:
            for module, old in zip(self.branches.values(), previous):
                module.enabled = old

    def state_dict(self):
        return {name + '.' + role: getattr(module, role).detach().cpu().clone()
                for name, module in self.branches.items() for role in ['a', 'b']}

    def load_state_dict(self, state):
        expected = {name + '.' + role for name in self.branches for role in ['a', 'b']}
        assert set(state) == expected
        with torch.no_grad():
            for name, module in self.branches.items():
                for role in ['a', 'b']:
                    getattr(module, role).copy_(state[name + '.' + role])

    def assert_trainable(self, model, base, writer):
        allowed = {id(p) for p in self.parameters()}
        assert {id(p) for p in model.parameters() if p.requires_grad} == allowed
        assert not any(p.requires_grad or p.grad is not None for p in base.parameters())
        assert not any(p.requires_grad or p.grad is not None for p in writer.parameters())
        assert all(p.grad is None for p in model.parameters() if id(p) not in allowed)

    def gradient_norms(self):
        return {target: sum(float(p.grad.detach().float().square().sum())
                            for name, module in self.branches.items() if name.endswith(target)
                            for p in [module.a, module.b] if p.grad is not None) ** 0.5
                for target in ['q_proj', 'o_proj']}


def cpu_test():
    torch.manual_seed(1234)
    base = nn.Linear(11, 17, bias=True).requires_grad_(False)
    layer = LoRAProjection(base, rank=3, alpha=3)
    x = torch.randn(2, 5, 11)
    original = layer(x).detach().clone()
    layer.enabled = True
    assert torch.equal(original, layer(x))
    opt = torch.optim.AdamW([layer.a, layer.b], lr=1e-3, weight_decay=0)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        layer(x).square().mean().backward()
        assert torch.isfinite(layer.b.grad).all() and layer.b.grad.norm() > 0
        assert base.weight.grad is None and base.bias.grad is None
        opt.step()
    assert layer.a.grad.norm() > 0
    assert not torch.equal(original, layer(x))
    layer.enabled = False
    assert torch.equal(original, layer(x))
    return dict(zero_init_bitwise_identity=True, disabled_after_training_bitwise_identity=True,
                frozen_base_no_grad=True, finite_nonzero_lora_gradient=True)
