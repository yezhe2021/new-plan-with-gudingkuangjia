"""Memory-slot-only residuals and matched Native/translated hidden-state capture."""
from contextlib import contextmanager

import torch
from torch import nn

LAYERS = (6, 12, 18, 24)
ADAPTER_RANK = 16
HIDDEN_WEIGHT = 0.1
SEED = 1234


class Recovery(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.blocks = nn.ModuleDict({str(l): nn.Sequential(nn.Linear(hidden, ADAPTER_RANK, bias=False), nn.GELU(),
                                                              nn.Linear(ADAPTER_RANK, hidden, bias=False)) for l in LAYERS})
        for block in self.blocks.values():
            nn.init.zeros_(block[-1].weight)
        self._hidden = None
        self._teacher = False

    @contextmanager
    def capture(self, *, teacher):
        assert self._hidden is None
        saved = {}
        self._hidden, self._teacher = saved, teacher
        try:
            yield saved
        finally:
            self._hidden, self._teacher = None, False

    def forward(self, layer, hidden):
        assert layer in LAYERS and hidden.shape[1] == 32
        if not self._teacher:
            delta = self.blocks[str(layer)](hidden.float())
            hidden = (hidden.float() + delta).to(hidden.dtype)
        if self._hidden is not None:
            self._hidden[layer] = hidden.detach() if self._teacher else hidden
        return hidden


def attach(memory):
    memory.requires_grad_(False)
    width = memory.memory.prompt.shape[-1]
    assert memory.M == 32 and set(memory.memory.adapters) == {str(layer) for layer in LAYERS}
    original_adapt = memory.memory.adapt
    memory.recovery = Recovery(width).to(memory.memory.prompt.device)

    def adapt(layer, hidden):
        return memory.recovery(layer, original_adapt(layer, hidden))

    memory.memory.adapt = adapt
    return memory


def hidden_loss(student, teacher):
    assert set(student) == set(teacher) == set(LAYERS)
    errors = []
    for l in LAYERS:
        output, target = student[l].float(), teacher[l].float()
        assert output.shape == target.shape and not target.requires_grad
        errors.append((output-target).square().mean() / target.square().mean().clamp_min(1e-8))
    return torch.stack(errors).mean()
