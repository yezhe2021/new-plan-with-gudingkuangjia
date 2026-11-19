"""Independent attention over a sample-conditioned Qwen Memory bank."""

from contextlib import contextmanager
from dataclasses import dataclass

from experiment_common import C, K, torch
from heterofold.attention import rotary_cos_sin, rotate_half
from memory_dynamic import DynamicMemory
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class AnswerCache:
    body: object
    bank: tuple


class GatedDynamicMemory(DynamicMemory):
    def __init__(self, engine, config):
        super().__init__(engine, config)
        blocks = C.layers_of(engine.model)
        self.gates = torch.nn.Parameter(torch.zeros(len(blocks), device=self.memory.prompt.device))
        self._answer = None
        self._queries = {}
        self._handles = []
        for index, block in enumerate(blocks):
            attention = block.self_attn

            def capture(_module, _args, normalized, layer=index):
                if self._answer is not None:
                    self._queries[layer] = normalized.transpose(1, 2)

            def combine(_module, args, layer=index, groups=attention.num_key_value_groups, scale=attention.scaling):
                if self._answer is None:
                    return None
                bank, gates = self._answer
                query = self._queries.pop(layer)
                key, value = bank[2 * layer:2 * layer + 2]
                attended = scaled_dot_product_attention(
                    query, key.repeat_interleave(groups, dim=1),
                    value.repeat_interleave(groups, dim=1), scale=scale,
                )
                body = args[0]
                attended = attended.transpose(1, 2).reshape_as(body)
                return ((body.float() + gates[layer].tanh() * attended.float()).to(body.dtype),)

            self._handles.append(attention.q_norm.register_forward_hook(capture))
            self._handles.append(attention.o_proj.register_forward_pre_hook(combine))

    @contextmanager
    def answering(self, bank, gates):
        assert self._answer is None and not self._queries
        self._answer = (bank, gates)
        try:
            yield
        finally:
            self._answer = None
            self._queries.clear()

    def construct(self, row, translated):
        length = translated.get_seq_length()
        model = self.engine.model
        constructed = super().construct(row, translated)
        positions = torch.arange(length, length + self.M, device=self.memory.prompt.device)
        bank = []
        for layer, (key, value) in enumerate(K.pairs(constructed)):
            cosine, sine = rotary_cos_sin(model, positions, layer, model.dtype)
            cosine, sine = cosine[None, None].float(), sine[None, None].float()
            rotated = key[:, :, length:, :].float()
            # Memory already passed k_norm. Undo its construction-time RoPE.
            raw = (rotated * cosine - rotate_half(rotated) * sine) / (cosine.square() + sine.square()).clamp_min(1e-12)
            bank.extend((raw.to(key.dtype), value[:, :, length:, :].clone()))
        return AnswerCache(translated, tuple(bank))

    def logits(self, row, cache, tokens=(), positions=(0,), *, rn=True):
        if not rn:
            return super().logits(row, cache, tokens, positions, rn=False)

        def forward(gates, *bank):
            with self.answering(bank, gates):
                return super(GatedDynamicMemory, self).logits(row, cache.body, tokens, positions, rn=False)

        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(forward, self.gates, *cache.bank, use_reentrant=False)
        return forward(self.gates, *cache.bank)

    @torch.no_grad()
    def rollout(self, row, cache, *, rn, cap=64):
        if not rn:
            return super().rollout(row, cache, rn=False, cap=cap)
        with self.answering(cache.bank, self.gates):
            return super().rollout(row, cache.body, rn=False, cap=cap)
