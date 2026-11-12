"""Independent Static Memory attention with the original Qwen body forward."""

from contextlib import contextmanager
from dataclasses import dataclass

from gated_common import C, CFG, ROOT, identity, reference, torch
from engine_frozen import Engine
from static_memory import StaticMemory, rephase
from heterofold.attention import rotary_cos_sin
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class AnswerCache:
    body: object
    bank: tuple


class GatedMemory(StaticMemory):
    def __init__(self, engine, config=CFG):
        super().__init__(engine, config)
        blocks = C.layers_of(engine.model)
        self.gates = torch.nn.Parameter(torch.zeros(len(blocks), device=self.memory.prompt.device))
        self._answer = None
        self._queries = {}
        self._bank = None
        self._bank_source = None
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
                memory = scaled_dot_product_attention(
                    query, key.repeat_interleave(groups, dim=1),
                    value.repeat_interleave(groups, dim=1), scale=scale,
                )
                body = args[0]
                memory = memory.transpose(1, 2).reshape_as(body)
                # Keep the residual sum in FP32 until the original o_proj input.
                combined = body.float() + gates[layer].tanh() * memory.float()
                return (combined.to(body.dtype),)

            self._handles.append(attention.q_norm.register_forward_hook(capture))
            self._handles.append(attention.o_proj.register_forward_pre_hook(combine))

    @contextmanager
    def answering(self, bank, gates):
        assert self._answer is None and not self._queries
        self._answer = (bank, gates)
        try:
            yield
        finally:
            # Non-reentrant checkpointing may stop the recomputation early.
            self._answer = None
            self._queries.clear()

    def construct(self, row, translated):
        native = self.canonical()
        reuse = not self.training and not torch.is_grad_enabled()
        if reuse and native is self._bank_source:
            return AnswerCache(translated, self._bank)
        start = len(self.engine.prefix_ids)
        model = self.engine.model
        positions = torch.arange(start, start + self.M, device=self.memory.prompt.device)
        bank = []
        for layer in range(len(self.gates)):
            cos, sin = rotary_cos_sin(model, positions, layer, model.dtype)
            # Q and K both omit answer-side RoPE in this branch. Qwen's
            # canonical Memory construction still uses its ordinary RoPE.
            key = rephase(native[2 * layer], cos, sin, torch.ones_like(cos), torch.zeros_like(sin))
            bank.extend((key, native[2 * layer + 1]))
        bank = tuple(bank)
        if reuse:
            self._bank_source, self._bank = native, bank
        return AnswerCache(translated, bank)

    def logits(self, row, cache, tokens=(), positions=(0,), *, rn=True):
        if not rn:
            return super().logits(row, cache, tokens, positions, rn=False)

        def forward(gates, *bank):
            with self.answering(bank, gates):
                return super(GatedMemory, self).logits(row, cache.body, tokens, positions, rn=False)

        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(forward, self.gates, *cache.bank, use_reentrant=False)
        return forward(self.gates, *cache.bank)

    @torch.no_grad()
    def rollout(self, row, cache, *, rn, cap=64):
        if not rn:
            return super().rollout(row, cache, rn=False, cap=cap)
        with self.answering(cache.bank, self.gates):
            return super().rollout(row, cache.body, rn=False, cap=cap)


def load(final=False):
    engine = Engine()
    module = GatedMemory(engine)
    assert len(module.gates) == 36 and sum(p.numel() for p in module.parameters()) == 409636
    if final:
        value = torch.load(ROOT / "runs/epoch4_step1536.pt", map_location="cpu", weights_only=True)
        assert value["manifest_sha256"] == identity() and value["step"] == CFG["updates"]
        module.load_state_dict(value["state"], strict=True)
    return engine, engine.correction, module
