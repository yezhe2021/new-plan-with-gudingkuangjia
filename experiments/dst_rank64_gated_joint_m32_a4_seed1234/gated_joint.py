"""Differentiable gated readout with explicit checkpoint inputs."""

from dataclasses import dataclass

import experiment_common as G
from experiment_common import C, K, ROOT, torch
from heterofold.attention import rotary_cos_sin, rotate_half
from torch.utils.checkpoint import checkpoint

original = G.reference.Z.scoped_module(
    "gated_dynamic_original", ROOT / "initial/gated_memory.py", G,
)


@dataclass(frozen=True)
class AnswerCache:
    body: object
    bank: tuple


class GatedJointMemory(original.GatedDynamicMemory):
    def construct(self, row, translated, *, gradient="both"):
        assert gradient in ("both", "body", "memory")
        length = translated.get_seq_length()
        engine, model = self.engine, self.engine.model
        body = K.flat(translated)
        construction = tuple(x.detach() for x in body) if gradient == "body" else body
        slots = torch.arange(length, length + self.M, device=self.gates.device)

        def build(prompt, *values):
            cache = K.from_values(values, model)
            with self.routing.context("construct", len(engine.prefix_ids), length, self.M, slots):
                result = C.decoder(model)(
                    inputs_embeds=prompt[None].to(model.dtype), past_key_values=cache,
                    attention_mask=torch.ones(1, length + self.M, device=slots.device, dtype=torch.long),
                    position_ids=slots[None], use_cache=True,
                )
            bank = []
            for layer, (key, value) in enumerate(K.pairs(result.past_key_values)):
                assert key.shape[2] == value.shape[2] == length + self.M
                cosine, sine = rotary_cos_sin(model, slots, layer, model.dtype)
                cosine, sine = cosine[None, None].float(), sine[None, None].float()
                rotated = key[:, :, length:].float()
                raw = (rotated * cosine - rotate_half(rotated) * sine) / (cosine.square() + sine.square()).clamp_min(1e-12)
                bank.extend((raw.to(key.dtype), value[:, :, length:].clone()))
            return tuple(bank)

        inputs = (self.memory.prompt, *construction)
        bank = (checkpoint(build, *inputs, use_reentrant=False)
                if self.activation_checkpointing and torch.is_grad_enabled() else build(*inputs))
        answer_body = K.from_values(tuple(x.detach() for x in body), model) if gradient == "memory" else translated
        return AnswerCache(answer_body, bank)

    def logits(self, row, cache, tokens=(), positions=(0,), *, rn=True):
        if not rn:
            return super().logits(row, cache, tokens, positions, rn=False)
        length = cache.body.get_seq_length()
        engine, model = self.engine, self.engine.model
        suffix = row["suffix_ids"]
        assert length == len(engine.prefix_ids) + len(row["content_ids"])
        take = torch.tensor([len(suffix)-1+i for i in positions], device=self.gates.device)
        bank_size = len(cache.bank)

        def forward(gates, *values):
            body = K.from_values(values[bank_size:], model)
            with self.answering(values[:bank_size], gates):
                result = engine.forward(suffix + list(tokens), body, length, use_cache=False)
                return model.lm_head(result.last_hidden_state[0, take]).float()

        inputs = (self.gates, *cache.bank, *K.flat(cache.body))
        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(forward, *inputs, use_reentrant=False)
        return forward(*inputs)
