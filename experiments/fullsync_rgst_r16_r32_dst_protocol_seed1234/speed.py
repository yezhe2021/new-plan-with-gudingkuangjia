"""Capture frozen Qwen/Llama projections once and append decoding KV in place."""

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


@torch.no_grad()
def capture(legacy, model, tokens, selected):
    backbone = legacy.backbone(model)
    indices = torch.as_tensor(selected, device="cuda", dtype=torch.long)
    captures = ({}, {})
    handles = []

    def save(role, layer, attention):
        def hook(module, inputs, output):
            shape = (1, len(tokens), -1, attention.head_dim)
            captures[role][layer] = output.detach().reshape(shape)[0].index_select(0, indices)
        return hook

    for layer, block in enumerate(backbone.layers):
        attn = block.self_attn
        handles.append(getattr(attn, "k_norm", attn.k_proj).register_forward_hook(save(0, layer, attn)))
        handles.append(attn.v_proj.register_forward_hook(save(1, layer, attn)))
    try:
        x = torch.tensor([tokens], device="cuda")
        backbone(input_ids=x, attention_mask=torch.ones_like(x),
                 position_ids=torch.arange(len(tokens), device="cuda")[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return tuple(torch.stack([values[i] for i in range(len(backbone.layers))]).cpu() for values in captures)


class AppendLayer(DynamicLayer):
    """Expose the same valid-length views as DynamicLayer without copying the prefix per step."""

    def __init__(self, keys, values, extra):
        super().__init__()
        self.lazy_initialization(keys, values)
        self.length = keys.shape[-2]
        self.key_storage = keys.new_empty((*keys.shape[:-2], self.length + extra, keys.shape[-1]))
        self.value_storage = values.new_empty((*values.shape[:-2], self.length + extra, values.shape[-1]))
        self.key_storage[..., :self.length, :].copy_(keys)
        self.value_storage[..., :self.length, :].copy_(values)
        self.keys = self.key_storage[..., :self.length, :]
        self.values = self.value_storage[..., :self.length, :]

    def update(self, key_states, value_states, *args, **kwargs):
        end = self.length + key_states.shape[-2]
        assert end <= self.key_storage.shape[-2]
        self.key_storage[..., self.length:end, :].copy_(key_states)
        self.value_storage[..., self.length:end, :].copy_(value_states)
        self.length = end
        self.keys = self.key_storage[..., :end, :]
        self.values = self.value_storage[..., :end, :]
        return self.keys, self.values


class AppendCache(DynamicCache):
    def __init__(self, config, pairs, extra):
        super().__init__(config=config)
        assert all(type(layer) is DynamicLayer for layer in self.layers)
        self.layers = [AppendLayer(k, v, extra) for k, v in pairs]


@torch.no_grad()
def rollout(engine, row, inputs, cap, C, K):
    model = engine.model
    num = len(inputs)
    cache = AppendCache(model.config, [
        tuple(torch.cat(tensors, dim=0) for tensors in zip(*layers))
        for layers in zip(*(K.pairs(c) for c in inputs))
    ], len(row["suffix_ids"]) + cap)
    length = cache.get_seq_length()
    ids = torch.tensor(row["suffix_ids"], device="cuda").repeat(num, 1)
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos]) | {engine.tok.convert_tokens_to_ids("<|im_end|>")}
    active = [True] * num
    generated = [[] for _ in range(num)]
    for _ in range(cap):
        positions = torch.arange(length, length + ids.shape[1], device=ids.device)
        length += ids.shape[1]
        result = C.decoder(model)(ids, past_key_values=cache, use_cache=True,
            attention_mask=torch.ones((num, length), device=ids.device, dtype=torch.long),
            position_ids=positions[None].expand(num, -1))
        next_ids = model.lm_head(result.last_hidden_state[:, -1]).float().argmax(-1)
        for i, token in enumerate(next_ids.tolist()):
            if active[i]:
                generated[i].append(token)
                active[i] = token not in stops
        if not any(active):
            break
        cache = result.past_key_values
        ids = next_ids[:, None]
    return generated
