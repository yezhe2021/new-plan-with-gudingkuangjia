"""Source-specific Q/O reading with a dynamic bank log-prior.

The physical Full-Sync cache/positions are unchanged. A joint softmax with
bank-specific query and log-prior is algebraically a partition-weighted blend
of separately normalized native/external attention. Zero residuals execute
the same matmuls/softmax as Qwen eager attention (no sigmoid rounding drift).
"""
from contextlib import contextmanager
from types import MethodType
import math
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


class HeadResidual(nn.Module):
    def __init__(self, heads, dim, rank):
        super().__init__()
        self.a = nn.Parameter(torch.empty(heads, dim, rank))
        self.b = nn.Parameter(torch.zeros(heads, rank, dim))
        nn.init.normal_(self.a, std=1 / math.sqrt(dim))

    def forward(self, x):
        return torch.einsum('bhtd,hdr,hrf->bhtf', x, self.a.to(x.dtype), self.b.to(x.dtype))


class ReaderLayer(nn.Module):
    def __init__(self, hidden, heads, dim, rank):
        super().__init__()
        self.query = HeadResidual(heads, dim, rank)
        self.output = HeadResidual(heads, dim, rank)
        self.gate = nn.Sequential(nn.Linear(hidden, rank, bias=False),
                                  nn.Linear(rank, heads, bias=True))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)


class ExternalReader(nn.Module):
    def __init__(self, config, rank=8):
        super().__init__()
        self.layers = nn.ModuleList([ReaderLayer(config.hidden_size, config.num_attention_heads,
                                                 config.head_dim, rank)
                                     for _ in range(config.num_hidden_layers)])
        self.end = None  # external positions are [1, end); anchor 0 stays native
        self.mode = 'full'
        self.originals = []
        self.statistics = None

    @contextmanager
    def scope(self, end, mode=None):
        old = self.end, self.mode
        self.end = end
        if mode is not None:
            self.mode = mode
        try:
            yield
        finally:
            self.end, self.mode = old

    def attach(self, receiver):
        if self.originals:
            raise RuntimeError('Reader already attached')
        owner = self
        for index, layer in enumerate(receiver.model.layers):
            attn = layer.self_attn
            original = attn.forward
            self.originals.append(original)

            def forward(module, hidden_states, position_embeddings, attention_mask,
                        past_key_values=None, _index=index, _original=original, **kwargs):
                if owner.end is None or owner.end <= 1 or owner.mode == 'disabled' or past_key_values is None:
                    return _original(hidden_states=hidden_states, position_embeddings=position_embeddings,
                                     attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
                return owner.attend(module, owner.layers[_index], hidden_states, position_embeddings,
                                    attention_mask, past_key_values, _index)
            attn.forward = MethodType(forward, attn)

    def attend(self, module, branch, hidden, position_embeddings, mask, cache, index):
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        q0 = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
        v = module.v_proj(hidden).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q0, k, cos, sin)
        k, v = cache.update(k, v, module.layer_idx)
        if k.shape[2] < self.end:
            raise RuntimeError('External memory boundary exceeds cache')
        k = repeat_kv(k, module.num_key_value_groups)
        v = repeat_kv(v, module.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(2, 3)) * module.scaling
        memory = slice(1, self.end)
        correction = torch.zeros_like(scores[..., memory])
        if self.mode not in ('gate_only', 'masked'):
            dq = branch.query(q0)
            dq, _ = apply_rotary_pos_emb(dq, torch.zeros_like(dq), cos, sin)
            correction = correction + torch.matmul(dq, k[..., memory, :].transpose(2, 3)) * module.scaling
        if self.mode not in ('read_only', 'masked'):
            gate = branch.gate(hidden.float()).transpose(1, 2).unsqueeze(-1)
            correction = correction + gate.to(scores.dtype)
        # No in-place mutation of tensors needed for backward; original position mask remains exact.
        scores = torch.cat((scores[..., :1], scores[..., memory] + correction, scores[..., self.end:]), -1)
        if mask is not None:
            scores = scores + mask
        if self.mode == 'masked':
            scores = torch.cat((scores[..., :1], torch.full_like(scores[..., memory], -torch.inf),
                                scores[..., self.end:]), -1)
        weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v)
        if self.mode not in ('gate_only', 'masked'):
            external = torch.matmul(weights[..., memory], v[..., memory, :])
            out = out + branch.output(external)
        if self.statistics is not None:
            self.statistics.append({'layer': index, 'external_mass': float(weights[..., memory].float().sum(-1).mean())})
        out = out.transpose(1, 2).contiguous().reshape(*hidden.shape[:-1], -1)
        return module.o_proj(out), weights
