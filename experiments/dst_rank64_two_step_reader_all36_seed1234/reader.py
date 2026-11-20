"""Two conditional reads of immutable, post-RoPE DST body KV."""
from contextlib import contextmanager
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from engine import BodyReader
from experiment_common import C, K


class ReadOnlyCache(C.DynamicCache):
    """Supply normal self-attention keys without changing checkpoint inputs."""

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        past = self.layers[layer_idx]
        return torch.cat((past.keys, key_states), dim=2), torch.cat((past.values, value_states), dim=2)


class LowRank(nn.Module):
    def __init__(self, source, target, rank):
        super().__init__()
        self.down = nn.Linear(source, rank, bias=False)
        self.up = nn.Linear(rank, target, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        hidden = F.linear(hidden, self.down.weight.to(hidden.dtype))
        return F.linear(hidden, self.up.weight.to(hidden.dtype))


class TwoReads(nn.Module):
    def __init__(self, attention, rank, initial_scale):
        super().__init__()
        self.query = LowRank(attention.q_proj.in_features, attention.q_proj.out_features, rank)
        self.output = LowRank(attention.o_proj.in_features, attention.o_proj.out_features, rank)
        self.scales = nn.Parameter(torch.full((2,), initial_scale))

    def read(self, block, hidden, body, rotation):
        attention = block.self_attn
        normalized = block.post_attention_layernorm(hidden)
        projected = attention.q_proj(normalized) + self.query(normalized)
        query = attention.q_norm(projected.unflatten(-1, (-1, attention.head_dim))).transpose(1, 2)
        cosine, sine = (item[:, None] for item in rotation)
        query = (query * cosine + rotate_half(query) * sine).contiguous()
        key, value = body
        readout = F.scaled_dot_product_attention(query, key, value, scale=attention.scaling, enable_gqa=True)
        readout = readout.transpose(1, 2).flatten(2)
        return attention.o_proj(readout) + self.output(readout)

    def forward(self, block, hidden, body, rotation, strength):
        for index in range(2):
            delta = self.read(block, hidden, body, rotation)
            hidden = (hidden.float() + strength * self.scales[index].tanh() * delta.float()).to(hidden.dtype)
        return hidden


class Reader(nn.Module, BodyReader):
    M = 0

    def __init__(self, engine, config):
        super().__init__()
        self.engine = engine
        model = engine.model
        self.routing = None
        self.body = None
        self.strength = 1.0
        self.timing = None
        self.activation_checkpointing = True
        blocks = C.layers_of(model)
        assert len(blocks) == config["receiver_layers"] and model.config.model_type == "qwen3"
        self.readers = nn.ModuleList(
            TwoReads(block.self_attn, config["rank"], config["initial_read_scale"]) for block in blocks
        ).to(device=model.device, dtype=torch.float32)
        for index, block in enumerate(blocks):
            block.forward = self.decoder_forward(block, index)

    def decoder_forward(self, block, index):
        original = block.forward
        reader = self.readers[index]

        def forward(hidden_states, *, position_embeddings=None, past_key_values=None, use_cache=False, **kwargs):
            if self.body is None:
                return original(hidden_states, position_embeddings=position_embeddings,
                                past_key_values=past_key_values, use_cache=use_cache, **kwargs)
            body = self.body[index]
            strength = self.strength

            def decode(hidden):
                attended, _ = block.self_attn(
                    hidden_states=block.input_layernorm(hidden), position_embeddings=position_embeddings,
                    past_key_values=past_key_values, use_cache=use_cache, **kwargs,
                )
                hidden = reader(block, hidden + attended, body, position_embeddings, strength)
                return hidden + block.mlp(block.post_attention_layernorm(hidden))

            if self.activation_checkpointing and torch.is_grad_enabled():
                assert isinstance(past_key_values, ReadOnlyCache) and not use_cache
                return checkpoint(decode, hidden_states, use_reentrant=False, preserve_rng_state=False)
            return decode(hidden_states)

        return forward

    @contextmanager
    def answering(self, row, cache):
        assert self.body is None
        header = len(self.engine.prefix_ids)
        length = cache.get_seq_length()
        assert length == header + len(row["content_ids"])
        self.body = tuple((key[:, :, header:length], value[:, :, header:length]) for key, value in K.pairs(cache))
        try:
            yield length
        finally:
            self.body = None

    def logits(self, row, cache, tokens=(), positions=(0,), *, rn=False):
        model = self.engine.model
        ids = row["suffix_ids"] + list(tokens)
        indices = torch.tensor([len(row["suffix_ids"]) - 1 + pos for pos in positions], device=model.device)
        with self.answering(row, cache) as length:
            current = ReadOnlyCache(ddp_cache_data=K.pairs(cache), config=model.config)
            result = self.engine.forward(ids, current, length, use_cache=False)
            if self.timing is not None:
                self.timing["candidate_forward_tokens"] += len(ids)
            return model.lm_head(result.last_hidden_state[0, indices]).float()

    @torch.no_grad()
    def rollout(self, row, cache, *, rn=False, cap=64):
        engine = self.engine
        model = engine.model
        tokens = []
        ids = row["suffix_ids"]
        eos = model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        stops.add(engine.tok.convert_tokens_to_ids("<|im_end|>"))
        with self.answering(row, cache) as length:
            # All tensors are detached; DynamicCache stores concatenated tensors
            # in its own layers and leaves the original DST body untouched.
            current = K.from_values(K.flat(cache), model)
            started = time.perf_counter()
            for _ in range(cap):
                result = engine.forward(ids, current, length, use_cache=True)
                token = int(model.lm_head(result.last_hidden_state[0, -1]).float().argmax())
                tokens.append(token)
                if self.timing is not None:
                    if len(tokens) == 1:
                        self.timing["receiver_ttft_seconds"] = time.perf_counter() - started
                    self.timing["decode_forward_tokens"] += len(ids)
                length += len(ids)
                current = result.past_key_values
                ids = [token]
                if token in stops:
                    break
            if self.timing is not None and len(tokens) > 1:
                self.timing["seconds_per_decode_token"] = (
                    time.perf_counter() - started - self.timing["receiver_ttft_seconds"]
                ) / (len(tokens) - 1)
        return tokens
