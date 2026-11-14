"""Gemma cache construction and readout with native sliding-window metadata."""

from copy import copy

from reverse_common import C, CFG, K, M, ROOT, torch
from heterofold.capture import capture_kv
from torch.utils.checkpoint import checkpoint


class GemmaText(torch.nn.Module):
    def __init__(self, loaded):
        super().__init__()
        self.model = loaded.model.language_model
        self.lm_head = loaded.lm_head
        self.config = self.model.config
        self.generation_config = loaded.generation_config

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype


def copy_cache(cache, values=None):
    """Copy tensors while retaining the local layers' absolute seen lengths."""
    if values is None:
        values = K.flat(cache)
    out = copy(cache)
    out.layers = []
    for i, old in enumerate(cache.layers):
        layer = copy(old)
        layer.keys = values[2 * i].clone()
        layer.values = values[2 * i + 1].clone()
        out.layers.append(layer)
    return out


class Engine:
    def __init__(self):
        from transformers import Gemma3ForConditionalGeneration

        self.sender_tok, self.sender = C.load_model(CFG["sender"], "cuda", dtype=torch.float16)
        self.tok = C.AutoTokenizer.from_pretrained(CFG["receiver"], local_files_only=True)
        loaded = Gemma3ForConditionalGeneration.from_pretrained(
            CFG["receiver"], dtype=torch.float32, attn_implementation="sdpa", local_files_only=True,
        )
        self.model = GemmaText(loaded).cuda().eval().requires_grad_(False)
        del loaded
        assert len(C.layers_of(self.sender)) == 36
        assert len(C.layers_of(self.model)) == 34
        assert self.model.config.sliding_window == 1024 and self.model.config.final_logit_softcapping is None
        assert self.model.config.num_key_value_heads * self.model.config.head_dim == CFG["model"]["target_width"]
        manifest = C.read(ROOT / "manifest.json")
        self.prefix_ids = manifest["prefix_ids"]
        self.stop_ids = set(self.model.generation_config.eos_token_id) | {manifest["stop_token"]}
        with torch.no_grad():
            self.header = self.forward(self.prefix_ids, None, 0).past_key_values
        self.mode = "train"
        self.student_test_native_prefills = 0
        self._versions = [C.versions(x) for x in (self.sender, self.model)]
        self.frozen_base = None
        self.base_versions = None

    def unchanged(self):
        assert self._versions == [C.versions(x) for x in (self.sender, self.model)]
        frozen = [self.sender, self.model]
        if self.frozen_base is not None:
            assert C.versions(self.frozen_base) == self.base_versions
            frozen.append(self.frozen_base)
        assert all(not p.requires_grad and p.grad is None for model in frozen for p in model.parameters())

    def forward(self, ids, cache, length, *, use_cache=True):
        assert cache is None or cache.get_seq_length() == length
        return self.model.model(
            torch.tensor([ids], device="cuda"), past_key_values=cache,
            position_ids=torch.arange(length, length + len(ids), device="cuda")[None],
            attention_mask=torch.ones(1, length + len(ids), device="cuda", dtype=torch.long),
            use_cache=use_cache,
        )

    @torch.no_grad()
    def source(self, row):
        assert self.sender_tok.encode(row["prompt"], add_special_tokens=False) == row["sender_ids"]
        _, keys, values = capture_kv(self.sender_tok, self.sender, row["prompt"], range(36),
                                      device="cpu", dtype=torch.float16)
        result = tuple(tuple(group[l] for l in range(36)) for group in (keys, values))
        assert all(t.shape == (len(row["sender_ids"]), 1024) and torch.isfinite(t).all()
                   for group in result for t in group)
        return result

    @torch.no_grad()
    def native_cache(self, row, *, baseline=False):
        if self.mode == "test" and not baseline:
            self.student_test_native_prefills += 1
            raise RuntimeError("Native body is available only to the isolated baseline at test time")
        return self.forward(self.prefix_ids + row["content_ids"], None, 0).past_key_values

    @torch.no_grad()
    def teacher(self, row):
        if self.mode == "test":
            raise RuntimeError("Native distillation targets are training-only")
        keys, values, hooks = {}, {}, []
        for l, block in enumerate(C.layers_of(self.model)):
            for name, result in (("k_proj", keys), ("v_proj", values)):
                def capture(_module, _args, out, result=result, layer=l):
                    result[layer] = out[0, len(self.prefix_ids):].detach().cpu().contiguous()
                hooks.append(getattr(block.self_attn, name).register_forward_hook(capture))
        try:
            native = self.native_cache(row)
        finally:
            for hook in hooks:
                hook.remove()
        key, value = (torch.stack([group[l] for l in range(34)]) for group in (keys, values))
        assert key.shape == value.shape == (34, len(row["content_ids"]), 1024)
        assert torch.isfinite(key).all() and torch.isfinite(value).all()
        return key, value, native

    def finalize(self, key, value, start):
        positions = torch.arange(start, start + key.shape[1], device=key.device)
        tables = C._rotary_tables(self.model, positions)
        pairs = [C.finalize_layer(self.model, l, key[l], value[l], *tables(l)) for l in range(34)]
        return tuple(torch.stack([pair[i] for pair in pairs]) for i in range(2))

    def assemble(self, chunks):
        data = [
            (torch.cat([k.detach(), *[part[0][l][None] for part in chunks]], dim=2),
             torch.cat([v.detach(), *[part[1][l][None] for part in chunks]], dim=2))
            for l, (k, v) in enumerate(K.pairs(self.header))
        ]
        cache = C.DynamicCache(ddp_cache_data=data, config=self.model.config)
        for layer in cache.layers:
            if layer.is_sliding:
                layer.keys = layer.keys.contiguous()
                layer.values = layer.values.contiguous()
        return cache

    def translated(self, row, module, source=None, *, checkpointed=True):
        source = self.source(row) if source is None else source
        indices = row["sync_source_indices"]
        chunks = []

        def build(begin, end):
            key, value = module(*source, indices[begin:end])
            return self.finalize(key, value, len(self.prefix_ids) + begin)

        for begin in range(0, len(indices), CFG["token_chunk"]):
            end = min(len(indices), begin + CFG["token_chunk"])
            if checkpointed and torch.is_grad_enabled():
                chunks.append(checkpoint(build, begin, end, use_reentrant=False))
            else:
                chunks.append(build(begin, end))
        cache = self.assemble(chunks)
        assert cache.get_seq_length() == len(self.prefix_ids) + len(row["content_ids"])
        return cache


class BodyReader:
    token_logits = M.RNKV.token_logits
    score = M.RNKV.score

    def __init__(self, engine):
        self.engine = engine

    def logits(self, row, cache, tokens=(), positions=(0,), *, rn=False):
        assert not rn
        length = len(self.engine.prefix_ids) + len(row["content_ids"])
        assert cache.get_seq_length() == length
        ids = row["suffix_ids"] + list(tokens)
        at = torch.tensor([len(row["suffix_ids"]) - 1 + i for i in positions], device="cuda")

        def forward(*values):
            result = self.engine.forward(ids, copy_cache(cache, values), length, use_cache=False)
            return self.engine.model.lm_head(result.last_hidden_state[0, at]).float()

        values = K.flat(cache)
        if torch.is_grad_enabled() and any(v.requires_grad for v in values):
            return checkpoint(forward, *values, use_reentrant=False)
        return forward(*values)

    @torch.no_grad()
    def rollout(self, row, cache, *, rn=False, cap=64):
        assert not rn
        ids = row["suffix_ids"]
        cache = copy_cache(cache)
        tokens = []
        for _ in range(cap):
            out = self.engine.forward(ids, cache, cache.get_seq_length())
            token = int(self.engine.model.lm_head(out.last_hidden_state[0, -1]).argmax())
            tokens.append(token)
            if token in self.engine.stop_ids:
                break
            cache, ids = out.past_key_values, [token]
        return tokens
