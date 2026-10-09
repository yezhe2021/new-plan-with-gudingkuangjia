"""Only the model-side KV path changes; harness owns prompts, filters and generation."""
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT / "base_code"))
from common import load_model, save_json, seed_all
from protocol import make_cache
from translator import NativeKVTranslator, ResidualKVAdapter


def encoding(tok, text):
    # Same default special-token behavior as HFLM.tok_encode(add_special_tokens=False).
    return list(tok.encode(text, add_special_tokens=False))


@torch.no_grad()
def capture(model, ids):
    keys, values, handles = {}, {}, []
    def hook(index):
        def call(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"): key = module.k_norm(key)
            keys[index] = key[0].cpu()
            values[index] = module.v_proj(hidden).view(shape)[0].cpu()
        return call
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
        model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False)
    finally:
        for h in handles: h.remove()
    return torch.stack([keys[i] for i in range(len(keys))]), torch.stack([values[i] for i in range(len(values))])


def source_map(text, tok_s, tok_t, target_ids):
    s = tok_s(text, add_special_tokens=False, return_offsets_mapping=True)
    t = tok_t(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(t["input_ids"]) != list(target_ids):
        raise RuntimeError("Harness context tokenization differs from offset tokenization; do not silently retokenize")
    # Preserve original Full-Sync copy/drop rule, anchor token0, last native readout.
    bos = [tok_s.bos_token_id] if tok_s.bos_token_id is not None else []
    source_ids = bos + list(s["input_ids"])
    sb = [(i + len(bos), a, b) for i, (a, b) in enumerate(s["offset_mapping"]) if a != b]
    tb = [(i, a, b) for i, (a, b) in enumerate(t["offset_mapping"]) if a != b]
    boundaries = sorted({0} | ({b for _, _, b in sb} & {b for _, _, b in tb}))
    mapping, counts = {}, Counter()
    wanted = set(range(1, len(target_ids) - 1))
    for left, right in zip(boundaries, boundaries[1:]):
        si = [i for i, _, b in sb if left < b <= right]
        ti = [i for i, _, b in tb if left < b <= right]
        if not si or not ti: continue
        counts["copy" if len(si) < len(ti) else "drop" if len(si) > len(ti) else "equal"] += 1
        for rank, index in enumerate(ti):
            if index in wanted:
                rank_s = math.ceil((rank + 1) * len(si) / len(ti)) - 1 if len(si) > len(ti) else rank * len(si) // len(ti)
                mapping[index] = si[rank_s]
    if set(mapping) != wanted:
        raise RuntimeError(f"Incomplete sync mapping: {len(wanted - set(mapping))} tokens")
    return source_ids, [mapping[i] for i in sorted(wanted)], dict(counts)


def pair(sender, receiver, tok_s, tok_t, context):
    context = context.rstrip()  # harness _encode_pair moves this whitespace to continuation.
    ids = encoding(tok_t, context)
    if not ids: raise RuntimeError("Empty context requires harness prefix-token handling")
    if len(ids) <= 2:
        # The declared anchor+native-readout schema leaves no external token, not an invalid example.
        empty_s = torch.empty(28,0,8,128,dtype=torch.float16)
        empty_t = torch.empty(36,0,8,128,dtype=torch.float16)
        ak,av = capture(receiver,ids[:-1]) if len(ids)>1 else (empty_t,empty_t.clone())
        return {"context":context,"ids":ids,"source_k":empty_s,"source_v":empty_s.clone(),
                "target_k":empty_t,"target_v":empty_t.clone(),"anchor_k":ak,"anchor_v":av,
                "counts":{"no_external_tokens":1},"tokens":0}
    source_ids, selected, counts = source_map(context, tok_s, tok_t, ids)
    sk, sv = capture(sender, source_ids)
    tk, tv = capture(receiver, ids[:-1])
    return {"context": context, "ids": ids, "source_k": sk[:, selected], "source_v": sv[:, selected],
            "target_k": tk[:, 1:], "target_v": tv[:, 1:],
            "anchor_k": tk[:, :1], "anchor_v": tv[:, :1], "counts": counts, "tokens": len(selected)}


def map_base(base, item, chunk=64):
    if item["tokens"]==0:
        return item["target_k"].cuda(),item["target_v"].cuda()
    kk, vv = [], []
    with torch.no_grad():
        for start in range(0, item["tokens"], chunk):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                k, v = base(item["source_k"][:, start:start+chunk][None].cuda(),
                            item["source_v"][:, start:start+chunk][None].cuda())
            kk.append(k[0]); vv.append(v[0])
    return torch.cat(kk, 1), torch.cat(vv, 1)


def assemble(item, k, v, adapter=None):
    if adapter is not None:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            k, v, _, _ = adapter(k[None], v[None])
        k, v = k[0], v[0]
    return torch.cat((item["anchor_k"].cuda(), k), 1), torch.cat((item["anchor_v"].cuda(), v), 1)


def forward_continuation(receiver, context_ids, continuation_ids, key=None, value=None):
    if key is None:
        ids = context_ids + continuation_ids
        first = len(context_ids) - 1
        prefix, cache = 0, None
    else:
        prefix = key.shape[1]
        if prefix != len(context_ids) - 1: raise RuntimeError("Native last-readout position mismatch")
        ids = context_ids[-1:] + continuation_ids
        first = 0
        cache = make_cache(receiver, key, value, torch.arange(prefix, device="cuda"))
    tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
    out = receiver.model(input_ids=tensor,
                         attention_mask=torch.ones((1, prefix + len(ids)), device="cuda", dtype=torch.long),
                         position_ids=torch.arange(prefix, prefix+len(ids), device="cuda")[None],
                         past_key_values=cache, use_cache=False, return_dict=True)
    # Project only scored tokens rather than the complete long few-shot prompt.
    logits = receiver.lm_head(out.last_hidden_state[:, first:first+len(continuation_ids)+1])[0]
    return logits[:-1], logits[-1]


def continuation_score(receiver, context_ids, continuation_ids, key=None, value=None, softmax_dtype=torch.float32):
    # Match harness exactly: feed context + continuation[:-1], not an extra EOS-scoring token.
    prefix = 0 if key is None else key.shape[1]
    ids = (context_ids if key is None else context_ids[-1:]) + continuation_ids[:-1]
    cache = None if key is None else make_cache(receiver,key,value,torch.arange(prefix,device="cuda"))
    tensor = torch.tensor([ids],device="cuda",dtype=torch.long)
    output = receiver(input_ids=tensor,
                      attention_mask=torch.ones((1,prefix+len(ids)),device="cuda",dtype=torch.long),
                      position_ids=torch.arange(prefix,prefix+len(ids),device="cuda")[None],
                      past_key_values=cache,use_cache=False,return_dict=True)
    logits = output.logits[0,-len(continuation_ids):]
    log_probs = F.log_softmax(logits, -1, dtype=softmax_dtype)
    target = torch.tensor(continuation_ids, device="cuda", dtype=torch.long)
    return log_probs.gather(-1, target[:, None]).sum(), bool((log_probs.argmax(-1) == target).all())


class FullSyncHFLM(HFLM):
    def __init__(self, receiver, mode="native", sender=None, base=None, adapter=None, audit_path=None):
        super().__init__(pretrained=receiver, tokenizer=CFG["models"]["qwen"], batch_size=1,
                         backend="causal", max_length=CFG["max_length"], logits_cache=False)
        self.mode, self.sender, self.base, self.adapter = mode, sender, base, adapter
        self.sender_tok = AutoTokenizer.from_pretrained(CFG["models"]["llama"], local_files_only=True) if sender else None
        self.audit_path = Path(audit_path) if audit_path else None
        self._active_contexts = {}
        self._cached_context = None
        self._cached_kv = None

    def _prefix(self, context, ids):
        clean = context.rstrip()
        if encoding(self.tokenizer, clean) != list(ids):
            raise RuntimeError("Context IDs differ from harness encoding; no implicit truncation or BOS rewrite")
        digest = hashlib.sha256((self.mode + clean).encode()).hexdigest()
        if digest == self._cached_context: return self._cached_kv
        self._cached_kv = None  # release previous CUDA prefix before new capture
        if self.mode == "oracle":
            k, v = capture(self.model, ids[:-1]); k, v = k.cuda(), v.cuda()
        else:
            source_ids, selected, counts = source_map(clean, self.sender_tok, self.tokenizer, ids)
            sk, sv = capture(self.sender, source_ids)
            ak, av = capture(self.model, ids[:1])
            item = {"source_k": sk[:, selected], "source_v": sv[:, selected],
                    "anchor_k": ak, "anchor_v": av, "tokens": len(selected)}
            bk, bv = map_base(self.base, item, CFG["chunk_tokens"])
            k, v = assemble(item, bk, bv, self.adapter)
        self._cached_context, self._cached_kv = digest, (k, v)
        return k, v

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        if self.mode == "native":
            return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)
        result = []
        for cache_key, context_ids, continuation_ids in requests:
            if cache_key is None: raise NotImplementedError("Rolling likelihood not part of this benchmark")
            context, continuation = cache_key
            if len(context_ids)<=2:
                # Anchor/readout consume all context tokens; no Translator invocation exists here.
                result.extend(super()._loglikelihood_tokens([(cache_key,context_ids,continuation_ids)],
                                                            disable_tqdm=True,override_bs=1))
                continue
            if len(context_ids) + len(continuation_ids) > self.max_length:
                raise RuntimeError("Context exceeds evaluated full-sync budget; do not silently change official requests")
            with torch.no_grad():
                key, value = self._prefix(context, context_ids)
                score, exact = continuation_score(self.model, context_ids, continuation_ids, key, value,
                                                  softmax_dtype=self.softmax_dtype)
            result.append((float(score), exact))
            self.cache_hook.add_partial("loglikelihood", cache_key, result[-1])
        return result

    def generate_until(self, requests, disable_tqdm=False):
        if self.mode != "native":
            self._active_contexts = {tuple(self.tok_encode(r.args[0])): r.args[0] for r in requests}
        # ALL stop handling, EOS, token limits, postprocessing stay inherited.
        result = super().generate_until(requests, disable_tqdm=disable_tqdm)
        self._active_contexts = {}
        return result

    def _model_generate(self, context, max_length, stop, **kwargs):
        if self.mode == "native" or context.shape[1]<=2:
            return super()._model_generate(context, max_length, stop, **kwargs)
        if context.shape[0] != 1: raise RuntimeError("Full-Sync benchmark batch_size must be1")
        ids = context[0].tolist()
        text = self._active_contexts.get(tuple(ids))
        if text is None: raise RuntimeError("Truncated/padded context cannot be silently substituted")
        with torch.no_grad():
            # No rstrip here: generation uses actual full context, not _encode_pair.
            if text != text.rstrip():
                raise RuntimeError("Generation context trailing whitespace requires exact offset handling")
            k, v = self._prefix(text, ids)
            cache = make_cache(self.model, k, v, torch.arange(k.shape[1], device="cuda"))
            return super()._model_generate(context, max_length, stop, past_key_values=cache, **kwargs)


def load_base(path):
    base = NativeKVTranslator("full28_mlp", hidden_dim=CFG["mlp_hidden_dim"]).cuda().eval().requires_grad_(False)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != CFG["protocol"]: raise RuntimeError("Wrong checkpoint protocol")
    base.load_state_dict(payload["state"], strict=True)
    return base


def load_adapter(path):
    adapter = ResidualKVAdapter(rank=CFG["adapter_rank"]).cuda().eval().requires_grad_(False)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != CFG["protocol"]: raise RuntimeError("Wrong checkpoint protocol")
    adapter.load_state_dict(payload["state"], strict=True)
    return adapter
