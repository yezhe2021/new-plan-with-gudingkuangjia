"""lm-evaluation-harness backend for frozen Llama->Qwen Full-Sync Gen-B."""

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
REF = Path(CFG["reference_eval"])
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))

import evaluate as ref  # noqa: E402
import evaluate_gsm8k_generation as gen  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_native_answer_suffix(context):
    marker = "Answer:"
    index = context.rfind(marker)
    if index < 0 or context[index:] != marker:
        raise RuntimeError("lm-eval GSM8K context must end exactly in the native suffix 'Answer:'")
    return context[:index], marker


def load_translators(use_stage_b=True):
    module = ref.load_module("lm_eval_generation_translator", CFG["translator_module"])
    base = module.NativeKVTranslator("full28_mlp", hidden_dim=1024)
    stage_a = torch.load(CFG["stage_a_checkpoint"], map_location="cpu", weights_only=True)
    base.load_state_dict(stage_a["state"], strict=True)
    base = base.cuda().eval().requires_grad_(False)
    adapter = None
    if use_stage_b:
        adapter = module.ResidualKVAdapter(rank=64)
        stage_b = torch.load(CFG["generation_stage_b_checkpoint"], map_location="cpu", weights_only=True)
        adapter.load_state_dict(stage_b["state"], strict=True)
        adapter = adapter.cuda().eval().requires_grad_(False)
    return base, adapter


@torch.no_grad()
def translate_entry(base, adapter, source_k, source_v, chunk=64):
    out_k, out_v = [], []
    for begin in range(0, source_k.shape[1], chunk):
        stop = min(source_k.shape[1], begin + chunk)
        sk, sv = source_k[:, begin:stop][None].cuda(), source_v[:, begin:stop][None].cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            key, value = base(sk, sv)
            if adapter is not None:
                key, value, _, _ = adapter(key, value)
        out_k.append(key[0]); out_v.append(value[0])
    return torch.cat(out_k, dim=1), torch.cat(out_v, dim=1)


@torch.no_grad()
def greedy_generate(model, tok, input_ids, max_new_tokens, until, key=None, value=None):
    if key is None:
        past = None
        current = torch.tensor([input_ids], device="cuda", dtype=torch.long)
    else:
        positions = torch.arange(key.shape[1], device="cuda")
        past = ref.make_cache(model, key, value, positions)
        current = torch.tensor([input_ids], device="cuda", dtype=torch.long)
    generated = []
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids}
    stop_reason = "max_gen_toks"
    text = ""
    for _ in range(max_new_tokens):
        prefix = int(past.get_seq_length()) if past is not None else 0
        mask = torch.ones((1, prefix + current.shape[1]), device="cuda", dtype=torch.long)
        pos = torch.arange(prefix, prefix + current.shape[1], device="cuda")[None]
        output = ref.backbone(model)(input_ids=current, attention_mask=mask, position_ids=pos,
                                     past_key_values=past, use_cache=True, return_dict=True)
        token = int(model.lm_head(output.last_hidden_state[:, -1])[0].argmax().item())
        past = output.past_key_values
        if token in eos_ids:
            stop_reason = "eos"
            break
        generated.append(token)
        text = tok.decode(generated, skip_special_tokens=True)
        hits = [(text.find(marker), marker) for marker in until if marker and text.find(marker) >= 0]
        if hits:
            first, marker = min(hits, key=lambda item: item[0])
            text = text[:first]
            stop_reason = f"until:{marker}"
            break
        current = torch.tensor([[token]], device="cuda", dtype=torch.long)
    return text, len(generated), stop_reason


@register_model("fullsync_l3_to_q4")
class FullSyncLM(LM):
    def __init__(self, mode="fullsync", audit_path="", max_gen_toks=256, max_length=32768,
                 batch_size=1, device="cuda:0", **kwargs):
        super().__init__()
        if mode not in {"fullsync", "fullsync_stagea", "native_qwen"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.audit_path = Path(audit_path) if audit_path else HERE / "results/backend_audit.jsonl"
        self._max_gen_toks = int(max_gen_toks)
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
        if self._batch_size != 1:
            raise ValueError("The pilot Full-Sync backend currently requires batch_size=1")
        self._device = torch.device(device)
        self._tok = AutoTokenizer.from_pretrained(CFG["qwen_model"], local_files_only=True)

    @property
    def eot_token_id(self):
        return self._tok.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests):
        raise NotImplementedError("GSM8K uses generate_until only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("GSM8K uses generate_until only")

    def _kwargs(self, raw):
        kwargs = dict(raw)
        if kwargs.get("do_sample", False):
            raise NotImplementedError("Pilot supports greedy generation only")
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)
        if temperature != 0.0:
            raise NotImplementedError("Pilot requires temperature=0")
        until = kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        max_tokens = int(kwargs.get("max_gen_toks", kwargs.get("max_new_tokens", self.max_gen_toks)))
        return list(until), max_tokens

    def _write_audits(self, rows):
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def generate_until(self, requests):
        request_rows = []
        for index, request in enumerate(requests):
            context, raw_kwargs = request.args
            until, max_tokens = self._kwargs(raw_kwargs)
            qwen_ids = gen.encode_prefix(self._tok, context)
            if len(qwen_ids) + max_tokens > self.max_length:
                raise RuntimeError("Pilot context exceeds fixed max_length; implicit truncation is forbidden")
            request_rows.append({"index": index, "context": context, "until": until,
                                 "max_tokens": max_tokens, "qwen_context_tokens": len(qwen_ids)})

        if self.mode == "native_qwen":
            qwen = ref.load_model("qwen")
            results, audits = [], []
            try:
                for row in request_rows:
                    ids = gen.encode_prefix(self._tok, row["context"])
                    text, count, reason = greedy_generate(qwen, self._tok, ids, row["max_tokens"], row["until"])
                    results.append(text)
                    audits.append({**row, "mode": self.mode, "generated_tokens": count,
                                   "stop_reason": reason, "generated_text": text})
            finally:
                del qwen; gc.collect(); torch.cuda.empty_cache()
            self._write_audits(audits)
            return results

        llama_tok = AutoTokenizer.from_pretrained(CFG["llama_model"], local_files_only=True)
        sources = []
        sender = ref.load_model("llama")
        try:
            for row in request_rows:
                prefix, suffix = split_native_answer_suffix(row["context"])
                aligned = gen.alignment_row(f"lm_eval_{row['index']}", prefix, "llama", llama_tok, self._tok)
                target, source, counts = ref.fullsync_map(aligned, llama_tok, self._tok, "llama")
                fields = aligned["encoded"]["llama"]
                key, value, _ = ref.capture(sender, fields["body"], len(fields["body"]), [0])
                sources.append({**row, "prefix": prefix, "suffix": suffix, "counts": counts,
                                "receiver_prefix_tokens": len(aligned["encoded"]["qwen"]["body"]),
                                "sender_prefix_tokens": len(fields["body"]),
                                "source_k": key[:, source].half(), "source_v": value[:, source].half(),
                                "mapped_tokens": len(target)})
        finally:
            del sender; gc.collect(); torch.cuda.empty_cache()

        qwen = ref.load_model("qwen")
        use_stage_b = self.mode == "fullsync"
        base, adapter = load_translators(use_stage_b=use_stage_b)
        results, audits = [], []
        try:
            for row in sources:
                translated_k, translated_v = translate_entry(base, adapter, row["source_k"], row["source_v"])
                native_ids = gen.encode_prefix(self._tok, row["prefix"])
                native_k, native_v, _ = ref.capture(qwen, native_ids, len(native_ids), [0])
                key = torch.cat((native_k[:, :1].cuda(), translated_k), dim=1)
                value = torch.cat((native_v[:, :1].cuda(), translated_v), dim=1)
                if key.shape[1] != row["receiver_prefix_tokens"]:
                    raise RuntimeError("Translated cache length does not match receiver Full-Sync prefix")
                suffix_ids = list(self._tok.encode(row["suffix"], add_special_tokens=False))
                text, count, reason = greedy_generate(qwen, self._tok, suffix_ids,
                                                       row["max_tokens"], row["until"], key, value)
                results.append(text)
                audits.append({key: value for key, value in row.items()
                               if key not in {"source_k", "source_v", "prefix", "context"}})
                audits[-1].update(
                    mode=self.mode,
                    generated_tokens=count,
                    stop_reason=reason,
                    generated_text=text,
                    stage_a_checkpoint=CFG["stage_a_checkpoint"],
                    generation_stage_b_checkpoint=(
                        CFG["generation_stage_b_checkpoint"] if use_stage_b else None
                    ),
                )
                del native_k, native_v, key, value, translated_k, translated_v
        finally:
            del qwen, base, adapter, sources; gc.collect(); torch.cuda.empty_cache()
        self._write_audits(audits)
        return results
