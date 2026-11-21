"""Append local digit tokens between two transmitted pieces of an evidence cache."""

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from heterofold.transfer import _cache_layer, _rotary_tables, finalize_layer


@dataclass(frozen=True)
class DigitHandoff:
    before: tuple
    token_ids: list
    after: tuple


def digit_handoff(raw_kv, ids, slots, header_length):
    begin, end = slots[0] - header_length, slots[-1] + 1 - header_length
    assert 0 < begin < end <= len(ids)
    assert slots == list(range(header_length + begin, header_length + end))
    assert all(t.shape[1] == len(ids) for t in raw_kv)
    # Use independent buffers so the packet never retains the omitted KV rows.
    return DigitHandoff(
        tuple(t[:, :begin].clone().contiguous() for t in raw_kv),
        ids[begin:end],
        tuple(t[:, end:].clone().contiguous() for t in raw_kv),
    )


def append_raw(model, cache, raw):
    start = cache.get_seq_length()
    length = raw[0].shape[1]
    if length == 0:
        return cache
    ks, vs = (t.cuda() for t in raw)
    table = _rotary_tables(model, torch.arange(start, start + length, device=ks.device))
    pairs = []
    for layer, (key, value) in enumerate(zip(ks, vs)):
        k, v = finalize_layer(model, layer, key, value, *table(layer))
        past_k, past_v = _cache_layer(cache, layer)
        pairs.append((torch.cat((past_k, k[None]), dim=2), torch.cat((past_v, v[None]), dim=2)))
    return DynamicCache(ddp_cache_data=pairs, config=model.config)


def assemble_digit_handoff(model, forward, header, handoff, raw_pr):
    prefix = append_raw(model, header, handoff.before)
    start = prefix.get_seq_length()
    local = forward(handoff.token_ids, prefix).past_key_values
    assert local.get_seq_length() == start + len(handoff.token_ids)
    suffix = tuple(torch.cat((after, pr), dim=1) for after, pr in zip(handoff.after, raw_pr))
    return append_raw(model, local, suffix)


def assert_only_local_slots_changed(base, actual, slots, layers):
    length = base.get_seq_length()
    assert length == actual.get_seq_length()
    before, after = slots[0], slots[-1] + 1
    max_abs = {"k": 0.0, "v": 0.0}
    for layer in layers:
        for role, a, b in zip(max_abs, _cache_layer(base, layer), _cache_layer(actual, layer)):
            assert torch.equal(a[:, :, :before], b[:, :, :before])
            assert torch.equal(a[:, :, after:], b[:, :, after:])
            delta = float((a[:, :, before:after].float() - b[:, :, before:after].float()).abs().max())
            max_abs[role] = max(max_abs[role], delta)
    return max_abs
