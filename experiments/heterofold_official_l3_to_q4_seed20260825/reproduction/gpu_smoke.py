"""Actual-model checks; all tiny calibration artifacts are diagnostic only."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from heterofold import calibration, recolor
from heterofold.data.calibration_corpus import load_manifest
from heterofold.maps import atomic_save, materialize
from heterofold.modeling import chat_affixes, kv_geometry, load_model
from heterofold.registry import resolve_model
from heterofold.transfer import capture_sender, build_receiver_cache


ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'smoke'
    out.mkdir(exist_ok=True)
    if (out / 'completed.json').exists():
        print('GPU SMOKE already completed', flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 25 * 2**30:
        raise RuntimeError('GPU unavailable or less than 25 GiB free; refusing smoke')
    torch.manual_seed(20260825)
    sender = load_model(resolve_model('llama3'))
    receiver = load_model(resolve_model('qwen4'))
    tok, model = receiver
    prefix, assistant = chat_affixes(tok)
    text = 'Explain briefly why ice melts when heated.'
    ids = tok(prefix + text + assistant, add_special_tokens=False,
              return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        full = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        cached = model(ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
        split = model(ids[:, -1:], past_key_values=cached, use_cache=True,
                      logits_to_keep=1).logits[:, -1].float()
    gap = float((full - split).abs().mean())
    same = bool(torch.equal(full.argmax(-1), split.argmax(-1)))
    if not same or gap > 0.05:
        raise AssertionError({'native_cache_argmax_equal': same, 'logit_mae': gap})
    del cached, full, split
    # Synthetic fixtures are solely interface diagnostics, never formal calibration data.
    questions = [
        'Why does water freeze when its temperature falls below zero degrees Celsius?',
        'If a box contains twelve balls and three are removed, how many balls remain?',
        'Why do plants usually need sunlight and water to grow and remain healthy?',
        'If five children each receive four apples, how many apples are distributed?',
        'What happens to a solid piece of ice when it is heated in a warm room?',
        'A farmer has eight sheep and buys another six sheep. How many sheep are there?']
    tiny = []
    for i, question in enumerate(questions):
        prompt = 'Background: This is a short reasoning and reading comprehension example.\nQuestion: ' + question + '\nAnswer:'
        tiny.append(dict(prompt=prompt, dataset='synthetic interface fixture',
            split='train' if i < 4 else 'validation',
            question_char_start=prompt.index(question),
            question_char_end=prompt.index(question) + len(question)))
    corpora = ['synthetic interface fixture']
    per_corpus = {}
    for corpus in corpora:
        texts = [r['prompt'] for r in tiny if r['dataset'] == corpus and r['split'] == 'train']
        per_corpus[corpus] = recolor.collect_statistics(sender, receiver, texts, log=print)
    stats = recolor.balance_corpora(per_corpus)
    del per_corpus
    base = recolor.fit_map(stats, log=print)
    del stats
    base.update(sender='llama3', receiver='qwen4', diagnostic_only=True)
    atomic_save(base, out / 'recolor.pt')
    result = calibration.calibrate(sender, receiver, base, tiny, device='cuda:0',
        out_path=out / 'heterofold.pt', checkpoint_path=out / 'train_checkpoint.pt',
        rank=16, epochs=1, grad_accum=2, log=print)
    history = result['functional_correction']['history']
    if not all(math.isfinite(e[s]['loss']) for e in history for s in ('train', 'validation')):
        raise FloatingPointError('Non-finite smoke calibration')
    diagnostics = result['functional_correction']['diagnostics']
    if not any(r['k_relative_update'] > 0 or r['v_relative_update'] > 0 for r in diagnostics):
        raise AssertionError('Calibration parameters did not change')
    mapping = materialize(result, device='cuda:0', dtype=model.dtype)
    with torch.no_grad():
        offsets, keys, values = capture_sender(*sender, mapping, text, device='cuda:0')
        cache, length = build_receiver_cache(tok, model, mapping, offsets, keys, values, text, prefix)
        tail = tok(assistant, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
        prediction = model(tail, past_key_values=cache, use_cache=False,
            attention_mask=torch.ones(1, length + tail.shape[1], device='cuda', dtype=torch.long),
            position_ids=torch.arange(length, length + tail.shape[1], device='cuda').unsqueeze(0),
            logits_to_keep=1).logits
        assert torch.isfinite(prediction).all(), 'Transfer logits are not finite'
    payload = {'native_cache_argmax_equal': same, 'native_cache_logit_mae': gap,
        'sender_geometry': kv_geometry(sender[1]), 'receiver_geometry': kv_geometry(model),
        'dtype': str(model.dtype), 'history': history,
        'diagnostic_only': True, 'transfer_logits_finite': True}
    (out / 'completed.json').write_text(json.dumps(payload, indent=2))
    print('GPU SMOKE PASSED', json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
