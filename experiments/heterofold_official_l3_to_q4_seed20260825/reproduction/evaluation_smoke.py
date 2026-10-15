"""Numerical audit of evaluation-only memory adaptations; no weight updates."""
import gc
import json
import os
from pathlib import Path

import torch

from heterofold.evaluation.runtime import Runtime, candidate_scores
from heterofold.evaluation.short_context import load, format_item, ANSWER_CUE
from heterofold.capture import capture_kv
from heterofold.maps import sender_layers

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 25 * 2**30:
        raise RuntimeError('CUDA unavailable or less than 25 GiB free')
    record = {}
    native = Runtime(receiver='qwen4')
    reference = json.loads((ROOT / 'results/native/arc/items_00000_00032.json').read_text())
    dataset = load('arc')
    gaps, divergences = [], []
    for row in reference['rows'][:4]:
        prompt, candidates, gold, scoring, metadata = format_item('arc', dataset[row['index']])
        result = candidate_scores(native, prompt, candidates, scoring == 'continuation',
                                  native.assistant + ANSWER_CUE[scoring])
        prediction = max(range(len(result['scores'])), key=result['scores'].__getitem__)
        assert prediction == row['prediction'], (row['index'], row['prediction'], prediction)
        gaps.extend(abs(a - b) for a, b in zip(row['scores'], result['scores']))
        old_logp = torch.tensor(row['scores'], dtype=torch.float64).log_softmax(0)
        new_logp = torch.tensor(result['scores'], dtype=torch.float64).log_softmax(0)
        divergences.append(float((old_logp.exp() * (old_logp - new_logp)).sum()))
    record['native_arc4_predictions_identical'] = True
    record['native_arc4_max_score_abs_error'] = max(gaps)
    # FP16 kernels are not bitwise identical; audit the decision distribution,
    # rather than a non-invariant absolute log-likelihood offset. Short tasks
    # continue using their original dispatch regardless of this audit.
    record['native_arc4_max_candidate_distribution_kl'] = max(divergences)
    assert max(divergences) < 0.01, record
    del native
    gc.collect()
    torch.cuda.empty_cache()
    runtime = Runtime(receiver='qwen4', sender='llama3',
        map_path=str(ROOT / 'maps/llama3_to_qwen4/heterofold.pt'))
    text = 'Water changes from a liquid into a solid when it freezes. Why does ice melt when heated?'
    layers = sender_layers(runtime.mapping)
    _, cpu_k, cpu_v = capture_kv(runtime.sender_tokenizer, runtime.sender_model,
        text, layers, device='cpu')
    _, gpu_k, gpu_v = capture_kv(runtime.sender_tokenizer, runtime.sender_model,
        text, layers, device=runtime.receiver_device)
    assert all(torch.equal(cpu_k[l], gpu_k[l].cpu()) and torch.equal(cpu_v[l], gpu_v[l].cpu())
               for l in layers), 'CPU staging changed KV values'
    record['cpu_staging_kv_bitwise_identical'] = True
    del cpu_k, cpu_v, gpu_k, gpu_v
    cache, length, _ = runtime.transfer_cache(text)
    tail = runtime.suffix_ids(runtime.assistant)
    with torch.inference_mode():
        logits = runtime.model(tail, past_key_values=cache, use_cache=False,
            attention_mask=torch.ones(1, length + tail.shape[1], device=tail.device, dtype=torch.long),
            position_ids=torch.arange(length, length + tail.shape[1], device=tail.device).unsqueeze(0),
            logits_to_keep=1).logits
        assert torch.isfinite(logits).all()
    record['translated_logits_finite'] = True
    (ROOT / 'smoke/evaluation_memory_adaptation.json').write_text(json.dumps(record, indent=2))
    print('EVALUATION MEMORY SMOKE PASSED', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
