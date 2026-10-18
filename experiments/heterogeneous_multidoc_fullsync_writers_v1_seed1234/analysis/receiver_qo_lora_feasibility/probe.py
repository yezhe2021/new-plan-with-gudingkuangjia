"""Server-A resource/gradient probe only; synthetic inputs, no benchmark or saved weights."""
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parents[1]
sys.path.insert(0, str(PARENT))
import experiment as exp


def save(result):
    result['timestamp'] = datetime.now(timezone.utc).isoformat()
    (ROOT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


class LoRA(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device='cuda', dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device='cuda', dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        delta = F.linear(F.linear(x.float(), self.a), self.b).to(original.dtype)
        return original + delta


def run():
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    result = {'status': 'running', 'server': 'A', 'rank': 8, 'dtype': 'float16 frozen base; float32 LoRA',
        'gpu': torch.cuda.get_device_name(0), 'free_before_gib': free / 2**30,
        'scope': 'Synthetic native external-KV prefix + trainable Q/O; tests feasibility only, not real-writer quality',
        'gradient_checkpointing': False, 'cases': []}
    if free < 11 * 2**30:
        result.update(status='skipped', reason='Less than11GiB currently free; do not interfere with other jobs')
        save(result); return
    save(result)
    path = exp.CFG['models']['qwen']
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    adapters = []
    for layer in model.model.layers:
        for name in ('q_proj', 'o_proj'):
            module = LoRA(getattr(layer.self_attn, name))
            setattr(layer.self_attn, name, module)
            adapters.append((name, module))
    parameters = [p for p in model.parameters() if p.requires_grad]
    result['trainable_parameters'] = sum(p.numel() for p in parameters)
    assert result['trainable_parameters'] == 3833856
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    text = 'Alice lives in Paris. Bob lives in London. The documents describe their cities. '
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for prefix_length, suffix_length in [(512, 32), (2048, 64)]:
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for _, module in adapters:
            module.enabled = False
        model.eval()
        prefix = (tokens * (prefix_length // len(tokens) + 1))[:prefix_length]
        memory = tuple(x.clone() for x in exp.capture(model, prefix))
        torch.cuda.empty_cache()
        for _, module in adapters:
            module.enabled = True
        model.train()
        optimizer.zero_grad(set_to_none=True)
        suffix = (tokens * (suffix_length // len(tokens) + 1))[:suffix_length]
        inputs = torch.tensor([suffix], device='cuda')
        cache = exp.make_cache(model, memory)
        out = model.model(input_ids=inputs,
            attention_mask=torch.ones((1, prefix_length + suffix_length), device='cuda', dtype=torch.long),
            position_ids=torch.arange(prefix_length, prefix_length + suffix_length, device='cuda')[None],
            past_key_values=cache, use_cache=False, return_dict=True)
        logits = model.lm_head(out.last_hidden_state[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), inputs[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        grad_norms = {name: math.sqrt(sum(float(module.b.grad.float().square().sum())
                         for kind, module in adapters if kind == name)) for name in ('q_proj', 'o_proj')}
        assert all(x > 0 for x in grad_norms.values()), grad_norms
        scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        result['cases'].append({'prefix_tokens': prefix_length, 'suffix_tokens': suffix_length,
            'loss': float(loss.detach()), 'gradient_norms_B': grad_norms,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'gpu_free_after_gib': torch.cuda.mem_get_info()[0] / 2**30,
            'seconds': time.monotonic() - start, 'status': 'passed'})
        save(result)
        del memory, cache, inputs, out, logits, loss
        optimizer.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
    result['status'] = 'passed'
    save(result)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        old = json.loads((ROOT / 'result.json').read_text()) if (ROOT / 'result.json').exists() else {}
        old.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        save(old)
        traceback.print_exc()
        raise
