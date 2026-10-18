"""Resume unchanged pilot outputs with conservative runtime weight placement."""
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
import experiment as exp

ROOT = Path(__file__).resolve().parent
POLICY = {'gemma_fp32_weight_cap_bytes': 6 << 30, 'qwen_fp32_weight_cap_bytes': 6 << 30,
          'fp16_weight_cap_bytes': 8 << 30, 'activation_reserve_bytes': 4 << 30,
          'minimum_free_before_load_bytes': 10 << 30, 'maximum_load_attempts': 3,
          'computation': 'original model dtype, all operations on CUDA; only CPU residence of layer weights changes'}


def partial_summary():
    items = [json.loads(p.read_text()) for p in (ROOT / 'runs/pilot/per_sample').glob('*.json')]
    if not items:
        return
    metrics = {arm: {key: sum(r['predictions'][arm][key] for r in items) / len(items)
                     for key in ['em', 'f1', 'full_first_token_kl']} for arm in exp.CFG['arms']}
    exp.save(ROOT / 'partial_summary.json', {'status': 'incomplete', 'sample_count': len(items),
        'planned_sample_count': exp.CFG['samples'], 'timestamp': exp.datetime.now(exp.timezone.utc).isoformat(),
        'config_signature': exp.signature(exp.CFG), 'metrics': metrics,
        'note': 'Partial subset only; formal paired statistics and final summary require all64 samples'})


def place_model(family, fp32=False):
    from transformers import AutoModelForCausalLM
    from accelerate import dispatch_model
    while torch.cuda.mem_get_info()[0] < POLICY['minimum_free_before_load_bytes']:
        exp.log('Recovery waiting for available GPU memory; unrelated jobs remain untouched')
        time.sleep(20)
    dtype = torch.float32 if family == 'gemma' or fp32 else torch.float16
    exp.log(f'Recovery load {family} {dtype}')
    model = AutoModelForCausalLM.from_pretrained(exp.CFG['models'][family], local_files_only=True,
        dtype=dtype, attn_implementation=exp.CFG['attention_implementation'])
    if family == 'gemma':
        for attr in ['vision_tower', 'multi_modal_projector']:
            if hasattr(model.model, attr):
                delattr(model.model, attr)
    model.eval().requires_grad_(False)
    cfg = exp.text_config(model)
    geometry = (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim)
    assert geometry == {'llama': (28, 8, 128), 'gemma': (34, 4, 256), 'qwen': (36, 8, 128)}[family]
    free = torch.cuda.mem_get_info()[0]
    cap = POLICY['gemma_fp32_weight_cap_bytes'] if family == 'gemma' else POLICY['qwen_fp32_weight_cap_bytes'] if fp32 else POLICY['fp16_weight_cap_bytes']
    budget = min(cap, exp.CFG['model_loading']['gpu_weight_budget_bytes'], free - POLICY['activation_reserve_bytes'])
    if budget <= 0:
        raise torch.OutOfMemoryError('No room for the recovery activation reserve')
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    resident, offloaded = size, []
    if size > budget:
        layers = exp.backbone(model).layers
        prefix = next(n for n, m in model.named_modules() if m is layers)
        for index in reversed(range(len(layers))):
            if resident <= budget:
                break
            offloaded.append(f'{prefix}.{index}')
            resident -= sum(p.numel() * p.element_size() for p in layers[index].parameters())
        if resident > budget:
            raise torch.OutOfMemoryError('Insufficient space for non-layer weights')
        mapping = {n: 'cpu' for n in offloaded}
        for name, module in model.named_modules():
            if any(name == n or name.startswith(n + '.') for n in offloaded):
                continue
            if any(True for _ in module.parameters(recurse=False)) or any(True for _ in module.buffers(recurse=False)):
                mapping[name] = 'cuda:0'
        model = dispatch_model(model, device_map=mapping, main_device='cuda:0', offload_buffers=False)
    else:
        model = model.cuda()
    events_path = ROOT / 'resource_recovery_load_events.json'
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    events.append({'timestamp': exp.datetime.now(exp.timezone.utc).isoformat(), 'family': family,
        'dtype': str(dtype), 'gpu_free_before_bytes': free, 'weight_budget_bytes': budget,
        'resident_weight_bytes': resident, 'offloaded_layer_weights': offloaded, 'compute_device': 'cuda:0'})
    exp.save(events_path, events)
    exp.log(f'Recovery loaded {family}; resident={resident / (1 << 30):.2f}GiB; CPU weight layers={len(offloaded)}')
    return model


def load_model(family, fp32=False):
    for attempt in range(POLICY['maximum_load_attempts']):
        try:
            return place_model(family, fp32)
        except torch.OutOfMemoryError:
            exp.log(f'Recovery transient allocation failure for {family}, attempt {attempt + 1}')
        # Leave the exception block before collecting its model traceback references.
        gc.collect(); torch.cuda.empty_cache()
        time.sleep(20)
    raise torch.OutOfMemoryError(f'Recovery could not load {family} after bounded retries')


def run():
    manifest = exp.prepare()  # Original code/config signatures must still match.
    partial_summary()
    metadata = {'timestamp': exp.datetime.now(exp.timezone.utc).isoformat(), 'policy': POLICY,
        'resume_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_manifest_signature': manifest['manifest_signature'],
        'original_config_signature': exp.signature(exp.CFG), 'experiment_code_sha256': exp.sha_file(ROOT / 'experiment.py'),
        'original_completed_samples': len(list((ROOT / 'runs/pilot/per_sample').glob('*.json'))),
        'protocol_change': False, 'runtime_weight_placement_change': True}
    exp.save(ROOT / 'execution_recovery.json', metadata)
    status = {'status': 'running', 'phase': 'pilot_recovery_gate',
              'timestamp': exp.datetime.now(exp.timezone.utc).isoformat(), 'recovery': True}
    exp.save(ROOT / 'status.json', status)
    exp.load_model = load_model
    # Actual greedy output must agree with one already-completed native baseline.
    first = manifest['rows'][0]
    saved = json.loads((ROOT / f'runs/pilot/per_sample/{first["id"]}.json').read_text())
    toks = exp.load_tokenizers()
    proto = exp.build_protocol(toks, first['bodies'], first['question'])
    model = load_model('qwen')
    pred, _ = exp.generate(model, toks['qwen'], proto['full_ids'])
    assert pred['generated_ids'] == saved['predictions']['native_full']['generated_ids'], 'Recovery placement changed native baseline generation'
    exp.save(ROOT / 'gates/recovery_native.json', {'status': 'passed', 'id': first['id'],
        'criterion': 'Exact greedy32 token match to original saved native_full output', 'policy': POLICY})
    del model, proto, toks
    exp.release()
    status['phase'] = 'pilot'; exp.save(ROOT / 'status.json', status)
    exp.evaluate('pilot')  # Reuses all32 completed outputs and evaluates only missing IDs.
    status.update(status='completed', timestamp=exp.datetime.now(exp.timezone.utc).isoformat())
    exp.save(ROOT / 'status.json', status)
    exp.log('Recovery completed all64 pilot rows and automatic summary')


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        failure = {'status': 'failed', 'phase': 'pilot_recovery', 'recovery': True,
                   'timestamp': exp.datetime.now(exp.timezone.utc).isoformat(), 'error': type(exc).__name__ + ': ' + str(exc)}
        exp.save(ROOT / 'status.json', failure)
        exp.save(ROOT / 'runs/pilot/recovery_failure.json', failure)
        partial_summary()
        exp.update_shared(failure)
        traceback.print_exc()
        raise
