"""Real frozen Full-Sync caches -> Qwen Q/O LoRA, server-A pilot."""
import argparse
import collections
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
PARENT = Path(CFG['parent'])
sys.path.insert(0, str(PARENT))
import experiment as exp
OUT = ROOT
SMOKE = False


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def status(phase, **kwargs):
    save('status.json', {'status': 'running', 'phase': phase, 'timestamp': now(), **kwargs})


def release():
    gc.collect(); torch.cuda.empty_cache()


def wait_gpu(kind):
    threshold = CFG['minimum_gpu_free_gib'][kind] * 2**30
    while torch.cuda.mem_get_info()[0] < threshold:
        status('waiting_GPU', next_phase=kind, minimum_gib=threshold / 2**30)
        log(f'Waiting for {threshold / 2**30:.1f}GiB GPU free for {kind}; unrelated jobs untouched')
        time.sleep(20)


def model_load(family):
    wait_gpu('receiver' if family == 'qwen' else 'source')
    dtype = torch.float32 if family == 'gemma' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(exp.CFG['models'][family], local_files_only=True,
        dtype=dtype, attn_implementation='eager').eval().requires_grad_(False)
    if family == 'gemma':
        for name in ('vision_tower', 'multi_modal_projector'):
            if hasattr(model.model, name): delattr(model.model, name)
    if family == 'qwen':
        return model.cuda()
    cap = int(CFG['source_weight_cap_gib'][family] * 2**30)
    budget = min(cap, torch.cuda.mem_get_info()[0] - 4 * 2**30)
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    resident, offloaded = size, []
    layers = exp.backbone(model).layers
    prefix = next(n for n, m in model.named_modules() if m is layers)
    for index in reversed(range(len(layers))):
        if resident <= budget: break
        offloaded.append(f'{prefix}.{index}')
        resident -= sum(p.numel() * p.element_size() for p in layers[index].parameters())
    assert resident <= budget
    if offloaded:
        from accelerate import dispatch_model
        mapping = {name: 'cpu' for name in offloaded}
        for name, module in model.named_modules():
            if any(name == p or name.startswith(p + '.') for p in offloaded): continue
            if any(True for _ in module.parameters(recurse=False)) or any(True for _ in module.buffers(recurse=False)):
                mapping[name] = 'cuda:0'
        model = dispatch_model(model, device_map=mapping, main_device='cuda:0', offload_buffers=False)
    else: model = model.cuda()
    event = {'family': family, 'dtype': str(dtype), 'resident_gib': resident / 2**30,
             'CPU_offload_layers': len(offloaded), 'compute_device': 'cuda:0', 'timestamp': now()}
    log('Model placement ' + json.dumps(event))
    save(f'placement_{family}.json', event)
    return model


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        r = CFG['rank']
        self.a = nn.Parameter(torch.empty(r, base.in_features, device='cuda', dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device='cuda', dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled: return y
        delta = F.linear(F.linear(x.float(), self.a), self.b).to(y.dtype)
        return y + delta * (CFG['lora_alpha'] / CFG['rank'])


def install(model):
    modules = []
    for layer in model.model.layers:
        for name in ('q_proj', 'o_proj'):
            adapter = LoRA(getattr(layer.self_attn, name))
            setattr(layer.self_attn, name, adapter)
            modules.append((name, adapter))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3833856
    return modules


def enable(modules, flag):
    for _, module in modules: module.enabled = flag


def lora_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore(model, state):
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name, tensor in state.items(): params[name].copy_(tensor.to(params[name].device))


def checkpoint(name, model, optimizer, scaler, epoch, updates, manifest, validation=None):
    path = OUT / 'checkpoints' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({'lora': lora_state(model), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'epoch': epoch, 'updates': updates, 'manifest_signature': manifest['signature'],
        'validation': validation, 'config': CFG, 'timestamp': now()}, tmp)
    tmp.replace(path)


def prepare(tokenizers):
    status('prepare_data')
    existing = OUT / 'manifest.json'
    frozen_dev = json.loads((PARENT / 'manifest.json').read_text())['rows']
    candidates = json.loads(Path(CFG['train_dataset']).read_text())
    random.Random(CFG['seed']).shuffle(candidates)
    ntrain, nval, ndev = (2, 1, 1) if SMOKE else (CFG['train_samples'], CFG['validation_samples'], CFG['evaluation_samples'])
    rows, rejected = [], collections.Counter()
    known_dev_ids = {r['id'] for r in frozen_dev}
    def attach(row, split):
        proto = exp.build_protocol(tokenizers, row['bodies'], row['question'])
        tok = tokenizers['qwen']
        target = tok.encode(' ' + row['answer'], add_special_tokens=False)
        if split != 'evaluation' and (not target or len(target) > CFG['max_answer_tokens']): raise ValueError('answer_length')
        suffix_text = 'Question: ' + row['question'] + '\nAnswer:'
        combined = tok.encode(suffix_text + ' ' + row['answer'], add_special_tokens=False)
        if split != 'evaluation' and combined != proto['suffix_ids'] + target: raise ValueError('answer_boundary_tokenization')
        return {**row, 'split': split, 'target_ids': target + [tok.eos_token_id], 'protocol': proto}
    for item in candidates:
        if item['type'] != 'bridge': rejected['not_bridge'] += 1; continue
        if item['_id'] in known_dev_ids: rejected['dev_id_overlap'] += 1; continue
        titles = {title for title, _ in item['supporting_facts']}
        docs = [(title, sentences) for title, sentences in item['context'] if title in titles]
        if len(docs) != 2: rejected['not_gold2'] += 1; continue
        bodies = [f'Document {i + 1}:\nTitle: {title}\n' + ' '.join(sents) + '\n\n' for i, (title, sents) in enumerate(docs)]
        row = {'id': item['_id'], 'question': item['question'], 'answer': item['answer'], 'bodies': bodies,
               'titles': [x[0] for x in docs]}
        try: row = attach(row, 'train' if len(rows) < ntrain else 'validation')
        except ValueError as e: rejected[str(e)] += 1; continue
        rows.append(row)
        if len(rows) == ntrain + nval: break
    assert len(rows) == ntrain + nval
    rows += [attach(row, 'evaluation') for row in frozen_dev[:ndev]]
    assert len({r['id'] for r in rows}) == len(rows)
    del candidates
    files = [ROOT / 'config.json', ROOT / 'train_receiver.py', PARENT / 'config.json', PARENT / 'experiment.py',
             PARENT / 'manifest.json', PARENT / 'fullsync_units.py', PARENT / 'writer_selections.json']
    identity = {'config': CFG, 'smoke': SMOKE, 'parent_manifest_signature': json.loads((PARENT / 'manifest.json').read_text())['manifest_signature'],
        'files': {str(p): exp.sha_file(p) for p in files}, 'train_dataset_sha256': exp.sha_file(Path(CFG['train_dataset'])),
        'samples': [{k: r[k] for k in ('id', 'split', 'question', 'answer', 'bodies', 'target_ids')} for r in rows],
        'rejected': dict(rejected), 'scope': {'train': ntrain, 'validation': nval, 'evaluation': ndev},
        'parent_writer_artifacts': json.loads((PARENT / 'manifest.json').read_text())['artifacts']}
    identity['signature'] = exp.signature(identity)
    if existing.exists(): assert json.loads(existing.read_text())['signature'] == identity['signature'], 'Immutable inputs changed'
    else: save('manifest.json', identity)
    # Upper bound for retained mapped joint/independent caches plus native memory and source-stage peak.
    tokens = sum(len(r['protocol']['units']['qwen']['joint']['body_ids']) for r in rows)
    retained = tokens * 147456 * 6.5
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    assert retained < min(CFG['max_cpu_kv_gib'] * 2**30, available * .7), 'CPU cache estimate exceeds configured budget'
    log(f'Prepared train={ntrain}, validation={nval}, evaluation={ndev}; estimated retained KV={retained / 2**30:.1f}GiB')
    return rows, identity


def translated_caches(rows):
    memories = {r['id']: {} for r in rows}
    for family in ('llama', 'gemma'):
        status('capture_source', family=family, sample_index=0, sample_count=len(rows))
        source = model_load(family)
        captured = {}
        for index, row in enumerate(rows):
            proto = row['protocol']
            captured[row['id']] = {unit: exp.capture(source, proto['units'][family][unit]['ids']) for unit in ('a', 'b', 'joint')}
            status('capture_source', family=family, sample_index=index + 1, sample_count=len(rows))
            log(f'Capture {family} {index + 1}/{len(rows)} {row["id"]}')
        del source
        release()
        wait_gpu('writer')
        writer = exp.load_writer(family)
        for index, row in enumerate(rows):
            rid = row['id']
            memories[rid][family] = {unit: tuple(t.clone() for t in exp.translate(writer, captured[rid][unit], row['protocol']['maps'][family][unit]))
                                       for unit in ('a', 'b', 'joint')}
            del captured[rid]
            status('translate_frozen_writer', family=family, sample_index=index + 1, sample_count=len(rows))
            log(f'Translate {family} {index + 1}/{len(rows)}')
        del writer, captured
        release()
    return memories


def complete_native(rows, memory, model):
    model.eval()
    for index, row in enumerate(rows):
        proto = row['protocol']; n = len(proto['prefix_ids'])
        native = {unit: tuple(t.clone() for t in exp.capture(model, proto['units']['qwen'][unit]['ids'])) for unit in ('a', 'b', 'joint')}
        memory[row['id']]['native_joint'] = native['joint']
        memory[row['id']]['prefix'] = tuple(t[:, :n].clone() for t in native['joint'])
        memory[row['id']]['native_parts'] = {unit: tuple(t[:, n:].clone() for t in native[unit]) for unit in ('a', 'b')}
        status('capture_native_teacher', sample_index=index + 1, sample_count=len(rows))
        log(f'Native memory {index + 1}/{len(rows)}')


def arm_memory(row, memories, arm):
    item = memories[row['id']]
    prefix = item['prefix']
    n = item['native_parts']
    if arm == 'native_full': return item['native_joint']
    if arm == 'query_only': return prefix
    if arm == 'native_independent': return exp.join_pairs(prefix, n['a'], n['b'])
    if arm.endswith('_joint'): return exp.join_pairs(prefix, item[arm.split('_')[0]]['joint'])
    if arm.endswith('_independent'):
        f = arm.split('_')[0]; return exp.join_pairs(prefix, item[f]['a'], item[f]['b'])
    a, b = {'llama_a_gemma_b': ('llama','gemma'), 'gemma_a_llama_b': ('gemma','llama'),
        'llama_a_native_b': ('llama','native'), 'native_a_gemma_b': ('native','gemma')}[arm]
    return exp.join_pairs(prefix, n['a'] if a == 'native' else item[a]['a'], n['b'] if b == 'native' else item[b]['b'])


def forward_gold(model, row, memory):
    suffix, target = row['protocol']['suffix_ids'], row['target_ids']
    tokens = suffix + target[:-1]
    inp = torch.tensor([tokens], device='cuda')
    cache = exp.make_cache(model, memory)
    prefix = memory[0].shape[1]
    out = model.model(input_ids=inp, attention_mask=torch.ones((1, prefix + len(tokens)), device='cuda', dtype=torch.long),
        position_ids=torch.arange(prefix, prefix + len(tokens), device='cuda')[None], past_key_values=cache,
        use_cache=False, return_dict=True)
    hidden = out.last_hidden_state[:, len(suffix) - 1:len(suffix) - 1 + len(target)]
    return model.lm_head(hidden)[0].float()


def objective(logits, teacher, row):
    target = torch.tensor(row['target_ids'], device='cuda')
    ce = F.cross_entropy(logits, target)
    reference = teacher.float().cuda().log_softmax(-1)
    kl = (reference.exp() * (reference - logits.log_softmax(-1))).sum(-1).mean()
    return ce + CFG['kl_weight'] * kl, ce, kl


def teacher_logits(rows, memory, model, modules):
    enable(modules, False); model.eval()
    for index, row in enumerate(rows):
        if row['split'] == 'evaluation': continue
        with torch.no_grad():
            logits = forward_gold(model, row, arm_memory(row, memory, 'native_full'))
        memory[row['id']]['teacher'] = logits.half().cpu()
        log(f'Frozen teacher targets {index + 1}/{len(rows)}')
    enable(modules, True)


def validation(rows, memory, model, modules):
    model.eval(); enable(modules, True)
    records = []
    with torch.no_grad():
        for row in rows:
            for arm in ('llama_a_gemma_b', 'gemma_a_llama_b'):
                logits = forward_gold(model, row, arm_memory(row, memory, arm))
                loss, ce, kl = objective(logits, memory[row['id']]['teacher'], row)
                records.append({'id': row['id'], 'arm': arm, 'loss': float(loss), 'ce': float(ce), 'kl': float(kl)})
    return {'loss': sum(r['loss'] for r in records) / len(records), 'records': records}


def gates(rows, memory, model, modules):
    row = rows[0]
    model.eval(); enable(modules, False)
    with torch.no_grad(): reference = forward_gold(model, row, arm_memory(row, memory, 'llama_a_gemma_b'))
    enable(modules, True)
    with torch.no_grad(): same = forward_gold(model, row, arm_memory(row, memory, 'llama_a_gemma_b'))
    assert torch.equal(reference, same), 'Zero-LoRA identity failed'
    # Reconstruction is checked on first native decision logits, not benchmark correctness.
    enable(modules, False)
    full, _ = exp.forward(model, row['protocol']['full_ids'])
    rebuilt, _ = exp.forward(model, row['protocol']['suffix_ids'], exp.make_cache(model, memory[row['id']]['native_joint']))
    p, q = full.log_softmax(-1), rebuilt.log_softmax(-1)
    floor = float((p.exp() * (p - q)).sum())
    assert abs(floor) < 1e-3 and float((full - rebuilt).abs().mean()) < .05
    enable(modules, True)
    save('gates/real_KV.json', {'status': 'passed', 'id': row['id'], 'zero_LoRA_exact': True,
        'native_reconstruction_KL': floor, 'finite_fullsync_sources': ['llama','gemma'],
        'no_raw_documents_in_student': True, 'timestamp': now()})


def train(rows, memory, model, modules, manifest):
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'validation']
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    updates, start_epoch, best = 0, 1, float('inf')
    last = OUT / 'checkpoints/last.pt'
    if last.exists():
        payload = torch.load(last, map_location='cpu', weights_only=True)
        assert payload['manifest_signature'] == manifest['signature']
        restore(model, payload['lora']); optimizer.load_state_dict(payload['optimizer']); scaler.load_state_dict(payload['scaler'])
        start_epoch = payload['epoch'] + 1; updates = payload['updates']
        selection = OUT / 'selection.json'
        if selection.exists(): best = json.loads(selection.read_text())['validation_loss']
    epochs = 1 if SMOKE else CFG['epochs']
    history = []
    for epoch in range(start_epoch, epochs + 1):
        ordered = list(train_rows); random.Random(CFG['seed'] + epoch).shuffle(ordered)
        model.train(); enable(modules, True); optimizer.zero_grad(set_to_none=True)
        choices = CFG['curriculum']['1'] if epoch == 1 else CFG['curriculum']['2-4']
        losses = []
        accumulation = 1 if SMOKE else CFG['effective_batch']
        for index, row in enumerate(ordered):
            arm = choices[(index + epoch - 1) % len(choices)]
            start = time.monotonic()
            status('train', epoch=epoch, sample_index=index, sample_count=len(ordered), updates=updates, arm=arm)
            logits = forward_gold(model, row, arm_memory(row, memory, arm))
            loss, ce, kl = objective(logits, memory[row['id']]['teacher'], row)
            keep = 0.0
            if index % CFG['native_keep_every'] == 0:
                native_logits = forward_gold(model, row, arm_memory(row, memory, 'native_full'))
                native_loss, _, _ = objective(native_logits, memory[row['id']]['teacher'], row)
                loss = loss + CFG['native_keep_weight'] * native_loss
                keep = float(native_loss.detach())
            assert torch.isfinite(loss)
            window = min(accumulation, len(ordered) - (index // accumulation) * accumulation)
            scaler.scale(loss / window).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(ordered):
                scaler.unscale_(optimizer)
                finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                assert finite, 'Nonfinite gradient; checkpoint not advanced'
                norms = {name: math.sqrt(sum(float(m.b.grad.float().square().sum()) for kind, m in modules if kind == name))
                         for name in ('q_proj','o_proj')}
                assert all(n > 0 for n in norms.values()), norms
                torch.nn.utils.clip_grad_norm_(parameters, CFG['gradient_clip'])
                scaler.step(optimizer); scaler.update(); updates += 1
                optimizer.zero_grad(set_to_none=True)
                if updates == 1:
                    save('gates/training_gradient.json', {'status': 'passed', 'gradient_norms_B': norms,
                        'loss': float(loss.detach()), 'updates': updates, 'timestamp': now()})
                    checkpoint('first_update.pt', model, optimizer, scaler, 0, updates, manifest)
            record = {'epoch': epoch, 'sample': index + 1, 'id': row['id'], 'arm': arm, 'loss': float(loss.detach()),
                'ce': float(ce.detach()), 'kl': float(kl.detach()), 'native_keep_loss': keep,
                'updates': updates, 'seconds': time.monotonic() - start,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            with (OUT / 'train_metrics.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(record) + '\n')
            losses.append(record['loss'])
            log(f'Train epoch{epoch} {index+1}/{len(ordered)} {arm} CE={record["ce"]:.4f} KL={record["kl"]:.4f} update={updates}')
            del logits, loss, ce, kl
            if index % CFG['native_keep_every'] == 0: del native_logits, native_loss
        valid = validation(val_rows, memory, model, modules)
        save(f'validation/epoch_{epoch}.json', valid)
        checkpoint('last.pt', model, optimizer, scaler, epoch, updates, manifest, valid)
        if valid['loss'] < best:
            best = valid['loss']
            checkpoint('best.pt', model, optimizer, scaler, epoch, updates, manifest, valid)
            save('selection.json', {'epoch': epoch, 'updates': updates, 'validation_loss': best,
                'rule': CFG['checkpoint_selection'], 'manifest_signature': manifest['signature']})
        history.append({'epoch': epoch, 'train_loss': sum(losses)/len(losses), 'validation_loss': valid['loss'], 'updates': updates})
        save('training_summary.json', {'status': 'running', 'epochs': history, 'best_validation_loss': best})
        release()
    payload = torch.load(OUT / 'checkpoints/best.pt', map_location='cpu', weights_only=True)
    restore(model, payload['lora'])
    save('training_summary.json', {'status': 'completed', 'epochs': history, 'best_validation_loss': best,
        'selection': json.loads((OUT / 'selection.json').read_text())})
    return updates


def generate(model, tokenizer, row, memory, mask=None):
    tokens, first = [], None
    inputs = row['protocol']['suffix_ids']
    cache = exp.make_cache(model, memory)
    prefix = memory[0].shape[1]
    with torch.no_grad():
        for step in range(CFG['max_new_tokens']):
            start = prefix if step == 0 else prefix + len(row['protocol']['suffix_ids']) + step - 1
            inp = torch.tensor([inputs], device='cuda')
            total = prefix + len(row['protocol']['suffix_ids']) + step
            attn = torch.ones((1, total), device='cuda', dtype=torch.long)
            if mask is not None: attn[0, :prefix] = mask.to('cuda')
            out = model.model(input_ids=inp, attention_mask=attn, position_ids=torch.arange(start,start+len(inputs),device='cuda')[None],
                              past_key_values=cache, use_cache=True, return_dict=True)
            logits = model.lm_head(out.last_hidden_state[:, -1])[0].float()
            if first is None: first = logits.cpu()
            token = int(logits.argmax()); tokens.append(token)
            cache = out.past_key_values
            if token == tokenizer.eos_token_id: break
            inputs = [token]
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = exp.parsed_answer(raw)
    return {'raw_text': raw, 'generated_ids': tokens, 'parsed_answer': parsed,
            **exp.score(parsed, row['answer']), 'hit_max_tokens': len(tokens) == CFG['max_new_tokens']}, first


def evaluate(rows, memory, model, modules, tokenizer):
    model.eval()
    chosen = [r for r in rows if r['split'] == 'evaluation']
    arms = ['native_full','query_only','native_independent','llama_joint','gemma_joint','llama_independent',
            'gemma_independent','llama_a_gemma_b','gemma_a_llama_b','llama_a_native_b','native_a_gemma_b']
    outputs = []
    for index, row in enumerate(chosen):
        record = {'id': row['id'], 'answer': row['answer'], 'question': row['question'], 'predictions': {}}
        for flag in (False, True):
            enable(modules, flag)
            condition = 'adapted' if flag else 'frozen'
            record['predictions'][condition] = {}
            for arm in arms + ['wrong_KV','drop_A','drop_B']:
                status('evaluation', sample_index=index, sample_count=len(chosen), adapter=condition, arm=arm)
                mem = arm_memory(row, memory, 'llama_a_gemma_b' if arm in ['wrong_KV','drop_A','drop_B'] else arm)
                mask = None
                if arm in ['drop_A','drop_B']:
                    mask = torch.ones(mem[0].shape[1], dtype=torch.long)
                    n = len(row['protocol']['prefix_ids']); a = len(row['protocol']['units']['qwen']['a']['body_ids'])
                    if arm == 'drop_A': mask[n:n+a] = 0
                    else: mask[n+a:] = 0
                if arm == 'wrong_KV':
                    other = chosen[(index + 1) % len(chosen)] if len(chosen) > 1 else next(r for r in rows if r['id'] != row['id'])
                    wrong = arm_memory(other, memory, 'llama_a_gemma_b')
                    n = len(row['protocol']['prefix_ids']); other_n = len(other['protocol']['prefix_ids'])
                    idx = torch.linspace(other_n, wrong[0].shape[1]-1, mem[0].shape[1]-n).round().long()
                    mem = tuple(torch.cat((m[:, :n], w[:, idx]),1) for m,w in zip(mem,wrong))
                pred, first = generate(model, tokenizer, row, mem, mask)
                record['predictions'][condition][arm] = pred
                del first
            log(f'Evaluation {index+1}/{len(chosen)} {condition} mixed F1={record["predictions"][condition]["llama_a_gemma_b"]["f1"]:.3f}')
        save(f'per_sample/evaluation/{row["id"]}.json', record)
        outputs.append(record)
    metrics = {condition: {arm: {metric: sum(r['predictions'][condition][arm][metric] for r in outputs)/len(outputs)
                                for metric in ('em','f1','hit_max_tokens')}
                          for arm in outputs[0]['predictions'][condition]} for condition in ('frozen','adapted')}
    return {'samples': len(outputs), 'metrics': metrics, 'parser': 'Unchanged parent short-answer parser',
            'automatic_metrics': True, 'manual_audit_performed': False}


def shared(result):
    if SMOKE: return
    import fcntl
    path = Path(CFG['shared_results'])
    begin, end = '<!-- BEGIN ' + CFG['experiment_name'] + ' -->', '<!-- END ' + CFG['experiment_name'] + ' -->'
    section = '\n'.join([begin, '## ' + CFG['experiment_name'], '', '- Server A; status: ' + result['status'],
        '- Timestamp: ' + now(), '- Frozen non-lm-eval Full-Sync writers; Qwen Q/O LoRA rank8.',
        '- Train128 + validation16 official train; evaluation32 fixed existing dev IDs.',
        '- Config: `' + str(ROOT/'config.json') + '`; manifest: `' + str(OUT/'manifest.json') + '`',
        '- Summary: `' + str(OUT/'summary.json') + '`; log: `' + str(ROOT/'logs/pipeline.log') + '`',
        '- Details: ' + json.dumps(result.get('evaluation', result.get('error','')),ensure_ascii=False), '', end, ''])
    with (path.parent/'.experiment_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = path.read_text() if path.exists() else '# Experiment Results\n'
        if begin in text:
            start=text.index(begin); stop=text.index(end,start)+len(end); text=text[:start]+section+text[stop:]
        else:text+='\n'+section
        tmp=path.with_suffix('.md.tmp'); tmp.write_text(text,encoding='utf-8'); tmp.replace(path)


def run():
    assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.manual_seed(CFG['seed']); random.seed(CFG['seed'])
    tokenizers = exp.load_tokenizers()
    rows, manifest = prepare(tokenizers)
    memories = translated_caches(rows)
    model = model_load('qwen')
    complete_native(rows, memories, model)
    modules = install(model)
    teacher_logits(rows, memories, model, modules)
    gates(rows, memories, model, modules)
    updates = train(rows, memories, model, modules, manifest)
    evaluation = evaluate(rows, memories, model, modules, tokenizers['qwen'])
    result = {'status': 'completed', 'timestamp': now(), 'server': 'A', 'smoke': SMOKE,
        'manifest_signature': manifest['signature'], 'trainable_parameters': 3833856, 'optimizer_updates': updates,
        'selection': json.loads((OUT/'selection.json').read_text()), 'evaluation': evaluation}
    save('summary.json', result); shared(result)
    save('status.json', {'status': 'completed', 'timestamp': now(), 'optimizer_updates': updates})
    log('Completed receiver training, validation selection and paired evaluation')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    SMOKE=args.smoke; OUT=ROOT/'runs'/('smoke' if SMOKE else 'pilot'); OUT.mkdir(parents=True,exist_ok=True)
    try:run()
    except Exception as exc:
        failure={'status':'failed','timestamp':now(),'error':type(exc).__name__+': '+str(exc)}
        save('status.json',failure); save('failure.json',failure); shared(failure); traceback.print_exc(); raise
