"""Server-B evaluation-only pilot. Evidence KV never persists to disk."""
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
import re
import string
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
HEAD = ('Read the supplied documents and answer the question using their information. '
        'Return only the short answer. Do not explain or show reasoning.\n\n')


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def signature(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(answer):
    answer = ''.join(c for c in answer.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', answer).split())


def score(prediction, answer):
    p, a = normalize(prediction), normalize(answer)
    em = float(p == a)
    if p in ('yes', 'no', 'noanswer') or a in ('yes', 'no', 'noanswer'):
        return {'em': em, 'f1': em}
    pp, aa = p.split(), a.split()
    common = sum((collections.Counter(pp) & collections.Counter(aa)).values())
    f1 = 2 * common / (len(pp) + len(aa)) if common else 0.0
    return {'em': em, 'f1': f1}


def parsed_answer(raw):
    lines = [x.strip() for x in raw.strip().splitlines() if x.strip()]
    if not lines:
        return ''
    tagged = [x.split(':', 1)[1].strip() for x in lines if x.lower().startswith('answer:')]
    return tagged[-1] if tagged else lines[0]


def encoded(tok, text):
    out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return {'ids': bos + list(out['input_ids']),
            'offsets': [(0, 0)] * len(bos) + [tuple(x) for x in out['offset_mapping']]}


def split_at(out, cut):
    for a, b in out['offsets']:
        if a < cut < b:
            raise ValueError('token_crosses_text_boundary')
    return sum(b <= cut for a, b in out['offsets'])


def unit(tok, body):
    out = encoded(tok, HEAD + body)
    n = split_at(out, len(HEAD))
    spans = [(a - len(HEAD), b - len(HEAD)) for a, b in out['offsets'][n:]]
    if not spans or any(a == b for a, b in spans):
        raise ValueError('empty_or_special_evidence_token')
    return {'ids': out['ids'], 'prefix_count': n, 'spans': spans,
            'body_ids': out['ids'][n:], 'body_tokens': len(spans)}


def align(source, target):
    """Full-Sync copy/drop on common character-END boundaries, no native fallback."""
    ss, ts = source['spans'], target['spans']
    bounds = sorted({0} | ({b for a, b in ss} & {b for a, b in ts}))
    result = {}
    si = ti = 0
    for left, right in zip(bounds, bounds[1:]):
        s, t = [], []
        while si < len(ss) and ss[si][1] <= right:
            if left < ss[si][1]:
                s.append(si)
            si += 1
        while ti < len(ts) and ts[ti][1] <= right:
            if left < ts[ti][1]:
                t.append(ti)
            ti += 1
        if not s or not t:
            continue
        for rank, idx in enumerate(t):
            pos = math.ceil((rank + 1) * len(s) / len(t)) - 1 if len(s) > len(t) else rank * len(s) // len(t)
            result[idx] = source['prefix_count'] + s[pos]
    if set(result) != set(range(len(ts))):
        raise ValueError('incomplete_fullsync_alignment')
    return [result[i] for i in range(len(ts))]


def build_protocol(toks, bodies, question):
    units = {f: {name: unit(tok, body) for name, body in
                  [('a', bodies[0]), ('b', bodies[1]), ('joint', ''.join(bodies))]}
             for f, tok in toks.items()}
    q = units['qwen']
    suffix_text = 'Question: ' + question + '\nAnswer:'
    out = encoded(toks['qwen'], HEAD + ''.join(bodies) + suffix_text)
    stop = split_at(out, len(HEAD) + len(''.join(bodies)))
    n = q['joint']['prefix_count']
    if out['ids'][:stop] != q['joint']['ids']:
        raise ValueError('question_changed_evidence_tokenization')
    if q['a']['prefix_count'] != n or q['b']['prefix_count'] != n:
        raise ValueError('prefix_length_changed')
    if q['a']['ids'][:n] != q['joint']['ids'][:n] or q['b']['ids'][:n] != q['joint']['ids'][:n]:
        raise ValueError('public_prefix_ids_changed')
    if q['a']['body_ids'] + q['b']['body_ids'] != q['joint']['body_ids']:
        raise ValueError('independent_token_grid_mismatch')
    if any(units[f][x]['body_tokens'] > CFG['max_document_tokens'] for f in toks for x in ('a', 'b')):
        raise ValueError('document_overlength')
    if any(units[f]['joint']['body_tokens'] > CFG['max_joint_tokens'] for f in toks):
        raise ValueError('joint_overlength')
    maps = {f: {name: align(units[f][name], q[name]) for name in ('a', 'b', 'joint')}
            for f in ('llama', 'gemma')}
    return {'units': units, 'maps': maps, 'full_ids': out['ids'],
            'suffix_ids': out['ids'][stop:], 'prefix_ids': q['joint']['ids'][:n],
            'query_only_ids': encoded(toks['qwen'], HEAD + suffix_text)['ids']}


def load_tokenizers():
    from transformers import AutoTokenizer
    return {f: AutoTokenizer.from_pretrained(p, local_files_only=True) for f, p in CFG['models'].items()}


def prepare():
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config_signature'] != signature(CFG):
            raise RuntimeError('Immutable config changed after manifest creation')
        if manifest['script_sha256'] != sha_file(ROOT / 'experiment.py'):
            raise RuntimeError('Experiment code changed after manifest creation')
        return manifest
    toks = load_tokenizers()
    candidates = json.loads(path(CFG['dataset']).read_text())
    random.Random(CFG['seed']).shuffle(candidates)
    rows, rejected = [], collections.Counter()
    for item in candidates:
        if item['type'] != 'bridge':
            rejected['not_bridge'] += 1
            continue
        titles = {title for title, _ in item['supporting_facts']}
        docs = [(title, sentences) for title, sentences in item['context'] if title in titles]
        if len(docs) != 2:
            rejected['not_exactly_two_support_documents'] += 1
            continue
        bodies = [f'Document {i + 1}:\nTitle: {title}\n' + ' '.join(sents) + '\n\n'
                  for i, (title, sents) in enumerate(docs)]
        try:
            proto = build_protocol(toks, bodies, item['question'])
        except ValueError as e:
            rejected[str(e)] += 1
            continue
        rows.append({'id': item['_id'], 'question': item['question'], 'answer': item['answer'],
                     'bodies': bodies, 'titles': [x[0] for x in docs],
                     'counts': {f: {x: u['body_tokens'] for x, u in family.items()}
                                for f, family in proto['units'].items()}})
        if len(rows) == CFG['samples']:
            break
    if len(rows) != CFG['samples']:
        raise RuntimeError(f'Insufficient eligible data: {len(rows)}; rejected={dict(rejected)}')
    artifacts = {}
    for f, w in CFG['writers'].items():
        for kind in ('stage_a', 'residual', 'module'):
            p = path(w[kind])
            log(f'hash {f}/{kind}: {p}')
            artifacts[f + '/' + kind] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha_file(p)}
    import torch, transformers
    models = {f: {'path': p, 'config_sha256': sha_file(Path(p) / 'config.json'),
                  'tokenizer_sha256': sha_file(Path(p) / 'tokenizer.json') if (Path(p) / 'tokenizer.json').exists() else None,
                  'dtype': 'float32' if f == 'gemma' else 'float16'} for f, p in CFG['models'].items()}
    manifest = {'status': 'prepared', 'config': CFG, 'config_signature': signature(CFG),
                'script_sha256': sha_file(ROOT / 'experiment.py'), 'model_identities': models,
                'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
                'dataset_sha256': sha_file(path(CFG['dataset'])), 'artifacts': artifacts,
                'selection': 'Seeded bridge filtering, complete gold paragraphs, length/alignment checks only; no answer-based selection',
                'rejected_before_model_answers': dict(rejected), 'rows': rows}
    manifest['manifest_signature'] = signature(manifest)
    save(manifest_path, manifest)
    log(f'prepared {len(rows)} examples, rejects={dict(rejected)}')
    return manifest


def backbone(model):
    return getattr(model.model, 'language_model', model.model)


def text_config(model):
    return getattr(model.config, 'text_config', model.config)


def load_model(family, fp32=False):
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    dtype = torch.float32 if family == 'gemma' or fp32 else torch.float16
    log(f'load {family} {dtype}')
    m = AutoModelForCausalLM.from_pretrained(CFG['models'][family], local_files_only=True,
           dtype=dtype, attn_implementation=CFG['attention_implementation'])
    if family == 'gemma':
        for attr in ('vision_tower', 'multi_modal_projector'):
            if hasattr(m.model, attr):
                delattr(m.model, attr)
    m = m.cuda().eval().requires_grad_(False)
    cfg = text_config(m)
    observed = (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim)
    expected = {'llama': (28, 8, 128), 'gemma': (34, 4, 256), 'qwen': (36, 8, 128)}[family]
    if observed != expected:
        raise RuntimeError(f'Unexpected {family} geometry: {observed}')
    return m


def release():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def capture(model, ids):
    import torch
    keys, values, handles = {}, {}, []
    def hook(index):
        def receive(module, args, kwargs):
            h = kwargs.get('hidden_states', args[0] if args else None)
            shape = (*h.shape[:-1], -1, module.head_dim)
            k = module.k_proj(h).view(shape)
            if hasattr(module, 'k_norm'):
                k = module.k_norm(k)
            keys[index] = k[0].detach().cpu()
            values[index] = module.v_proj(h).view(shape)[0].detach().cpu()
        return receive
    for index, layer in enumerate(backbone(model).layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        with torch.inference_mode():
            inp = torch.tensor([ids], device='cuda', dtype=torch.long)
            backbone(model)(input_ids=inp, attention_mask=torch.ones_like(inp),
                position_ids=torch.arange(len(ids), device='cuda')[None], use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return (torch.stack([keys[i] for i in range(len(keys))]),
            torch.stack([values[i] for i in range(len(values))]))


def tensor_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_writer(family):
    import torch
    w = CFG['writers'][family]
    spec = importlib.util.spec_from_file_location('frozen_' + family, path(w['module']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_payload = torch.load(path(w['stage_a']), map_location='cpu', weights_only=True, mmap=True)
    residual_payload = torch.load(path(w['residual']), map_location='cpu', weights_only=True, mmap=True)
    if base_payload.get('step') != w['stage_a_step'] or residual_payload.get('step') != w['residual_step']:
        raise RuntimeError(f'{family} checkpoint steps differ from pinned selection')
    if residual_payload.get('kind') != 'residual':
        raise RuntimeError('Expected residual checkpoint')
    if base_payload.get('signature') != residual_payload.get('signature'):
        raise RuntimeError('Stage-A/Residual training signature mismatch')
    with torch.device('meta'):
        base = module.NativeKVTranslator()
        residual = module.ResidualKVAdapter(rank=64)
    base.load_state_dict(base_payload['state'], strict=True, assign=True)
    residual.load_state_dict(residual_payload['state'], strict=True, assign=True)
    if tensor_hash(base) != residual_payload.get('base_hash'):
        raise RuntimeError(f'{family} residual does not belong to this Stage-A')
    base = base.cuda().eval().requires_grad_(False)
    residual = residual.cuda().eval().requires_grad_(False)
    return base, residual


def translate(writer, source, indices):
    import torch
    base, residual = writer
    kk, vv = [], []
    for start in range(0, len(indices), 32):
        selected = indices[start:start + 32]
        keep = len(selected)
        padded = selected + [selected[-1]] * (32 - keep)
        k = source[0][:, padded].unsqueeze(0).cuda().float()
        v = source[1][:, padded].unsqueeze(0).cuda().float()
        with torch.inference_mode():
            k, v = base(k, v)
            k, v, _, _ = residual(k, v)
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise RuntimeError('Nonfinite mapped memory')
        kk.append(k[0, :, :keep].half().cpu())
        vv.append(v[0, :, :keep].half().cpu())
    return torch.cat(kk, 1), torch.cat(vv, 1)


def memory_digest(pair):
    h = hashlib.sha256()
    for x in pair:
        h.update(x.contiguous().numpy().tobytes())
    return h.hexdigest()


def join_pairs(*pairs):
    import torch
    return tuple(torch.cat([pair[i] for pair in pairs], 1) for i in (0, 1))


def make_cache(model, memory):
    import torch
    from transformers import DynamicCache
    k, v = [x.cuda().to(next(model.parameters()).dtype) for x in memory]
    pos = torch.arange(k.shape[1], device='cuda')[None]
    cos, sin = backbone(model).rotary_emb(k[0][None], pos)
    half1, half2 = k.chunk(2, dim=-1)
    rotated = k * cos[0, :, None] + torch.cat((-half2, half1), -1) * sin[0, :, None]
    data = [(rotated[l].permute(1, 0, 2)[None], v[l].permute(1, 0, 2)[None])
            for l in range(k.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=text_config(model))


def forward(model, ids, cache=None):
    import torch
    n = cache.get_seq_length() if cache is not None else 0
    inp = torch.tensor([ids], device='cuda', dtype=torch.long)
    with torch.inference_mode():
        out = backbone(model)(input_ids=inp, attention_mask=torch.ones((1, n + len(ids)), device='cuda', dtype=torch.long),
            position_ids=torch.arange(n, n + len(ids), device='cuda')[None], past_key_values=cache, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1])[0].float()
    return logits, out.past_key_values


def generate(model, tok, ids, memory=None, limit=None):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    cache = make_cache(model, memory) if memory is not None else None
    logits, cache = forward(model, ids, cache)
    first = logits.cpu()
    torch.cuda.synchronize()
    ttft = time.perf_counter() - start
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    output, stop = [], 'max_tokens'
    for _ in range(limit or CFG['max_new_tokens']):
        token = int(logits.argmax())
        if token in eos:
            stop = 'eos'
            break
        output.append(token)
        logits, cache = forward(model, [token], cache)
    torch.cuda.synchronize()
    raw = tok.decode(output, skip_special_tokens=True)
    result = {'raw_answer': raw, 'parsed_answer': parsed_answer(raw), 'generated_ids': output,
              'stop': stop, 'first_token_id': int(first.argmax()), 'ttft_seconds': ttft,
              'generation_seconds': time.perf_counter() - start}
    return result, first


def compare_kv(mapped, native):
    import torch.nn.functional as F
    result = {}
    for name, m, n in zip(('k', 'v'), mapped, native):
        m, n = m.float(), n.float()
        result[name + '_nmse'] = float((m - n).square().mean() / n.square().mean().clamp_min(1e-12))
        result[name + '_cosine'] = float(F.cosine_similarity(m, n, dim=-1).mean())
    return result


def cpu_checks():
    assert score('The Eiffel Tower.', 'Eiffel Tower') == {'em': 1.0, 'f1': 1.0}
    assert score('yes', 'no') == {'em': 0.0, 'f1': 0.0}
    assert score('red blue', 'blue green')['f1'] == 0.5
    assert align({'spans': [(0, 1), (1, 3)], 'prefix_count': 2},
                 {'spans': [(0, 1), (1, 2), (2, 3)]}) == [2, 3, 3]
    assert align({'spans': [(0, 1), (1, 2), (2, 3)], 'prefix_count': 1},
                 {'spans': [(0, 3)]}) == [3]
    try:
        split_at({'offsets': [(0, 0), (0, 4)]}, 2)
    except ValueError:
        pass
    else:
        raise AssertionError('Boundary crossing was not rejected')
    save(ROOT / 'gates/cpu.json', {'status': 'passed', 'checks': ['official_answer_metric_cases', 'copy_drop_alignment', 'token_boundary_rejection']})


def gpu_gate():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    toks = load_tokenizers()
    candidates = json.loads(path(CFG['dataset']).read_text())
    random.Random(CFG['seed']).shuffle(candidates)
    row = proto = None
    for item in candidates:
        if item['type'] != 'bridge':
            continue
        titles = {title for title, _ in item['supporting_facts']}
        docs = [(title, sentences) for title, sentences in item['context'] if title in titles]
        if len(docs) != 2:
            continue
        bodies = [f'Document {i + 1}:\nTitle: {title}\n' + ' '.join(sents) + '\n\n'
                  for i, (title, sents) in enumerate(docs)]
        try:
            proto = build_protocol(toks, bodies, item['question'])
        except ValueError:
            continue
        row = {'id': item['_id'], 'bodies': bodies, 'question': item['question']}
        break
    if row is None:
        raise RuntimeError('No eligible document pair for native gate')
    model = load_model('qwen', fp32=True)
    k, v = capture(model, proto['units']['qwen']['joint']['ids'])
    a, first_full = generate(model, toks['qwen'], proto['full_ids'], limit=16)
    b, first_split = generate(model, toks['qwen'], proto['suffix_ids'], (k, v), limit=16)
    err = float((first_full - first_split).abs().max())
    mae = float((first_full - first_split).abs().mean())
    ok = mae < 1e-4 and a['generated_ids'] == b['generated_ids'] and a['first_token_id'] == b['first_token_id']
    save(ROOT / 'gates/native_fp32.json', {'status': 'passed' if ok else 'failed', 'id': row['id'],
        'logit_max_abs': err, 'logit_mean_abs': mae, 'full': a, 'reconstructed': b,
        'criterion': 'FP32 MAE<1e-4 and exact first argmax and 16 generated tokens'})
    del model, k, v
    release()
    if not ok:
        raise RuntimeError('Native FP32 cache-reconstruction gate failed')
    log('FP32 native full/cache reconstruction gate passed')
    source_checks = {}
    for family in ('llama', 'gemma'):
        model = load_model(family)
        ids = proto['units'][family]['a']['ids']
        source = capture(model, ids)
        assert source[0].shape == source[1].shape and source[0].shape[1] == len(ids)
        source_checks[family] = {'capture_shape': list(source[0].shape),
                                'finite': bool(torch.isfinite(source[0]).all() and torch.isfinite(source[1]).all())}
        del model
        release()
        if not source_checks[family]['finite']:
            raise RuntimeError('Nonfinite source capture')
        if family == 'llama':
            writer = load_writer(family)
            indices = proto['maps'][family]['a'][:32]
            indices = indices + [indices[-1]] * (32 - len(indices))
            permutation = list(reversed(range(32)))
            one = translate(writer, source, indices)
            two = translate(writer, source, [indices[i] for i in permutation])
            error = max(float((a - b[:, permutation]).abs().max()) for a, b in zip(one, two))
            source_checks[family]['writer_token_permutation_max_abs'] = error
            if error > 1e-4:
                raise RuntimeError('Legacy writer is not invariant to token permutation')
            del writer, one, two
            release()
        del source
    save(ROOT / 'gates/source_interfaces.json', {'status': 'passed', 'checks': source_checks})
    log('Llama/Gemma source interfaces and frozen Llama writer gate passed')


def wait_inputs(names):
    plan = json.loads((ROOT / 'input_sources.json').read_text())
    started = time.monotonic()
    for name in names:
        p, spec = ROOT / name, plan[name]
        last_logged = None
        while not p.exists() or p.stat().st_size != spec['bytes']:
            current = p.stat().st_size if p.exists() else 0
            bucket = current // (64 << 20)
            if bucket != last_logged:
                log(f'waiting for verified input {name}: {current}/{spec["bytes"]} bytes')
                save(ROOT / 'input_wait.json', {'status': 'waiting', 'file': name,
                    'copied_bytes': current, 'expected_bytes': spec['bytes']})
                last_logged = bucket
            if time.monotonic() - started > 21600:
                raise RuntimeError('Input transfer did not complete within six hours')
            time.sleep(10)
        if sha_file(p) != spec['sha256']:
            raise RuntimeError('Source/target input hash mismatch: ' + name)
        log('verified input ' + name)
    save(ROOT / 'input_wait.json', {'status': 'verified', 'files': names})


def evaluate_group(rows, toks, mode, manifest):
    import torch
    protocols = {r['id']: build_protocol(toks, r['bodies'], r['question']) for r in rows}
    mapped, timings = {r['id']: {} for r in rows}, {}
    for family in ('llama', 'gemma'):
        start = time.perf_counter()
        model = load_model(family)
        captured = {r['id']: {x: capture(model, protocols[r['id']]['units'][family][x]['ids'])
                              for x in ('a', 'b', 'joint')} for r in rows}
        del model
        release()
        timings[family + '_capture_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        writer = load_writer(family)
        for r in rows:
            rid = r['id']
            mapped[rid][family] = {x: translate(writer, captured[rid][x], protocols[rid]['maps'][family][x])
                                   for x in ('a', 'b', 'joint')}
        del writer, captured
        release()
        timings[family + '_mapping_seconds'] = time.perf_counter() - start
    model = load_model('qwen')
    for row in rows:
        rid, proto = row['id'], protocols[row['id']]
        n = proto['units']['qwen']['joint']['prefix_count']
        native_captured = {x: capture(model, proto['units']['qwen'][x]['ids']) for x in ('a', 'b', 'joint')}
        prefix = capture(model, proto['prefix_ids'])
        native = {x: tuple(t[:, n:] for t in native_captured[x]) for x in ('a', 'b', 'joint')}
        m = mapped[rid]
        arms = {
            'native_independent': join_pairs(prefix, native['a'], native['b']),
            'llama_joint': join_pairs(prefix, m['llama']['joint']),
            'gemma_joint': join_pairs(prefix, m['gemma']['joint']),
            'llama_independent': join_pairs(prefix, m['llama']['a'], m['llama']['b']),
            'gemma_independent': join_pairs(prefix, m['gemma']['a'], m['gemma']['b']),
            'llama_a_gemma_b': join_pairs(prefix, m['llama']['a'], m['gemma']['b']),
            'gemma_a_llama_b': join_pairs(prefix, m['gemma']['a'], m['llama']['b']),
            'llama_a_native_b': join_pairs(prefix, m['llama']['a'], native['b']),
            'native_a_llama_b': join_pairs(prefix, native['a'], m['llama']['b']),
            'gemma_a_native_b': join_pairs(prefix, m['gemma']['a'], native['b']),
            'native_a_gemma_b': join_pairs(prefix, native['a'], m['gemma']['b'])}
        expected_tokens = len(proto['units']['qwen']['joint']['ids'])
        for arm, mem in arms.items():
            assert mem[0].shape == mem[1].shape and mem[0].shape[1] == expected_tokens, arm
        digest_before = {f: {x: memory_digest(pair) for x, pair in family.items()} for f, family in m.items()}
        predictions, logits = {}, {}
        for arm in CFG['arms']:
            ids = proto['full_ids'] if arm == 'native_full' else proto['query_only_ids'] if arm == 'query_only' else proto['suffix_ids']
            memory = None if arm in ('native_full', 'query_only') else arms[arm]
            pred, first = generate(model, toks['qwen'], ids, memory)
            pred.update(score(pred['parsed_answer'], row['answer']))
            predictions[arm], logits[arm] = pred, first
        if mode == 'smoke':
            rebuilt, first = generate(model, toks['qwen'], proto['suffix_ids'], native_captured['joint'])
            rebuilt['full_first_argmax_match'] = rebuilt['first_token_id'] == predictions['native_full']['first_token_id']
            rebuilt['full_generation_match'] = rebuilt['generated_ids'] == predictions['native_full']['generated_ids']
            rebuilt['full_first_logit_mae'] = float((first - logits['native_full']).abs().mean())
        else:
            rebuilt = None
        assert digest_before == {f: {x: memory_digest(pair) for x, pair in family.items()} for f, family in m.items()}, 'Shared memory mutated'
        reference = logits['native_full'].log_softmax(-1)
        for arm, first in logits.items():
            predictions[arm]['full_first_token_kl'] = float((reference.exp() * (reference - first.log_softmax(-1))).sum())
        result = {'status': 'completed', 'mode': mode, 'id': rid, 'config_signature': signature(CFG),
                  'manifest_signature': manifest['manifest_signature'], 'input': row,
                  'predictions': predictions, 'fp16_native_reconstruction': rebuilt,
                  'kv_diagnostics': {f: {x: compare_kv(pair, native[x]) for x, pair in family.items()} for f, family in m.items()},
                  'group_timings': timings, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'memory_integrity_verified': True, 'translated_receiver_evidence_text_access': False}
        save(ROOT / f'runs/{mode}/per_sample/{rid}.json', result)
        log(f'{mode} saved {rid}; native={predictions["native_full"]["f1"]:.3f}; mixed={predictions["llama_a_gemma_b"]["f1"]:.3f}')
        del native_captured, native, arms, predictions, logits, prefix
    del model, mapped, protocols
    release()


def bootstrap(diffs):
    rng = random.Random(CFG['seed'])
    values = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000))
    return [values[49], values[1949]]


def summarize(mode, manifest, rows):
    items = [json.loads((ROOT / f'runs/{mode}/per_sample/{r["id"]}.json').read_text()) for r in rows]
    metrics = {arm: {key: sum(x['predictions'][arm][key] for x in items) / len(items)
                     for key in ('em', 'f1', 'full_first_token_kl', 'ttft_seconds', 'generation_seconds')}
               for arm in CFG['arms']}
    pairs = [('native_full', 'native_independent'), ('llama_joint', 'llama_independent'),
             ('gemma_joint', 'gemma_independent'), ('native_independent', 'llama_a_gemma_b'),
             ('native_independent', 'gemma_a_llama_b'), ('llama_independent', 'llama_a_gemma_b'),
             ('gemma_independent', 'llama_a_gemma_b')]
    paired = {}
    for a, b in pairs:
        diffs = [x['predictions'][b]['f1'] - x['predictions'][a]['f1'] for x in items]
        paired[b + '_minus_' + a] = {'f1_difference': sum(diffs) / len(diffs), 'bootstrap_95_ci': bootstrap(diffs),
            'both_em_correct': sum(x['predictions'][a]['em'] and x['predictions'][b]['em'] for x in items),
            'only_first_em_correct': sum(x['predictions'][a]['em'] and not x['predictions'][b]['em'] for x in items),
            'only_second_em_correct': sum(x['predictions'][b]['em'] and not x['predictions'][a]['em'] for x in items)}
    correct_native = [x for x in items if x['predictions']['native_full']['em']]
    summary = {'status': 'completed', 'mode': mode, 'experiment_name': CFG['experiment_name'],
        'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': CFG['protocol'],
        'sample_count': len(items), 'seed': CFG['seed'], 'dataset_split': 'official HotpotQA DEV, bridge/gold2 mechanism pilot',
        'checkpoint_selection': 'pinned prior OpenBookQA validation selections; no tuning on pilot',
        'config_signature': signature(CFG), 'manifest_signature': manifest['manifest_signature'],
        'metrics': metrics, 'paired': paired,
        'native_correct_stratum': {'count': len(correct_native), 'metrics': {
            arm: {'em': sum(x['predictions'][arm]['em'] for x in correct_native) / len(correct_native),
                  'f1': sum(x['predictions'][arm]['f1'] for x in correct_native) / len(correct_native)}
            for arm in CFG['arms']} if correct_native else {}},
        'limitations': ['Gold supporting documents, not retrieval evaluation or official benchmark score',
            'Legacy sparse options32 MCQ writers applied densely to query-independent evidence; distribution/interface shift',
            'No receiver training; no universal claim that heterogeneous composition is impossible',
            'Warm read timings include cache assembly but exclude source/model loading and offline mapping'],
        'paths': {'manifest': str(ROOT / 'manifest.json'), 'predictions': str(ROOT / f'runs/{mode}/per_sample'),
                  'log': str(ROOT / 'logs/pipeline.log')}}
    save(ROOT / f'runs/{mode}/summary.json', summary)
    if mode == 'pilot':
        update_shared(summary)
    return summary


def update_shared(summary):
    import fcntl
    p = path(CFG['shared_results'])
    p.parent.mkdir(parents=True, exist_ok=True)
    begin = '<!-- BEGIN ' + CFG['experiment_name'] + ' -->'
    end = '<!-- END ' + CFG['experiment_name'] + ' -->'
    lines = [begin, '## ' + CFG['experiment_name'], '', '- Status: `' + summary['status'] + '`',
        '- Timestamp: ' + summary['timestamp'], '- Server: B; seed: 1234',
        '- Protocol: `' + CFG['protocol'] + '`', '- Config signature: `' + signature(CFG) + '`',
        '- Split: official HotpotQA DEV bridge/gold2; no training; 64 seeded eligible examples.',
        '- Checkpoints: fixed prior OpenBookQA validation selections, no new checkpoint selection.',
        '- Raw results: `' + str(ROOT / ('runs/pilot/summary.json' if summary['status'] == 'completed' else 'runs/pilot/failure.json')) + '`',
        '- Manifest/checkpoint hashes: `' + str(ROOT / 'manifest.json') + '`',
        '- Per-sample outputs: `' + str(ROOT / 'runs/pilot/per_sample') + '`',
        '- Log: `' + str(ROOT / 'logs/pipeline.log') + '`', '']
    if summary['status'] == 'completed':
        lines += ['| Arm | EM | F1 | First-token KL to Full |', '|---|---:|---:|---:|']
        for arm, m in summary['metrics'].items():
            lines.append(f'| {arm} | {m["em"]:.6f} | {m["f1"]:.6f} | {m["full_first_token_kl"]:.6f} |')
        lines += ['', 'Paired F1 differences and confidence intervals are stored in the raw summary.',
                  'Custom mechanism pilot; old writer interface/domain extrapolation is recorded; no impossibility claim.']
    else:
        lines += ['Failure: `' + summary.get('error', 'unknown').replace('`', '') + '`']
    lines += ['', end, '']
    section = '\n'.join(lines)
    with (p.parent / '.experiment_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = p.read_text() if p.exists() else '# Experiment Results\n\n'
        if begin in text:
            start = text.index(begin)
            stop = text.index(end, start) + len(end)
            text = text[:start] + section + text[stop:]
        else:
            text += '\n' + section
        tmp = p.with_suffix('.md.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(p)
    save(ROOT / 'shared_summary_update.json', {'status': 'completed', 'path': str(p),
        'run_status': summary['status'], 'timestamp': summary['timestamp']})


def evaluate(mode):
    import torch
    torch.manual_seed(CFG['seed'])
    torch.set_num_threads(4)
    manifest = prepare()
    rows = manifest['rows'][:CFG['smoke_samples']] if mode == 'smoke' else manifest['rows']
    pending = []
    for row in rows:
        p = ROOT / f'runs/{mode}/per_sample/{row["id"]}.json'
        if p.exists():
            saved = json.loads(p.read_text())
            if saved['config_signature'] != signature(CFG) or saved['manifest_signature'] != manifest['manifest_signature']:
                raise RuntimeError('Resume output identity mismatch')
            if set(saved['predictions']) != set(CFG['arms']):
                raise RuntimeError('Incomplete arms in saved output')
        else:
            pending.append(row)
    toks = load_tokenizers()
    for start in range(0, len(pending), CFG['group_size']):
        log(f'{mode} group {start}/{len(pending)}')
        evaluate_group(pending[start:start + CFG['group_size']], toks, mode, manifest)
    summarize(mode, manifest, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'cpu', 'gate', 'smoke', 'pilot', 'pipeline'])
    args = parser.parse_args()
    if args.command == 'cpu':
        cpu_checks()
        return
    if args.command == 'prepare':
        prepare()
        return
    if args.command == 'gate':
        gpu_gate()
        return
    if args.command in ('smoke', 'pilot'):
        evaluate(args.command)
        return
    status = {'status': 'running', 'phase': 'prepare', 'timestamp': datetime.now(timezone.utc).isoformat()}
    save(ROOT / 'status.json', status)
    try:
        status['phase'] = 'waiting_dataset'
        save(ROOT / 'status.json', status)
        wait_inputs(['data/hotpot_dev_distractor_v1.json'])
        for command in ('cpu', 'gate'):
            status['phase'] = command
            save(ROOT / 'status.json', status)
            subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), command], cwd=ROOT, check=True)
        status['phase'] = 'waiting_writers'
        save(ROOT / 'status.json', status)
        wait_inputs(['checkpoints/gemma_stage_a.pt', 'checkpoints/gemma_residual.pt', 'checkpoints/llama_residual.pt'])
        for command in ('prepare', 'smoke', 'pilot'):
            status['phase'] = command
            save(ROOT / 'status.json', status)
            subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), command], cwd=ROOT, check=True)
        status.update(status='completed', phase='pilot', timestamp=datetime.now(timezone.utc).isoformat())
        save(ROOT / 'status.json', status)
    except Exception as exc:
        status.update(status='failed', error=str(exc), timestamp=datetime.now(timezone.utc).isoformat())
        save(ROOT / 'status.json', status)
        save(ROOT / 'runs/pilot/failure.json', status)
        update_shared(status)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
