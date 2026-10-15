"""Frozen translated PCA directions applied to native KV; GPU-heavy, CPU-limited."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent
PARENT = ROOT.parents[2]
CFG = json.loads((ROOT / 'config.json').read_text())
ARMS = [('native_control', None, None, 128)] + [
    (f'{basis}__{kind}_{k}', basis, kind, k)
    for basis in CFG['basis_sources'] for k in CFG['retained_components']
    for kind in CFG['compression_targets']]


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require_gpu(gib=0):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU compute fallback is prohibited')
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < gib:
        raise RuntimeError(f'Only {free:.2f} GiB free, need {gib}; no other task will be stopped')


@torch.no_grad()
def capture_gpu(model, ids):
    """Same legacy capture math, without moving full K/V to CPU."""
    keys, values, hooks = {}, {}, []
    def hook(index):
        def call(attn, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, attn.head_dim)
            key = attn.k_proj(hidden).view(shape)
            if hasattr(attn, 'k_norm'):
                key = attn.k_norm(key)
            keys[index] = key[0].detach()
            values[index] = attn.v_proj(hidden).view(shape)[0].detach()
        return call
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True))
    try:
        x = torch.tensor([ids], device='cuda', dtype=torch.long)
        model.model(input_ids=x, attention_mask=torch.ones_like(x),
                    position_ids=torch.arange(len(ids), device='cuda')[None], use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([keys[i] for i in range(len(keys))]), torch.stack([values[i] for i in range(len(values))])


def positions(row, length):
    seed = int(hashlib.sha256(f'{CFG["seed"]}:{row["id"]}'.encode()).hexdigest()[:16], 16)
    return sorted(random.Random(seed).sample(range(length), min(32, length)))


def prepare(kb, ood, objects):
    original = read(REFERENCE / 'manifest.json')
    old_ood = read(REFERENCE / 'native_compression/ood_transfer/manifest.json')
    records = {'mcq': [], 'gsm8k': [], 'ood_mcq': [], 'ood_gsm8k': []}
    used_files = [ROOT / 'run.py', ROOT / 'config.json', REFERENCE / 'manifest.json',
        REFERENCE / 'native_compression/manifest.json',
        REFERENCE / 'native_compression/ood_transfer/manifest.json',
        REFERENCE / 'native_compression/run_compression.py',
        REFERENCE / 'native_compression/ood_transfer/run_ood.py', PARENT / 'kv_backend.py',
        PARENT / 'base_code/protocol.py', PARENT / 'base_code/translator.py',
        PARENT / 'base_code/common.py', PARENT / 'tasks_native.py', PARENT / 'config.json']
    for group, selected in original['samples'].items():
        for task in sorted({r['task'] for r in selected}):
            path = PARENT / f'runs/{group}/evaluation/qwen_native__{task}__samples.jsonl'
            used_files.append(path)
            filt = 'strict-match' if task == 'gsm8k' else 'none'
            lookup = {r['doc_id']: r for r in (json.loads(s) for s in path.read_text().splitlines() if s)
                      if r['filter'] == filt}
            for selected_row in [r for r in selected if r['task'] == task]:
                raw = lookup[selected_row['doc_id']]
                args = raw['arguments']
                context = args[0][0] if task == 'gsm8k' else args[0][0].rstrip()
                assert context == selected_row['context']
                assert raw['doc_hash'] == selected_row['original_doc_hash']
                records[group].append({**selected_row, 'arguments': args, 'doc': raw['doc'],
                    'group': group, 'basis_group': group,
                    'output_type': 'generate_until' if task == 'gsm8k' else 'multiple_choice'})
            assert sum(r['task'] == task and r['role'] == 'calibration' for r in records[group]) == 8
            assert sum(r['task'] == task and r['role'] == 'analysis' for r in records[group]) == 24
        for stage in ('stage_a', 'stage_b'):
            used_files += [PARENT / f'runs/{group}/{stage}/best.pt', PARENT / f'runs/{group}/{stage}/selection.json']
        used_files.append(REFERENCE / f'native_compression/basis/{group}.npz')
    calibration_hashes = {r['context_sha256'] for rows in records.values() for r in rows if r['role'] == 'calibration'}
    for group, selected in old_ood['samples'].items():
        for selected_row in selected:
            task, doc_id = selected_row['id'].rsplit(':', 1)
            doc_id = int(doc_id)
            path = PARENT / f'runs/mcq/evaluation_ood/qwen_native__{task}__samples.jsonl'
            used_files.append(path)
            filt = 'none' if group == 'hellaswag' else 'custom-extract'
            raw = next(r for r in (json.loads(s) for s in path.read_text().splitlines() if s)
                       if r['doc_id'] == doc_id and r['filter'] == filt)
            assert raw['arguments'] == selected_row['arguments'] and raw['doc_hash'] == selected_row['doc_hash']
            generation = selected_row['output_type'] == 'generate_until'
            context = raw['arguments'][0][0] if generation else raw['arguments'][0][0].rstrip()
            assert hashlib.sha256(context.encode()).hexdigest() == selected_row['context_sha256']
            assert selected_row['context_sha256'] not in calibration_hashes
            basis_group = CFG['ood_basis_group'][group]
            records['ood_' + basis_group].append({**selected_row, 'task': task, 'doc_id': doc_id,
                'doc': raw['doc'], 'context': context, 'role': 'analysis',
                'group': 'ood_' + basis_group, 'basis_group': basis_group})
    assert sum(r['role'] == 'analysis' for rows in records.values() for r in rows) == 110
    assert len(records['ood_mcq']) == 24 and len(records['ood_gsm8k']) == 14
    for group in ('mcq', 'gsm8k'):
        assert not ({r['context_sha256'] for r in records[group] if r['role'] == 'calibration'} &
                    {r['context_sha256'] for r in records[group] if r['role'] == 'analysis'})
    import inspect, importlib.metadata
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    used_files += [Path(inspect.getfile(ConfigurableTask)), Path(inspect.getfile(HFLM))]
    for model in kb.CFG['models'].values():
        used_files += [Path(model) / 'config.json', Path(model) / 'tokenizer.json']
    fingerprints = {}
    for path in dict.fromkeys(used_files):
        log('Fingerprint ' + str(path.relative_to(PARENT)) if path.is_relative_to(PARENT) else 'Fingerprint ' + path.name)
        value = {'sha256': sha(path), 'bytes': path.stat().st_size}
        fingerprints[str(path)] = value
        if str(path) in original['files']:
            assert value['sha256'] == original['files'][str(path)]['sha256'], f'Original input changed: {path}'
    body = {'config': CFG, 'files': fingerprints, 'original_checkpoints': original['checkpoints'],
        'samples': records, 'task_configs': {name: ood.stable_task_config(task) for name, task in objects.items()},
        'runtime': {'torch': torch.__version__, 'lm_eval': importlib.metadata.version('lm_eval'),
                    'gpu': torch.cuda.get_device_name(0), 'torch_threads': torch.get_num_threads()}}
    signature = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (ROOT / 'manifest.json').exists():
        assert read(ROOT / 'manifest.json')['signature'] == signature, 'Input changed; do not overwrite old run'
    else:
        save('manifest.json', {**body, 'signature': signature, 'timestamp': now()})
    # Replaying logged responses must reproduce the installed official score.
    replay = {}
    for task in objects:
        row = next(r for rows in records.values() for r in rows if r['task'] == task)
        requests = ood.request_list(row)
        source_path = PARENT / (f'runs/{row["basis_group"]}/evaluation/qwen_native__{task}__samples.jsonl'
            if not row['group'].startswith('ood_') else f'runs/mcq/evaluation_ood/qwen_native__{task}__samples.jsonl')
        filt = 'strict-match' if task == 'gsm8k' else 'custom-extract' if task.startswith('mmlu_pro') else 'none'
        raw = next(r for r in (json.loads(s) for s in source_path.read_text().splitlines() if s)
                   if r['doc_id'] == row['doc_id'] and r['filter'] == filt)
        class Replay:
            def loglikelihood(self, requests, **kwargs):
                return [x[0] for x in raw['resps']]
            generate_until = loglikelihood
        scored = ood.evaluate(Replay(), objects[task], row)
        for metric in raw['metrics']:
            assert scored['scores'][filt][metric] == raw[metric], (task, metric)
        replay[task] = scored['scores']
    save('gates/official_scoring_replay.json', {'status': 'passed', 'scores': replay})
    return records, signature


@torch.no_grad()
def accumulate(moment, x):
    x = x.permute(0, 2, 1, 3).double()
    moment['sum'].add_(x.sum(-2))
    moment['xx'].add_(x.transpose(-1, -2) @ x)
    moment['n'] += x.shape[-2]


def new_moment():
    return {'sum': torch.zeros(36, 8, 128, device='cuda', dtype=torch.float64),
            'xx': torch.zeros(36, 8, 128, 128, device='cuda', dtype=torch.float64), 'n': 0}


def covariance(moment):
    mean = moment['sum'] / moment['n']
    c = moment['xx'] / moment['n'] - mean[..., :, None] * mean[..., None, :]
    return mean, (c + c.transpose(-1, -2)) * .5


def load_bases(group, signature):
    result = {}
    for source in CFG['basis_sources']:
        path = ROOT / f'basis/{group}_{source}.pt'
        if not path.exists():
            return None
        payload = torch.load(path, map_location='cuda', weights_only=True)
        assert payload['signature'] == signature
        result[source] = payload['tensors']
    return result


def writer(kb, group, stage):
    path = PARENT / f'runs/{group}/{stage}/best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    expected = read(REFERENCE / 'manifest.json')['checkpoints'][group][stage]['selection']
    assert (payload['stage'], payload['epoch'], payload['step'], payload['protocol']) == (
        stage.removeprefix('stage_'), expected['epoch'], expected['step'], kb.CFG['protocol'])
    model = (kb.NativeKVTranslator('full28_mlp', hidden_dim=kb.CFG['mlp_hidden_dim']) if stage == 'stage_a'
             else kb.ResidualKVAdapter(rank=kb.CFG['adapter_rank'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['state'], strict=True)
    return model


@torch.no_grad()
def fit_bases(group, rows, signature, kb, tokenizer):
    existing = load_bases(group, signature)
    if existing is not None:
        log('Reuse frozen bases ' + group)
        return existing
    require_gpu(CFG['minimum_free_gpu_gib'])
    from transformers import AutoTokenizer
    sender = kb.load_model({'models': kb.CFG['models'], 'attention_implementation': kb.CFG['attention_implementation']}, 'llama')
    receiver = kb.load_model({'models': kb.CFG['models'], 'attention_implementation': kb.CFG['attention_implementation']}, 'qwen')
    source_tok = AutoTokenizer.from_pretrained(kb.CFG['models']['llama'], local_files_only=True)
    base, adapter = writer(kb, group, 'stage_a'), writer(kb, group, 'stage_b')
    moments = {source: {kind: new_moment() for kind in ('K', 'V')} for source in CFG['basis_sources']}
    selected_rows = [r for r in rows if r['role'] == 'calibration']
    for i, row in enumerate(selected_rows):
        save('status.json', {'status': 'running', 'phase': 'fit_gpu_pca', 'group': group,
            'sample': i + 1, 'samples': len(selected_rows), 'id': row['id'], 'timestamp': now()})
        item = kb.pair(sender, receiver, source_tok, tokenizer, row['context'])
        assert item['ids'] == kb.encoding(tokenizer, row['context']) and item['tokens'] > 0
        indices = positions(row, item['tokens'])
        a_k, a_v = kb.map_base(base, item, kb.CFG['chunk_tokens'])
        with torch.amp.autocast('cuda', dtype=torch.float16):
            b_k, b_v, _, _ = adapter(a_k[None], a_v[None])
        memories = {'native': (item['target_k'], item['target_v']),
                    'stage_a': (a_k.half(), a_v.half()), 'stage_b': (b_k[0].half(), b_v[0].half())}
        for source, memory in memories.items():
            for kind, tensor in zip(('K', 'V'), memory):
                assert tensor.shape == item['target_k'].shape and torch.isfinite(tensor).all()
                accumulate(moments[source][kind], tensor[:, indices])
        save(f'calibration/{group}/{row["id"].replace(":", "_")}.json', {
            'id': row['id'], 'context_sha256': row['context_sha256'], 'role': row['role'],
            'sampled_positions': [p + 1 for p in indices], 'external_tokens': item['tokens']})
        del item, a_k, a_v, b_k, b_v, memories
        log(f'GPU calibration {group} {i + 1}/{len(selected_rows)} {row["id"]}')
    result, audit = {source: {} for source in CFG['basis_sources']}, {}
    old_manifest = read(REFERENCE / 'native_compression/manifest.json')
    with np.load(REFERENCE / f'native_compression/basis/{group}.npz', allow_pickle=False) as old:
        assert str(old['signature']) == old_manifest['signature']
        for kind in ('K', 'V'):
            for suffix in ('mean', 'basis', 'eigenvalues'):
                result['native'][f'{kind}_{suffix}'] = torch.from_numpy(old[f'{kind}_{suffix}'].copy()).cuda()
    native_covariances = {}
    for kind in ('K', 'V'):
        mean, cov = covariance(moments['native'][kind])
        old_mean = result['native'][kind + '_mean'].double()
        u = result['native'][kind + '_basis'].double()
        eig = result['native'][kind + '_eigenvalues'].double()
        old_cov = (u * eig[..., None, :]) @ u.transpose(-1, -2)
        mean_error = float((mean - old_mean).abs().max())
        covariance_error = float((cov - old_cov).norm() / cov.norm().clamp_min(1e-20))
        assert mean_error < 1e-5 and covariance_error < 1e-4, (mean_error, covariance_error)
        audit[kind] = {'native_mean_max_error': mean_error, 'native_covariance_relative_error': covariance_error}
        native_covariances[kind] = cov
    for source in ('stage_a', 'stage_b'):
        for kind in ('K', 'V'):
            mean, cov = covariance(moments[source][kind])
            eigenvalues, axes = torch.linalg.eigh(cov)
            axes = axes.flip(-1).float()
            assert float((axes.transpose(-1, -2) @ axes - torch.eye(128, device='cuda')).abs().max()) < 2e-6
            result[source].update({kind + '_mean': mean.float(), kind + '_basis': axes,
                                   kind + '_eigenvalues': eigenvalues.flip(-1).clamp_min(0)})
    geometry = {}
    for source, basis in result.items():
        geometry[source] = {}
        for kind in ('K', 'V'):
            u, cov = basis[kind + '_basis'].double(), native_covariances[kind]
            diagonal = (u * (cov @ u)).sum(-2).clamp_min(0).cumsum(-1)
            trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-20)
            geometry[source][kind] = {str(k): {'mean_fraction': float((diagonal[..., k - 1] / trace).mean()),
                'per_layer_head': (diagonal[..., k - 1] / trace).tolist()} for k in CFG['retained_components']}
    save(f'gates/{group}_native_basis_replay.json', {'status': 'passed', 'checks': audit, 'fit_device': 'cuda',
         'calibration_prompts': len(selected_rows), 'tokens': moments['native']['K']['n']})
    save(f'metrics/{group}_calibration_native_variance_retained.json', geometry)
    (ROOT / 'basis').mkdir(exist_ok=True)
    for source, basis in result.items():
        path = ROOT / f'basis/{group}_{source}.pt'
        temp = path.with_suffix('.tmp')
        torch.save({'signature': signature, 'tensors': {k: v.cpu() for k, v in basis.items()}}, temp)
        temp.replace(path)
    del sender, receiver, base, adapter, moments, native_covariances
    gc.collect()
    torch.cuda.empty_cache()
    log('Frozen GPU PCA fitting completed ' + group)
    return result


@torch.no_grad()
def project(memory, bases, source, kind, k):
    result, geometry = [], {}
    for channel, raw in zip(('K', 'V'), memory):
        if kind is None or channel not in kind:
            result.append(raw)
            geometry[channel] = {'calibration_mean_energy': 1.0, 'centered_variance': 1.0}
            continue
        mean = bases['native'][channel + '_mean'].float()[:, :, None, :]
        u = bases[source][channel + '_basis'][..., :k].float()
        x = raw[:, 1:].permute(0, 2, 1, 3).float()
        centered = x - mean
        coordinates = centered @ u
        rebuilt = (coordinates @ u.transpose(-1, -2) + mean).permute(0, 2, 1, 3).half()
        assert torch.isfinite(rebuilt).all()
        projected_energy = coordinates.square().sum((-1, -2))
        energy = projected_energy / centered.square().sum((-1, -2)).clamp_min(1e-20)
        heldout = x - x.mean(-2, keepdim=True)
        variance = (heldout @ u).square().sum((-1, -2)) / heldout.square().sum((-1, -2)).clamp_min(1e-20)
        geometry[channel] = {'calibration_mean_energy': float(energy.mean()),
                             'centered_variance': float(variance.mean())}
        result.append(torch.cat((raw[:, :1], rebuilt), 1))
        assert torch.equal(result[-1][:, :1], raw[:, :1]) and result[-1].shape == raw.shape
    return tuple(result), geometry


@torch.inference_mode()
def probe(model, ids, memory=None):
    from protocol import final_logits
    if memory is None:
        x = torch.tensor([ids], device='cuda')
        out = model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
        return model.lm_head(out.last_hidden_state[:, -1])[0].float()
    return final_logits(model, ids[-1:], *memory, torch.arange(len(ids) - 1, device='cuda')).float()


def kl(reference, other):
    p, q = reference.float().log_softmax(-1), other.float().log_softmax(-1)
    return {'first_token_kl': max(0., float((p.exp() * (p - q)).sum())),
            'first_argmax_match': bool(reference.argmax() == other.argmax()),
            'logit_mean_abs': float((reference - other).abs().mean())}


def gate(group, row, bases, lm, kb, task, ood):
    if (ROOT / f'gates/{group}_gpu.json').exists():
        assert read(ROOT / f'gates/{group}_gpu.json')['status'] == 'passed'
        return
    ids = kb.encoding(lm.tokenizer, row['context'])
    memory = kb.capture(lm.model, ids[:-1])
    lm.expected_ids = ids
    reference = probe(lm.model, ids, memory)
    floor = kl(probe(lm.model, ids), reference)
    assert floor['first_token_kl'] < 1e-3 and floor['logit_mean_abs'] < .05, floor
    lm.mode = 'native'
    ordinary = ood.evaluate(lm, task, row, 32)
    lm.mode, lm.pinned = 'oracle', memory
    native = ood.evaluate(lm, task, row, 32)
    checks = {}
    for source in CFG['basis_sources']:
        reconstructed, _ = project(memory, bases, source, 'KV', 128)
        error = kl(reference, probe(lm.model, ids, reconstructed))
        assert error['first_token_kl'] < 1e-3 and error['logit_mean_abs'] < .05, (source, error)
        lm.pinned = reconstructed
        rebuilt = ood.evaluate(lm, task, row, 32)
        if row['output_type'] == 'generate_until':
            assert ordinary['responses'] == native['responses'] == rebuilt['responses'], (group, source, 'Generation parity')
        else:
            a, b, c = [np.array(v['responses'])[:, 0] for v in (ordinary, native, rebuilt)]
            assert max(float(abs(a - b).max()), float(abs(b - c).max())) < .15
            assert ordinary['scores'] == native['scores'] == rebuilt['scores']
        checks[source] = error
    lm.pinned = None
    save(f'gates/{group}_gpu.json', {'status': 'passed', 'native_floor': floor,
         'full128_per_basis': checks, 'gate_only_max_gen_toks': 32, 'id': row['id']})
    log('Native/full-rank GPU gates passed ' + group)


def compact(row):
    return {k: row[k] for k in ('id', 'task', 'group', 'basis_group')} | {
        'arms': {name: {'scores': r['evaluation']['scores'], 'output': r['output'], 'geometry': r['geometry']}
                 for name, r in row['arms'].items()}}


def summarize(final=False):
    rows = [read(p) for p in sorted((ROOT / 'metrics/samples').rglob('*.json'))]
    summary = {'status': 'completed' if final else 'running', 'timestamp': now(),
        'completed_samples': len(rows), 'expected_samples': 110, 'full_benchmark': False, 'tasks': {}}
    arm_names = [a[0] for a in ARMS]
    for task in sorted({r['task'] for r in rows}):
        chosen = [r for r in rows if r['task'] == task]
        task_result = {'samples': len(chosen), 'arms': {}}
        filters = chosen[0]['arms']['native_control']['scores']
        for name in arm_names:
            task_result['arms'][name] = {'scores': {}, 'first_token_kl_mean': float(torch.tensor([
                r['arms'][name]['output']['first_token_kl'] for r in chosen], device='cuda').mean()),
                'geometry': {channel: {key: float(torch.tensor([r['arms'][name]['geometry'][channel][key]
                    for r in chosen], device='cuda').mean()) for key in ('centered_variance', 'calibration_mean_energy')}
                    for channel in ('K', 'V')}}
        gen = torch.Generator(device='cuda').manual_seed(int(hashlib.sha256(task.encode()).hexdigest()[:12], 16))
        indices = torch.randint(len(chosen), (CFG['bootstrap_resamples'], len(chosen)), device='cuda', generator=gen)
        for filt, metrics in filters.items():
            for metric in metrics:
                scores = torch.tensor([[r['arms'][a]['scores'][filt][metric] for a in arm_names]
                                       for r in chosen], device='cuda', dtype=torch.float64)
                native = scores[:, :1]
                drop = native - scores
                means = scores.mean(0)
                ci = torch.quantile(drop[indices].mean(1), torch.tensor([.025, .975], device='cuda', dtype=torch.float64), dim=0)
                for j, name in enumerate(arm_names):
                    task_result['arms'][name]['scores'].setdefault(filt, {})[metric] = {
                        'score': float(means[j]), 'drop_from_native': float(drop[:, j].mean()),
                        'functional_retention_ratio': float(means[j] / means[0]) if float(means[0]) != 0 else None,
                        'paired_drop_ci95': ci[:, j].tolist(),
                        'both_correct': int(((native[:, 0] == 1) & (scores[:, j] == 1)).sum()),
                        'only_native_correct': int(((native[:, 0] == 1) & (scores[:, j] == 0)).sum()),
                        'only_this_correct': int(((native[:, 0] == 0) & (scores[:, j] == 1)).sum()),
                        'both_wrong': int(((native[:, 0] == 0) & (scores[:, j] == 0)).sum())}
        summary['tasks'][task] = task_result
    if final:
        assert len(rows) == 110 and all(len(r['arms']) == 37 for r in rows)
    save('summary.json' if final else 'partial_summary.json', summary)
    return summary


def report(summary):
    lines = ['# Native KV: Native vs translated PCA directions', '',
        'CPU-limited, GPU PCA/projection/inference/bootstrap; no retraining; Native mean in every arm.', '',
        '| Task | n | Arm | Score | Drop pp | Native variance retained K/V |',
        '|---|---:|---|---:|---:|---|']
    for task, entry in summary['tasks'].items():
        filt, metric = ('strict-match', 'exact_match') if task == 'gsm8k' else (
            ('custom-extract', 'exact_match') if task.startswith('mmlu_pro') else ('none', 'acc_norm'))
        for name, data in entry['arms'].items():
            score = data['scores'][filt][metric]
            energy = '/'.join(f'{data["geometry"][c]["centered_variance"]:.4f}' for c in ('K', 'V'))
            lines.append(f'| {task} | {entry["samples"]} | {name} | {100*score["score"]:.2f} | {100*score["drop_from_native"]:.2f} | {energy} |')
    lines += ['', 'All original filters/metrics, confidence intervals and paired correctness counts are in summary.json.',
        'MMLU-Pro has one sample/subject; subject-wise intervals are uninformative. This is a small mechanism pilot.',
        'No claim that OOD samples were absent from model pretraining; no OOD PCA fitting or test-driven rank selection.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def run():
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    sys.path.insert(0, str(PARENT))
    import kv_backend as kb
    from tasks_native import tasks
    from transformers import AutoTokenizer
    ood = module('frozen_ood_helpers', REFERENCE / 'native_compression/ood_transfer/run_ood.py')
    objects = ood.flatten(tasks(CFG['id_tasks'] + CFG['ood_tasks']))
    kb.seed_all(CFG['seed'])
    # Legacy seed_all sets four CPU threads; restore this experiment's limit.
    torch.set_num_threads(CFG['cpu_threads'])
    assert torch.get_num_threads() == 1
    require_gpu()
    torch.backends.cuda.preferred_linalg_library('cusolver')
    records, signature = prepare(kb, ood, objects)
    save('resource_policy.json', {'torch_intraop_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(), 'arrow_cpu_threads': pa.cpu_count(),
        'pca_fit_device': 'cuda', 'projection_device': 'cuda', 'inference_device': 'cuda',
        'bootstrap_device': 'cuda', 'no_full_kv_disk_cache': True})
    if '--preflight' in sys.argv:
        log('PREFLIGHT PASSED: frozen inputs, split isolation, original official score replay')
        return
    require_gpu(CFG['minimum_free_gpu_gib'])
    kb.capture = capture_gpu  # local process only; on-disk legacy backend remains unchanged.
    tokenizer = AutoTokenizer.from_pretrained(kb.CFG['models']['qwen'], local_files_only=True)
    for group in ('mcq', 'gsm8k'):
        bases = fit_bases(group, records[group], signature, kb, tokenizer)
        del bases
        gc.collect()
        torch.cuda.empty_cache()
    require_gpu(14)
    model = kb.load_model({'models': kb.CFG['models'], 'attention_implementation': kb.CFG['attention_implementation']}, 'qwen')
    class PinnedHFLM(kb.FullSyncHFLM):
        pinned = None
        expected_ids = None
        last_generation = None
        def _prefix(self, context, ids):
            assert list(ids) == self.expected_ids == kb.encoding(self.tokenizer, context.rstrip())
            assert self.pinned is not None and self.pinned[0].shape[1] == len(ids) - 1
            return self.pinned
        def _model_generate(self, context, max_length, stop, **kwargs):
            output = super()._model_generate(context, max_length, stop, **kwargs)
            generated = output[0, context.shape[1]:].detach().cpu().tolist()
            self.last_generation = {'new_tokens': len(generated), 'token_ids': generated}
            return output
    lm = PinnedHFLM(model, mode='oracle')
    for group in records:
        bases = load_bases('mcq' if group.endswith('mcq') else 'gsm8k', signature)
        row = next(r for r in records[group] if r['role'] == 'analysis')
        gate(group, row, bases, lm, kb, objects[row['task']], ood)
        del bases
        torch.cuda.empty_cache()
    if '--gate-only' in sys.argv:
        log('ALL GPU GATES PASSED')
        return
    for group, group_rows in records.items():
        bases = load_bases('mcq' if group.endswith('mcq') else 'gsm8k', signature)
        selected = [r for r in group_rows if r['role'] == 'analysis']
        for index, row in enumerate(selected):
            relative = f'per_sample/{group}/{row["id"].replace(":", "_")}.json'
            result = read(ROOT / relative) if (ROOT / relative).exists() else {
                **{k: row[k] for k in ('id', 'task', 'group', 'basis_group', 'context_sha256', 'arguments', 'doc')},
                'manifest_signature': signature, 'arms': {}}
            assert result['manifest_signature'] == signature
            if len(result['arms']) == len(ARMS):
                save(f'metrics/samples/{group}/{row["id"].replace(":", "_")}.json', compact(result))
                continue
            ids = kb.encoding(tokenizer, row['context'])
            memory = kb.capture(model, ids[:-1])
            lm.expected_ids = ids
            reference_logits = probe(model, ids, memory)
            for arm, source, kind, k in ARMS:
                if arm in result['arms']:
                    continue
                save('status.json', {'status': 'running', 'phase': 'evaluation', 'group': group,
                    'sample_index': index, 'sample_count': len(selected), 'id': row['id'], 'arm': arm, 'timestamp': now()})
                start = time.monotonic()
                with torch.inference_mode():
                    projected, geometry = project(memory, bases, source, kind, k)
                    lm.pinned = projected
                    output = kl(reference_logits, probe(model, ids, projected))
                    evaluated = ood.evaluate(lm, objects[row['task']], row)
                result['arms'][arm] = {'basis_source': source, 'kind': kind, 'rank': k,
                    'geometry': geometry, 'output': output, 'evaluation': evaluated,
                    'seconds': time.monotonic() - start}
                save(relative, result)
                lm.pinned = None
                del projected
                log(f'{group} {index+1}/{len(selected)} {row["id"]} {arm}: {evaluated["scores"]}; {result["arms"][arm]["seconds"]:.1f}s')
            save(f'metrics/samples/{group}/{row["id"].replace(":", "_")}.json', compact(result))
            del memory, reference_logits
            torch.cuda.empty_cache()
        del bases
        partial = summarize()
        report(partial)
        log('PHASE COMPLETED ' + group)
    result = summarize(final=True)
    report(result)
    save('status.json', {'status': 'completed', 'samples': 110, 'arms': 37, 'timestamp': now()})
    log('ALL EXPERIMENTS COMPLETED')


if __name__ == '__main__':
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            run()
    except Exception as exc:
        failure = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc), 'timestamp': now()}
        save('failure.json', failure)
        save('status.json', failure)
        traceback.print_exc()
        raise
