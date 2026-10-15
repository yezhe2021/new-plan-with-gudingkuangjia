"""Frozen Qwen native KV PCA intervention, exact original lm-eval requests."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent
PARENT = ROOT.parents[2]
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
ARMS = [('native', None, 128)] + [(f'{kind}_{k}', kind, k)
    for k in CFG['retained_components'] for kind in CFG['compression_targets']]


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def save(name, data):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def positions(row, length):
    n = min(32, length)
    seed = int(hashlib.sha256(f'{CFG["seed"]}:{row["id"]}'.encode()).hexdigest()[:16], 16)
    return sorted(random.Random(seed).sample(range(length), n))


def project_numpy(x, mean, basis, k):
    u = basis[..., :k]
    return mean + ((x - mean) @ u) @ u.T


def cpu_gate():
    rng = np.random.default_rng(1234)
    u, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    mean = rng.normal(size=8)
    x = rng.normal(size=(37, 8)) + mean
    assert np.allclose(project_numpy(x, mean, u, 8), x, atol=1e-12)
    y = project_numpy(x, mean, u, 3)
    assert np.linalg.matrix_rank(y - mean, tol=1e-10) <= 3
    assert np.allclose(project_numpy(mean[None], mean, u, 3), mean)
    assert np.allclose(project_numpy(y, mean, u, 3), y)
    save('gates/cpu.json', {'status': 'passed', 'checks': ['full_rank_identity', 'centered_rank_bound',
        'mean_preserved', 'idempotent_projection'], 'timestamp': now()})


def prepare(kb, objects):
    source = read(REFERENCE / 'manifest.json')
    files = [ROOT / 'config.json', ROOT / 'run_compression.py', REFERENCE / 'manifest.json',
             PARENT / 'config.json', PARENT / 'kv_backend.py', PARENT / 'base_code/protocol.py',
             PARENT / 'base_code/common.py', PARENT / 'tasks_native.py']
    import inspect
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    files += [Path(inspect.getfile(ConfigurableTask)), Path(inspect.getfile(HFLM))]
    for name in ('config.json', 'tokenizer.json'):
        files.append(Path(kb.CFG['models']['qwen']) / name)
    records = {}
    for group in CFG['groups']:
        records[group] = []
        for task in source['config']['groups'][group]:
            path = PARENT / 'runs' / group / 'evaluation' / f'qwen_native__{task}__samples.jsonl'
            files.append(path)
            original = {(r['doc_id'], r['filter']): r for r in
                        (json.loads(line) for line in path.read_text().splitlines() if line)}
            filt = 'strict-match' if task == 'gsm8k' else 'none'
            for selected in [r for r in source['samples'][group] if r['task'] == task]:
                r = original[(selected['doc_id'], filt)]
                args = r['arguments']
                context = args[0][0] if task == 'gsm8k' else args[0][0].rstrip()
                assert all(a[0] == args[0][0] for a in args)
                assert context == selected['context']
                assert hashlib.sha256(context.encode()).hexdigest() == selected['context_sha256']
                assert r['doc_hash'] == selected['original_doc_hash']
                records[group].append({**selected, 'arguments': args, 'doc': r['doc'],
                    'original_native_responses': r['resps'], 'original_native_scores': selected['original_task_scores']['native']})
            chosen = [r for r in records[group] if r['task'] == task]
            assert sum(r['role'] == 'calibration' for r in chosen) == 8
            assert sum(r['role'] == 'analysis' for r in chosen) == 24
            assert len({r['doc_id'] for r in chosen}) == 32
    for group in records:
        a = {r['id'] for r in records[group] if r['role'] == 'calibration'}
        b = {r['id'] for r in records[group] if r['role'] == 'analysis'}
        assert not a & b
    fingerprint = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
    for path, info in fingerprint.items():
        if path in source['files']:
            assert info['sha256'] == source['files'][path]['sha256'], f'Parent PCA input changed: {path}'
    task_configs = {name: task.dump_config() for name, task in objects.items()}
    import torch, transformers
    body = {'config': CFG, 'files': fingerprint, 'task_configs': task_configs,
        'samples': {g: [{'id': r['id'], 'role': r['role'], 'context_sha256': r['context_sha256']}
                        for r in rs] for g, rs in records.items()},
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                    'lm_eval': importlib.metadata.version('lm_eval'), 'gpu': torch.cuda.get_device_name(0)}}
    signature = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    body.update(signature=signature, timestamp=now())
    old = ROOT / 'manifest.json'
    if old.exists():
        assert read(old)['signature'] == signature, 'Inputs changed; preserve previous run and use a new directory'
    else:
        save('manifest.json', body)
    return records, signature


def capture(kb, model, tokenizer, row):
    ids = kb.encoding(tokenizer, row['context'])
    assert len(ids) > 2
    k, v = kb.capture(model, ids[:-1])
    assert k.shape == v.shape == (36, len(ids) - 1, 8, 128)
    assert k.dtype == v.dtype
    return ids, (k, v)


def fit_basis(group, records, kb, model, tokenizer, signature):
    path = ROOT / 'basis' / f'{group}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            assert str(data['signature']) == signature
            return {k: data[k].copy() for k in data.files if k != 'signature'}
    buffers = {'K': [], 'V': []}
    for index, row in enumerate(r for r in records if r['role'] == 'calibration'):
        ids, memory = capture(kb, model, tokenizer, row)
        chosen = positions(row, len(ids) - 2)
        for kind, tensor in zip(('K', 'V'), memory):
            buffers[kind].append(tensor[:, [i + 1 for i in chosen]].permute(1, 0, 2, 3).numpy().astype(np.float32))
        save(f'calibration/{group}/{row["id"].replace(":", "_")}.json', {'id': row['id'],
             'context_sha256': row['context_sha256'], 'sampled_positions': [i + 1 for i in chosen],
             'external_tokens': len(ids) - 2, 'role': 'calibration'})
        del memory
        log(f'{group} calibration {index + 1}: {row["id"]}')
    result = {}
    for kind in ('K', 'V'):
        x = np.concatenate(buffers[kind], 0).astype(np.float64)
        mean = x.mean(0)
        centered = x - mean
        covariance = np.einsum('nlhd,nlhe->lhde', centered, centered, optimize=True) / len(x)
        eig, vectors = np.linalg.eigh(covariance)
        result[kind + '_mean'] = mean.astype(np.float32)
        result[kind + '_basis'] = vectors[..., ::-1].copy().astype(np.float32)
        result[kind + '_eigenvalues'] = np.maximum(eig[..., ::-1], 0)
        result[kind + '_calibration_tokens'] = np.array(len(x))
        assert np.max(np.abs(np.swapaxes(result[kind + '_basis'], -1, -2) @
                             result[kind + '_basis'] - np.eye(128))) < 2e-6
        del x, centered, covariance
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, signature=np.array(signature), **result)
    path.with_suffix('.tmp').replace(path)
    del buffers
    gc.collect()
    return result


def transform(memory, basis, kind, k):
    import torch
    output, energy = [], {}
    for channel, raw in zip(('K', 'V'), memory):
        if kind is None or channel not in kind:
            output.append(raw.cuda())
            energy[channel] = 1.0
            continue
        mean = torch.from_numpy(basis[channel + '_mean']).cuda()[:, None]
        u = torch.from_numpy(basis[channel + '_basis'][..., :k].copy()).cuda()
        x = raw[:, 1:].cuda().float() - mean
        z = torch.einsum('lthd,lhdk->lthk', x, u)
        y = torch.einsum('lthk,lhdk->lthd', z, u) + mean
        reconstructed = y.half()
        assert torch.isfinite(reconstructed).all()
        ratio = ((reconstructed.float() - mean).square().sum((1, 3)) /
                 x.square().sum((1, 3)).clamp_min(1e-20)).mean()
        energy[channel] = float(ratio)
        result = torch.cat((raw[:, :1].cuda(), reconstructed), 1)
        assert torch.equal(result[:, :1].cpu(), raw[:, :1]), 'Anchor was modified'
        output.append(result)
    assert output[0].shape == output[1].shape == memory[0].shape
    return tuple(output), energy


def kl(reference, other):
    p, q = reference.float().log_softmax(-1), other.float().log_softmax(-1)
    return {'first_token_kl': max(0.0, float((p.exp() * (p - q)).sum())),
        'first_argmax_match': bool(reference.argmax() == other.argmax()),
        'logit_mean_abs': float((reference.float() - other.float()).abs().mean())}


def probe(model, ids, memory=None):
    import torch
    from protocol import final_logits
    with torch.inference_mode():
        if memory is None:
            inp = torch.tensor([ids], device='cuda')
            out = model.model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
            return model.lm_head(out.last_hidden_state[:, -1])[0].float().cpu()
        return final_logits(model, ids[-1:], *memory,
                            torch.arange(len(ids) - 1, device='cuda')).float().cpu()


def requests(row, generation_limit=None):
    from lm_eval.api.instance import Instance
    result = []
    for i, args in enumerate(row['arguments']):
        args = list(args)
        if row['task'] == 'gsm8k' and generation_limit is not None:
            args[1] = {**args[1], 'max_gen_toks': generation_limit}
        result.append(Instance(request_type='generate_until' if row['task'] == 'gsm8k' else 'loglikelihood',
            doc=row['doc'], arguments=tuple(args), idx=i, metadata=(row['task'], row['doc_id'], 1)))
    return result


def evaluate(lm, task, row, generation_limit=None):
    reqs = requests(row, generation_limit)
    outputs = lm.generate_until(reqs, disable_tqdm=True) if row['task'] == 'gsm8k' else lm.loglikelihood(reqs, disable_tqdm=True)
    for req, out in zip(reqs, outputs):
        req.resps = [out]
    task._instances = reqs
    task.apply_filters()
    filters = sorted(reqs[0].filtered_resps)
    scores = {name: task.process_results(row['doc'], [r.filtered_resps[name] for r in reqs]) for name in filters}
    return {'responses': outputs, 'filtered_responses': {name: [r.filtered_resps[name] for r in reqs] for name in filters},
            'scores': scores}


def gate(group, row, memory, basis, lm, task, kb, ids):
    import torch
    p = ROOT / 'gates' / f'{group}_gpu.json'
    if p.exists():
        assert read(p)['status'] == 'passed'
        return
    native = tuple(t.cuda() for t in memory)
    native_logits = probe(lm.model, ids, native)
    floor = kl(probe(lm.model, ids), native_logits)
    full, _ = transform(memory, basis, 'KV', 128)
    full_error = kl(native_logits, probe(lm.model, ids, full))
    assert floor['first_token_kl'] < 1e-3 and floor['logit_mean_abs'] < .05, floor
    assert full_error['first_token_kl'] < 1e-3 and full_error['logit_mean_abs'] < .05, full_error
    # The same official generator is exercised using a gate-only 32-token limit.
    lm.mode = 'native'
    ordinary = evaluate(lm, task, row, 32)
    lm.mode = 'oracle'; lm.pinned = native
    oracle = evaluate(lm, task, row, 32)
    lm.pinned = full
    reconstructed = evaluate(lm, task, row, 32)
    if row['task'] == 'gsm8k':
        assert ordinary['responses'] == oracle['responses'] == reconstructed['responses'], 'Native/full-rank generation gate mismatch'
    else:
        a, b = np.array(ordinary['responses'])[:, 0], np.array(oracle['responses'])[:, 0]
        c = np.array(reconstructed['responses'])[:, 0]
        assert np.max(abs(a - b)) < .15 and np.max(abs(b - c)) < .15, 'Likelihood parity failed'
        assert ordinary['scores'] == oracle['scores'] == reconstructed['scores'], 'Native/full-rank decisions differ'
    save(f'gates/{group}_gpu.json', {'status': 'passed', 'timestamp': now(), 'id': row['id'],
        'native_floor': floor, 'full_rank_error': full_error, 'short_gate_only_max_gen_toks': 32,
        'ordinary': ordinary, 'oracle': oracle, 'full_rank': reconstructed,
        'checks': ['native_cache_reconstruction', 'full_rank_reconstruction', 'native_anchor_exact',
                   'official_likelihood_or_generation_parity', 'unchanged_slot_count']})
    lm.pinned = None
    del native, full
    torch.cuda.empty_cache()
    log(f'{group} GPU gates passed: native KL={floor["first_token_kl"]:.6g}, full128 KL={full_error["first_token_kl"]:.6g}')


def mean_scores(rows, arm):
    filters = rows[0]['arms'][arm]['evaluation']['scores']
    return {f: {m: float(np.mean([r['arms'][arm]['evaluation']['scores'][f][m] for r in rows]))
                for m in metrics} for f, metrics in filters.items()}


def summarize(partial=False):
    rows = [read(p) for p in sorted((ROOT / 'per_sample').rglob('*.json'))]
    complete = [r for r in rows if len(r['arms']) == len(ARMS)]
    tasks = {}
    for task in sorted({r['task'] for r in complete}):
        chosen = [r for r in complete if r['task'] == task]
        tasks[task] = {'samples': len(chosen), 'arms': {arm: {
            'scores': mean_scores(chosen, arm),
            'first_token_kl_mean': float(np.mean([r['arms'][arm]['output']['first_token_kl'] for r in chosen])),
            'first_argmax_match_rate': float(np.mean([r['arms'][arm]['output']['first_argmax_match'] for r in chosen])),
            'retained_centered_energy': {kind: float(np.mean([r['arms'][arm]['retained_centered_energy'][kind] for r in chosen])) for kind in ('K', 'V')}
            } for arm, _, _ in ARMS}}
    result = {'status': 'running' if partial else 'completed', 'timestamp': now(), 'config': CFG,
        'manifest_signature': read(ROOT / 'manifest.json')['signature'], 'completed_samples': len(complete),
        'expected_samples': 72, 'tasks': tasks,
        'scores_recomputed': True, 'full_official_test': False}
    if not partial:
        assert len(complete) == 72 and all(v['samples'] == 24 for v in tasks.values())
    save('partial_summary.json' if partial else 'summary.json', result)
    return result


def report(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# Qwen 原生 KV PCA 压缩干预', '', '服务器 B；沿用原 PCA 的校准/分析划分，每任务 24 条分析样本。',
        '所有分数均重新评测；首 token KL 与完整生成评分分开。未训练模型，未删除 token，后续生成 KV 不压缩。', '',
        '| 任务 | 条件 | 首 token KL | acc / strict EM | acc_norm / flexible EM |', '|---|---|---:|---:|---:|']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (task, entry) in enumerate(summary['tasks'].items()):
        native = entry['arms']['native']
        filt = 'strict-match' if task == 'gsm8k' else 'none'
        metric = 'exact_match' if task == 'gsm8k' else 'acc'
        for arm, _, _ in ARMS:
            item = entry['arms'][arm]
            score = item['scores'][filt][metric]
            other = item['scores']['flexible-extract']['exact_match'] if task == 'gsm8k' else item['scores']['none']['acc_norm']
            lines.append(f'| {task} | {arm} | {item["first_token_kl_mean"]:.5f} | {score:.4f} | {other:.4f} |')
        for kind in CFG['compression_targets']:
            x = CFG['retained_components']
            items = [entry['arms'][f'{kind}_{k}'] for k in x]
            axes[0, col].plot(x, [a['scores'][filt][metric] for a in items], marker='o', label=kind)
            axes[1, col].plot(x, [a['first_token_kl_mean'] for a in items], marker='o', label=kind)
        axes[0, col].axhline(native['scores'][filt][metric], color='black', linestyle='--', label='Native')
        axes[0, col].set(title=task, ylabel=metric, ylim=(-.03, 1.03))
        axes[1, col].set(xlabel='Retained principal components / head', ylabel='First-token KL')
        for ax in axes[:, col]:
            ax.legend(); ax.grid(alpha=.2)
    folder = ROOT / 'figures'; folder.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'compression_vs_output.{ext}', dpi=180)
    plt.close(fig)
    lines += ['', '![压缩与输出](figures/compression_vs_output.png)', '',
        '本实验只检验在原生 PCA 方向上丢弃尾部成分的影响；不能仅凭相同 k 将它等同于翻译器的频谱、方向或误差。',
        '保留方差为相对校准均值的能量；不等于语义保留率。谱的计算说明见 README。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def shared(summary):
    import fcntl
    p = Path(CFG['shared_results'])
    begin, end = '<!-- BEGIN ' + CFG['experiment_name'] + ' -->', '<!-- END ' + CFG['experiment_name'] + ' -->'
    lines = [begin, '## ' + CFG['experiment_name'], '', '- Server B; status: ' + summary['status'],
        '- Timestamp: ' + now(), '- Report: `' + str(ROOT / 'REPORT.md') + '`',
        '- Config: `' + str(ROOT / 'config.json') + '`; manifest: `' + str(ROOT / 'manifest.json') + '`',
        '- Log: `' + str(ROOT / 'logs/pipeline.log') + '`; samples: `' + str(ROOT / 'per_sample') + '`']
    if summary['status'] == 'completed':
        lines += ['- Fresh scores; exact original PCA72 analysis prompts; 12 PCA interventions + native; no training.',
                  '- Raw summary: `' + str(ROOT / 'summary.json') + '`']
        for task, item in summary['tasks'].items():
            lines.append('- ' + task + ': ' + json.dumps({a: x['scores'] for a, x in item['arms'].items()}))
    elif 'error' in summary:
        lines.append('- Failure: ' + summary['error'])
    section = '\n'.join(lines + ['', end, ''])
    with (p.parent / '.experiment_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = p.read_text() if p.exists() else '# Experiment Results\n'
        if begin in text:
            start = text.index(begin); stop = text.index(end, start) + len(end)
            text = text[:start] + section + text[stop:]
        else:
            text += '\n' + section
        temp = p.with_suffix('.md.tmp'); temp.write_text(text, encoding='utf-8'); temp.replace(p)
    save('shared_summary_update.json', {'status': 'completed', 'run_status': summary['status'], 'timestamp': now()})


def spectra(group, sampled, basis):
    result = {'method': 'Held-out centered native covariance projected through calibration PCA; before FP16 rounding', 'kinds': {}}
    for kind in ('K', 'V'):
        x = np.concatenate(sampled[kind], 0).astype(np.float64)
        x -= x.mean(0)
        covariance = np.einsum('nlhd,nlhe->lhde', x, x, optimize=True) / len(x)
        u = basis[kind + '_basis'].astype(np.float64)
        small = np.swapaxes(u, -1, -2) @ covariance @ u
        result['kinds'][kind] = {}
        for k in CFG['retained_components'] + [128]:
            eig = np.maximum(np.linalg.eigvalsh(small[..., :k, :k])[..., ::-1], 0)
            p = eig / np.maximum(eig.sum(-1, keepdims=True), 1e-20)
            effective = np.exp(-(p * np.log(np.maximum(p, 1e-300))).sum(-1))
            cumulative = np.cumsum(p, -1)
            result['kinds'][kind][str(k)] = {'effective_rank_mean': float(effective.mean()),
                'effective_rank_per_layer_head': effective.tolist(),
                'covariance_spectrum_cumulative': cumulative.tolist()}
    save(f'metrics/{group}_spectra.json', result)


def run():
    import torch
    from threadpoolctl import threadpool_limits
    from transformers import AutoTokenizer
    sys.path.insert(0, str(PARENT))
    import kv_backend as kb
    from tasks_native import tasks
    cpu_gate()
    assert torch.cuda.is_available()
    kb.seed_all(CFG['seed'])
    objects = tasks(['openbookqa', 'arc_challenge', 'gsm8k'])
    records, signature = prepare(kb, objects)
    save('status.json', {'status': 'running', 'phase': 'loading', 'timestamp': now()})
    shared({'status': 'running'})
    while torch.cuda.mem_get_info()[0] < CFG['gpu_free_memory_before_load_gib'] * (1 << 30):
        log('Waiting for 14GiB free GPU memory; other jobs untouched'); time.sleep(20)
    model = kb.load_model({'models': kb.CFG['models'], 'attention_implementation': kb.CFG['attention_implementation']}, 'qwen')
    tokenizer = AutoTokenizer.from_pretrained(kb.CFG['models']['qwen'], local_files_only=True)

    class PinnedHFLM(kb.FullSyncHFLM):
        pinned = None
        expected_ids = None

        def _prefix(self, context, ids):
            assert list(ids) == self.expected_ids, 'Request changed context tokenization'
            assert kb.encoding(self.tokenizer, context.rstrip()) == self.expected_ids
            assert self.pinned is not None and self.pinned[0].shape[1] == len(ids) - 1
            return self.pinned

    lm = PinnedHFLM(model, mode='oracle')
    save('effective_harness_settings.json', {'max_gen_toks': lm.max_gen_toks, 'max_length': lm.max_length,
        'softmax_dtype': str(lm.softmax_dtype), 'mixed_precision_dtype': str(lm.mixed_precision_dtype),
        'model_dtype': str(next(model.parameters()).dtype), 'generation_kwargs_source': 'exact original sample arguments'})
    for group in CFG['groups']:
        with threadpool_limits(limits=4):
            basis = fit_basis(group, records[group], kb, model, tokenizer, signature)
        sampled = {'K': [], 'V': []}
        analysis = [r for r in records[group] if r['role'] == 'analysis']
        for index, row in enumerate(analysis):
            ids, memory = capture(kb, model, tokenizer, row)
            chosen = positions(row, len(ids) - 2)
            for kind, tensor in zip(('K', 'V'), memory):
                sampled[kind].append(tensor[:, [p + 1 for p in chosen]].permute(1, 0, 2, 3).numpy().astype(np.float32))
            lm.expected_ids = ids
            gate(group, row, memory, basis, lm, objects[row['task']], kb, ids)
            native = tuple(t.cuda() for t in memory)
            reference_logits = probe(model, ids, native)
            relative = f'per_sample/{group}/{row["id"].replace(":", "_")}.json'
            path = ROOT / relative
            result = read(path) if path.exists() else {'id': row['id'], 'task': row['task'], 'role': 'analysis',
                'doc_id': row['doc_id'], 'context_sha256': row['context_sha256'], 'arguments': row['arguments'],
                'doc': row['doc'], 'context_tokens': len(ids), 'external_tokens': len(ids) - 2,
                'original_native_scores_for_reference_only': row['original_native_scores'],
                'manifest_signature': signature, 'arms': {}}
            assert result['manifest_signature'] == signature
            for arm, kind, k in ARMS:
                if arm in result['arms']:
                    continue
                save('status.json', {'status': 'running', 'phase': 'evaluation', 'group': group,
                    'sample_index': index, 'sample_count': len(analysis), 'id': row['id'], 'arm': arm, 'timestamp': now()})
                start = time.monotonic()
                with torch.inference_mode():
                    transformed, energy = (native, {'K': 1.0, 'V': 1.0}) if kind is None else transform(memory, basis, kind, k)
                    lm.pinned = transformed
                    out_metrics = kl(reference_logits, probe(model, ids, transformed))
                    evaluation = evaluate(lm, objects[row['task']], row)
                result['arms'][arm] = {'kind': kind, 'retained_components': k, 'output': out_metrics,
                    'retained_centered_energy': energy, 'evaluation': evaluation, 'seconds': time.monotonic() - start}
                save(relative, result)
                lm.pinned = None
                del transformed
                log(f'{group} {index + 1}/{len(analysis)} {row["id"]} {arm}: KL={out_metrics["first_token_kl"]:.5f}; {evaluation["scores"]}; {result["arms"][arm]["seconds"]:.1f}s')
            del native, memory
            if index % 4 == 0 or index + 1 == len(analysis):
                summarize(partial=True)
            torch.cuda.empty_cache()
        with threadpool_limits(limits=4):
            spectra(group, sampled, basis)
        del sampled, basis
        gc.collect()
    summary = summarize()
    report(summary); shared(summary)
    save('status.json', {'status': 'completed', 'timestamp': now()})
    log('Completed compression intervention, fresh task scores, figures and shared summary')


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        failure = {'status': 'failed', 'timestamp': now(), 'error': type(exc).__name__ + ': ' + str(exc)}
        save('status.json', failure); save('failure.json', failure)
        try:
            shared(failure)
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        raise
