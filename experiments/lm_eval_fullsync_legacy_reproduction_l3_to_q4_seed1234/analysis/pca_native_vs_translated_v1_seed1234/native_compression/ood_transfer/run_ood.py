"""Frozen ID PCA transfer; exact existing official OOD requests and scoring."""
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np

ROOT = Path(__file__).resolve().parent
ID = ROOT.parent
PARENT = ID.parents[2]
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('id_compression_helpers', ID / 'run_compression.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.ROOT = ROOT
ARMS = helper.ARMS
save, read, log, sha, now = helper.save, helper.read, helper.log, helper.sha, helper.now


def flatten(objects):
    result = {}
    for key, value in objects.items():
        if isinstance(value, dict):
            result.update(flatten(value))
        elif hasattr(value, 'get_config'):
            result[key] = value
        elif isinstance(value, tuple) and hasattr(value[-1], 'get_config'):
            result[key] = value[-1]
    return result


def callable_config(value):
    """Stable task-function provenance, without process-specific memory addresses."""
    import functools
    import inspect
    if isinstance(value, functools.partial):
        return {'partial': value.func, 'args': value.args, 'keywords': value.keywords}
    if callable(value):
        name = getattr(value, '__module__', type(value).__module__) + '.' + getattr(value, '__qualname__', type(value).__qualname__)
        source = inspect.getsource(value)
        return {'callable': name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    raise TypeError(f'Unsupported task config value: {type(value).__name__}')


def stable_task_config(task):
    # dump_config stringifies top-level partials with process-specific addresses.
    # Replace those values with the actual functions before canonical encoding.
    import re
    config = task.dump_config()
    for key in config:
        raw = getattr(task.config, key, None)
        if callable(raw):
            config[key] = raw
    encoded = json.dumps(config, default=callable_config, sort_keys=True)
    assert not re.search(r'<function [^>]* at 0x[0-9a-fA-F]+>', encoded), 'Unstable function representation remains'
    return json.loads(encoded)


def request_list(row, generation_limit=None):
    from lm_eval.api.instance import Instance
    result = []
    for index, args in enumerate(copy.deepcopy(row['arguments'])):
        if row['output_type'] == 'generate_until' and generation_limit is not None:
            args[1]['max_gen_toks'] = generation_limit
        result.append(Instance(request_type='generate_until' if row['output_type'] == 'generate_until' else 'loglikelihood',
            doc=row['doc'], arguments=tuple(args), idx=index, metadata=(row['task'], row['doc_id'], 1)))
    return result


def evaluate(lm, task, row, generation_limit=None):
    requests = request_list(row, generation_limit)
    generation = row['output_type'] == 'generate_until'
    outputs = lm.generate_until(requests, disable_tqdm=True) if generation else lm.loglikelihood(requests, disable_tqdm=True)
    for request, response in zip(requests, outputs):
        request.resps = [response]
    task._instances = requests
    task.apply_filters()
    filters = sorted(requests[0].filtered_resps)
    result = {'responses': outputs,
        'filtered_responses': {f: [r.filtered_resps[f] for r in requests] for f in filters},
        'scores': {f: task.process_results(row['doc'], [r.filtered_resps[f] for r in requests]) for f in filters}}
    if generation and hasattr(lm, 'last_generation'):
        budget = generation_limit or row['arguments'][0][1].get('max_gen_toks', lm.max_gen_toks)
        result['generation'] = {**lm.last_generation, 'budget': budget,
            'hit_token_budget': lm.last_generation['new_tokens'] >= budget}
    return result


def prepare(kb, objects, tokenizer):
    import inspect
    import torch, transformers
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    old = read(ID / 'manifest.json')
    files = [ROOT / 'run_ood.py', ROOT / 'config.json', ID / 'run_compression.py', ID / 'manifest.json',
        ID / 'config.json', PARENT / 'config.json', PARENT / 'kv_backend.py', PARENT / 'base_code/protocol.py',
        PARENT / 'tasks_native.py', Path(inspect.getfile(ConfigurableTask)), Path(inspect.getfile(HFLM))]
    files += [Path(kb.CFG['models']['qwen']) / x for x in ('config.json', 'tokenizer.json')]
    for group in ('mcq', 'gsm8k'):
        files.append(ID / 'basis' / f'{group}.npz')
    records, replay = {}, {}
    calibration_hashes = {r['context_sha256'] for rs in read(ID.parent / 'manifest.json')['samples'].values()
                          for r in rs if r['role'] == 'calibration'}
    for group, settings in CFG['groups'].items():
        names = [name for name in sorted(objects) if name == group or name.startswith(group + '_')]
        if group == 'mmlu_pro':
            assert len(names) == settings['expected_subjects'], names
        records[group] = []
        for task_name in names:
            path = PARENT / 'runs/mcq/evaluation_ood' / f'qwen_native__{task_name}__samples.jsonl'
            files.append(path)
            raw = [json.loads(s) for s in path.read_text().splitlines() if s]
            candidates = {r['doc_id']: r for r in raw if r['filter'] == settings['primary_filter']}
            eligible, excluded = [], []
            for doc_id, r in sorted(candidates.items()):
                args = r['arguments']
                generation = objects[task_name].get_config('output_type') == 'generate_until'
                context = args[0][0] if generation else args[0][0].rstrip()
                assert all(a[0] == args[0][0] for a in args)
                if len(kb.encoding(tokenizer, context)) <= 2:
                    excluded.append(doc_id); continue
                if generation:
                    assert context == context.rstrip(), 'Do not rewrite trailing generation whitespace'
                eligible.append({**r, 'id': f'{task_name}:{doc_id}', 'task': task_name, 'group': group,
                    'context': context, 'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
                    'output_type': 'generate_until' if generation else 'multiple_choice'})
            random.Random(f'{CFG["seed"]}:ood:{task_name}').shuffle(eligible)
            count = settings.get('samples_per_subject', settings.get('sample_count'))
            assert len(eligible) >= count
            selected = sorted(eligible[:count], key=lambda r: r['doc_id'])
            assert all(r['context_sha256'] not in calibration_hashes for r in selected)
            records[group].extend(selected)
            # Replaying the original model responses must reproduce the official logged score.
            row = selected[0]
            class Replay:
                def loglikelihood(self, requests, **kwargs):
                    return [x[0] for x in row['resps']]
                generate_until = loglikelihood
            result = evaluate(Replay(), objects[task_name], row)
            for metric in row['metrics']:
                assert result['scores'][settings['primary_filter']][metric] == row[metric]
            replay[task_name] = {'status': 'passed', 'doc_id': row['doc_id'], 'scores': result['scores'],
                                 'excluded_no_external_tokens': excluded}
    assert sum(map(len, records.values())) == CFG['expected_samples']
    info = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
    for path, item in info.items():
        if path in old['files']:
            assert item['sha256'] == old['files'][path]['sha256'], f'ID dependency changed: {path}'
    body = {'config': CFG, 'files': info, 'id_manifest_signature': old['signature'],
        'task_configs': {name: stable_task_config(obj) for name, obj in objects.items()},
        'samples': {g: [{'id': r['id'], 'context_sha256': r['context_sha256'], 'arguments': r['arguments'],
                         'output_type': r['output_type'], 'doc_hash': r['doc_hash']} for r in rs] for g, rs in records.items()},
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                     'gpu': torch.cuda.get_device_name(0)}}
    signature = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    body.update(signature=signature, timestamp=now())
    if (ROOT / 'manifest.json').exists():
        assert read(ROOT / 'manifest.json')['signature'] == signature, 'Inputs changed; preserve old run in a separate directory'
    else:
        save('manifest.json', body)
    save('gates/official_scoring_replay.json', {'status': 'passed', 'tasks': replay})
    return records, signature


def gpu_gate(group, row, basis, lm, task, kb):
    import torch
    path = ROOT / 'gates' / f'{group}_gpu.json'
    if path.exists():
        assert read(path)['status'] == 'passed'; return
    ids, memory = helper.capture(kb, lm.model, lm.tokenizer, row)
    lm.expected_ids = ids
    native = tuple(x.cuda() for x in memory)
    logits = helper.probe(lm.model, ids, native)
    floor = helper.kl(helper.probe(lm.model, ids), logits)
    full, _ = helper.transform(memory, basis, 'KV', 128)
    error = helper.kl(logits, helper.probe(lm.model, ids, full))
    assert floor['first_token_kl'] < 1e-3 and floor['logit_mean_abs'] < .05, floor
    assert error['first_token_kl'] < 1e-3 and error['logit_mean_abs'] < .05, error
    lm.mode = 'native'
    ordinary = evaluate(lm, task, row, 32)
    lm.mode = 'oracle'; lm.pinned = native
    oracle = evaluate(lm, task, row, 32)
    lm.pinned = full
    rebuilt = evaluate(lm, task, row, 32)
    if row['output_type'] == 'generate_until':
        assert ordinary['responses'] == oracle['responses'] == rebuilt['responses'], 'Short generation parity failed'
    else:
        a, b, c = [np.array(x['responses'])[:, 0] for x in (ordinary, oracle, rebuilt)]
        assert max(np.max(abs(a - b)), np.max(abs(b - c))) < .15
        assert ordinary['scores'] == oracle['scores'] == rebuilt['scores']
    save(f'gates/{group}_gpu.json', {'status': 'passed', 'timestamp': now(), 'id': row['id'],
        'native_floor': floor, 'full_rank_error': error, 'gate_only_max_gen_toks': 32,
        'ordinary': ordinary, 'oracle': oracle, 'full_rank': rebuilt})
    lm.pinned = None
    del memory, native, full
    torch.cuda.empty_cache()
    log(f'{group} GPU gates passed; native KL={floor["first_token_kl"]:.6g}, full128 KL={error["first_token_kl"]:.6g}')


def score(row, arm, filt, metric):
    return float(row['arms'][arm]['evaluation']['scores'][filt][metric])


def summary(partial=False):
    rows = [read(p) for p in sorted((ROOT / 'per_sample').rglob('*.json'))]
    rows = [r for r in rows if len(r['arms']) == len(ARMS)]
    result = {'status': 'running' if partial else 'completed', 'timestamp': now(), 'config': CFG,
        'completed_samples': len(rows), 'expected_samples': CFG['expected_samples'], 'groups': {}}
    for group, settings in CFG['groups'].items():
        chosen = [r for r in rows if r['group'] == group]
        if not chosen:
            continue
        id_rows = [read(p) for p in sorted((ID / 'per_sample' / settings['basis_group']).glob('*.json'))]
        assert all(len(r['arms']) == len(ARMS) for r in id_rows)
        f, m = settings['primary_filter'], settings['primary_metric']
        id_f, id_m = settings['id_filter'], settings['id_metric']
        arms = {}
        for arm, _, _ in ARMS:
            delta = np.array([score(r, 'native', f, m) - score(r, arm, f, m) for r in chosen])
            id_delta = np.array([score(r, 'native', id_f, id_m) - score(r, arm, id_f, id_m) for r in id_rows])
            seed = int(hashlib.sha256(f'{CFG["seed"]}:{group}:{arm}'.encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed); b = CFG['bootstrap_resamples']
            boot = delta[rng.integers(0, len(delta), size=(b, len(delta)))].mean(-1)
            id_boot = id_delta[rng.integers(0, len(id_delta), size=(b, len(id_delta)))].mean(-1)
            gen = [r['arms'][arm]['evaluation']['generation'] for r in chosen
                   if 'generation' in r['arms'][arm]['evaluation']]
            arms[arm] = {'score': float(np.mean([score(r, arm, f, m) for r in chosen])),
                'paired_drop_from_native': float(delta.mean()), 'drop_bootstrap95': np.quantile(boot, [.025, .975]).tolist(),
                'id_drop_from_native': float(id_delta.mean()), 'ood_minus_id_drop': float(delta.mean() - id_delta.mean()),
                'ood_minus_id_drop_bootstrap95': np.quantile(boot - id_boot, [.025, .975]).tolist(),
                'first_token_kl_mean': float(np.mean([r['arms'][arm]['output']['first_token_kl'] for r in chosen])),
                'id_first_token_kl_mean': float(np.mean([r['arms'][arm]['output']['first_token_kl'] for r in id_rows])),
                'retained_centered_energy': {kind: float(np.mean([r['arms'][arm]['retained_centered_energy'][kind] for r in chosen])) for kind in ('K', 'V')},
                'generation': {'mean_tokens': float(np.mean([r['new_tokens'] for r in gen])),
                    'budget_hit_rate': float(np.mean([r['hit_token_budget'] for r in gen]))} if gen else None}
        result['groups'][group] = {'samples': len(chosen), 'id_samples': len(id_rows), 'filter': f, 'metric': m,
            'id_filter': id_f, 'id_metric': id_m, 'arms': arms,
            'context_tokens': {'min': min(r['context_tokens'] for r in chosen), 'max': max(r['context_tokens'] for r in chosen),
                               'mean': float(np.mean([r['context_tokens'] for r in chosen]))}}
    if not partial:
        assert len(rows) == CFG['expected_samples']
    save('partial_summary.json' if partial else 'summary.json', result)
    return result


def report(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    lines = ['# 冻结原生 PCA 基：OOD 压缩对照', '', '服务器 B；全部分数重新评测；不在 OOD 上拟合或选择 PCA 基。',
        'HellaSwag 使用 MCQ 基，MMLU-Pro 使用 GSM8K 基。MMLU-Pro2048 token，ID GSM8K256 token；格式与长度差异是解释限制。', '',
        '| 数据集 | 条件 | 分数 | 相对原生下降 pp | ID下降 pp | 首token KL |', '|---|---|---:|---:|---:|---:|']
    for col, (group, entry) in enumerate(result['groups'].items()):
        for arm, _, _ in ARMS:
            x = entry['arms'][arm]
            lines.append(f'| {group} | {arm} | {x["score"]:.4f} | {100*x["paired_drop_from_native"]:.2f} | {100*x["id_drop_from_native"]:.2f} | {x["first_token_kl_mean"]:.4f} |')
        for kind in CFG['compression_targets']:
            points = [entry['arms'][f'{kind}_{k}'] for k in CFG['retained_components']]
            axes[0, col].plot(CFG['retained_components'], [100*x['paired_drop_from_native'] for x in points], marker='o', label=kind + ' OOD')
            axes[0, col].plot(CFG['retained_components'], [100*x['id_drop_from_native'] for x in points], linestyle='--', alpha=.6, label=kind + ' ID')
            axes[1, col].plot(CFG['retained_components'], [x['first_token_kl_mean'] for x in points], marker='o', label=kind)
        axes[0, col].set(title=f'{group}, n={entry["samples"]}', ylabel='Drop from native (percentage points)')
        axes[0, col].axhline(0, color='black', linewidth=.7)
        axes[1, col].set(xlabel='Retained components per KV head', ylabel='First-token KL')
        for ax in axes[:, col]:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    folder = ROOT / 'figures'; folder.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'ood_vs_id.{ext}', dpi=180)
    plt.close(fig)
    lines += ['', '![OOD与ID](figures/ood_vs_id.png)', '', '## 解释范围', '',
        '- OOD 指偏离校准任务分布，不代表模型预训练未见。',
        '- 先比较各数据集内压缩相对原生的下降，不用 OOD 原始得分更低证明压缩更敏感。',
        '- 每学科仅一题；置信区间及逐样本输出见 summary/per_sample，不能视为完整 benchmark。',
        '- MMLU-Pro 与 GSM8K 的预算、答案格式、上下文长度和任务难度不同，跨任务下降差异不能全部归因于 OOD。',
        '- 相同 k 的保留能量可能因数据集而异；报告实际能量，不将低秩自动等同于信息丢失。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def shared(result):
    import fcntl
    p = Path(CFG['shared_results'])
    begin, end = '<!-- BEGIN ' + CFG['experiment_name'] + ' -->', '<!-- END ' + CFG['experiment_name'] + ' -->'
    lines = [begin, '## ' + CFG['experiment_name'], '', '- Server B; status: ' + result['status'],
        '- Timestamp: ' + now(), '- Config: `' + str(ROOT / 'config.json') + '`; manifest: `' + str(ROOT / 'manifest.json') + '`',
        '- Log: `' + str(ROOT / 'logs/pipeline.log') + '`; report: `' + str(ROOT / 'REPORT.md') + '`',
        '- Frozen ID PCA only; HellaSwag24 + MMLU-Pro14; original official OOD prompts, filters and budgets.']
    if 'error' in result:
        lines.append('- Error: ' + result['error'])
    if result['status'] == 'completed':
        for group, item in result['groups'].items():
            lines.append('- ' + group + ': ' + json.dumps({a: {'score': x['score'], 'drop': x['paired_drop_from_native']} for a, x in item['arms'].items()}))
    section = '\n'.join(lines + ['', end, ''])
    with (p.parent / '.experiment_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = p.read_text() if p.exists() else '# Experiment Results\n'
        if begin in text:
            start = text.index(begin); stop = text.index(end, start) + len(end)
            text = text[:start] + section + text[stop:]
        else:
            text += '\n' + section
        tmp = p.with_suffix('.md.tmp'); tmp.write_text(text, encoding='utf-8'); tmp.replace(p)
    save('shared_summary_update.json', {'status': 'completed', 'run_status': result['status'], 'timestamp': now()})


def run(preflight_only=False):
    import torch
    import datasets
    from transformers import AutoTokenizer
    datasets.disable_progress_bars()
    sys.path.insert(0, str(PARENT))
    import kv_backend as kb
    from tasks_native import tasks
    helper.cpu_gate()
    kb.seed_all(CFG['seed'])
    objects = flatten(tasks(['hellaswag', 'mmlu_pro']))
    tokenizer = AutoTokenizer.from_pretrained(kb.CFG['models']['qwen'], local_files_only=True)
    records, signature = prepare(kb, objects, tokenizer)
    bases = {}
    for group, settings in CFG['groups'].items():
        with np.load(ID / 'basis' / f'{settings["basis_group"]}.npz', allow_pickle=False) as data:
            assert str(data['signature']) == read(ID / 'manifest.json')['signature']
            bases[group] = {k: data[k].copy() for k in data.files if k != 'signature'}
    save('gates/frozen_basis.json', {'status': 'passed', 'checks': ['exact_ID_basis_signature', 'no_OOD_fit', 'disjoint_calibration_contexts']})
    if preflight_only:
        log('CPU preflight passed; exact OOD sample manifest and frozen basis verified')
        return
    while True:
        predecessor = read(ID / 'status.json')
        if predecessor['status'] == 'completed':
            break
        if predecessor['status'] == 'failed':
            raise RuntimeError('ID predecessor failed; preserve it and do not load another GPU model')
        save('status.json', {'status': 'waiting', 'phase': 'ID_predecessor', 'timestamp': now()})
        log('Waiting for ID predecessor completion; GPU model not loaded'); time.sleep(20)
    assert torch.cuda.is_available()
    while torch.cuda.mem_get_info()[0] < CFG['minimum_free_gpu_gib'] * (1 << 30):
        save('status.json', {'status': 'waiting', 'phase': 'GPU_memory', 'timestamp': now()})
        log('Waiting for14GiB free GPU memory; other jobs untouched'); time.sleep(20)
    save('status.json', {'status': 'running', 'phase': 'loading', 'timestamp': now()}); shared({'status': 'running'})
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
    for group, rows in records.items():
        gpu_gate(group, rows[0], bases[group], lm, objects[rows[0]['task']], kb)
    for group, rows in records.items():
        basis = bases[group]
        for index, row in enumerate(rows):
            ids, memory = helper.capture(kb, model, tokenizer, row)
            lm.expected_ids = ids
            native = tuple(x.cuda() for x in memory)
            reference = helper.probe(model, ids, native)
            relative = f'per_sample/{group}/{row["id"].replace(":", "_")}.json'
            p = ROOT / relative
            result = read(p) if p.exists() else {'id': row['id'], 'task': row['task'], 'group': group,
                'output_type': row['output_type'], 'context_tokens': len(ids), 'external_tokens': len(ids) - 2,
                'context_sha256': row['context_sha256'], 'arguments': row['arguments'], 'doc': row['doc'],
                'basis_group': CFG['groups'][group]['basis_group'], 'manifest_signature': signature, 'arms': {}}
            assert result['manifest_signature'] == signature
            for arm, kind, k in ARMS:
                if arm in result['arms']:
                    continue
                save('status.json', {'status': 'running', 'phase': 'evaluation', 'group': group,
                    'sample_index': index, 'sample_count': len(rows), 'id': row['id'], 'arm': arm, 'timestamp': now()})
                start = time.monotonic()
                with torch.inference_mode():
                    transformed, energy = (native, {'K': 1.0, 'V': 1.0}) if kind is None else helper.transform(memory, basis, kind, k)
                    lm.pinned = transformed
                    output = helper.kl(reference, helper.probe(model, ids, transformed))
                    evaluated = evaluate(lm, objects[row['task']], row)
                result['arms'][arm] = {'kind': kind, 'retained_components': k, 'output': output,
                    'retained_centered_energy': energy, 'evaluation': evaluated, 'seconds': time.monotonic() - start}
                save(relative, result)
                lm.pinned = None
                del transformed
                log(f'{group} {index + 1}/{len(rows)} {row["id"]} {arm}: KL={output["first_token_kl"]:.5f}; {evaluated["scores"]}; {result["arms"][arm]["seconds"]:.1f}s')
            del memory, native
            summary(partial=True)
            torch.cuda.empty_cache()
    final = summary(); report(final); shared(final)
    save('status.json', {'status': 'completed', 'timestamp': now()})
    log('Completed OOD frozen-basis intervention, paired ID comparison, figures and shared summary')


if __name__ == '__main__':
    try:
        run(preflight_only='--preflight' in sys.argv)
    except Exception as exc:
        failure = {'status': 'failed', 'timestamp': now(), 'error': type(exc).__name__ + ': ' + str(exc)}
        save('failure.json', failure); save('status.json', failure)
        try:
            shared(failure)
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        raise
