"""Full native Llama covariance spectra and source-alignment control; no KV files."""
import gc
import hashlib
import json
import math
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
sys.path.insert(0, str(REFERENCE))
import analyze as original

CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
VIEWS = ['llama_native', 'llama_aligned']
COLORS = {'llama_native': '#059669', 'llama_aligned': '#9333ea', 'qwen_native': '#2563eb', 'stage_b': '#dc2626'}
LABELS = {'llama_native': 'Llama native tokens', 'llama_aligned': 'Llama aligned source slots',
          'qwen_native': 'Qwen native', 'stage_b': 'Translated A+B'}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, obj):
    path = ROOT / name; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def log(message):
    print(f'[{now()}] {message}', flush=True)


def counts(cumulative, threshold):
    assert np.all(cumulative[:, -1] >= threshold)
    return np.argmax(cumulative >= threshold, axis=1) + 1


def describe(cumulative, ranks):
    output = {'layer_head_count': len(cumulative), 'effective_rank_mean': float(np.mean(ranks)),
              'pc1_pc2_variance_mean': float(cumulative[:, 1].mean()), 'thresholds': {}}
    for threshold in CFG['thresholds']:
        k = counts(cumulative, threshold)
        output['thresholds'][f'{threshold:.2f}'] = {'median': float(np.median(k)),
            'median_ceil': int(math.ceil(float(np.median(k)))), 'mean': float(k.mean()),
            'p10': float(np.percentile(k, 10)), 'p90': float(np.percentile(k, 90)),
            'min': int(k.min()), 'max': int(k.max())}
    return output


def native_positions(source_offsets, target_offsets, target_length, bos_count):
    left, right = target_offsets[1][0], target_offsets[target_length - 2][1]
    indices = [i + bos_count for i, (a, b) in enumerate(source_offsets) if a < b and a < right and b > left]
    boundary = [i + bos_count for i, (a, b) in enumerate(source_offsets)
                if a < b and a < right and b > left and (a < left or b > right)]
    assert indices and len(indices) == len(set(indices))
    return indices, boundary, [left, right]


def gate(model, ids, memory):
    import torch
    from protocol import final_logits
    with torch.inference_mode():
        full = final_logits(model, ids).float().cpu()
        k, v = [t[:, :-1].cuda() for t in memory]
        rebuilt = final_logits(model, ids[-1:], k, v, torch.arange(len(ids) - 1, device='cuda')).float().cpu()
    lp, lq = full.log_softmax(-1), rebuilt.log_softmax(-1)
    kl = float((lp.exp() * (lp - lq)).sum())
    mae = float((full - rebuilt).abs().mean())
    assert kl < 1e-3 and mae < .05, f'Llama native reconstruction failed: KL={kl}, MAE={mae}'
    save('gates/native_cache.json', {'status': 'passed', 'kl': kl, 'logit_mean_abs': mae,
        'first_argmax_match': bool(full.argmax() == rebuilt.argmax()),
        'criterion': 'Native same-KV injection KL<1e-3 and mean absolute logit error<0.05 in original FP16'})


def collect(group, samples, model, kb, source_tok, target_tok):
    buffers = {role: {kind: {view: [] for view in VIEWS} for kind in ['K', 'V']} for role in ['calibration', 'analysis']}
    for number, row in enumerate(samples):
        save('status.json', {'status': 'running', 'phase': 'capture', 'group': group,
                            'sample_index': number, 'sample_count': len(samples), 'timestamp': now()})
        context = row['context']
        assert hashlib.sha256(context.encode()).hexdigest() == row['context_sha256']
        target_ids = kb.encoding(target_tok, context)
        source_ids, mapping, sync_counts = kb.source_map(context, source_tok, target_tok, target_ids)
        s = source_tok(context, add_special_tokens=False, return_offsets_mapping=True)
        t = target_tok(context, add_special_tokens=False, return_offsets_mapping=True)
        assert list(t['input_ids']) == target_ids
        bos_count = len(source_ids) - len(s['input_ids'])
        assert bos_count in [0, 1]
        eligible, boundary, interval = native_positions(s['offset_mapping'], t['offset_mapping'], len(target_ids), bos_count)
        seed = int(hashlib.sha256(f'{CFG["seed"]}:{row["id"]}:native_unique'.encode()).hexdigest()[:16], 16)
        selected = sorted(random.Random(seed).sample(eligible, min(CFG['sampled_tokens_per_prompt'], len(eligible))))
        old_path = REFERENCE / f'per_sample/{group}/{row["id"].replace(":", "_")}.json'
        old = json.loads(old_path.read_text(encoding='utf-8'))
        assert old['context_sha256'] == row['context_sha256'] and old['role'] == row['role']
        target_positions = old['sampled_target_positions']
        aligned = [mapping[p - 1] for p in target_positions]
        k, v = kb.capture(model, source_ids)
        assert k.shape == v.shape == (28, len(source_ids), 8, 128)
        import torch
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        if group == 'mcq' and number == 0:
            gate(model, source_ids, (k, v))
        for view, positions in [('llama_native', selected), ('llama_aligned', aligned)]:
            for kind, tensor in [('K', k), ('V', v)]:
                buffers[row['role']][kind][view].append(tensor[:, positions].permute(1, 0, 2, 3).numpy().astype(np.float32))
        result = {'id': row['id'], 'task': row['task'], 'role': row['role'], 'context_sha256': row['context_sha256'],
            'source_tokens_with_bos': len(source_ids), 'bos_count': bos_count,
            'target_context_tokens': len(target_ids), 'external_character_interval': interval,
            'eligible_native_source_positions': len(eligible), 'boundary_overlap_positions': boundary,
            'sampled_unique_native_source_positions': selected,
            'original_sampled_target_positions': target_positions, 'aligned_source_positions': aligned,
            'aligned_duplicate_slots': len(aligned) - len(set(aligned)), 'fullsync_unit_counts': sync_counts}
        save(f'per_sample/{group}/{row["id"].replace(":", "_")}.json', result)
        del k, v
        log(f'{group} {number + 1}/{len(samples)} {row["id"]}; unique_native={len(selected)}, aligned={len(aligned)}')
    return buffers


def analyze_group(group, buffers):
    metrics, projections, comparisons = [], [], {}
    prior = json.loads((REFERENCE / f'metrics/{group}_layer_head.json').read_text())
    for kind in ['K', 'V']:
        calibration = np.concatenate(buffers['calibration'][kind]['llama_native'], axis=0)
        values = {view: np.concatenate(buffers['analysis'][kind][view], axis=0) for view in VIEWS}
        comparisons[kind] = {}
        for layer in range(28):
            for head in range(8):
                pca = original.fit_native_pca(calibration[:, layer, head])
                for view, points in values.items():
                    x = points[:, layer, head]
                    covariance = original.cov(x)
                    eigenvalues = original.spectrum(covariance)
                    cumulative = np.cumsum(eigenvalues) / max(float(eigenvalues.sum()), 1e-20)
                    assert np.isfinite(cumulative).all() and abs(cumulative[-1] - 1) < 1e-8
                    metrics.append({'kind': kind, 'view': view, 'layer': layer, 'kv_head': head,
                        'analysis_tokens': len(x), 'calibration_tokens': len(calibration),
                        'effective_rank': original.effective_rank(eigenvalues),
                        'variance_trace': float(np.trace(covariance)),
                        'covariance_spectrum_cumulative': cumulative.tolist()})
                if layer in CFG['plot_layers_zero_based'] and head == CFG['plot_kv_head']:
                    projections.append({'kind': kind, 'layer': layer, 'kv_head': head,
                        'calibration_pc1_pc2_variance': float(pca['evr'][:2].sum()),
                        'views': {view: np.round((points[:, layer, head] - pca['mean']) @ pca['axes'][:, :2], 6).tolist()
                                  for view, points in values.items()}})
            log(f'{group} {kind} native PCA layer {layer + 1}/28')
        for view in VIEWS:
            selected = [m for m in metrics if m['kind'] == kind and m['view'] == view]
            matrix = np.asarray([m['covariance_spectrum_cumulative'] for m in selected])
            comparisons[kind][view] = {'summary': describe(matrix, [m['effective_rank'] for m in selected]),
                                       'spectra': matrix}
        for arm, name in [('native', 'qwen_native'), ('stage_b', 'stage_b')]:
            selected = [m for m in prior if m['kind'] == kind]
            matrix = np.asarray([m['arms'][arm]['covariance_spectrum_cumulative'] for m in selected])
            comparisons[kind][name] = {'summary': describe(matrix, [m['arms'][arm]['effective_rank'] for m in selected]),
                                      'spectra': matrix}
    save(f'metrics/{group}_layer_head.json', metrics)
    save(f'metrics/{group}_projection_points.json', projections)
    summary = {kind: {view: item['summary'] for view, item in views.items()} for kind, views in comparisons.items()}
    save(f'metrics/{group}_summary.json', summary)
    plot_scatter(group, projections)
    return summary, comparisons


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def finish(plt, fig, name):
    folder = ROOT / 'figures'; folder.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(folder / f'{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


def plot_scatter(group, projections):
    plt = plotting()
    for kind in ['K', 'V']:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, p in zip(axes.flat, [p for p in projections if p['kind'] == kind]):
            for view in VIEWS:
                points = np.asarray(p['views'][view])
                ax.scatter(points[:, 0], points[:, 1], s=7, alpha=.3, color=COLORS[view], label=LABELS[view], rasterized=True)
            ax.set_title(f'Layer {p["layer"] + 1}, head {p["kv_head"]} | PC1+2 {p["calibration_pc1_pc2_variance"]:.1%}')
            ax.set_xlabel('PC1, Llama-native calibration basis'); ax.set_ylabel('PC2')
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{group.upper()} | native Llama {kind}; alignment changes selected positions/multiplicity')
        finish(plt, fig, f'{group}_llama_pca_{kind}')


def plot_comparison(comparisons):
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, group in enumerate(['mcq', 'gsm8k']):
        for col, kind in enumerate(['K', 'V']):
            ax = axes[row, col]
            for name, item in comparisons[group][kind].items():
                spectra = item['spectra']; x = np.arange(1, 129)
                if name in ['llama_native', 'qwen_native']:
                    ax.fill_between(x, np.percentile(spectra, 10, 0), np.percentile(spectra, 90, 0), color=COLORS[name], alpha=.08)
                ax.plot(x, spectra.mean(0), color=COLORS[name], ls='--' if name == 'llama_aligned' else '-', label=LABELS[name])
            ax.axhline(.90, color='#94a3b8', ls=':', lw=.7); ax.axhline(.95, color='#94a3b8', ls=':', lw=.7)
            ax.set_title(f'{group.upper()} | {kind}'); ax.set_xlabel('Principal components per 128D head')
            ax.set_ylabel('Cumulative explained variance'); ax.set_xlim(1, 128); ax.set_ylim(0, 1.015); ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Full native/source/translated spectra | each curve uses its own per-head covariance basis\nMean over28x8 Llama or36x8 Qwen layer/head combinations; no layer/head identity assumed', fontsize=11)
    finish(plt, fig, 'native_vs_translated_spectra')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, group in enumerate(['mcq', 'gsm8k']):
        for col, kind in enumerate(['K', 'V']):
            ax = axes[row, col]; names = list(comparisons[group][kind])
            series = [counts(comparisons[group][kind][n]['spectra'], .95) for n in names]
            boxes = ax.boxplot(series, patch_artist=True, showfliers=False, whis=(10, 90))
            for box, name in zip(boxes['boxes'], names):box.set_facecolor(COLORS[name]); box.set_alpha(.35)
            ax.set_xticks(range(1, len(names) + 1), ['Llama\nnative', 'Llama\naligned', 'Qwen\nnative', 'Translated\nA+B'])
            ax.set_ylim(0, 128); ax.set_ylabel('Components for95% variance'); ax.set_title(f'{group.upper()} | {kind}'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Per-layer/head distribution | box: quartiles; whiskers: 10th-90th percentile, not confidence intervals', fontsize=11)
    finish(plt, fig, 'components_for_95pct_comparison')


def report(summary):
    lines = ['# Llama 原生 KV 的完整 PCA 分析', '',
        '使用原 PCA 实验完全相同的96个 prompt 和校准/对照划分。Llama 为28层×8 KV heads，Qwen 为36层×8 KV heads，每个 head 都是128维。',
        'Llama 保留原 writer 的 sender 编码，包括 BOS；K 是 RoPE 之前的原生 key，V 为原生 value。',
        'Llama native：在相同外部文本区间内，按原生 token 位置不重复采样。Llama aligned：原 Full-Sync copy/drop 后、进入翻译器前的精确 source slots，包含重复位置。',
        '各条件均基于各自对照样本的中心化协方差谱统计，不能把不同模型的 PCA 坐标方向直接对应。', '',
        '| 分组 | KV | 条件 | 80% | 90% | 95% | 99% | 平均有效秩 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for group, kinds in summary['groups'].items():
        for kind, views in kinds.items():
            for view, item in views.items():
                values = [item['thresholds'][f'{t:.2f}']['median_ceil'] for t in CFG['thresholds']]
                lines.append(f'| {group} | {kind} | {LABELS[view]} | ' + ' | '.join(map(str, values)) + f' | {item["effective_rank_mean"]:.2f} |')
    lines += ['', '成分数为所有层/head 的所需数量中位数，向上取整；原始逐层逐 head 结果保留在 metrics。', '',
        '![完整谱比较](figures/native_vs_translated_spectra.png)', '',
        '![95%方差所需成分数分布](figures/components_for_95pct_comparison.png)', '',
        '## 对齐抽样检查', '', '```json', json.dumps(summary['sampling'], ensure_ascii=False, indent=2), '```', '',
        '## 二维投影，仅供观察', '']
    for group in ['mcq', 'gsm8k']:
        for kind in ['K', 'V']:
            lines += [f'![{group} Llama {kind}](figures/{group}_llama_pca_{kind}.png)', '']
    lines += ['## 解释边界', '',
        '- 本实验对比同一批文本，但原生分词点集和对齐后点集不同；字符边界采用重叠规则并逐样本记录。',
        '- Llama 与 Qwen 层数不同、head 坐标不同；没有假定它们的同编号层/head 具有语义对应。',
        '- 方差谱集中不等于语义信息丢失；仅凭源空间与目标空间维度不同也不能证明翻译器导致信息瓶颈。',
        '- 对齐视图与已有翻译后统计使用精确相同的 target positions，便于检查 copy/drop 和 token 权重的影响。',
        '- 没有重新评测任务准确率，没有训练接收方，也没有进行混合缓存实验。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def shared(summary):
    import fcntl
    p = Path(CFG['shared_results']); begin = '<!-- BEGIN ' + CFG['experiment_name'] + ' -->'; end = '<!-- END ' + CFG['experiment_name'] + ' -->'
    lines = [begin, '## ' + CFG['experiment_name'], '', '- Status: `' + summary['status'] + '`', '- Timestamp: ' + summary['timestamp'],
        '- Server B; seed1234; exact same original PCA96 prompts and calibration/analysis roles; no training.',
        '- Native model: fixed parent Llama3.2-3B-Instruct, FP16/eager; no checkpoint selection.',
        '- Config: `' + str(ROOT / 'config.json') + '`; manifest: `' + str(ROOT / 'manifest.json') + '`',
        '- Raw summary: `' + str(ROOT / 'summary.json') + '`; samples: `' + str(ROOT / 'per_sample') + '`',
        '- Log: `' + str(ROOT / 'logs/pipeline.log') + '`; report: `' + str(ROOT / 'REPORT.md') + '`', '']
    if summary['status'] == 'completed':
        lines += ['| Group | KV | View | PC90 median ceil | PC95 median ceil | Effective rank mean |', '|---|---|---|---:|---:|---:|']
        for group, kinds in summary['groups'].items():
            for kind, views in kinds.items():
                for view, item in views.items():
                    lines.append(f'| {group} | {kind} | {view} | {item["thresholds"]["0.90"]["median_ceil"]} | {item["thresholds"]["0.95"]["median_ceil"]} | {item["effective_rank_mean"]:.4f} |')
    else:
        lines += ['Failure: ' + summary.get('error', 'unknown')]
    section = '\n'.join(lines + ['', end, ''])
    with (p.parent / '.experiment_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = p.read_text() if p.exists() else '# Experiment Results\n\n'
        if begin in text:
            start = text.index(begin); stop = text.index(end, start) + len(end); text = text[:start] + section + text[stop:]
        else:text += '\n' + section
        temp = p.with_suffix('.md.tmp'); temp.write_text(text, encoding='utf-8'); temp.replace(p)
    save('shared_summary_update.json', {'status': 'completed', 'run_status': summary['status'], 'path': str(p), 'timestamp': now()})


def run():
    import torch
    from transformers import AutoTokenizer
    from threadpoolctl import threadpool_limits
    assert torch.cuda.is_available()
    assert counts(np.array([[.5, .8, .9, 1]]), .9).tolist() == [3]
    assert native_positions([(0, 2), (2, 5)], [(0, 1), (1, 3), (3, 5)], 3, 1)[0] == [1, 2]
    save('gates/cpu.json', {'status': 'passed', 'checks': ['component_threshold', 'unique_native_boundary_overlap']})
    save('status.json', {'status': 'running', 'phase': 'prepare', 'timestamp': now()})
    source_manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    kb = original.backend(); kb.seed_all(CFG['seed'])
    manifest = {'config': CFG, 'source_manifest_sha256': original.digest(REFERENCE / 'manifest.json'),
        'script_sha256': original.digest(ROOT / 'analyze_llama.py'),
        'model_path': kb.CFG['models']['llama'], 'parent_backend_sha256': original.digest(kb.ROOT / 'kv_backend.py'),
        'model_config_sha256': original.digest(Path(kb.CFG['models']['llama']) / 'config.json'),
        'model_tokenizer_sha256': original.digest(Path(kb.CFG['models']['llama']) / 'tokenizer.json'),
        'samples': {g: [{'id': r['id'], 'role': r['role'], 'context_sha256': r['context_sha256']} for r in rs]
                    for g, rs in source_manifest['samples'].items()},
        'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}}
    save('manifest.json', manifest)
    while torch.cuda.mem_get_info()[0] < (8 << 30):
        log('Waiting for8GiB free GPU memory; other jobs untouched'); time.sleep(20)
    model = kb.load_model({'models': kb.CFG['models'], 'attention_implementation': kb.CFG['attention_implementation']}, 'llama')
    source_tok = AutoTokenizer.from_pretrained(kb.CFG['models']['llama'], local_files_only=True)
    target_tok = AutoTokenizer.from_pretrained(kb.CFG['models']['qwen'], local_files_only=True)
    summaries, comparisons = {}, {}
    for group, samples in source_manifest['samples'].items():
        buffers = collect(group, samples, model, kb, source_tok, target_tok)
        save('status.json', {'status': 'running', 'phase': 'pca', 'group': group, 'timestamp': now()})
        with threadpool_limits(limits=4):
            summaries[group], comparisons[group] = analyze_group(group, buffers)
        del buffers; gc.collect()
    del model; gc.collect(); torch.cuda.empty_cache()
    plot_comparison(comparisons)
    all_rows = [json.loads(p.read_text()) for p in (ROOT / 'per_sample').rglob('*.json')]
    sampling = {group: {role: {
        'prompts': len(chosen := [r for r in all_rows if r['id'].split(':')[0] in original.CFG['groups'][group] and r['role'] == role]),
        'native_sampled_tokens': sum(len(r['sampled_unique_native_source_positions']) for r in chosen),
        'aligned_sampled_slots': sum(len(r['aligned_source_positions']) for r in chosen),
        'aligned_duplicate_slots': sum(r['aligned_duplicate_slots'] for r in chosen)}
        for role in ['calibration', 'analysis']} for group in summaries}
    summary = {'status': 'completed', 'timestamp': now(), 'experiment_name': CFG['experiment_name'],
        'server': 'B', 'seed': CFG['seed'], 'config': CFG, 'groups': summaries, 'sampling': sampling,
        'dataset_split': 'same original official lm-eval test-pilot96 prompts;24 calibration and72 analysis prompts',
        'paths': {'report': str(ROOT / 'REPORT.md'), 'manifest': str(ROOT / 'manifest.json'), 'log': str(ROOT / 'logs/pipeline.log')}}
    save('summary.json', summary); report(summary); shared(summary)
    save('status.json', {'status': 'completed', 'phase': 'report', 'timestamp': now()})
    log('Completed Llama native and aligned-source PCA, comparisons and automatic shared summary')


if __name__ == '__main__':
    try:run()
    except Exception as exc:
        failure = {'status': 'failed', 'timestamp': now(), 'error': type(exc).__name__ + ': ' + str(exc)}
        save('status.json', failure); save('failure.json', failure); shared(failure)
        traceback.print_exc(); raise
