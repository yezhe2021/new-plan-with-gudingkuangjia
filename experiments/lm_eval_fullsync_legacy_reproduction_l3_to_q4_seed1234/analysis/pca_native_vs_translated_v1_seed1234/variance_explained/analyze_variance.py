"""Count components from the saved complete per-head covariance spectra. No GPU."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parent
THRESHOLDS = [0.80, 0.90, 0.95, 0.99]
ARMS = ['native', 'stage_a', 'stage_b']
LABELS = {'native': 'Qwen native', 'stage_a': 'Stage A', 'stage_b': 'Stage A+B'}
COLORS = {'native': '#2563eb', 'stage_a': '#f59e0b', 'stage_b': '#dc2626'}


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def component_counts(spectra, threshold):
    spectra = np.asarray(spectra, dtype=float)
    assert np.isfinite(spectra).all()
    assert np.all(np.diff(spectra, axis=-1) >= -1e-10)
    assert np.all(spectra[:, -1] >= threshold), 'Threshold not reached; never silently return one component'
    return np.argmax(spectra >= threshold, axis=-1) + 1


def statistics(values):
    return {'median': float(np.median(values)), 'median_ceil': int(math.ceil(float(np.median(values)))),
            'mean': float(np.mean(values)), 'p10': float(np.percentile(values, 10)),
            'p90': float(np.percentile(values, 90)), 'min': int(np.min(values)), 'max': int(np.max(values))}


def run(input_root):
    original = json.loads((input_root / 'summary.json').read_text(encoding='utf-8'))
    assert original['status'] == 'completed'
    assert component_counts([[.5, .8, .9, 1]], .9).tolist() == [3]
    spectra_by_group, summary, detailed = {}, {}, []
    for group in ['mcq', 'gsm8k']:
        records = json.loads((input_root / f'metrics/{group}_layer_head.json').read_text(encoding='utf-8'))
        summary[group], spectra_by_group[group] = {}, {}
        for kind in ['K', 'V']:
            chosen = sorted([r for r in records if r['kind'] == kind], key=lambda r: (r['layer'], r['kv_head']))
            assert [(r['layer'], r['kv_head']) for r in chosen] == [(l, h) for l in range(36) for h in range(8)]
            summary[group][kind], spectra_by_group[group][kind] = {}, {}
            for arm in ARMS:
                spectra = np.asarray([r['arms'][arm]['covariance_spectrum_cumulative'] for r in chosen])
                assert spectra.shape == (288, 128) and np.allclose(spectra[:, -1], 1)
                spectra_by_group[group][kind][arm] = spectra
                counts = {f'{t:.2f}': component_counts(spectra, t) for t in THRESHOLDS}
                matrix = np.stack(list(counts.values()), axis=1)
                assert np.all(np.diff(matrix, axis=1) >= 0)
                summary[group][kind][arm] = {
                    'layer_head_count': len(chosen),
                    'thresholds': {t: statistics(v) for t, v in counts.items()},
                    'coverage_at_fixed_components': {str(k): {
                        'mean': float(spectra[:, k - 1].mean()),
                        'median': float(np.median(spectra[:, k - 1])),
                        'p10': float(np.percentile(spectra[:, k - 1], 10)),
                        'p90': float(np.percentile(spectra[:, k - 1], 90))}
                        for k in [2, 4, 8, 16, 32, 64, 96, 128]}}
                for index, record in enumerate(chosen):
                    detailed.append({'group': group, 'kind': kind, 'arm': arm,
                        'layer_zero_based': record['layer'], 'kv_head_zero_based': record['kv_head'],
                        'dimension': 128, 'components_required': {t: int(v[index]) for t, v in counts.items()}})
    output = {'status': 'completed', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source_experiment': original['experiment_name'], 'source_seed': original['seed'],
        'method': 'Minimum k such that sum of the first k centered-covariance eigenvalues reaches the variance threshold',
        'basis': 'Each arm uses its own held-out covariance spectrum; this measures variance concentration, not cross-arm alignment',
        'aggregation': '36 layers x8 KV heads, independently,128 coordinates per head; tables use ceil(median required k)',
        'data': {group: {'analysis_prompts': item['analysis_prompt_count'], 'sampled_analysis_tokens': item['analysis_token_count']}
                 for group, item in original['groups'].items()},
        'thresholds': THRESHOLDS, 'results': summary,
        'limits': ['Explained variance is not task information or guaranteed model accuracy',
                   'This is a per-head result, not a joint PCA of all layers and heads',
                   'Native calibration PCA plots in the original report use a different basis and split',
                   'Token samples from a prompt are correlated; GSM8K includes few-shot demonstration tokens']}
    save('summary.json', output)
    save('per_layer_head.json', detailed)
    plot(spectra_by_group)
    report(output)
    print(json.dumps({'status': 'completed', 'layer_head_arm_records': len(detailed),
                      'png_figures': 3, 'report': str(ROOT / 'REPORT.md')}, ensure_ascii=False))


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    def finish(fig, name):
        fig.savefig(figures / (name + '.png'), bbox_inches='tight')
        fig.savefig(figures / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, group in enumerate(['mcq', 'gsm8k']):
        for col, kind in enumerate(['K', 'V']):
            ax = axes[row, col]
            for arm in ARMS:
                spectra = data[group][kind][arm]
                x = np.arange(1, 129)
                if arm != 'stage_a':
                    ax.fill_between(x, np.percentile(spectra, 10, axis=0), np.percentile(spectra, 90, axis=0), color=COLORS[arm], alpha=.10)
                ax.plot(x, spectra.mean(0), color=COLORS[arm], ls='--' if arm == 'stage_a' else '-', label=LABELS[arm])
            for threshold in [.80, .90, .95, .99]:
                ax.axhline(threshold, color='#94a3b8', ls=':', lw=.7)
            ax.set_title(f'{group.upper()} | {kind}')
            ax.set_xlabel('Number of principal components per 128D KV head')
            ax.set_ylabel('Cumulative explained variance')
            ax.set_xlim(1, 128); ax.set_ylim(0, 1.015)
            ax.set_xticks([2, 8, 16, 32, 64, 96, 128]); ax.grid(alpha=.15)
    axes[0, 0].legend()
    fig.suptitle('Complete covariance spectra | lines: mean; bands: 10th-90th percentile over layer/head\nEach arm uses its own PCA basis; Stage A and A+B nearly overlap', fontsize=12)
    finish(fig, 'full_variance_coverage')
    for threshold in [.90, .95]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row, group in enumerate(['mcq', 'gsm8k']):
            for col, kind in enumerate(['K', 'V']):
                ax = axes[row, col]
                for arm in ARMS:
                    counts = component_counts(data[group][kind][arm], threshold).reshape(36, 8)
                    x = np.arange(1, 37)
                    if arm != 'stage_a':
                        ax.fill_between(x, counts.min(1), counts.max(1), color=COLORS[arm], alpha=.10)
                    ax.plot(x, np.median(counts, axis=1), color=COLORS[arm], ls='--' if arm == 'stage_a' else '-', label=LABELS[arm])
                ax.set_title(f'{group.upper()} | {kind}')
                ax.set_xlabel('Layer (1-based)'); ax.set_ylabel('Required components per 128D KV head')
                ax.set_ylim(0, 128); ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle(f'Components needed for {threshold:.0%} variance | lines: median over8 heads; bands: min-max\nOwn covariance basis per arm; no model rerun', fontsize=12)
        finish(fig, f'components_for_{int(threshold * 100)}pct')


def report(summary):
    lines = ['# 完整方差谱：解释 KV 需要多少个主成分', '',
        '使用已完成实验保存的完整 128 维协方差谱，无需重跑模型。',
        '每一层、每一个 KV head 的 K/V 分开计算：从最大特征值开始累加，第一次达到目标方差比例的成分数即为所需数量。',
        '下表是 36 层 × 8 KV head 共 288 个组合的所需数量中位数，向上取整。每个 head 原始维度为 128。', '',
        '各组用各自对照样本的中心化协方差谱，回答自身方差集中在多少方向；不表示翻译 KV 与原生 KV 使用相同方向。',
        '原始二维图使用原生校准集的坐标和解释方差比例；这里使用对照样本自身完整谱，统计口径不同。', '',
        '| 任务组 | KV | 条件 | 80% 方差 | 90% 方差 | 95% 方差 | 99% 方差 |',
        '|---|---|---|---:|---:|---:|---:|']
    names = {'native': 'Qwen 原生', 'stage_a': 'Stage-A', 'stage_b': 'Stage-A+B'}
    for group, kinds in summary['results'].items():
        for kind, arms in kinds.items():
            for arm, item in arms.items():
                values = [item['thresholds'][f'{t:.2f}']['median_ceil'] for t in THRESHOLDS]
                lines.append(f'| {group} | {kind} | {names[arm]} | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## 层和 head 的差异', '',
        '以下是达到 95% 方差所需成分数的第 10–90 百分位范围（向外取整），用于展示异质性，不是置信区间。', '',
        '| 组 | KV | 条件 | P10–P90 | 全部 head 均达到 95% 所需上限 |', '|---|---|---|---:|---:|']
    for group, kinds in summary['results'].items():
        for kind, arms in kinds.items():
            for arm in ['native', 'stage_b']:
                s = arms[arm]['thresholds']['0.95']
                lines.append(f'| {group} | {kind} | {names[arm]} | {math.floor(s["p10"])}–{math.ceil(s["p90"])} | {s["max"]} |')
    lines += ['', '## 对结果的解释', '',
        '- 原生 V 的 95% 方差通常需要约 98–99 个成分，99% 方差约需 120 个，明显不能用二维或三维充分概括。',
        '- 原生 K 达到 90% 方差的典型数量约 43–47，翻译后约 20–21；翻译后的方差更集中。',
        '- Stage-A 与 Stage-A+B 所需数量非常接近，Stage-B 没有明显恢复原生的方差谱形状。',
        '- 这是方差集中程度的证据；不能直接证明丢失了多少语义、低方差方向不重要，或压缩后任务效果会保持。',
        '- 这些数量针对每个 head 的 128 维空间，不是整个模型跨层跨 head 联合表示的维度。', '',
        '## 图表', '',
        '![完整累积方差曲线](figures/full_variance_coverage.png)', '',
        '![逐层解释 90% 方差所需数量](figures/components_for_90pct.png)', '',
        '![逐层解释 95% 方差所需数量](figures/components_for_95pct.png)', '',
        '[完整统计](summary.json) · [每层每个 head 的精确数量](per_layer_head.json) · [计算脚本](analyze_variance.py)', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default = ROOT.parent if (ROOT.parent / 'metrics').is_dir() else ROOT.parent / 'downloaded_results'
    parser.add_argument('--input-root', type=Path, default=default)
    args = parser.parse_args()
    run(args.input_root)
