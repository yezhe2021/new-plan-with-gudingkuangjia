"""Frozen native/translated KV geometry and controlled receiver probes; no KV files."""
import argparse
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
PARENT = ROOT.parent.parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
ARMS = CFG['arms']
COLORS = {'native': '#2563eb', 'stage_a': '#f59e0b', 'stage_b': '#dc2626'}
LABELS = {'native': 'Qwen native', 'stage_a': 'Stage A', 'stage_b': 'Stage A+B'}


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def save(name, data):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def backend():
    sys.path.insert(0, str(PARENT))
    import kv_backend
    return kv_backend


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare(kb):
    from transformers import AutoTokenizer
    target_tok = AutoTokenizer.from_pretrained(kb.CFG['models']['qwen'], local_files_only=True)
    manifest = {'experiment_name': CFG['experiment_name'], 'timestamp': now(), 'config': CFG,
                'script_sha256': digest(ROOT / 'analyze.py'), 'parent_config': kb.CFG,
                'files': {}, 'checkpoints': {}, 'samples': {}}
    files = [PARENT / 'config.json', PARENT / 'kv_backend.py', PARENT / 'base_code/protocol.py',
             PARENT / 'base_code/translator.py', ROOT / 'config.json']
    for model in kb.CFG['models'].values():
        files += [Path(model) / 'config.json', Path(model) / 'tokenizer.json']
    for group, tasks in CFG['groups'].items():
        records = []
        manifest['checkpoints'][group] = {}
        for stage in ('stage_a', 'stage_b'):
            folder = PARENT / 'runs' / group / stage
            selection = json.loads((folder / 'selection.json').read_text())
            files += [folder / 'selection.json', folder / 'best.pt']
            manifest['checkpoints'][group][stage] = {'path': str(folder / 'best.pt'), 'selection': selection['best']}
        for task in tasks:
            by_arm = {}
            for arm, original in [('native', 'qwen_native'), ('stage_a', 'stage_a'), ('stage_b', 'stage_b')]:
                p = PARENT / 'runs' / group / 'evaluation' / f'{original}__{task}__samples.jsonl'
                files.append(p)
                by_arm[arm] = {(r['doc_id'], r['filter']): r for r in rows(p)}
            filt = 'strict-match' if task == 'gsm8k' else 'none'
            ids = sorted(k[0] for k in by_arm['native'] if k[1] == filt)
            excluded = [doc_id for doc_id in ids if len(kb.encoding(target_tok,
                by_arm['native'][(doc_id, filt)]['arguments'][0][0].rstrip())) <= 2]
            manifest.setdefault('excluded_no_external_tokens', {})[task] = excluded
            ids = [doc_id for doc_id in ids if doc_id not in excluded]
            random.Random(f'{CFG["seed"]}:{group}:{task}').shuffle(ids)
            count = CFG['calibration_prompts_per_task'] + CFG['analysis_prompts_per_task']
            assert len(ids) >= count
            for index, doc_id in enumerate(ids[:count]):
                original = by_arm['native'][(doc_id, filt)]
                args = original['arguments']
                context = args[0][0]
                assert all(a[0] == context for a in args), 'Different contexts among harness candidates'
                if task != 'gsm8k':
                    context = context.rstrip()
                else:
                    assert context == context.rstrip(), 'Generation prompt requires exact whitespace handling'
                scores = {}
                for arm in ARMS:
                    r = by_arm[arm][(doc_id, filt)]
                    assert r['arguments'] == args and r['doc_hash'] == original['doc_hash'], 'Sample/prompt mismatch'
                    scores[arm] = {k: r[k] for k in ['acc', 'acc_norm', 'exact_match'] if k in r}
                    if task == 'gsm8k':
                        other = by_arm[arm].get((doc_id, 'flexible-extract'))
                        if other is not None:
                            assert other['arguments'] == args
                            scores[arm]['flexible_exact_match'] = other['exact_match']
                records.append({'id': f'{task}:{doc_id}', 'task': task, 'doc_id': doc_id,
                    'role': 'calibration' if index < CFG['calibration_prompts_per_task'] else 'analysis',
                    'context': context, 'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
                    'original_prompt_hash': original['prompt_hash'], 'original_doc_hash': original['doc_hash'],
                    'original_task_scores': scores, 'original_score_filter': filt})
        manifest['samples'][group] = records
    for p in dict.fromkeys(files):
        log('hash input ' + str(p))
        manifest['files'][str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    import torch, transformers
    manifest['runtime'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                           'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
    save('manifest.json', manifest)
    return manifest


def cov(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(0)
    return centered.T @ centered / len(x)


def spectrum(c):
    return np.maximum(np.linalg.eigvalsh(c)[::-1], 0)


def effective_rank(values):
    total = values.sum()
    if total <= 1e-20:
        return 0.0
    p = values[values > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def fit_native_pca(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    eigenvalues, vectors = np.linalg.eigh(cov(x))
    values, axes = np.maximum(eigenvalues[::-1], 0), vectors[:, ::-1]
    evr = values / max(float(values.sum()), 1e-20)
    rank = min(len(evr), int(np.searchsorted(np.cumsum(evr), CFG['subspace_explained_variance'])) + 1)
    return {'mean': mean, 'axes': axes, 'evr': evr, 'rank': rank}


def geometry(x, y, pca):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    cx, cy = x - x.mean(0), y - y.mean(0)
    xx, yy, xy = cx.T @ cx, cy.T @ cy, cx.T @ cy
    native_cov, other_cov = xx / len(x), yy / len(y)
    native_values, other_values = spectrum(native_cov), spectrum(other_cov)
    centered = y - pca['mean']
    projection = centered @ pca['axes'][:, :pca['rank']]
    outside = np.maximum((centered * centered).sum(1) - (projection * projection).sum(1), 0).sum()
    return {
        'paired_nmse': float(((y - x) ** 2).sum() / max(float((x * x).sum()), 1e-20)),
        'paired_cosine': float(np.mean((x * y).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-20))),
        'norm_ratio': float(np.mean(np.linalg.norm(y, axis=1)) / max(float(np.mean(np.linalg.norm(x, axis=1))), 1e-20)),
        'mean_shift_variance_units': float(np.linalg.norm(y.mean(0) - x.mean(0)) / max(math.sqrt(float(np.trace(native_cov))), 1e-10)),
        'variance_ratio': float(np.trace(other_cov) / max(float(np.trace(native_cov)), 1e-20)),
        'covariance_relative_frobenius': float(np.linalg.norm(other_cov - native_cov) / max(float(np.linalg.norm(native_cov)), 1e-20)),
        'effective_rank': effective_rank(other_values),
        'native_effective_rank': effective_rank(native_values),
        'linear_cka': float((xy * xy).sum() / max(float(np.linalg.norm(xx) * np.linalg.norm(yy)), 1e-20)),
        'outside_native_90pct_subspace_energy_fraction': float(outside / max(float((centered * centered).sum()), 1e-20)),
        'covariance_spectrum_cumulative': (np.cumsum(other_values) / max(float(other_values.sum()), 1e-20)).tolist()
    }


def attention_metrics(q, native_k, native_v, other_k, other_v):
    """Inputs are RoPE-applied query/head and full-prefix KV, all in matched GQA order."""
    import torch
    q, native_k, native_v, other_k, other_v = [x.float() for x in (q, native_k, native_v, other_k, other_v)]
    native_scores = torch.einsum('hd,thd->ht', q, native_k) / math.sqrt(q.shape[-1])
    other_scores = torch.einsum('hd,thd->ht', q, other_k) / math.sqrt(q.shape[-1])
    lp, lq = native_scores.log_softmax(-1), other_scores.log_softmax(-1)
    p, w = lp.exp(), lq.exp()
    reference = torch.einsum('ht,thd->hd', p, native_v)
    output = torch.einsum('ht,thd->hd', w, other_v)
    ref_centered = native_scores - native_scores.mean(-1, keepdim=True)
    changed_centered = other_scores - other_scores.mean(-1, keepdim=True)
    return {
        'attention_kl_native_to_translated': float((p * (lp - lq)).sum(-1).mean()),
        'centered_qk_relative_rmse': float(((changed_centered - ref_centered).square().mean() / ref_centered.square().mean().clamp_min(1e-12)).sqrt()),
        'weighted_v_relative_rmse': float(((output - reference).square().mean() / reference.square().mean().clamp_min(1e-12)).sqrt()),
        'attention_entropy': float(-(w * lq).sum(-1).mean())
    }


def self_test():
    rng = np.random.default_rng(1234)
    x = rng.normal(size=(256, 8))
    pca = fit_native_pca(x)
    same = geometry(x, x, pca)
    assert same['paired_nmse'] == 0 and abs(same['linear_cka'] - 1) < 1e-10
    scaled = geometry(x, 2 * x, pca)
    assert abs(scaled['variance_ratio'] - 4) < 1e-10 and abs(scaled['linear_cka'] - 1) < 1e-10
    low = np.zeros((256, 8)); low[:, :2] = x[:, :2]
    shifted = low.copy(); shifted[:, 7] = 5
    basis = fit_native_pca(low)
    assert np.allclose((low - basis['mean']) @ basis['axes'][:, :2], (shifted - basis['mean']) @ basis['axes'][:, :2])
    assert geometry(low, shifted, basis)['outside_native_90pct_subspace_energy_fraction'] > 0.8
    import torch
    torch.manual_seed(1234)
    q, k, v = torch.randn(2, 8), torch.randn(16, 2, 8), torch.randn(16, 2, 8)
    same_read = attention_metrics(q, k, v, k, v)
    assert same_read['weighted_v_relative_rmse'] == 0 and same_read['attention_kl_native_to_translated'] == 0
    scaled_v = attention_metrics(q, k, v, k, 2 * v)
    assert abs(scaled_v['weighted_v_relative_rmse'] - 1) < 1e-6 and scaled_v['attention_kl_native_to_translated'] == 0
    shifted_k = attention_metrics(q, k, v, k + 0.7, v)
    assert abs(shifted_k['attention_kl_native_to_translated']) < 1e-6
    save('gates/cpu.json', {'status': 'passed', 'checks': ['identical_geometry', 'CKA_scale_invariance',
        '2D_overlap_can_hide_orthogonal_error', 'identical_fixed_query', 'V_error_with_identical_attention',
        'constant_key_shift_attention_invariance']})
    log('CPU mathematical checks passed')


def native_probe(receiver, ids):
    import torch
    queries, final_k, final_v, handles = {}, {}, {}, []
    def hook(index):
        def capture(module, args, kwargs):
            h = kwargs.get('hidden_states', args[0] if args else None)
            shape = (1, h.shape[1], -1, module.head_dim)
            q, k, v = module.q_proj(h).view(shape), module.k_proj(h).view(shape), module.v_proj(h).view(shape)
            if hasattr(module, 'q_norm'): q = module.q_norm(q)
            if hasattr(module, 'k_norm'): k = module.k_norm(k)
            queries[index], final_k[index], final_v[index] = q[0, -1:].cpu(), k[0, -1:].cpu(), v[0, -1:].cpu()
        return capture
    for index, layer in enumerate(receiver.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        with torch.inference_mode():
            inp = torch.tensor([ids], device='cuda')
            out = receiver.model(input_ids=inp, attention_mask=torch.ones_like(inp),
                position_ids=torch.arange(len(ids), device='cuda')[None], use_cache=False)
            logits = receiver.lm_head(out.last_hidden_state[:, -1])[0].float().cpu()
    finally:
        for handle in handles: handle.remove()
    return torch.stack([queries[i] for i in range(len(queries))]), torch.stack([final_k[i] for i in range(len(final_k))]), torch.stack([final_v[i] for i in range(len(final_v))]), logits


def controlled_reading(receiver, query, self_k, self_v, memories, prefix_tokens):
    import torch
    from protocol import apply_rope
    output = {arm: [] for arm in ARMS}
    repeat = receiver.config.num_attention_heads // receiver.config.num_key_value_heads
    with torch.inference_mode():
        for layer in range(receiver.config.num_hidden_layers):
            q = apply_rope(receiver, query[layer].cuda(), torch.tensor([prefix_tokens], device='cuda'))[0]
            kv = {}
            for arm in ARMS:
                k = torch.cat((memories[arm][0][layer], self_k[layer]), 0).half().cuda()
                v = torch.cat((memories[arm][1][layer], self_v[layer]), 0).half().cuda()
                k = apply_rope(receiver, k, torch.arange(prefix_tokens + 1, device='cuda'))
                kv[arm] = (k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1))
            for arm in ARMS:
                output[arm].append(attention_metrics(q, *kv['native'], *kv[arm]))
    return output


def readout_logits(receiver, ids, memory):
    import torch
    from protocol import final_logits
    k, v = (t.cuda().half() for t in memory)
    with torch.inference_mode():
        result = final_logits(receiver, ids[-1:], k, v, torch.arange(len(ids) - 1, device='cuda'))
    return result.float().cpu()


def kl_logits(reference, other):
    p, q = reference.log_softmax(-1), other.log_softmax(-1)
    return {'output_first_token_kl': float((p.exp() * (p - q)).sum()),
            'first_argmax_match': bool(reference.argmax() == other.argmax()),
            'logit_mean_abs': float((reference - other).abs().mean())}


def collect_group(group, samples, manifest, kb, sender, receiver, source_tok, target_tok):
    import torch
    from threadpoolctl import threadpool_limits
    folder = PARENT / 'runs' / group
    base, adapter = kb.load_base(folder / 'stage_a/best.pt'), kb.load_adapter(folder / 'stage_b/best.pt')
    for stage, checkpoint in [('stage_a', folder / 'stage_a/best.pt'), ('stage_b', folder / 'stage_b/best.pt')]:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        best = manifest['checkpoints'][group][stage]['selection']
        assert (payload['stage'], payload['epoch'], payload['step'], payload['protocol']) == (stage, best['epoch'], best['step'], kb.CFG['protocol'])
        del payload
    buffers = {role: {kind: {arm: [] for arm in ARMS} for kind in ['K', 'V']} for role in ['calibration', 'analysis']}
    metadata, diagnostics = {role: [] for role in buffers}, []
    for number, row in enumerate(samples):
        save('status.json', {'status': 'running', 'phase': 'capture', 'group': group, 'sample_index': number,
                            'sample_count': len(samples), 'sample_id': row['id'], 'timestamp': now()})
        item = kb.pair(sender, receiver, source_tok, target_tok, row['context'])
        assert item['tokens'] > 0, 'No external tokens; do not silently replace sampled documents'
        with torch.inference_mode():
            a_k, a_v = kb.map_base(base, item, kb.CFG['chunk_tokens'])
            with torch.amp.autocast('cuda', dtype=torch.float16):
                b_k, b_v, _, _ = adapter(a_k[None], a_v[None])
            translated = {'stage_a': (a_k.half().cpu(), a_v.half().cpu()),
                          'stage_b': (b_k[0].half().cpu(), b_v[0].half().cpu())}
        del a_k, a_v, b_k, b_v
        external = {'native': (item['target_k'], item['target_v']), **translated}
        assert all(k.shape == item['target_k'].shape and v.shape == item['target_v'].shape and
                   torch.isfinite(k).all() and torch.isfinite(v).all() for k, v in external.values())
        n = min(CFG['sampled_tokens_per_prompt'], item['tokens'])
        seed = int(hashlib.sha256(f'{CFG["seed"]}:{row["id"]}'.encode()).hexdigest()[:16], 16)
        selected = sorted(random.Random(seed).sample(range(item['tokens']), n))
        offsets = target_tok(row['context'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        for arm, (k, v) in external.items():
            for kind, tensor in [('K', k), ('V', v)]:
                buffers[row['role']][kind][arm].append(tensor[:, selected].permute(1, 0, 2, 3).numpy().astype(np.float32))
        for pos in selected:
            absolute = pos + 1
            lo, hi = offsets[absolute]
            metadata[row['role']].append({'sample_id': row['id'], 'task': row['task'], 'target_position': absolute,
                'relative_position': absolute / max(len(item['ids']) - 1, 1), 'token_text': row['context'][lo:hi]})
        result = {'id': row['id'], 'task': row['task'], 'role': row['role'], 'context_sha256': row['context_sha256'],
                  'context_tokens': len(item['ids']), 'external_tokens': item['tokens'],
                  'sampled_target_positions': [p + 1 for p in selected], 'copy_drop_counts': item['counts'],
                  'original_task_scores': row['original_task_scores'], 'task_scores_reused_not_new': True}
        if row['role'] == 'analysis' or number == 0:
            q, self_k, self_v, native_logits = native_probe(receiver, item['ids'])
            memories = {arm: (torch.cat((item['anchor_k'], k), 1), torch.cat((item['anchor_v'], v), 1))
                        for arm, (k, v) in external.items()}
            result['controlled_reading'] = controlled_reading(receiver, q, self_k, self_v, memories, len(item['ids']) - 1)
            oracle_logits = readout_logits(receiver, item['ids'], memories['native'])
            result['native_reconstruction_floor'] = kl_logits(native_logits, oracle_logits)
            result['actual_injection'] = {arm: kl_logits(native_logits, readout_logits(receiver, item['ids'], memories[arm]))
                                          for arm in ['stage_a', 'stage_b']}
            if number == 0:
                floor = result['native_reconstruction_floor']
                native_read = result['controlled_reading']['native']
                assert floor['output_first_token_kl'] < 1e-3 and floor['logit_mean_abs'] < 0.05, 'Native reconstruction failed'
                assert all(abs(x['attention_kl_native_to_translated']) < 1e-7 and x['weighted_v_relative_rmse'] == 0 for x in native_read)
                save(f'gates/{group}_gpu.json', {'status': 'passed', 'id': row['id'], 'native_floor': floor,
                     'checks': ['selected_checkpoint_metadata', 'finite_matched_token_KV', 'actual_native_cache_reconstruction', 'fixed_query_identity']})
            del q, self_k, self_v, memories, oracle_logits, native_logits
        diagnostics.append(result)
        save(f'per_sample/{group}/{row["id"].replace(":", "_")}.json', result)
        del item, external, translated
        torch.cuda.empty_cache()
        log(f'{group} {number + 1}/{len(samples)} {row["role"]} {row["id"]}; tokens={result["external_tokens"]}')
    del base, adapter
    gc.collect(); torch.cuda.empty_cache()
    save('status.json', {'status': 'running', 'phase': 'pca', 'group': group, 'timestamp': now()})
    with threadpool_limits(limits=4):
        return finalize_group(group, buffers, metadata, diagnostics)


def finalize_group(group, buffers, metadata, diagnostics):
    metrics, projections = [], {'group': group, 'analysis_tokens': metadata['analysis'], 'panels': []}
    for kind in ['K', 'V']:
        calibration = np.concatenate(buffers['calibration'][kind]['native'], 0)
        analysis = {arm: np.concatenate(buffers['analysis'][kind][arm], 0) for arm in ARMS}
        for layer in range(calibration.shape[1]):
            for head in range(calibration.shape[2]):
                pca = fit_native_pca(calibration[:, layer, head])
                entry = {'kind': kind, 'layer': layer, 'kv_head': head, 'calibration_tokens': len(calibration),
                    'analysis_tokens': len(analysis['native']), 'pc1_pc2_calibration_explained_variance': float(pca['evr'][:2].sum()),
                    'native_90pct_subspace_rank': pca['rank'], 'arms': {}}
                for arm in ARMS:
                    entry['arms'][arm] = geometry(analysis['native'][:, layer, head], analysis[arm][:, layer, head], pca)
                metrics.append(entry)
                if layer in CFG['plot_layers'] and head == CFG['plot_head']:
                    projections['panels'].append({'kind': kind, 'layer': layer, 'kv_head': head,
                        'pc1_pc2_calibration_explained_variance': entry['pc1_pc2_calibration_explained_variance'],
                        'calibration_spectrum': np.cumsum(pca['evr']).tolist(),
                        'arms': {arm: np.round((analysis[arm][:, layer, head] - pca['mean']) @ pca['axes'][:, :2], 6).tolist() for arm in ARMS}})
            log(f'{group} {kind} PCA layer {layer + 1}/{calibration.shape[1]}')
    save(f'metrics/{group}_layer_head.json', metrics)
    save(f'metrics/{group}_projection_points.json', projections)
    held = [r for r in diagnostics if r['role'] == 'analysis']
    reading = {arm: {key: [float(np.mean([r['controlled_reading'][arm][l][key] for r in held])) for l in range(36)]
                       for key in held[0]['controlled_reading'][arm][0]} for arm in ARMS}
    tasks = {}
    for task in CFG['groups'][group]:
        chosen = [r for r in held if r['task'] == task]
        tasks[task] = {arm: {key: float(np.mean([r['original_task_scores'][arm][key] for r in chosen]))
                            for key in chosen[0]['original_task_scores'][arm]} for arm in ARMS}
    aggregated = {kind: {arm: {key: float(np.mean([m['arms'][arm][key] for m in metrics if m['kind'] == kind]))
                               for key in ['paired_nmse', 'paired_cosine', 'norm_ratio', 'mean_shift_variance_units',
                                           'variance_ratio', 'effective_rank', 'linear_cka', 'outside_native_90pct_subspace_energy_fraction']}
                         for arm in ARMS} for kind in ['K', 'V']}
    summary = {'group': group, 'calibration_prompt_count': sum(r['role'] == 'calibration' for r in diagnostics),
        'analysis_prompt_count': len(held), 'calibration_token_count': len(metadata['calibration']),
        'analysis_token_count': len(metadata['analysis']), 'geometry_equal_weight_layer_head_mean': aggregated,
        'controlled_reading_layer_means': reading,
        'actual_injection': {arm: {key: float(np.mean([r['actual_injection'][arm][key] for r in held]))
                                  for key in held[0]['actual_injection'][arm]} for arm in ['stage_a', 'stage_b']},
        'native_reconstruction_floor': {key: float(np.mean([r['native_reconstruction_floor'][key] for r in held]))
                                        for key in held[0]['native_reconstruction_floor']},
        'original_official_scores_on_selected_analysis_documents': tasks,
        'pc1_pc2_explained_variance': {kind: {'mean': float(np.mean([m['pc1_pc2_calibration_explained_variance'] for m in metrics if m['kind'] == kind])),
            'min': float(min(m['pc1_pc2_calibration_explained_variance'] for m in metrics if m['kind'] == kind)),
            'max': float(max(m['pc1_pc2_calibration_explained_variance'] for m in metrics if m['kind'] == kind))} for kind in ['K', 'V']}}
    save(f'metrics/{group}_summary.json', summary)
    plot_group(group, metrics, projections, summary)
    return summary


def plot_group(group, metrics, projections, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120, 'savefig.dpi': 160, 'axes.spines.top': False, 'axes.spines.right': False})
    output = ROOT / 'figures'; output.mkdir(exist_ok=True)
    def finish(fig, suffix):
        fig.savefig(output / f'{group}_{suffix}.png', bbox_inches='tight')
        fig.savefig(output / f'{group}_{suffix}.pdf', bbox_inches='tight')
        plt.close(fig)
    for kind in ['K', 'V']:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, panel in zip(axes.flat, [p for p in projections['panels'] if p['kind'] == kind]):
            for arm in ARMS:
                points = np.asarray(panel['arms'][arm])
                ax.scatter(points[:, 0], points[:, 1], s=7, alpha=.32, color=COLORS[arm], label=LABELS[arm], rasterized=True)
            # Same-token arrows are descriptive, drawn at fixed indices independent of error.
            n = np.asarray(panel['arms']['native']); b = np.asarray(panel['arms']['stage_b'])
            for i in np.linspace(0, len(n) - 1, min(16, len(n))).astype(int):
                ax.annotate('', xy=b[i], xytext=n[i], arrowprops={'arrowstyle': '->', 'color': '#64748b', 'alpha': .4, 'lw': .6})
            ax.set_title(f'Layer {panel["layer"] + 1}, KV head {panel["kv_head"]} | PC1+2 {panel["pc1_pc2_calibration_explained_variance"]:.1%}')
            ax.set_xlabel('PC1, native calibration axes'); ax.set_ylabel('PC2')
        axes.flat[0].legend(markerscale=2, fontsize=9)
        fig.suptitle(f'{group.upper()} | {kind} | same held-out tokens, original scale', fontsize=14)
        finish(fig, f'pca_{kind}')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, kind in enumerate(['K', 'V']):
        for col, (key, title) in enumerate([('paired_nmse', 'Paired NMSE'), ('variance_ratio', 'Centered variance / native'), ('linear_cka', 'Linear CKA')]):
            for arm in ['stage_a', 'stage_b']:
                values = [np.mean([m['arms'][arm][key] for m in metrics if m['kind'] == kind and m['layer'] == l]) for l in range(36)]
                axes[row, col].plot(range(1, 37), values, color=COLORS[arm], label=LABELS[arm])
            axes[row, col].set_title(f'{kind}: {title}'); axes[row, col].set_xlabel('Layer'); axes[row, col].grid(alpha=.2)
            if key == 'paired_nmse': axes[row, col].set_yscale('log')
    axes[0, 0].legend(); fig.suptitle(f'{group.upper()} | high-dimensional geometry, mean over 8 KV heads')
    finish(fig, 'layer_metrics')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for row, kind in enumerate(['K', 'V']):
        matrices = [np.array([[next(m for m in metrics if m['kind'] == kind and m['layer'] == l and m['kv_head'] == h)['arms'][arm]['paired_nmse'] for h in range(8)] for l in range(36)]) for arm in ['stage_a', 'stage_b']]
        top = max(float(np.max(m)) for m in matrices)
        for col, (arm, matrix) in enumerate(zip(['stage_a', 'stage_b'], matrices)):
            im = axes[row, col].imshow(matrix, aspect='auto', vmin=0, vmax=top, cmap='magma')
            axes[row, col].set_title(f'{kind}: {LABELS[arm]} paired NMSE'); axes[row, col].set_xlabel('KV head (0-based)'); axes[row, col].set_ylabel('Layer (0-based)')
            fig.colorbar(im, ax=axes[row, col], shrink=.7)
    finish(fig, 'head_heatmaps')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (key, title) in zip(axes, [('centered_qk_relative_rmse', 'Fixed-Q centered QK relative RMSE'), ('attention_kl_native_to_translated', 'Fixed-Q attention KL'), ('weighted_v_relative_rmse', 'Fixed-Q weighted-V relative RMSE')]):
        for arm in ['stage_a', 'stage_b']:
            ax.plot(range(1, 37), summary['controlled_reading_layer_means'][arm][key], color=COLORS[arm], label=LABELS[arm])
        ax.set_title(title); ax.set_xlabel('Layer'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle(f'{group.upper()} | full-prefix controlled memory reading')
    finish(fig, 'reading')
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for row, kind in enumerate(['K', 'V']):
        for col, layer in enumerate(CFG['plot_layers']):
            entry = next(m for m in metrics if m['kind'] == kind and m['layer'] == layer and m['kv_head'] == CFG['plot_head'])
            for arm in ARMS:
                axes[row, col].plot(range(1, 129), entry['arms'][arm]['covariance_spectrum_cumulative'], color=COLORS[arm], label=LABELS[arm])
            axes[row, col].set_title(f'{kind}, layer {layer + 1}, head {CFG["plot_head"]}'); axes[row, col].set_xlabel('Principal components'); axes[row, col].set_ylim(0, 1.02); axes[row, col].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); fig.suptitle(f'{group.upper()} | held-out centered covariance spectra')
    finish(fig, 'spectra')


def report(summary):
    lines = ['# lm-eval translator KV PCA', '', '**Completed frozen analysis. No KV tensors saved.**', '',
        'Original validation-selected lm-eval retrained checkpoints; exact original harness prompts.',
        'Eight calibration and24 disjoint analysis documents per task. MCQ and GSM8K are independent.',
        'PCA is fitted to native calibration vectors per layer/head. All arms use the same center and axes.',
        'Plots use KV head0 at prespecified layers1/12/24/36; full36x8 metrics are retained.', '',
        'Task scores below reuse original official lm-eval outputs on the same selected documents.',
        'Fixed-Q attention probes isolate memory changes; actual cache-injection output KL is reported separately.', '']
    for group, item in summary['groups'].items():
        lines += [f'## {group}', '', f'Calibration: {item["calibration_prompt_count"]} prompts / {item["calibration_token_count"]} tokens. Analysis: {item["analysis_prompt_count"]} prompts / {item["analysis_token_count"]} sampled tokens.', '',
            '| KV | Arm | Paired NMSE | Cosine | Variance/native | CKA | Outside native90% subspace |',
            '|---|---|---:|---:|---:|---:|---:|']
        for kind in ['K', 'V']:
            for arm in ['stage_a', 'stage_b']:
                m = item['geometry_equal_weight_layer_head_mean'][kind][arm]
                lines.append(f'| {kind} | {LABELS[arm]} | {m["paired_nmse"]:.4f} | {m["paired_cosine"]:.4f} | {m["variance_ratio"]:.4f} | {m["linear_cka"]:.4f} | {m["outside_native_90pct_subspace_energy_fraction"]:.4f} |')
        lines += ['', '| Arm | Actual first-token output KL | First argmax agrees |', '|---|---:|---:|']
        for arm, m in item['actual_injection'].items():
            lines.append(f'| {LABELS[arm]} | {m["output_first_token_kl"]:.6f} | {m["first_argmax_match"]:.1%} |')
        lines += ['', f'Native reconstruction numerical floor KL: {item["native_reconstruction_floor"]["output_first_token_kl"]:.8f}.', '',
                  'Original official scores on selected analysis documents:', '```json',
                  json.dumps(item['original_official_scores_on_selected_analysis_documents'], indent=2), '```', '']
        for suffix in ['pca_K', 'pca_V', 'layer_metrics', 'head_heatmaps', 'reading', 'spectra']:
            lines += [f'![{group} {suffix}](figures/{group}_{suffix}.png)', '']
    lines += ['## Interpretation limits', '',
        '- PC1+PC2 explained variance is shown in every panel;2D overlap does not establish full-space alignment.',
        '- Native90% subspace excludes some native held-out energy too; compare to the native baseline in raw metrics.',
        '- CKA is insensitive to isotropic scaling and does not establish direct cache compatibility.',
        '- Positions are sampled uniformly within prompts; GSM8K includes few-shot demonstration tokens.',
        '- MCQ aggregates two task distributions; token metadata retains task and position for stratified inspection.',
        '- PCA calibration uses a disjoint subset of official evaluation documents, without model fitting or checkpoint tuning.',
        '- Receiver query trajectories can change after injection; fixed-native-Q probes are a controlled diagnostic.',
        '- No mixed-source composition or receiver adaptation is tested here.', '',
        'Raw metrics: metrics/*_layer_head.json and metrics/*_summary.json. Paired2D coordinates and token identities: metrics/*_projection_points.json.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def update_shared(summary):
    import fcntl
    path = Path(CFG['shared_results']); path.parent.mkdir(parents=True, exist_ok=True)
    begin, end = '<!-- BEGIN ' + CFG['experiment_name'] + ' -->', '<!-- END ' + CFG['experiment_name'] + ' -->'
    lines = [begin, '## ' + CFG['experiment_name'], '', '- Status: `' + summary['status'] + '`',
             '- Timestamp: ' + summary['timestamp'], '- Server: B; seed:1234',
             '- Config: `' + str(ROOT / 'config.json') + '`; manifest/checkpoint hashes: `' + str(ROOT / 'manifest.json') + '`',
             '- Official evaluation prompt subset: OBQA/ARC/GSM each8 calibration +24 analysis documents; no training.',
             '- Comparison: native, lm-eval Stage-A, lm-eval Stage-A+Stage-B; original validation-loss-only selections.',
             '- Raw summary: `' + str(ROOT / 'summary.json') + '`; report: `' + str(ROOT / 'REPORT.md') + '`',
             '- Per-sample outputs: `' + str(ROOT / 'per_sample') + '`; logs: `' + str(ROOT / 'logs/pipeline.log') + '`', '']
    if summary['status'] == 'completed':
        lines += ['| Group | KV | Arm | NMSE | Cosine | Variance ratio | CKA |', '|---|---|---|---:|---:|---:|---:|']
        for group, item in summary['groups'].items():
            for kind in ['K', 'V']:
                for arm in ['stage_a', 'stage_b']:
                    m = item['geometry_equal_weight_layer_head_mean'][kind][arm]
                    lines.append(f'| {group} | {kind} | {arm} | {m["paired_nmse"]:.6f} | {m["paired_cosine"]:.6f} | {m["variance_ratio"]:.6f} | {m["linear_cka"]:.6f} |')
        lines += ['', 'PCA is native-calibrated, per layer/head; controlled reading, output KL and reused official task scores are in the raw summary. No composition/impossibility claim.']
    else:
        lines += ['Error: ' + summary.get('error', 'unknown')]
    section = '\n'.join(lines + ['', end, ''])
    with (path.parent / '.experiment_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = path.read_text() if path.exists() else '# Experiment Results\n\n'
        if begin in text:
            start = text.index(begin); stop = text.index(end, start) + len(end)
            text = text[:start] + section + text[stop:]
        else:
            text += '\n' + section
        temp = path.with_suffix('.md.tmp'); temp.write_text(text, encoding='utf-8'); temp.replace(path)
    save('shared_summary_update.json', {'status': 'completed', 'run_status': summary['status'], 'path': str(path), 'timestamp': now()})


def run():
    import torch
    from transformers import AutoTokenizer
    if (ROOT / 'summary.json').exists() and json.loads((ROOT / 'summary.json').read_text())['status'] == 'completed':
        raise RuntimeError('Analysis already completed; do not overwrite')
    self_test()
    kb = backend(); kb.seed_all(CFG['seed'])
    assert torch.cuda.is_available()
    manifest = prepare(kb)
    model_cfg = {'models': kb.CFG['models'], 'attention_implementation': kb.CFG['attention_implementation']}
    sender, receiver = kb.load_model(model_cfg, 'llama'), kb.load_model(model_cfg, 'qwen')
    source_tok = AutoTokenizer.from_pretrained(kb.CFG['models']['llama'], local_files_only=True)
    target_tok = AutoTokenizer.from_pretrained(kb.CFG['models']['qwen'], local_files_only=True)
    groups = {}
    for group, samples in manifest['samples'].items():
        groups[group] = collect_group(group, samples, manifest, kb, sender, receiver, source_tok, target_tok)
    del sender, receiver; gc.collect(); torch.cuda.empty_cache()
    summary = {'status': 'completed', 'timestamp': now(), 'experiment_name': CFG['experiment_name'],
        'seed': CFG['seed'], 'server': CFG['server'], 'groups': groups, 'config': CFG,
        'checkpoint_selection': CFG['checkpoint_rule'], 'dataset_split': 'original official lm-eval128-document test pilots, disjoint analysis calibration and held-out subsets',
        'paths': {'manifest': str(ROOT / 'manifest.json'), 'report': str(ROOT / 'REPORT.md'),
                  'figures': str(ROOT / 'figures'), 'per_sample': str(ROOT / 'per_sample')},
        'limitations': ['Frozen geometry and reading diagnostics, not composition or receiver adaptation',
                        'Official task metrics are reused on selected documents, not newly evaluated',
                        'Uniform within-prompt token sampling includes few-shot demonstrations',
                        'Two-dimensional PCA and CKA do not prove direct KV compatibility']}
    save('summary.json', summary); report(summary); update_shared(summary)
    save('status.json', {'status': 'completed', 'phase': 'report', 'timestamp': now()})
    log('Completed analysis, plots and shared summary')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['self-test', 'run'])
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test()
    else:
        try:
            run()
        except Exception as exc:
            failure = {'status': 'failed', 'timestamp': now(), 'experiment_name': CFG['experiment_name'], 'error': str(exc)}
            save('status.json', failure); save('failure.json', failure)
            update_shared(failure)
            traceback.print_exc(); raise
