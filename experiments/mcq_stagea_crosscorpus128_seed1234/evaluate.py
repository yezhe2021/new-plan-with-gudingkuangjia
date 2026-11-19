"""Frozen MCQ/GSM8K Stage-A comparison on a fixed official-test prefix."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234'
sys.path.insert(0, str(LEGACY))
import torch
from lm_eval import evaluator
from kv_backend import CFG, FullSyncHFLM, load_base, load_model, save_json, seed_all
from tasks_native import tasks, stable_id, jsonl

NAMES = ['openbookqa', 'arc_challenge']
PER_TASK = 32
SAMPLES = {name: list(range(PER_TASK)) for name in NAMES}


def log(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class ProgressLM(FullSyncHFLM):
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        values = []
        for begin in range(0, len(requests), 32):
            values.extend(super()._loglikelihood_tokens(requests[begin:begin+32], True, override_bs))
            log(f'Candidate likelihood {min(begin+32, len(requests))}/{len(requests)}')
        return values


def main():
    if (ROOT / 'results/completed.json').exists():
        log('Already completed; preserving results')
        return
    seed_all(CFG['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    objects = tasks(NAMES)
    blocked = set()
    for group in ('mcq', 'gsm8k'):
        for split in ('train', 'validation'):
            blocked.update(r['id'] for r in jsonl(LEGACY / f'runs/{group}/{split}.jsonl'))
    manifest = []
    for name in NAMES:
        task = objects[name]
        split = task.get_config('test_split') or task.get_config('validation_split')
        for index in SAMPLES[name]:
            doc = task.dataset[split][index]
            key = stable_id(name, doc)
            if key in blocked:
                raise RuntimeError(f'Train/validation overlap: {key}')
            manifest.append({'task': name, 'doc_id': index, 'split': split, 'stable_id': key})
    checkpoints = {g: LEGACY / f'runs/{g}/stage_a/best.pt' for g in ('mcq', 'gsm8k')}
    save_json(ROOT / 'config.json', {
        'server': 'B', 'seed': CFG['seed'], 'protocol': CFG['protocol'],
        'lm_eval_version': CFG['lm_eval_version'], 'apply_chat_template': False,
        'official_task_prompt_and_scoring': True, 'stage_b': False, 'training': False,
        'samples': SAMPLES, 'total_questions': PER_TASK * len(NAMES), 'models': CFG['models'],
        'checkpoint_paths': {g: str(p) for g, p in checkpoints.items()},
        'checkpoint_sha256': {g: digest(p) for g, p in checkpoints.items()},
        'selection': f'first{PER_TASK} official test documents per task; diagnostic subset, not full benchmark',
    })
    save_json(ROOT / 'manifest.json', {'samples': manifest, 'training_validation_overlap': 0})
    cfg = {'models': CFG['models'], 'attention_implementation': CFG['attention_implementation']}
    log('Loading frozen Llama Sender and Qwen Receiver')
    sender, receiver = load_model(cfg, 'llama'), load_model(cfg, 'qwen')
    summaries, outcomes = {}, {}
    for group, checkpoint in checkpoints.items():
        path = ROOT / f'results/{group}_stage_a.json'
        if not path.exists():
            log(f'START {group} Stage-A only')
            base = load_base(checkpoint)
            lm = ProgressLM(receiver, 'translated', sender, base, None)
            result = evaluator.simple_evaluate(
                model=lm, tasks=NAMES, samples=SAMPLES, num_fewshot=None,
                bootstrap_iters=1000, log_samples=True, apply_chat_template=False,
                random_seed=CFG['seed'], numpy_random_seed=CFG['seed'],
                torch_random_seed=CFG['seed'], fewshot_random_seed=CFG['seed'])
            payload = json.loads(json.dumps(result, ensure_ascii=False,
                default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
            save_json(path, payload)
            del lm, base, result
            gc.collect(); torch.cuda.empty_cache()
        payload = json.loads(path.read_text())
        summaries[group], outcomes[group] = {}, {}
        for name in NAMES:
            rows = payload['samples'][name]
            if len(rows) != PER_TASK:
                raise RuntimeError(f'Wrong sample count: {group}/{name}')
            with (ROOT / f'results/{group}__{name}__samples.jsonl').open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            metrics = []
            for row in rows:
                key = stable_id(name, row['doc'])
                score = objects[name].process_results(row['doc'], row['filtered_resps'])
                outcomes[group][key] = {k: bool(score[k]) for k in ('acc', 'acc_norm')}
                metrics.append(score)
            summaries[group][name] = {metric: {'correct': sum(bool(r[metric]) for r in metrics),
                'count': len(metrics), 'accuracy': sum(bool(r[metric]) for r in metrics)/len(metrics)}
                for metric in ('acc', 'acc_norm')}
        log(f'FINISHED {group}: {summaries[group]}')
    total = PER_TASK * len(NAMES)
    if set(outcomes['mcq']) != set(outcomes['gsm8k']) or len(outcomes['mcq']) != total:
        raise RuntimeError(f'Conditions did not evaluate exactly the same {total} questions')
    overlap = {}
    for metric in ('acc', 'acc_norm'):
        a, b = outcomes['mcq'], outcomes['gsm8k']
        overlap[metric] = {
            'both_correct': sum(a[k][metric] and b[k][metric] for k in a),
            'mcq_only_correct': sum(a[k][metric] and not b[k][metric] for k in a),
            'gsm8k_only_correct': sum(not a[k][metric] and b[k][metric] for k in a),
            'both_wrong': sum(not a[k][metric] and not b[k][metric] for k in a)}
        for group in summaries:
            n = sum(summaries[group][name][metric]['correct'] for name in NAMES)
            summaries[group].setdefault(f'combined{total}', {})[metric] = {'correct': n, 'count': total, 'accuracy': n/total}
    save_json(ROOT / 'results/summary.json', {'conditions': summaries, 'paired_overlap': overlap})
    save_json(ROOT / 'results/completed.json', {'completed': True, 'questions_per_condition': total})
    log(f'ALL MCQ STAGE-A CROSS-CORPUS{total} EVALUATIONS COMPLETED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-task', type=int, default=32)
    args = parser.parse_args()
    if args.per_task < 1:
        parser.error('--per-task must be positive')
    PER_TASK = args.per_task
    SAMPLES = {name: list(range(PER_TASK)) for name in NAMES}
    main()
