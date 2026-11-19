"""Official GSM8K5-shot: frozen MCQ vs GSM8K Stage-A on the same32 test docs."""
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
from kv_backend import CFG, FullSyncHFLM, load_model, load_base, save_json, seed_all
from tasks_native import tasks, jsonl, stable_id


def log(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class ProgressLM(FullSyncHFLM):
    def generate_until(self, requests, disable_tqdm=False):
        output = []
        for index, request in enumerate(requests, 1):
            log(f'GENERATE {index}/{len(requests)} START')
            text = super().generate_until([request], disable_tqdm=True)[0]
            output.append(text)
            with self.partial_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'request_index': index, 'context': request.args[0],
                    'generation': text}, ensure_ascii=False)+'\n')
            log(f'GENERATE {index}/{len(requests)} DONE')
        return output


def main():
    (ROOT/'results').mkdir(parents=True, exist_ok=True)
    if (ROOT/'results/completed.json').exists():
        log('Already completed; preserving results'); return
    seed_all(CFG['seed'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    task = tasks(['gsm8k'])['gsm8k']
    blocked = set()
    for group in ('mcq', 'gsm8k'):
        for split in ('train', 'validation'):
            blocked.update(r['id'] for r in jsonl(LEGACY/f'runs/{group}/{split}.jsonl'))
    manifest = []
    for index in range(32):
        doc = task.dataset['test'][index]
        key = stable_id('gsm8k', doc)
        if key in blocked: raise RuntimeError(f'Training/validation overlap: {key}')
        manifest.append({'doc_id': index, 'stable_id': key})
    checkpoints = {g: LEGACY/f'runs/{g}/stage_a/best.pt' for g in ('mcq', 'gsm8k')}
    save_json(ROOT/'config.json', {
        'server': 'B', 'protocol': CFG['protocol'], 'lm_eval_version': CFG['lm_eval_version'],
        'seed': CFG['seed'], 'models': CFG['models'], 'samples': {'gsm8k': list(range(32))},
        'apply_chat_template': False, 'stage_b': False, 'training': False,
        'num_fewshot': task.get_config('num_fewshot'),
        'generation_kwargs': task.get_config('generation_kwargs'),
        'checkpoint_paths': {g: str(p) for g,p in checkpoints.items()},
        'checkpoint_sha256': {g: digest(p) for g,p in checkpoints.items()},
        'subset': 'first32 official test docs; not full benchmark'})
    save_json(ROOT/'manifest.json', {'samples': manifest, 'training_validation_overlap': 0})
    cfg = {'models': CFG['models'], 'attention_implementation': CFG['attention_implementation']}
    log('Loading frozen Llama Sender and Qwen Receiver')
    sender, receiver = load_model(cfg, 'llama'), load_model(cfg, 'qwen')
    summaries, outcomes = {}, {}
    for group, path in checkpoints.items():
        result_path = ROOT/f'results/{group}_stage_a.json'
        if not result_path.exists():
            log(f'START {group} Stage-A only')
            base = load_base(path)
            lm = ProgressLM(receiver, 'translated', sender, base, None)
            lm.partial_path = ROOT/f'results/{group}_generation_progress.jsonl'
            log(f'Harness default max_gen_toks={lm.max_gen_toks}')
            result = evaluator.simple_evaluate(model=lm, tasks=['gsm8k'],
                samples={'gsm8k': list(range(32))}, num_fewshot=None, log_samples=True,
                apply_chat_template=False, bootstrap_iters=1000,
                random_seed=CFG['seed'], numpy_random_seed=CFG['seed'],
                torch_random_seed=CFG['seed'], fewshot_random_seed=CFG['seed'])
            payload = json.loads(json.dumps(result, ensure_ascii=False,
                default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
            save_json(result_path, payload)
            del lm, base, result
            gc.collect(); torch.cuda.empty_cache()
        payload = json.loads(result_path.read_text())
        rows = payload['samples']['gsm8k']
        with (ROOT/f'results/{group}__gsm8k__samples.jsonl').open('w', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
        outcomes[group] = {}
        for row in rows:
            filt = row['filter']
            key = stable_id('gsm8k', row['doc'])
            score = task.process_results(row['doc'], row['filtered_resps'])['exact_match']
            outcomes[group].setdefault(filt, {})[key] = bool(score)
        summaries[group] = {}
        for filt, values in outcomes[group].items():
            if len(values) != 32: raise RuntimeError(f'Wrong sample count {group}/{filt}')
            summaries[group][filt] = {'correct': sum(values.values()), 'count': 32,
                'accuracy': sum(values.values())/32}
        log(f'FINISHED {group}: {summaries[group]}')
    overlap = {}
    if set(outcomes['mcq']) != set(outcomes['gsm8k']): raise RuntimeError('Filter mismatch')
    for filt in outcomes['mcq']:
        a,b = outcomes['mcq'][filt], outcomes['gsm8k'][filt]
        if set(a) != set(b): raise RuntimeError('Question mismatch')
        overlap[filt] = {'both_correct': sum(a[k] and b[k] for k in a),
            'mcq_only_correct': sum(a[k] and not b[k] for k in a),
            'gsm8k_only_correct': sum(not a[k] and b[k] for k in a),
            'both_wrong': sum(not a[k] and not b[k] for k in a)}
    save_json(ROOT/'results/summary.json', {'conditions': summaries, 'paired_overlap': overlap})
    save_json(ROOT/'results/completed.json', {'completed': True, 'questions_per_condition': 32})
    log('ALL GSM8K STAGE-A CROSS-CORPUS32 EVALUATIONS COMPLETED')


if __name__ == '__main__': main()
