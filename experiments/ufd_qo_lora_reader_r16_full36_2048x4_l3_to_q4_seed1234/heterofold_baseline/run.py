"""Fill missing HeteroFold rows; combine six arms without changing any Writer."""
import os
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '1'
import sys
import json
import time
import fcntl
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parent
READER = ROOT.parent
CFG = json.loads((ROOT / 'config.json').read_text())
UFD = Path(CFG['ufd_root'])
spec = importlib.util.spec_from_file_location('hf_baseline_ufd', UFD / 'experiment.py')
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)
C, torch = U.C, U.torch
from heterofold.maps import load_map, describe
from heterofold.transfer import capture_sender, build_receiver_cache
HF = Path(C.CFG['hf_root'])
LLAMA = READER / 'llama_native'


def identity(): return C.sha(ROOT / 'manifest.json')
def status(**kw): C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **kw))


def shared(state, results):
    record = dict(experiment=CFG['name'], timestamp=time.strftime('%F %T'), status=state, config=CFG,
                  split_counts=CFG['counts'], checkpoint_selection=CFG['checkpoint_selection'], results=results,
                  raw_results=str(ROOT / 'evaluation/per_sample.jsonl'), log=str(ROOT / 'logs/pipeline.log'))
    path = READER.parent / 'EXPERIMENT_RESULTS.md'
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n\n## ' + CFG['name'] + '\n\n```json\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n```\n')
        stream.flush(); os.fsync(stream.fileno())
    C.save(ROOT / 'shared_summary_updated.json', dict(status=state, path=str(path)))


def baseline(row):
    path = LLAMA / 'evaluation' / row['task'] / 'per_sample' / (U.fname(row) + '.json')
    value = json.loads(path.read_text())
    assert value['llama_native_manifest_sha256'] == CFG['llama_native_manifest_sha256']
    for k in ['id', 'task', 'prompt', 'receiver_suffix', 'candidates', 'gold']:
        assert value[k] == row[k]
    return value


def old_results():
    values, files = {}, []
    for path in sorted((HF / 'evaluation_128_seed20260825/results/heterofold/arc').glob('*.json')):
        files.append(path)
        for item in json.loads(path.read_text())['rows']:
            values['arc_' + str(item['index'])] = dict(item, reused_source=str(path))
    path = HF / 'results/heterofold/gsm8k/items_00000_00032.json'; files.append(path)
    for item in json.loads(path.read_text())['rows']:
        values['gsm8k_test_' + str(item['index'])] = dict(item, reused_source=str(path))
    path = HF / 'evaluation_crossdataset_arc128_hellaswag64_seed1234/outputs/heterofold/formal/per_sample.jsonl'; files.append(path)
    for item in C.read_jsonl(path):
        values[item['id']] = dict(item['conditions']['heterofold'], reused_source=str(path),
                                  old_prompt=item['prompt'], old_candidates=item['candidates'], old_gold=item['gold'])
    return values, files


def prepare():
    assert U.identity() == CFG['ufd_manifest_sha256']
    assert C.sha(READER / 'manifest.json') == CFG['reader_manifest_sha256']
    assert C.sha(LLAMA / 'manifest.json') == CFG['llama_native_manifest_sha256']
    assert (LLAMA / 'completed.json').exists()
    assert C.sha(CFG['map_path']) == CFG['map_sha256']
    for name, expected in C.CFG['reference_hashes'].items():
        assert C.sha(HF / 'heterofold' / name) == expected
    old, files = old_results()
    previous = json.loads((Path(C.CFG['previous_root']) / 'previous_prepared.json').read_text())
    old_inputs = {r['id']: r for r in previous}
    native_gsm = json.loads((HF / 'results/native/gsm8k/items_00000_00032.json').read_text())['rows']
    native_gsm = {r['index']: r for r in native_gsm}
    rr = U.rows('test')
    assert {t: sum(r['task'] == t for r in rr) for t in CFG['counts']} == CFG['counts']
    for row in rr:
        saved = baseline(row)
        if row['task'] in ['arc_challenge', 'hellaswag']:
            prev = old_inputs[row['id']]
            for k in ['prompt', 'candidates', 'gold', 'normalize', 'cue', 'native_context_ids']:
                assert prev[k] == row[k]
        if row['task'] in CFG['reuse_existing']:
            item = old[row['id']]
            gold = item.get('old_gold', item.get('gold'))
            if gold is not None: assert str(gold) == str(row['gold'])
            if row['task'] == 'hellaswag':
                assert item['old_prompt'] == row['prompt'] and item['old_candidates'] == row['candidates']
            if row['task'] == 'gsm8k':
                native = native_gsm[int(row['id'].split('_')[-1])]
                assert C.S.normalize_number(row['gold']) == native['gold']
                for k in ['prediction', 'correct', 'generation', 'generated_tokens']:
                    assert native[k] == saved['conditions']['qwen_native'][k]
    info = dict(config=CFG, map_sha256=CFG['map_sha256'], parent_reader_manifest=CFG['reader_manifest_sha256'],
                parent_ufd_manifest=U.identity(), native_summary_sha256=C.sha(LLAMA / 'summary.json'),
                reference_hashes={str(p.relative_to(HF / 'heterofold')): C.sha(p) for p in sorted((HF / 'heterofold').rglob('*.py'))},
                reused_results={str(f): C.sha(f) for f in files},
                prepared_rows_sha256=C.digest(rr), code={p.name: C.sha(p) for p in ROOT.glob('*.py')},
                model_configs=json.loads((UFD / 'manifest.json').read_text())['model_configs'],
                versions=json.loads((UFD / 'manifest.json').read_text())['versions'])
    path = ROOT / 'manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == info
    else: C.save(path, info)
    C.save(ROOT / 'prepared/completed.json', dict(status='completed', counts=CFG['counts'],
                                               exact_existing_rows=True, original_gsm_native_outputs_identical=True,
                                               manifest_sha256=identity()))
    C.log('HETEROFOLD SIX-TASK INPUT AND FROZEN MAP AUDIT PASSED')
    return old


def save_row(row, value):
    result = dict(baseline(row), conditions=dict(baseline(row)['conditions'], heterofold=value),
                  heterofold_manifest_sha256=identity(), heterofold_map_sha256=CFG['map_sha256'])
    C.save(ROOT / 'evaluation' / row['task'] / 'per_sample' / (U.fname(row) + '.json'), result)


def finish_task(task):
    rr = [r for r in U.rows('test') if r['task'] == task]
    rows = [json.loads((ROOT / 'evaluation' / task / 'per_sample' / (U.fname(r) + '.json')).read_text()) for r in rr]
    C.write_jsonl(ROOT / 'evaluation' / task / 'per_sample.jsonl', rows)
    C.save(ROOT / 'evaluation' / task / 'completed.json', dict(status='completed', n=len(rows), manifest_sha256=identity()))


def reuse(old):
    for task in CFG['reuse_existing']:
        for row in U.rows('test'):
            if row['task'] != task: continue
            value = {k: v for k, v in old[row['id']].items() if k not in ['old_prompt', 'old_candidates', 'old_gold']}
            save_row(row, dict(value, reused=True))
        finish_task(task)
        C.log(f'HETEROFOLD reused completed {task}: {CFG["counts"][task]} samples')


@torch.no_grad()
def score(row, rt, sender_tok, sender, mapping):
    # Exact original HeteroFold: raw text -> pre-norm folded cache -> native chat suffix.
    offsets, keys, values = capture_sender(sender_tok, sender, mapping, row['prompt'], device='cpu')
    cache, length = build_receiver_cache(rt.tokenizer, rt.model, mapping, offsets, keys, values, row['prompt'], rt.prefix)
    scores, suffix_tokens = C.R.transfer_candidate_scores(rt, cache, length, row['candidates'], row['normalize'], rt.assistant + row['cue'])
    value = C.record(scores, row['gold'])
    value.update(transfer_cache_tokens=length, receiver_suffix_tokens=suffix_tokens, reused=False,
                 sender_input='original raw task prompt; no chat template', protocol='original frozen HeteroFold')
    return value


@torch.no_grad()
def smoke(rt, sender_tok, sender, mapping, old):
    if (ROOT / 'smoke/completed.json').exists(): return
    checks = []
    rr = U.rows('test')
    for task in ['arc_challenge', 'hellaswag']:
        for row in [r for r in rr if r['task'] == task][:2]:
            fresh = score(row, rt, sender_tok, sender, mapping)
            previous = old[row['id']]
            error = max(abs(a-b) for a, b in zip(fresh['scores'], previous['scores']))
            assert error < 0.02 and fresh['prediction'] == previous['prediction'], ('Known HeteroFold replay mismatch', row['id'], error)
            checks.append(dict(id=row['id'], max_score_difference=error, prediction_identical=True))
    for task in CFG['newly_evaluate']:
        row = next(r for r in rr if r['task'] == task)
        assert list(rt.native_ids(row['prompt'], rt.assistant + row['cue'])[0].cpu().tolist()) == row['native_context_ids']
        value = score(row, rt, sender_tok, sender, mapping)
        assert len(value['scores']) == len(row['candidates'])
        checks.append(dict(id=row['id'], finite_candidate_scores=True, actual_option_count=len(row['candidates']), same_receiver_native_context=True))
    C.save(ROOT / 'smoke/completed.json', dict(status='completed', checks=checks, manifest_sha256=identity()))
    C.log('ORIGINAL HETEROFOLD REPLAY AND THREE NEW TASK SMOKE PASSED')


@torch.no_grad()
def evaluate(task, rt, sender_tok, sender, mapping):
    folder = ROOT / 'evaluation' / task
    if (folder / 'completed.json').exists(): return
    rr = [r for r in U.rows('test') if r['task'] == task]
    for i, row in enumerate(rr, 1):
        path = folder / 'per_sample' / (U.fname(row) + '.json')
        if path.exists():
            assert json.loads(path.read_text())['heterofold_manifest_sha256'] == identity()
            continue
        status(status='running', phase='heterofold_evaluation', task=task, sample=i, total=len(rr))
        value = score(row, rt, sender_tok, sender, mapping)
        save_row(row, value)
        if i % 8 == 0 or i == len(rr): C.log(f'HETEROFOLD {task} {i}/{len(rr)}')
    finish_task(task)


def summarize():
    all_rows, tasks = [], {}
    for task, expected in CFG['counts'].items():
        rr = C.read_jsonl(ROOT / f'evaluation/{task}/per_sample.jsonl')
        assert len(rr) == expected
        all_rows.extend(rr); metrics = {}
        for arm in rr[0]['conditions']:
            vv = [r['conditions'][arm] for r in rr]
            metrics[arm] = dict(n=len(vv), correct=sum(v['correct'] for v in vv), accuracy=sum(v['correct'] for v in vv) / len(vv))
            if task == 'gsm8k': metrics[arm].update(mean_generated_tokens=sum(v['generated_tokens'] for v in vv) / len(vv), cap_count=sum(v.get('reached_cap', v['generated_tokens'] >= 4096) for v in vv))
        pairs, names = {}, list(metrics)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                aa, bb = [r['conditions'][a]['correct'] for r in rr], [r['conditions'][b]['correct'] for r in rr]
                pairs[a + '__vs__' + b] = dict(both_correct=sum(x and y for x, y in zip(aa, bb)), only_a=sum(x and not y for x, y in zip(aa, bb)),
                                             only_b=sum(not x and y for x, y in zip(aa, bb)), both_wrong=sum(not x and not y for x, y in zip(aa, bb)))
        tasks[task] = dict(metrics=metrics, paired=pairs)
    result = dict(experiment=CFG['name'], status='completed', config=CFG, tasks=tasks, manifest_sha256=identity(),
                  cuda_allocator_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  caveats=['Same evaluation rows and Receiver scoring, but HeteroFold RAW Sender/cache interface and calibration differ from our methods: NOT same-budget architecture ablation.',
                           'MMLU-Pro remains approved custom heldout and A-J label extension, not official test/CoT.',
                           'Only OBQA/MMLU-Pro/SciQ HeteroFold outputs are new; ARC128/Hella64/GSM32 reused after input/identity verification.'])
    C.write_jsonl(ROOT / 'evaluation/per_sample.jsonl', all_rows)
    C.save(ROOT / 'summary.json', result)
    C.save(READER / 'summary_with_llama_native_and_heterofold.json', result)
    shared('completed', result)
    C.save(ROOT / 'completed.json', dict(status='completed', manifest_sha256=identity()))
    status(status='completed', phase='all'); C.log('ALL SIX-TASK HETEROFOLD BASELINES COMPLETED')


if __name__ == '__main__':
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (ROOT / 'completed.json').exists(): C.log('Already completed')
            else:
                C.seed(); old = prepare(); reuse(old)
                assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]/2**30 >= 20, 'Need20GiB free for both frozen models; do not preempt'
                sender_tok, sender = C.model('llama')
                tok, model = C.model('qwen'); rt = C.runtime(model, tok)
                mapping = load_map(CFG['map_path'], device='cuda', dtype=torch.float16)
                C.save(ROOT / 'map_description.json', describe(mapping))
                smoke(rt, sender_tok, sender, mapping, old)
                for task in CFG['newly_evaluate']: evaluate(task, rt, sender_tok, sender, mapping)
                summarize()
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            C.save(ROOT / 'failure.json', dict(status='failed_or_incomplete', **error)); status(status='failed', **error)
            shared('failed_or_incomplete', error); raise
