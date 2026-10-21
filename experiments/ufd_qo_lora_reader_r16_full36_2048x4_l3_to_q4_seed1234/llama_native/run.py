"""Complete the existing six-task comparison with the frozen Sender native arm."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
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
spec = importlib.util.spec_from_file_location('baseline_ufd', UFD / 'experiment.py')
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)
C, torch = U.C, U.torch


def identity():
    return C.sha(ROOT / 'manifest.json')


def status(**kwargs):
    C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **kwargs))


def shared(state, results):
    record = dict(experiment=CFG['name'], timestamp=time.strftime('%F %T'), status=state,
                  config=CFG, split_counts=CFG['counts'], checkpoint_selection=CFG['checkpoint_selection'],
                  results=results, raw_results=str(ROOT / 'evaluation/per_sample.jsonl'), log=str(ROOT / 'logs/pipeline.log'))
    path = READER.parent / 'EXPERIMENT_RESULTS.md'
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n\n## ' + CFG['name'] + '\n\n```json\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n```\n')
        stream.flush()
        os.fsync(stream.fileno())
    C.save(ROOT / 'shared_summary_updated.json', dict(status=state, path=str(path)))


def context(tok, row):
    frame = row['frames']['llama']
    rendered = frame['rendered_full']
    assert list(tok.encode(rendered, add_special_tokens=False)) == frame['full_ids']
    prefix = frame['rendered_prefix']
    assert rendered.startswith(prefix + row['prompt'])
    assistant = rendered[len(prefix) + len(row['prompt']):]
    assert prefix + row['prompt'] + assistant == rendered
    full = rendered + row['cue']
    ids = list(tok.encode(full, add_special_tokens=False))
    assert full.count(row['prompt']) == 1
    return dict(rendered=full, ids=ids, prefix=prefix, suffix=assistant + row['cue'])


def runtime(tok, model, ctx):
    rt = C.R.Runtime.__new__(C.R.Runtime)
    rt.model, rt.tokenizer, rt.mode = model, tok, 'native'
    rt.prefix, rt.assistant = ctx['prefix'], ctx['suffix']
    return rt


def prepare(tok):
    assert U.identity() == CFG['ufd_manifest_sha256']
    assert C.sha(READER / 'manifest.json') == CFG['reader_manifest_sha256']
    assert (READER / 'completed.json').exists() and (UFD / 'completed.json').exists()
    pm = json.loads((UFD / 'manifest.json').read_text())
    assert C.digest(U.rows('test')) == pm['prepared_sha256']['test']
    for file, expected in C.CFG['reference_hashes'].items():
        assert C.sha(Path(C.CFG['hf_root']) / 'heterofold' / file) == expected
    rr = U.rows('test')
    assert {task: sum(r['task'] == task for r in rr) for task in CFG['counts']} == CFG['counts']
    for row in rr:
        context(tok, row)
        path = READER / 'evaluation' / row['task'] / 'per_sample' / (U.fname(row) + '.json')
        baseline = json.loads(path.read_text())
        assert baseline['reader_manifest_sha256'] == CFG['reader_manifest_sha256']
        for field in ['id', 'task', 'prompt', 'receiver_suffix', 'candidates', 'gold']:
            assert baseline[field] == row[field]
    metadata = dict(config=CFG, model_path=C.CFG['models']['llama'], model_hashes=pm['model_configs']['llama'],
                    original_template_date=C.CFG['llama_template_date'], reference_hashes=C.CFG['reference_hashes'],
                    versions=pm['versions'], test_rows_sha256=pm['prepared_sha256']['test'],
                    reader_summary_sha256=C.sha(READER / 'summary.json'), ufd_summary_sha256=C.sha(UFD / 'summary.json'),
                    code={p.name: C.sha(p) for p in ROOT.glob('*.py')}, ids=[r['id'] for r in rr])
    path = ROOT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == metadata, 'Native baseline definition changed'
    else:
        C.save(path, metadata)
    C.save(ROOT / 'prepared/completed.json', dict(status='completed', n=len(rr), counts=CFG['counts'], manifest_sha256=identity()))
    C.log('LLAMA NATIVE DATA / CHAT / BASELINE AUDIT PASSED')


@torch.no_grad()
def smoke(tok, model):
    if (ROOT / 'smoke/completed.json').exists():
        return
    seen, checks = set(), []
    for row in U.rows('test'):
        if row['task'] in seen:
            continue
        seen.add(row['task'])
        ctx = context(tok, row)
        rt = runtime(tok, model, ctx)
        ids = torch.tensor([ctx['ids']], device='cuda')
        assert torch.equal(ids, rt.native_ids(row['prompt'], ctx['suffix']))
        if row['task'] == 'gsm8k':
            expected = model(ids, use_cache=False).logits[:, -1].float()
            actual = model.lm_head(model.model(ids, use_cache=False).last_hidden_state[:, -1]).float()
            kl = float(U.full_kl(expected, actual))
            assert kl < 0.001 and torch.equal(expected.argmax(-1), actual.argmax(-1))
            checks.append(dict(task=row['task'], native_first_logit_kl=kl, top1_identical=True))
        else:
            official, _ = C.R.native_candidate_scores(rt, row['prompt'], row['candidates'], row['normalize'], ctx['suffix'])
            direct = []
            for candidate in row['candidates']:
                target = C.R.candidate_ids(tok, candidate)
                full = torch.tensor([ctx['ids'] + target], device='cuda')
                hidden = model.model(full, use_cache=False).last_hidden_state[:, len(ctx['ids'])-1:len(ctx['ids'])-1+len(target)]
                logits = model.lm_head(hidden).float()
                direct.append(C.R._candidate_log_likelihood(logits, torch.tensor([target], device='cuda'), row['normalize']))
            error = max(abs(a-b) for a, b in zip(official, direct))
            assert error < 0.02 and C.record(official, row['gold'])['prediction'] == C.record(direct, row['gold'])['prediction']
            checks.append(dict(task=row['task'], max_candidate_score_difference=error, prediction_identical=True))
    assert len(checks) == 6
    C.save(ROOT / 'smoke/completed.json', dict(status='completed', manifest_sha256=identity(), checks=checks))
    C.log('LLAMA NATIVE SIX-TASK SMOKE PASSED')


@torch.no_grad()
def evaluate(task, tok, model):
    folder = ROOT / 'evaluation' / task
    if (folder / 'completed.json').exists():
        return
    rows = [r for r in U.rows('test') if r['task'] == task]
    for i, row in enumerate(rows, 1):
        path = folder / 'per_sample' / (U.fname(row) + '.json')
        if path.exists():
            assert json.loads(path.read_text())['llama_native_manifest_sha256'] == identity()
            continue
        status(status='running', phase='evaluation', task=task, sample=i, total=len(rows))
        ctx = context(tok, row)
        rt = runtime(tok, model, ctx)
        if task == 'gsm8k':
            text, count = C.R.greedy_decode(rt, torch.tensor([ctx['ids']], device='cuda'), U.CFG['gsm8k_formal_max_new_tokens'])
            prediction = C.S.extract_numeric_answer(text)
            value = dict(prediction=prediction, correct=int(prediction == C.S.normalize_number(row['gold'])),
                         generation=text, generated_tokens=count, reached_cap=int(count >= U.CFG['gsm8k_formal_max_new_tokens']))
        else:
            scores, count = C.R.native_candidate_scores(rt, row['prompt'], row['candidates'], row['normalize'], ctx['suffix'])
            value = C.record(scores, row['gold'])
            assert count == len(ctx['ids'])
        baseline_path = READER / 'evaluation' / task / 'per_sample' / (U.fname(row) + '.json')
        baseline = json.loads(baseline_path.read_text())
        result = dict(baseline, conditions=dict(baseline['conditions'], llama_native=value),
                      llama_native_manifest_sha256=identity(), llama_rendered_prompt=ctx['rendered'], llama_context_ids=ctx['ids'])
        C.save(path, result)
        if i % 8 == 0 or task == 'gsm8k':
            C.log(f'LLAMA NATIVE {task} {i}/{len(rows)}')
    results = [json.loads((folder / 'per_sample' / (U.fname(r) + '.json')).read_text()) for r in rows]
    C.write_jsonl(folder / 'per_sample.jsonl', results)
    C.save(folder / 'completed.json', dict(status='completed', n=len(results), manifest_sha256=identity()))


def summarize():
    records, tasks = [], {}
    for task, expected in CFG['counts'].items():
        assert (ROOT / f'evaluation/{task}/completed.json').exists()
        rr = C.read_jsonl(ROOT / f'evaluation/{task}/per_sample.jsonl')
        assert len(rr) == expected
        records.extend(rr)
        metrics = {}
        for arm in rr[0]['conditions']:
            vv = [r['conditions'][arm] for r in rr]
            metrics[arm] = dict(n=len(vv), correct=sum(v['correct'] for v in vv), accuracy=sum(v['correct'] for v in vv) / len(vv))
            if task == 'gsm8k':
                metrics[arm].update(mean_generated_tokens=sum(v['generated_tokens'] for v in vv) / len(vv), cap_count=sum(v['reached_cap'] for v in vv))
        pairs = {}
        names = list(metrics)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                aa, bb = [r['conditions'][a]['correct'] for r in rr], [r['conditions'][b]['correct'] for r in rr]
                pairs[a + '__vs__' + b] = dict(both_correct=sum(x and y for x, y in zip(aa, bb)),
                    only_a=sum(x and not y for x, y in zip(aa, bb)), only_b=sum(not x and y for x, y in zip(aa, bb)),
                    both_wrong=sum(not x and not y for x, y in zip(aa, bb)))
        tasks[task] = dict(metrics=metrics, paired=pairs)
    result = dict(experiment=CFG['name'], status='completed', config=CFG, tasks=tasks,
                  manifest_sha256=identity(), raw_results=str(ROOT / 'evaluation/per_sample.jsonl'),
                  caveats=['MMLU-Pro uses same approved custom heldout split and label protocol.',
                           'Native Llama uses its OWN recorded Sender chat template, not Qwen template.',
                           'Other four arms reused unchanged on the exact same rows; only llama_native newly computed.'])
    C.write_jsonl(ROOT / 'evaluation/per_sample.jsonl', records)
    C.save(ROOT / 'summary.json', result)
    C.save(READER / 'summary_with_llama_native.json', result)
    shared('completed', result)
    C.save(ROOT / 'completed.json', dict(status='completed', manifest_sha256=identity()))
    status(status='completed', phase='all')
    C.log('ALL LLAMA NATIVE BASELINES AND FIVE-ARM COMPARISONS COMPLETED')


if __name__ == '__main__':
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (ROOT / 'completed.json').exists():
                C.log('Already completed')
            else:
                C.seed()
                tok = C.AutoTokenizer.from_pretrained(C.CFG['models']['llama'], local_files_only=True)
                prepare(tok)
                tok, model = C.model('llama')
                smoke(tok, model)
                for task in ['arc_challenge', 'openbookqa', 'hellaswag', 'mmlu_pro', 'sciq', 'gsm8k']:
                    evaluate(task, tok, model)
                summarize()
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            C.save(ROOT / 'failure.json', dict(status='failed_or_incomplete', **error))
            status(status='failed', **error)
            shared('failed_or_incomplete', error)
            raise
