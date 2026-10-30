"""Frozen A/C/D: paired native K/V substitution and attention masking, 64/task."""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
PARENT = Path(CFG['parent'])
OLD = PARENT / CFG['mask_experiment']
sys.path.insert(0, str(PARENT))
import core as C
import train as T
import evaluate as E
spec = importlib.util.spec_from_file_location('previous_mask', OLD / 'experiment.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
torch = C.torch
METHODS = CFG['methods']


def identity(): return C.sha(ROOT / 'manifest.json')
def mark(path, **v): C.save(ROOT / path, dict(manifest_sha256=identity(), **v))
def status(**v): C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **v))
def rows(task=None):
    return [r for r in C.read_jsonl(ROOT / 'prepared/test.jsonl') if task is None or r['task'] == task]


def conditions():
    return [dict(name='unmasked', mode='none', layers=[])] + [
        dict(name=f'g{g+1}_{mode}', mode=mode, layers=list(range(g*6, (g+1)*6)))
        for g in range(6) for mode in ['k', 'v', 'kv', 'mask']]


def prepare():
    if (ROOT / 'manifest.json').exists(): verify(); return
    C.verify(); M.verify()
    assert C.identity() == CFG['parent_manifest_sha256']
    assert M.identity() == CFG['mask_manifest_sha256']
    assert C.read(OLD / 'smoke/completed.json')['status'] == 'completed'
    allrows = C.rows('test')
    selected = [r for task, n in CFG['test_counts'].items()
                for r in [v for v in allrows if v['task'] == task][:n]]
    assert len(selected) == 192
    assert all(sum(r['task'] == t for r in selected) == 64 for t in CFG['test_counts'])
    refs = [PARENT / f for f in ['manifest.json', 'config.json', 'completed.json',
            'core.py', 'evaluate.py', 'train.py', 'prepared/test.jsonl']]
    refs += [PARENT / 'runs' / a / 'epoch4_step800.pt' for a in ['C', 'D']]
    refs += [OLD / f for f in ['manifest.json', 'config.json', 'experiment.py', 'smoke/completed.json']]
    reuse = []
    for row in selected:
        refs.append(PARENT / 'evaluation' / row['task'] / 'per_sample' / (row['id'] + '.json'))
        assert row == next(r for r in M.rows(row['task']) if r['id'] == row['id'])
        for method in METHODS:
            for con in conditions():
                if con['mode'] not in ['none', 'mask']: continue
                oldname = 'unmasked' if con['mode'] == 'none' else con['name'].split('_')[0]
                source = OLD / 'evaluation' / row['task'] / 'per_config' / method / oldname / (row['id'] + '.json')
                item = C.read(source)
                assert item['manifest_sha256'] == M.identity() and item['id'] == row['id']
                assert item['method'] == method
                assert item['condition']['layers'] == con['layers']
                refs.append(source)
                reuse.append(dict(task=row['task'], id=row['id'], method=method, condition=con, source=str(source)))
    C.write_jsonl(ROOT / 'prepared/test.jsonl', selected)
    refs.append(ROOT / 'prepared/test.jsonl')
    C.save(ROOT / 'manifest.json', dict(config=CFG, config_sha256=C.sha(ROOT / 'config.json'),
        code={p.name: C.sha(p) for p in ROOT.glob('*.py')}, sources={str(p): C.sha(p) for p in refs},
        conditions=conditions(), test_ids={t: [r['id'] for r in selected if r['task'] == t] for t in CFG['test_counts']},
        counts=CFG['test_counts'], training=False, gsm8k=False, sample_configurations=14592,
        selection='First64 in immutable parent task order, no outcome-dependent selection.',
        intervention='Replace only original transferred body positions, after receiver k_norm/RoPE/cache assembly; preserve public header and positions. Each group independent. Native KV is offline oracle information.'))
    for item in reuse:
        value=C.read(item['source'])['result']
        mark(f'evaluation/{item["task"]}/per_config/{item["method"]}/{item["condition"]["name"]}/{item["id"]}.json',
             id=item['id'], task=item['task'], method=item['method'], condition=item['condition'], result=value,
             reused_from=item['source'], source_manifest_sha256=M.identity())
    mark('prepared/completed.json', status='completed', n=192, reused_configurations=len(reuse),
         conditions_per_method=25, native_baseline_per_question=1, sample_configurations=14592)
    C.log('ORACLE64 PREPARED', CFG['test_counts'], 'reused', len(reuse))


def verify():
    C.verify(); m=C.read(ROOT / 'manifest.json')
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    for name, digest in m['code'].items(): assert C.sha(ROOT / name) == digest, name
    for path, digest in m['sources'].items(): assert C.sha(path) == digest, path


@torch.no_grad()
def replace(engine, row, original, native, layers, mode):
    """Mix finalized cache tensors on exactly the same Qwen token/position grid."""
    assert mode in ['none', 'mask', 'k', 'v', 'kv']
    end=len(engine.prefix_ids) + len(row['content_ids']); start=len(engine.prefix_ids)
    assert original.get_seq_length() == native.get_seq_length() == end
    mixed=C.clone_cache(original, engine.model)
    for layer in layers if mode in ['k', 'v', 'kv'] else []:
        target=C._cache_layer(mixed, layer); source=C._cache_layer(native, layer)
        for index in ([0] if mode=='k' else [1] if mode=='v' else [0, 1]):
            assert target[index].shape == source[index].shape
            target[index][:, :, start:end, :].copy_(source[index][:, :, start:end, :])
    return mixed


def models(): return M.models()


def native_parity(row, value):
    old=M.historical(row)['conditions']['native']
    match=value['prediction'] == old['prediction']
    return dict(historical_prediction_matches=match,
        historical_generation_matches=value.get('generation') == old.get('generation'),
        max_score_difference=max([abs(a-b) for a,b in zip(value.get('scores', []), old.get('scores', []))] or [0.]))


@torch.no_grad()
def smoke():
    verify(); engine, modules, mask=models(); checks=[]
    chosen=[rows(t)[0] for t in CFG['test_counts']]
    chosen.append(max(rows('hotpotqa'), key=lambda r: len(r['content_ids'])))
    for row in {r['id']: r for r in chosen}.values():
        base=engine.base(row); native=engine.native_cache(row, baseline=True)
        native_value=M.score(engine, mask, row, C.clone_cache(native, engine.model), [])
        # Native cached path is checked against the original direct-prefill scorer.
        direct=E.result(engine, row, None, None, native=True)
        assert native_value['prediction'] == direct['prediction'], ('Native cached/direct prediction', row['id'])
        if row['task']=='hotpotqa': assert native_value['generation'] == direct['generation'], row['id']
        else:
            p=torch.tensor(native_value['scores'], dtype=torch.float64).log_softmax(-1)
            q=torch.tensor(direct['scores'], dtype=torch.float64).log_softmax(-1)
            assert abs(float((p.exp()*(p-q)).sum())) < .001
        start=len(engine.prefix_ids); end=native.get_seq_length()
        for method, module in modules.items():
            original=engine.cache(row, base, module)
            value=M.score(engine, mask, row, C.clone_cache(original, engine.model), [])
            M.parity(row, method, value)
            original_logits=M.first_logits(engine, mask, row, original, [])
            same=replace(engine, row, original, original, list(range(36)), 'kv')
            same_logits=M.first_logits(engine, mask, row, same, [])
            assert torch.equal(original_logits, same_logits), 'Self-substitution changed logits'
            del same, same_logits, original_logits
            # Check every group/type, all36 physical cache tensors, protected header,
            # untouched counterpart (K-only/V-only), and no original-cache mutation.
            for con in conditions()[1:]:
                mixed=replace(engine, row, original, native, con['layers'], con['mode'])
                for l in range(36):
                    for index, tensor in enumerate(C._cache_layer(mixed, l)):
                        orig=C._cache_layer(original, l)[index]
                        assert torch.equal(tensor[:, :, :start, :], orig[:, :, :start, :])
                        selected=l in con['layers'] and (con['mode']=='kv' or con['mode']==('k' if index==0 else 'v'))
                        expected=C._cache_layer(native, l)[index] if selected else orig
                        assert torch.equal(tensor[:, :, start:end, :], expected[:, :, start:end, :])
                logits=M.first_logits(engine, mask, row, mixed, con['layers'] if con['mode']=='mask' else [])
                assert torch.isfinite(logits).all()
                del mixed, logits
            all_native=replace(engine, row, original, native, list(range(36)), 'kv')
            endpoint=M.score(engine, mask, row, all_native, [])
            assert endpoint['prediction'] == native_value['prediction'], ('All-native endpoint', row['id'], method)
            if row['task']=='hotpotqa': assert endpoint['generation'] == native_value['generation']
            else: assert max(abs(a-b) for a,b in zip(endpoint['scores'], native_value['scores'])) < .01
            # Run complete original decoding/scoring for K-only, V-only, and KV.
            for mode in ['k', 'v', 'kv']:
                mixed=replace(engine, row, original, native, list(range(18, 24)), mode)
                value=M.score(engine, mask, row, mixed, [])
                assert ('f1' in value) if row['task']=='hotpotqa' else ('correct' in value)
                del mixed
            del original
        engine.unchanged(); del base, native
        checks.append(dict(id=row['id'], task=row['task'], unmasked_historical_parity=True,
            self_substitution_bitwise_logits=True, all_native_endpoint_parity=True,
            all_groups_components_and_protected_header_checked=True,
            native_cached_direct_prefill_parity=True, **native_parity(row, native_value)))
        C.log('ORACLE64 SMOKE', row['id'], 'PASSED')
    assert engine.student_test_native_prefills==0
    mark('smoke/completed.json', status='completed', checks=checks, training=False,
        native_kv_usage='Explicit offline oracle substitutions and native baseline only; never ordinary A/C/D or masks.',
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.no_grad()
def evaluate(task):
    verify(); assert C.read(ROOT / 'smoke/completed.json')['manifest_sha256']==identity()
    assert task in CFG['test_counts'] and task!='gsm8k'
    engine, modules, mask=models(); records=[]; started=time.monotonic(); done=0
    for i, row in enumerate(rows(task), 1):
        base=engine.base(row); native=engine.native_cache(row, baseline=True)
        path=f'evaluation/{task}/per_config/native/{row["id"]}.json'
        if (ROOT/path).exists(): native_value=C.read(ROOT/path)['result']
        else:
            native_value=M.score(engine, mask, row, C.clone_cache(native, engine.model), [])
            mark(path, id=row['id'], task=task, result=native_value, **native_parity(row, native_value))
        values={}; done+=1
        for method, module in modules.items():
            original=engine.cache(row, base, module); values[method]={}
            for con in conditions():
                path=f'evaluation/{task}/per_config/{method}/{con["name"]}/{row["id"]}.json'
                if (ROOT/path).exists():
                    item=C.read(ROOT/path); assert item['manifest_sha256']==identity(); value=item['result']
                else:
                    mixed=replace(engine, row, original, native, con['layers'], con['mode'])
                    value=M.score(engine, mask, row, mixed, con['layers'] if con['mode']=='mask' else [])
                    del mixed
                    mark(path, id=row['id'], task=task, method=method, condition=con, result=value)
                if con['mode']=='none': M.parity(row, method, value)
                values[method][con['name']]=value; done+=1
                status(status='running', phase='evaluation', task=task, sample=i, total=64, method=method,
                    condition=con['name'], task_configurations_completed=done, task_configurations=4864,
                    elapsed_seconds=time.monotonic()-started)
            del original
        record=dict(id=row['id'], task=task, gold=row['gold'], native=native_value, conditions=values)
        mark(f'evaluation/{task}/per_sample/{row["id"]}.json', **record); records.append(record)
        del base, native; engine.unchanged()
        if i%4==0: C.log('ORACLE64', task, f'{i}/64')
    assert engine.student_test_native_prefills==0; verify()
    C.write_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl', records)
    make_summary(task, records)


def make_summary(task, records):
    keys=['em', 'f1'] if task=='hotpotqa' else ['correct']; primary=keys[0]
    common=[r for r in records if all(r['conditions'][a]['unmasked'][primary] for a in METHODS)]
    metrics={}; comparisons={}; contrasts={}
    for method in METHODS:
        metrics[method]={}; comparisons[method]={}; contrasts[method]={}
        for con in conditions():
            name=con['name']; vals=[r['conditions'][method][name] for r in records]
            metrics[method][name]={k:sum(v[k] for v in vals)/len(records) for k in keys}
            metrics[method][name]['cap_count']=sum(v.get('reached_cap', 0) for v in vals)
            comparisons[method][name]=dict(
                delta_vs_unmasked={k:C.paired([v[k]-r['conditions'][method]['unmasked'][k] for r,v in zip(records, vals)]) for k in keys},
                rescued_ids=[r['id'] for r,v in zip(records, vals) if v[primary] and not r['conditions'][method]['unmasked'][primary]],
                harmed_ids=[r['id'] for r,v in zip(records, vals) if not v[primary] and r['conditions'][method]['unmasked'][primary]],
                common_correct_n=len(common), common_correct_retained=sum(r['conditions'][method][name][primary] for r in common))
            if con['mode'] in ['k', 'v', 'kv']:
                maskname=name.split('_')[0]+'_mask'
                contrasts[method][name]={k:C.paired([v[k]-r['conditions'][method][maskname][k] for r,v in zip(records, vals)]) for k in keys}
    native_metrics={k:sum(r['native'][k] for r in records)/len(records) for k in keys}
    recovery={a:{c['name']:{k:((metrics[a][c['name']][k]-metrics[a]['unmasked'][k]) /
        (native_metrics[k]-metrics[a]['unmasked'][k]) if native_metrics[k]>metrics[a]['unmasked'][k] else None)
        for k in keys} for c in conditions()} for a in METHODS}
    mark(f'evaluation/{task}/summary.json', status='completed', task=task, n=len(records),
        metrics=metrics, native=native_metrics, comparisons=comparisons,
        substitution_minus_same_group_mask=contrasts, recovery_ratio=recovery,
        common_correct_ids=[r['id'] for r in common], training=False,
        native_historical_prediction_mismatches=[r['id'] for r in records if not native_parity(next(v for v in rows(task) if v['id']==r['id']), r['native'])['historical_prediction_matches']])
    mark(f'evaluation/{task}/completed.json', status='completed', finished_at=time.strftime('%F %T'))


def summarize():
    verify(); tasks={t:C.read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    assert all(x['status']=='completed' and x['manifest_sha256']==identity() for x in tasks.values())
    mark('summary.json', status='completed', tasks=tasks, config=CFG, training=False, gsm8k=False)
    lines=['# A/C/D native KV substitution and masking:64 per task', '',
        '| Method | Condition | ARC64 accuracy | HellaSwag64 accuracy | HotpotQA64 EM | HotpotQA64 F1 |',
        '|---|---|---:|---:|---:|---:|']
    def table_row(method, name, value):
        return '| '+method+' | '+name+' | '+' | '.join(f'{x*100:.2f}%' for x in value)+' |'
    lines.append(table_row('Qwen', 'native', [tasks[t]['native']['correct'] for t in ['arc','hellaswag']]+
        [tasks['hotpotqa']['native'][k] for k in ['em','f1']]))
    for method in METHODS:
        for con in conditions():
            name=con['name']; value=([tasks[t]['metrics'][method][name]['correct'] for t in ['arc','hellaswag']]+
                [tasks['hotpotqa']['metrics'][method][name][k] for k in ['em','f1']])
            lines.append(table_row(method, name, value))
    lines+=['', *['- '+v for v in CFG['caveats']]]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json', status='completed', finished_at=time.strftime('%F %T')); status(status='completed')


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare': prepare()
    elif action=='smoke': smoke()
    elif action=='evaluate': evaluate(sys.argv[2])
    elif action=='summarize': summarize()
    else: raise ValueError(action)
