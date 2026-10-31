"""Wrong-donor KV probe: layer relevance vs translation accuracy, immutable A/C/D."""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
PARENT = Path(CFG['parent'])
ORACLE = PARENT / CFG['oracle_experiment']
sys.path.insert(0, str(PARENT))
import core as C
spec = importlib.util.spec_from_file_location('previous_oracle', ORACLE / 'experiment.py')
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)
torch = C.torch
METHODS = CFG['methods']
GROUPS = CFG['groups']
TASKS = ['arc', 'hellaswag', 'hotpotqa']


def identity(): return C.sha(ROOT / 'manifest.json')
def mark(path, **v): C.save(ROOT / path, dict(manifest_sha256=identity(), **v))
def status(**v): C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **v))
def rows(task=None): return [r for r in C.read_jsonl(ROOT / 'prepared/test.jsonl')
                            if task is None or r['task'] == task]


def conditions():
    cs = [dict(name='unmasked', mode='none', layers=[])]
    cs += [dict(name=f'g{g+1}_native', mode='kv', layers=list(GROUPS[g])) for g in range(6)]
    cs += [dict(name=f'g{g+1}_donor', mode='donor', layers=list(GROUPS[g])) for g in range(6)]
    return cs


# ---------------------------------------------------------------- selection
def select_rows():
    allrows = C.rows('test')
    out = []
    for task in TASKS:
        cand = [r for r in allrows if r['task'] == task]
        assert len(cand) >= CFG['test_counts'][task], (task, len(cand))
        out += cand[:CFG['test_counts'][task]]
    return out


def build_donors(rs):
    """Deterministic closest-length donor from a different question of the same task."""
    table = {}
    for r in rs:
        length = len(r['content_ids'])
        cand = [q for q in rs if q['id'] != r['id'] and
                (r['task'] != 'hotpotqa' or q.get('question') != r.get('question'))]
        assert cand, ('no eligible donor', r['id'])
        donor = min(cand, key=lambda q: (abs(len(q['content_ids']) - length), q['id']))
        assert donor['id'] != r['id']
        table[r['id']] = donor['id']
    return table


# ---------------------------------------------------------------- injection
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def reposition_key(key, cos_src, sin_src, cos_dst, sin_dst):
    """Move post-RoPE keys from their source positions to the target positions.

    The stored cache key is R(theta_src) * k_normed. Rotating by the angle
    difference gives R(theta_dst) * k_normed, i.e. the same content sitting at
    the target position. When src == dst this is exactly the identity.
    """
    cs = cos_src[None, None]; ss = sin_src[None, None]
    cd = cos_dst[None, None]; sd = sin_dst[None, None]
    cos_delta = cs * cd + ss * sd
    sin_delta = sd * cs - cd * ss
    moved = key * cos_delta + rotate_half(key) * sin_delta
    return moved.to(key.dtype)


@torch.no_grad()
def replace_donor(engine, row, original, donor, layers):
    """Copy a DIFFERENT question's native body KV onto this row's body positions."""
    start = len(engine.prefix_ids)
    tlen = len(row['content_ids'])
    dlen = donor.get_seq_length() - start
    assert dlen > 0 and tlen > 0
    if dlen == 1:
        idx = torch.zeros(tlen, dtype=torch.long, device='cuda')
    else:
        idx = (torch.arange(tlen, device='cuda', dtype=torch.float32) *
               (dlen - 1) / (tlen - 1)).round().long().clamp_(0, dlen - 1)
    assert int(idx.min()) >= 0 and int(idx.max()) < dlen
    source_positions = idx + start
    target_positions = torch.arange(start, start + tlen, device='cuda')
    # Equal source and target positions mean the phase is already correct; skip the
    # rotation entirely so exact-length pairs stay bitwise identical to a plain copy
    # (cos^2+sin^2 is only 1 up to fp rounding and would otherwise perturb the bits).
    positions_equal = torch.equal(source_positions, target_positions)
    if not positions_equal:
        src_tables = C._rotary_tables(engine.model, source_positions)
        dst_tables = C._rotary_tables(engine.model, target_positions)
    mixed = C.clone_cache(original, engine.model)
    for layer in layers:
        target = C._cache_layer(mixed, layer)
        source = C._cache_layer(donor, layer)
        for index in (0, 1):
            assert target[index].shape[0] == source[index].shape[0]
            assert target[index].shape[1] == source[index].shape[1]
            assert target[index].shape[3] == source[index].shape[3]
            picked = source[index].index_select(2, idx + start)
            if index == 0 and not positions_equal:
                cos_src, sin_src = src_tables(layer)
                cos_dst, sin_dst = dst_tables(layer)
                picked = reposition_key(picked, cos_src, sin_src, cos_dst, sin_dst)
            assert picked.shape[2] == tlen and torch.isfinite(picked).all()
            target[index][:, :, start:start + tlen, :].copy_(picked)
    return mixed


# ---------------------------------------------------------------- prepare
def oracle_path(task, method, condition, row_id):
    if method == 'native':
        return ORACLE / 'evaluation' / task / 'per_config/native' / (row_id + '.json')
    return ORACLE / 'evaluation' / task / 'per_config' / method / condition / (row_id + '.json')


def reuse_value(task, method, condition, row_id):
    item = C.read(oracle_path(task, method, condition, row_id))
    assert item['manifest_sha256'] == CFG['oracle_manifest_sha256'], (task, method, condition, row_id)
    assert item['id'] == row_id
    return item['result']


def prepare():
    if (ROOT / 'manifest.json').exists():
        verify(); return
    O.verify()
    assert C.identity() == CFG['parent_manifest_sha256'], 'parent manifest drift'
    assert O.identity() == CFG['oracle_manifest_sha256'], 'oracle manifest drift'
    assert C.read(ORACLE / 'completed.json')['status'] == 'completed'
    selected = select_rows()
    assert len(selected) == sum(CFG['test_counts'].values()) == 256
    donors = {}
    for task in TASKS:
        donors.update(build_donors([r for r in selected if r['task'] == task]))
    assert set(donors) == {r['id'] for r in selected}
    prepared = [dict(r, donor_id=donors[r['id']]) for r in selected]
    # Reused anchors must match the oracle 64-question set exactly where they exist.
    reused = []
    for r in prepared:
        if r['task'] == 'arc':
            continue                                  # ARC is recomputed in full (128)
        for method in METHODS:
            reused.append((r, method, 'unmasked'))
            for g in range(6):
                reused.append((r, method, f'g{g+1}_kv'))
        reused.append((r, None, 'native'))
    for r, method, condition in reused:
        if method is None:
            item = C.read(oracle_path(r['task'], 'native', None, r['id']))
            assert item['manifest_sha256'] == CFG['oracle_manifest_sha256'] and item['id'] == r['id']
        else:
            item = C.read(oracle_path(r['task'], method, condition, r['id']))
            assert item['manifest_sha256'] == CFG['oracle_manifest_sha256']
            assert item['id'] == r['id'] and item['method'] == method
            assert item['condition']['name'] == condition
    refs = [ORACLE / f for f in ['manifest.json', 'config.json', 'experiment.py',
                                 'completed.json', 'prepared/test.jsonl', 'smoke/completed.json']]
    refs += [PARENT / f for f in ['manifest.json', 'core.py', 'train.py', 'evaluate.py',
                                  'prepared/test.jsonl']]
    refs += [PARENT / 'runs' / a / 'epoch4_step800.pt' for a in ['C', 'D']]
    C.write_jsonl(ROOT / 'prepared/test.jsonl', prepared)
    refs.append(ROOT / 'prepared/test.jsonl')
    C.save(ROOT / 'manifest.json', dict(
        config=CFG, config_sha256=C.sha(ROOT / 'config.json'),
        code={p.name: C.sha(p) for p in ROOT.glob('*.py')},
        sources={str(p): C.sha(p) for p in refs},
        test_ids={t: [r['id'] for r in prepared if r['task'] == t] for t in TASKS},
        donor_of={r['id']: r['donor_id'] for r in prepared},
        conditions=conditions(),
        anchors='ARC recomputed in full at 128; HellaSwag/HotpotQA unmasked and g*_native reused under oracle identity, never recomputed.',
        native_baseline='ARC recomputed; HellaSwag/HotpotQA reused from oracle.',
        n=256, training=False, gsm8k=False,
        design='translated vs same-prompt native vs different-question donor, per 6-layer group'))
    for r, method, condition in reused:
        if method is None:
            value = reuse_value(r['task'], 'native', None, r['id'])
            mark(f'evaluation/{r["task"]}/per_config/native/{r["id"]}.json',
                 id=r['id'], task=r['task'], method='native',
                 condition=dict(name='native', mode='native', layers=list(range(36))),
                 result=value, reused_from=str(oracle_path(r['task'], 'native', None, r['id'])),
                 source_manifest_sha256=O.identity())
        else:
            public = 'unmasked' if condition == 'unmasked' else condition.replace('_kv', '_native')
            con = next(c for c in conditions() if c['name'] == public)
            value = reuse_value(r['task'], method, condition, r['id'])
            mark(f'evaluation/{r["task"]}/per_config/{method}/{public}/{r["id"]}.json',
                 id=r['id'], task=r['task'], method=method, condition=con, result=value,
                 reused_from=str(oracle_path(r['task'], method, condition, r['id'])),
                 reused_condition=f'{r["task"]}/per_config/{method}/{condition}/{r["id"]}.json',
                 source_manifest_sha256=O.identity())
    mark('prepared/completed.json', status='completed', n=len(prepared),
         new_configurations=7296, reused_configurations=len(reused))
    C.log('WRONGDONOR PREPARED', CFG['test_counts'], 'reused', len(reused))


def verify():
    C.verify()
    m = C.read(ROOT / 'manifest.json')
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    for name, digest in m['code'].items():
        assert C.sha(ROOT / name) == digest, name
    for path, digest in m['sources'].items():
        assert C.sha(path) == digest, path


# ---------------------------------------------------------------- smoke
@torch.no_grad()
def smoke():
    verify()
    engine, modules, mask = O.models()
    checks = []
    for task in TASKS:
        rs = rows(task)
        pick = [rs[0], max(rs, key=lambda r: len(r['content_ids']))]
        for row in {r['id']: r for r in pick}.values():
            donor_row = next(r for r in rows(task) if r['id'] == row['donor_id'])
            assert donor_row['id'] != row['id'], ('donor equals target', row['id'])
            base = engine.base(row)
            native = O.native_cache(engine, row)
            donor = O.native_cache(engine, donor_row)
            start = len(engine.prefix_ids)
            end = start + len(row['content_ids'])
            for method, module in modules.items():
                original = engine.cache(row, base, module)
                # 1. identity resampling: self-mode donor must reproduce same-prompt native KV.
                identity_mix = replace_donor(engine, row, original, native, list(range(36)))
                reference = O.replace(engine, row, original, native, list(range(36)), 'kv')
                for layer in range(36):
                    for a, b in zip(C._cache_layer(identity_mix, layer), C._cache_layer(reference, layer)):
                        assert torch.equal(a, b), ('identity resample drift', row['id'], method, layer)
                del identity_mix, reference
                # 1b. phase repositioning must be the identity when source == target position.
                pos = torch.arange(start, end, device='cuda')
                tb = C._rotary_tables(engine.model, pos)
                cos_t, sin_t = tb(0)
                probe = C._cache_layer(original, 0)[0][:, :, start:end, :]
                moved = reposition_key(probe, cos_t, sin_t, cos_t, sin_t)
                assert moved.dtype == probe.dtype, 'phase dtype drift'
                scale = float(probe.float().abs().max())
                diff = (moved.float() - probe.float()).abs()
                C.log('PHASE identity', row['id'], 'scale=%.2f max_abs=%.3e' % (scale, float(diff.max())))
                assert float(diff.max()) <= 1e-2 * scale, 'phase identity failed'
                far = C._rotary_tables(engine.model, pos + 7)
                cos_f, sin_f = far(0)
                there = reposition_key(probe, cos_t, sin_t, cos_f, sin_f)
                back = reposition_key(there, cos_f, sin_f, cos_t, sin_t)
                rback = (back.float() - probe.float()).abs()
                C.log('PHASE inverse', row['id'], 'max_abs=%.3e' % float(rback.max()))
                assert float(rback.max()) <= 3e-2 * scale, 'phase inverse failed'
                del probe, moved, there, back
                # 2. donor injection touches only body positions of the selected layers.
                for g in range(6):
                    mixed = replace_donor(engine, row, original, donor, GROUPS[g])
                    for layer in range(36):
                        for index, tensor in enumerate(C._cache_layer(mixed, layer)):
                            orig = C._cache_layer(original, layer)[index]
                            assert torch.equal(tensor[:, :, :start, :], orig[:, :, :start, :]), \
                                ('header/eartext changed', row['id'], layer)
                            assert torch.equal(tensor[:, :, end:, :], orig[:, :, end:, :]), \
                                ('suffix changed', row['id'], layer)
                            if layer in GROUPS[g]:
                                assert torch.isfinite(tensor[:, :, start:end, :]).all()
                    logits = O.M.first_logits(engine, mask, row, mixed, [])
                    assert torch.isfinite(logits).all(), ('non-finite logits', row['id'], method, g)
                    del mixed, logits
                del original
            engine.unchanged()
            del base, native, donor
            checks.append(dict(id=row['id'], task=task,
                               donor_id=row['donor_id'],
                               identity_resampling_bitwise_exact=True,
                               donor_is_different_question=True,
                               header_and_suffix_preserved=True,
                               resample_index_in_range=True,
                               finite_logits=True))
            C.log('WRONGDONOR SMOKE', task, row['id'], 'PASSED')
    assert engine.student_test_native_prefills == 0
    mark('smoke/completed.json', status='completed', checks=checks, training=False,
         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)


# ---------------------------------------------------------------- evaluate
def distances(original, native, start, end):
    """Relative K/V distance to same-prompt native, per 6-layer group."""
    out = {}
    for g, layers in enumerate(GROUPS):
        numk = denk = numv = denv = 0.0
        for layer in layers:
            kt, vt = C._cache_layer(original, layer)
            kn, vn = C._cache_layer(native, layer)
            a = kt[:, :, start:end, :].float(); b = kn[:, :, start:end, :].float()
            numk += float((a - b).norm()); denk += float(b.norm())
            a = vt[:, :, start:end, :].float(); b = vn[:, :, start:end, :].float()
            numv += float((a - b).norm()); denv += float(b.norm())
        out[str(g + 1)] = dict(k=numk / max(denk, 1e-12), v=numv / max(denv, 1e-12))
    return out


@torch.no_grad()
def evaluate(task):
    verify()
    assert C.read(ROOT / 'smoke/completed.json')['manifest_sha256'] == identity()
    engine, modules, mask = O.models()
    records = []; done = 0; started = time.monotonic(); dist = []
    rs = rows(task); total = len(rs)
    by_id = {r['id']: r for r in rs}
    for i, row in enumerate(rs, 1):
        base = engine.base(row)
        native = O.native_cache(engine, row)
        donor = O.native_cache(engine, by_id[row['donor_id']])
        start = len(engine.prefix_ids)
        end = start + len(row['content_ids'])
        if (ROOT / f'evaluation/{task}/per_config/native/{row["id"]}.json').exists():
            native_value = C.read(ROOT / f'evaluation/{task}/per_config/native/{row["id"]}.json')['result']
        else:
            native_value = O.M.score(engine, mask, row, C.clone_cache(native, engine.model), [])
            mark(f'evaluation/{task}/per_config/native/{row["id"]}.json',
                 id=row['id'], task=task, method='native',
                 condition=dict(name='native', mode='native', layers=list(range(36))),
                 result=native_value)
        done += 1
        values = {}
        for method, module in modules.items():
            original = engine.cache(row, base, module); values[method] = {}
            dist.append(dict(row=row['id'], method=method, **distances(original, native, start, end)))
            for con in conditions():
                name = con['name']
                path = f'evaluation/{task}/per_config/{method}/{name}/{row["id"]}.json'
                if (ROOT / path).exists():
                    item = C.read(ROOT / path)
                    assert item['manifest_sha256'] == identity()
                    assert item['id'] == row['id'] and item['method'] == method
                    assert item['condition'] == con
                    value = item['result']
                elif con['mode'] == 'none':
                    value = O.M.score(engine, mask, row, C.clone_cache(original, engine.model), [])
                    mark(path, id=row['id'], task=task, method=method, condition=con, result=value)
                elif con['mode'] == 'donor':
                    mixed = replace_donor(engine, row, original, donor, con['layers'])
                    value = O.M.score(engine, mask, row, mixed, []); del mixed
                    mark(path, id=row['id'], task=task, method=method, condition=con, result=value)
                else:
                    mixed = O.replace(engine, row, original, native, con['layers'], 'kv')
                    value = O.M.score(engine, mask, row, mixed, []); del mixed
                    mark(path, id=row['id'], task=task, method=method, condition=con, result=value)
                values[method][name] = value; done += 1
                status(status='running', phase='evaluation', task=task, sample=i, total=total,
                       method=method, condition=name, completed=done,
                       elapsed_seconds=time.monotonic() - started)
            del original
        record = dict(id=row['id'], task=task, gold=row['gold'], donor_id=row['donor_id'],
                      native=native_value, conditions=values)
        mark(f'evaluation/{task}/per_sample/{row["id"]}.json', **record)
        records.append(record)
        engine.unchanged(); del base, native, donor
        if i % 4 == 0: C.log('WRONGDONOR', task, f'{i}/{total}')
    assert engine.student_test_native_prefills == 0
    verify()
    C.write_jsonl(ROOT / f'evaluation/{task}/per_sample.jsonl', records)
    make_summary(task, records, dist)


# ---------------------------------------------------------------- summary
def mean(values): return sum(values) / len(values) if values else 0.0


def make_summary(task, records, dist):
    keys = ['em', 'f1'] if task == 'hotpotqa' else ['correct']
    primary = keys[0]; n = len(records)
    native = {k: mean([r['native'][k] for r in records]) for k in keys}
    grid = {}; comparisons = {}; table = []
    for method in METHODS:
        grid[method] = {}
        for con in conditions():
            name = con['name']
            grid[method][name] = {k: mean([r['conditions'][method][name][k] for r in records]) for k in keys}
            grid[method][name]['cap_count'] = sum(r['conditions'][method][name].get('reached_cap', 0) for r in records)
        comparisons[method] = {}
        for g in range(6):
            nat = f'g{g+1}_native'; don = f'g{g+1}_donor'; base = 'unmasked'
            row = dict(group=g + 1, layers=GROUPS[g],
                       translated=grid[method][base][primary],
                       native=grid[method][nat][primary],
                       donor=grid[method][don][primary])
            for key in keys:
                row[f'donor_minus_translated_{key}'] = C.paired(
                    [r['conditions'][method][don][key] - r['conditions'][method][base][key] for r in records])
                row[f'native_minus_translated_{key}'] = C.paired(
                    [r['conditions'][method][nat][key] - r['conditions'][method][base][key] for r in records])
                row[f'donor_minus_native_{key}'] = C.paired(
                    [r['conditions'][method][don][key] - r['conditions'][method][nat][key] for r in records])
            row['donor_minus_translated_rescued'] = [r['id'] for r in records
                if r['conditions'][method][don][primary] and not r['conditions'][method][base][primary]]
            row['donor_minus_translated_harmed'] = [r['id'] for r in records
                if not r['conditions'][method][don][primary] and r['conditions'][method][base][primary]]
            row['reading'] = reading(row, primary)
            comparisons[method][g + 1] = row; table.append((method, row))
    dist_mean = {}
    for method in METHODS:
        sub = [d for d in dist if d['method'] == method]
        dist_mean[method] = {str(g): dict(k=mean([d[str(g)]['k'] for d in sub]),
                                          v=mean([d[str(g)]['v'] for d in sub])) for g in range(1, 7)}
    if task == 'arc':
        first = [r for r in records if r['id'] in {x['id'] for x in rows('arc')[:64]}]
        subset = {m: {k: mean([r['conditions'][m]['unmasked'][k] for r in first]) for k in keys} for m in METHODS}
    else:
        subset = None
    mark(f'evaluation/{task}/summary.json', status='completed', task=task, n=n, primary_metric=primary,
         native=native, grid=grid, comparisons=comparisons, distances=dist_mean,
         arc_first64_subset=subset, training=False, gsm8k=False,
         native_usage='Offline oracle body cache only; shared header; no training.')
    mark(f'evaluation/{task}/completed.json', status='completed', finished_at=time.strftime('%F %T'))
    return table


def reading(row, primary):
    dn = row[f'donor_minus_translated_{primary}']['mean']
    nn = row[f'native_minus_translated_{primary}']['mean']
    lo_d, hi_d = row[f'donor_minus_translated_{primary}']['ci95']
    lo_n, hi_n = row[f'native_minus_translated_{primary}']['ci95']
    donor_hurts = hi_d < 0
    native_helps = lo_n > 0
    if donor_hurts and not native_helps:
        return 'content-critical, translation already accurate'
    if not donor_hurts:
        return 'layers not content-critical here'
    if native_helps:
        return 'translation inaccurate here'
    return 'mixed'


def summarize():
    verify()
    tasks = {t: C.read(ROOT / f'evaluation/{t}/summary.json') for t in TASKS}
    assert all(d['status'] == 'completed' and d['manifest_sha256'] == identity() for d in tasks.values())
    mark('summary.json', status='completed', tasks=tasks, config=CFG, training=False, gsm8k=False)
    lines = ['# Wrong-donor KV probe: A/C/D (ARC128, HellaSwag64, HotpotQA64)', '',
             'Three sources per 6-layer group: **translated** (no injection), **native** (same-question '
             'native KV, oracle repair), **donor** (different-question native KV, same shape and position '
             'grid but wrong content).', '',
             'If donor is far below translated while native sits within noise of translated, the group '
             'matters and the translation there is already accurate. If donor is within noise of '
             'translated, that group is not content-critical.', '']
    for task in TASKS:
        d = tasks[task]; p = d['primary_metric']
        lines += [f'## {task} (n={d["n"]}, primary={p}, native={d["native"][p]*100:.2f}%)', '']
        lines += [f'| Method | Group | Layers | translated | native | donor | donor−translated (95% CI) | reading |',
                  '|---|---:|---|---:|---:|---:|---|---|']
        for method in METHODS:
            for g in range(1, 7):
                row = d['comparisons'][method][str(g)]
                dd = row[f'donor_minus_translated_{p}']
                lines.append('|' + method + f'|{g}|{row["layers"][0]+1}–{row["layers"][-1]+1}|'
                             f'{row["translated"]*100:.2f}%|{row["native"]*100:.2f}%|{row["donor"]*100:.2f}%|'
                             f'{dd["mean"]*100:+.2f} [{dd["ci95"][0]*100:+.1f},{dd["ci95"][1]*100:+.1f}]|'
                             f'{row["reading"]}|')
        lines += ['', f'Per-group relative K/V distance to same-question native ({task}):', '',
                  '| Method | Group | K rel.dist | V rel.dist |', '|---|---:|---:|---:|']
        for method in METHODS:
            for g in range(1, 7):
                e = d['distances'][method][str(g)]
                lines.append(f'|{method}|{g}|{e["k"]:.4f}|{e["v"]:.4f}|')
        if d.get('arc_first64_subset'):
            lines += ['', 'ARC first-64 subset (comparable to the 64-question experiments):', '']
            for method in METHODS:
                v = d['arc_first64_subset'][method][p]
                lines.append(f'- {method}: {v*100:.2f}%')
        lines.append('')
    lines += [*['- ' + s for s in CFG['caveats']]]
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    mark('completed.json', status='completed', finished_at=time.strftime('%F %T'))
    status(status='completed')


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    elif action == 'smoke': smoke()
    elif action == 'evaluate': evaluate(sys.argv[2])
    elif action == 'summarize': summarize()
    else: raise ValueError(action)
