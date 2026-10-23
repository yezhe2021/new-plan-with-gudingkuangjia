"""Independent grouped translator experiment; parent artifacts are read-only."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import sys
import json
import time
import random
import shutil
import fcntl
import importlib.util
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


U = load_module('grouped_parent_ufd', Path(CFG['ufd_parent']) / 'experiment.py')
C, torch, F = U.C, U.torch, U.F
T = load_module('grouped_parent_reconstruction', C.ROOT / 'train.py')
from grouped_translator import GroupedKVTranslator
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def identity():
    return C.sha(ROOT / 'manifest.json')


def status(**kw):
    C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **kw))


def mark(path, **kw):
    C.save(path, dict(manifest_sha256=identity(), **kw))


def done(path):
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    assert record['manifest_sha256'] == identity(), ('Stale artifact', str(path))
    assert record['status'] == 'completed'
    return True


def shared(state, result):
    value = dict(experiment=CFG['name'], time=time.strftime('%F %T'), status=state,
                 manifest=str(ROOT / 'manifest.json'), config=CFG, results=result,
                 raw_results=str(ROOT / 'evaluation/per_sample.jsonl'))
    with (ROOT.parent / 'EXPERIMENT_RESULTS.md').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n\n## ' + CFG['name'] + '\n\n```json\n' +
                     json.dumps(value, ensure_ascii=False, indent=2) + '\n```\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare():
    parents = {}
    for directory, expected in CFG['parent_manifest_sha256'].items():
        parent = Path(directory)
        assert C.sha(parent / 'manifest.json') == expected
        manifest = json.loads((parent / 'manifest.json').read_text())
        for name, digest in manifest['code'].items():
            assert C.sha(parent / name) == digest, ('Parent code changed', directory, name)
        if 'config_sha256' in manifest:
            assert C.sha(parent / 'config.json') == manifest['config_sha256']
        parents[directory] = dict(manifest_sha256=expected,
                                 config_sha256=C.sha(parent / 'config.json'),
                                 code={p.name: C.sha(p) for p in parent.glob('*.py')})
    datasets = {}
    for label, parent in [('a', C.ROOT), ('b', U.ROOT)]:
        for split in (['train', 'validation'] if label == 'a' else ['train', 'validation', 'test']):
            path = parent / 'prepared' / (split + '.jsonl')
            rr = C.read_jsonl(path)
            recorded = json.loads((parent / 'manifest.json').read_text())['prepared_sha256'][split]
            assert C.digest(rr) == recorded, ('Parent prepared-content digest mismatch', str(path))
            for row in rr:
                if label == 'a':
                    for family in ['llama', 'qwen']:
                        assert C.cache_path(split, row, family).exists()
                else:
                    assert U.source_path(row).exists()
            datasets[str(path)] = dict(sha256=C.sha(path), n=len(rr))
    assert len(C.rows('train')) == 2048 and len(C.rows('validation')) == 256
    assert len(U.rows('train')) == 2048
    parent_completed = json.loads((U.ROOT / 'completed.json').read_text())
    assert parent_completed['status'] == 'completed' and parent_completed['manifest_sha256'] == U.identity()
    for row in U.rows('test'):
        baseline_path = U.ROOT / 'evaluation' / row['task'] / 'per_sample' / (U.fname(row) + '.json')
        baseline = json.loads(baseline_path.read_text())
        assert baseline['manifest_sha256'] == U.identity()
        for name in ['id', 'task', 'prompt', 'receiver_suffix', 'candidates', 'gold']:
            assert baseline[name] == row[name], ('Baseline mismatch', row['id'], name)
        datasets[str(baseline_path)] = dict(sha256=C.sha(baseline_path))
    for task, count in CFG['stage_b_train_counts'].items():
        assert sum(r['task'] == task for r in U.rows('train')) == count
    train_keys = {U.key(U.qdoc(r['doc'])) for r in C.rows('train')}
    train_keys |= {r['question_key'] for r in U.rows('train')}
    heldout = {r['question_key'] for s in ['validation', 'test'] for r in U.rows(s)}
    assert not train_keys & heldout, 'Training/held-out question overlap'
    references = {str(Path(C.CFG['hf_root']) / 'heterofold' / name): digest
                  for name, digest in C.CFG['reference_hashes'].items()}
    for path, digest in references.items():
        assert C.sha(path) == digest
    assert shutil.disk_usage(ROOT).free / 2**30 >= CFG['disk_reserve_gib'] + 8
    value = dict(config=CFG, config_sha256=C.sha(ROOT / 'config.json'), parents=parents,
                 datasets=datasets, reference_code=references,
                 code={p.name: C.sha(p) for p in ROOT.glob('*.py')},
                 source_cache_policy='Read existing identity-checked caches; no recapture or overwrite')
    if (ROOT / 'manifest.json').exists():
        assert json.loads((ROOT / 'manifest.json').read_text()) == value
    else:
        C.save(ROOT / 'manifest.json', value)
    mark(ROOT / 'prepared/completed.json', status='completed', datasets=datasets)
    status(status='prepared', phase='waiting_for_previous_experiment')


def verify():
    m = json.loads((ROOT / 'manifest.json').read_text())
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    for name, digest in m['code'].items():
        assert C.sha(ROOT / name) == digest, ('Experiment code changed', name)
    for directory, record in m['parents'].items():
        assert C.sha(Path(directory) / 'manifest.json') == record['manifest_sha256']
        assert C.sha(Path(directory) / 'config.json') == record['config_sha256']
        for name, digest in record['code'].items():
            assert C.sha(Path(directory) / name) == digest
    for path, record in m['datasets'].items():
        assert C.sha(path) == record['sha256']
    for path, digest in m['reference_code'].items():
        assert C.sha(path) == digest


def fresh():
    model = GroupedKVTranslator(**CFG['architecture'])
    assert sum(p.numel() for p in model.parameters()) == CFG['expected_parameters']
    return model


def load_weights(path, trainable=False):
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert value['manifest_sha256'] == identity()
    module = fresh()
    module.load_state_dict(value['state'], strict=True)
    del value
    return module.cuda().requires_grad_(trainable).train(trainable)


def map_source(module, source):
    pieces = []
    for start in range(0, source['key'].shape[1], CFG['chunk_tokens']):
        key = source['key'][:, start:start + CFG['chunk_tokens']][None].cuda()
        value = source['value'][:, start:start + CFG['chunk_tokens']][None].cuda()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            if torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters()):
                pair = activation_checkpoint(module, key, value, use_reentrant=False)
            else:
                pair = module(key, value)
        pieces.append(pair)
    return tuple(torch.cat([pair[i][0] for pair in pieces], dim=1) for i in range(2))


def native_prefix(model, tok, row, native):
    parent_path = U.ROOT / 'native_prefix' / (U.fname(row) + '.json')
    if parent_path.exists():
        record = json.loads(parent_path.read_text())
        assert record['manifest_sha256'] == U.identity() and record['id'] == row['id']
        return record
    path = ROOT / 'native_prefix' / (U.fname(row) + '.json')
    if path.exists():
        record = json.loads(path.read_text())
        assert record['manifest_sha256'] == identity() and record['id'] == row['id']
        return record
    result = U.rollout(model, tok, row, U.clone_cache(native, model), U.CFG['prefix_max_new_tokens'])
    mark(path, id=row['id'], **result)
    return result


def functional_loss(module, model, tok, row, epoch, student=False, validation=False):
    source = U.loaded_source(row)
    with torch.no_grad():
        native = U.native_body(model, row)
        if student:
            key, value = map_source(module, source)
            trajectory = U.rollout(model, tok, row, U.translated_cache(model, row, native, key, value),
                                   U.CFG['prefix_max_new_tokens'])
            del key, value
        else:
            trajectory = native_prefix(model, tok, row, native)
    positions = U.positions_for(trajectory['tokens'], row, epoch, validation)
    prefix = trajectory['tokens'][:max(positions)]
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        teacher = U.suffix_logits(model, row, U.clone_cache(native, model), prefix, positions)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        key, value = map_source(module, source)
        logits = U.suffix_logits(model, row, U.translated_cache(model, row, native, key, value), prefix, positions)
    loss = U.full_kl(teacher, logits)
    assert torch.isfinite(loss), 'Nonfinite functional KL'
    return loss, dict(kl=float(loss.detach()), positions=len(positions),
                      prefix_source='student' if student else 'native',
                      rollout_reached_cap=int(trajectory['reached_cap']))


def update(module, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), CFG['gradient_clip'])
    assert torch.isfinite(norm) and norm > 0, 'Invalid translator gradient'
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def save_checkpoint(path, module, stage, epoch, step, position, complete,
                    history, optimizer=None, scaler=None):
    assert shutil.disk_usage(ROOT).free / 2**30 > CFG['disk_reserve_gib']
    record = dict(manifest_sha256=identity(), state={k: v.detach().cpu() for k, v in module.state_dict().items()},
                  stage=stage, epoch=epoch, step=step, position=position,
                  epoch_completed=complete, history=history)
    if optimizer is not None:
        record.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
    C.save_tensor(path, record)


@torch.no_grad()
def validate_a(module):
    totals = defaultdict(float)
    rr = C.rows('validation')
    for row in rr:
        metrics = T.stage_a_backward(module, C.state('validation', row, 'llama'),
                                     C.state('validation', row, 'qwen'), backward=False)
        for name, value in metrics.items():
            totals[name] += value
    return {name: value / len(rr) for name, value in totals.items()}


@torch.no_grad()
def validate_b(module, model, tok):
    values = defaultdict(list)
    for row in U.rows('validation'):
        if len(values[row['task']]) >= CFG['monitor_validation_per_task']:
            continue
        _, record = functional_loss(module, model, tok, row, 0, validation=True)
        values[row['task']].append(record['kl'])
    tasks = {task: dict(n=len(v), kl=sum(v) / len(v)) for task, v in values.items()}
    return dict(tasks=tasks, macro_kl=sum(v['kl'] for v in tasks.values()) / len(tasks))


def smoke(stage):
    path = ROOT / 'smoke' / (stage + '.json')
    if done(path):
        return
    C.seed()
    module = fresh().cuda() if stage == 'a' else load_weights(ROOT / 'runs/stage_a/best.pt', True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=CFG['stage_' + stage + '_learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    checks = []
    if stage == 'a':
        for row in C.rows('train', True):
            metrics = T.stage_a_backward(module, C.state('train', row, 'llama'),
                                         C.state('train', row, 'qwen'), scaler=scaler)
            checks.append(dict(id=row['id'], grad_norm=update(module, optimizer, scaler), metrics=metrics))
    else:
        tok, model = C.model('qwen')
        # Native cache reconstruction gate validates the reused physical cache interface.
        row = U.rows('train', True)[0]
        ids = row['frames']['qwen']['body_ids']
        anchor = row['frames']['qwen']['native_prefix_length']
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            key, value, _ = C.capture(model, ids, len(ids), [0])
            native = U.native_body(model, row)
            a = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0])
            b = U.suffix_logits(model, row, U.translated_cache(model, row, native, key[:, anchor:].cuda(), value[:, anchor:].cuda()), [], [0])
            parity = float(U.full_kl(a, b))
            assert parity < U.CFG['native_parity_kl_max'] and int(a.argmax()) == int(b.argmax())
        del key, value, native, a, b
        for student in [False, True]:
            for row in U.rows('train', True):
                loss, metrics = functional_loss(module, model, tok, row, 2, student=student)
                scaler.scale(loss).backward()
                assert all(p.grad is None for p in model.parameters())
                # All groups and both roles must receive gradients through the real receiver.
                assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in group.parameters())
                           for group in list(module.k_encoders) + list(module.v_encoders))
                checks.append(dict(id=row['id'], grad_norm=update(module, optimizer, scaler), metrics=metrics))
        checks.append(dict(native_cache_kl=parity))
    mark(path, status='completed', checks=checks, parameters=sum(p.numel() for p in module.parameters()))
    C.log('GROUPED SMOKE PASSED ' + stage)


def train(stage):
    folder = ROOT / ('runs/stage_' + stage)
    folder.mkdir(parents=True, exist_ok=True)
    if done(folder / 'completed.json'):
        return
    assert done(ROOT / 'smoke' / (stage + '.json'))
    C.seed()
    module = fresh().cuda() if stage == 'a' else load_weights(ROOT / 'runs/stage_a/best.pt', True)
    tok = model = None
    if stage == 'b':
        tok, model = C.model('qwen')
    optimizer = torch.optim.AdamW(module.parameters(), lr=CFG['stage_' + stage + '_learning_rate'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    epoch0, position0, step, history = 1, 0, 0, []
    if (folder / 'last.pt').exists():
        record = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        assert record['manifest_sha256'] == identity() and record['stage'] == stage
        module.load_state_dict(record['state'], strict=True)
        optimizer.load_state_dict(record['optimizer'])
        scaler.load_state_dict(record['scaler'])
        epoch0 = record['epoch'] + int(record['epoch_completed'])
        position0 = 0 if record['epoch_completed'] else record['position']
        step, history = record['step'], record['history']
        del record
    epochs = CFG['stage_' + stage + '_epochs']
    with (folder / 'steps.jsonl').open('a') as stream:
        for epoch in range(epoch0, epochs + 1):
            module.train()
            if stage == 'a':
                rr = C.rows('train')
                random.Random(CFG['seed'] + epoch).shuffle(rr)
                order = [(r, False) for r in rr]
            else:
                order = U.epoch_order(epoch)
            optimizer.zero_grad(set_to_none=True)
            values = []
            for position, (row, student) in enumerate(order, 1):
                if epoch == epoch0 and position <= position0:
                    continue
                if stage == 'a':
                    metrics = T.stage_a_backward(module, C.state('train', row, 'llama'), C.state('train', row, 'qwen'),
                                                 CFG['effective_batch'], scaler=scaler)
                    values.append(metrics['total'])
                else:
                    loss, metrics = functional_loss(module, model, tok, row, epoch, student)
                    scaler.scale(loss / CFG['effective_batch']).backward()
                    values.append(metrics['kl'])
                    del loss
                if position % CFG['effective_batch'] == 0:
                    if model is not None:
                        assert all(p.grad is None for p in model.parameters())
                    gradient = update(module, optimizer, scaler)
                    step += 1
                    record = dict(stage=stage, epoch=epoch, step=step, position=position,
                                  total_samples=len(order), loss=sum(values) / len(values), grad_norm=gradient)
                    values = []
                    stream.write(json.dumps(record) + '\n')
                    stream.flush()
                    status(status='running', phase='stage_' + stage, **record)
                    if step % 8 == 0:
                        C.log(str(record))
                    if step % CFG['checkpoint_every_steps'] == 0 and position < len(order):
                        save_checkpoint(folder / 'last.pt', module, stage, epoch, step, position, False, history, optimizer, scaler)
            module.eval()
            metrics = validate_a(module) if stage == 'a' else validate_b(module, model, tok)
            entry = dict(epoch=epoch, step=step, validation=metrics)
            history = [r for r in history if r['epoch'] != epoch] + [entry]
            if stage == 'a' and min(history, key=lambda r: (r['validation']['total'], r['epoch'])) == entry:
                save_checkpoint(folder / 'best.pt', module, stage, epoch, step, len(order), True, history)
            mark(folder / 'validation.json', history=history)
            save_checkpoint(folder / 'last.pt', module, stage, epoch, step, len(order), True, history, optimizer, scaler)
    assert step == epochs * 256
    result = dict(status='completed', stage=stage, steps=step, history=history,
                  selected='best.pt' if stage == 'a' else 'last.pt')
    mark(folder / 'completed.json', **result)
    shared('stage_' + stage + '_completed', result)


@torch.no_grad()
def evaluate(task):
    folder = ROOT / 'evaluation' / task
    if done(folder / 'completed.json'):
        return
    assert done(ROOT / 'runs/stage_b/completed.json')
    module = load_weights(ROOT / 'runs/stage_b/last.pt')
    tok, model = C.model('qwen')
    runtime = C.runtime(model, tok)
    rr = [r for r in U.rows('test') if r['task'] == task]
    assert len(rr) == U.CFG['formal_counts'][task]
    records = []
    for index, row in enumerate(rr, 1):
        path = folder / 'per_sample' / (U.fname(row) + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            assert record['manifest_sha256'] == identity()
            records.append(record)
            continue
        original = json.loads((U.ROOT / 'evaluation' / task / 'per_sample' / (U.fname(row) + '.json')).read_text())
        assert original['manifest_sha256'] == U.identity()
        for name in ['id', 'task', 'prompt', 'receiver_suffix', 'candidates', 'gold']:
            assert original[name] == row[name]
        native = U.native_body(model, row)
        key, value = map_source(module, U.loaded_source(row))
        if task == 'gsm8k':
            text, generated = C.R.generate(runtime, torch.tensor([row['suffix_ids']], device='cuda'),
                U.CFG['gsm8k_formal_max_new_tokens'], past_key_values=U.translated_cache(model, row, native, key, value),
                past_length=len(row['frames']['qwen']['body_ids']), temperature=0.0)
            prediction = C.S.extract_numeric_answer(text)
            result = dict(prediction=prediction, correct=int(prediction == C.S.normalize_number(row['gold'])),
                          generation=text, generated_tokens=generated,
                          reached_cap=int(generated >= U.CFG['gsm8k_formal_max_new_tokens']))
        else:
            result = U.candidate_scores(model, tok, row, native, key, value)
        conditions = {k: original['conditions'][k] for k in ['qwen_native', 'new_ufd_stage_b']}
        conditions['grouped_translator'] = result
        record = dict(id=row['id'], task=task, prompt=row['prompt'], gold=row['gold'], conditions=conditions,
                      manifest_sha256=identity(), baseline_source=str(U.ROOT / 'evaluation' / task / 'per_sample' / (U.fname(row) + '.json')))
        C.save(path, record)
        records.append(record)
        status(status='running', phase='evaluation', task=task, sample=index, total=len(rr))
    C.write_jsonl(folder / 'per_sample.jsonl', records)
    mark(folder / 'completed.json', status='completed', samples=len(rr))


def summarize():
    results, all_records = {}, []
    for task in CFG['evaluation_tasks']:
        assert done(ROOT / 'evaluation' / task / 'completed.json')
        records = C.read_jsonl(ROOT / 'evaluation' / task / 'per_sample.jsonl')
        all_records.extend(records)
        results[task] = {arm: dict(n=len(records), correct=sum(r['conditions'][arm]['correct'] for r in records),
                            accuracy=sum(r['conditions'][arm]['correct'] for r in records) / len(records))
                         for arm in ['qwen_native', 'new_ufd_stage_b', 'grouped_translator']}
    C.write_jsonl(ROOT / 'evaluation/per_sample.jsonl', all_records)
    mark(ROOT / 'summary.json', status='completed', results=results, caveats=CFG['caveats'])
    shared('completed', results)
    mark(ROOT / 'completed.json', status='completed')
    status(status='completed', phase='all')


if __name__ == '__main__':
    try:
        action = sys.argv[1]
        if action == 'prepare':
            prepare()
        else:
            verify()
            if action == 'verify':
                C.log('MANIFEST VERIFIED')
            elif action == 'smoke':
                smoke(sys.argv[2])
            elif action == 'train':
                train(sys.argv[2])
            elif action == 'evaluate':
                evaluate(sys.argv[2])
            elif action == 'summarize':
                summarize()
            else:
                raise ValueError(action)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        C.save(ROOT / 'failure.json', error)
        status(status='failed', phase='exception', **error)
        raise
