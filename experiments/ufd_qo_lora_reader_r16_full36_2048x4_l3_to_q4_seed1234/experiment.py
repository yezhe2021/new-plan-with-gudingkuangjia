"""Frozen UFD Writer + all36-layer Q/O-LoRA continuation Reader pilot."""
import os
for env_name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[env_name] = '1'
import sys
import json
import time
import fcntl
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
PARENT = Path(CFG['parent'])
spec = importlib.util.spec_from_file_location('frozen_ufd', PARENT / 'experiment.py')
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)
C, torch = U.C, U.torch
sys.path.insert(0, str(ROOT))
from reader import QOReader, cpu_test


def identity():
    return C.sha(ROOT / 'manifest.json')


def status(**kwargs):
    C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **kwargs))


def shared(state, value):
    path = ROOT.parent / 'EXPERIMENT_RESULTS.md'
    record = dict(experiment=CFG['name'], timestamp=time.strftime('%F %T'), status=state,
                  config=CFG, checkpoint_selection=CFG['selection'], results=value,
                  raw_results=str(ROOT / 'evaluation/per_sample.jsonl'), log=str(ROOT / 'logs/pipeline.log'))
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n\n## ' + CFG['name'] + '\n\n```json\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n```\n')
        stream.flush()
        os.fsync(stream.fileno())
    C.save(ROOT / 'shared_summary_updated.json', dict(status=state, path=str(path)))


def prepare():
    assert U.identity() == CFG['parent_manifest_sha256']
    pm = json.loads((PARENT / 'manifest.json').read_text())
    for name, sha in pm['code'].items():
        assert C.sha(PARENT / name) == sha, ('Parent code changed', name)
    for name, sha in pm['parent_code'].items():
        assert C.sha(Path(U.CFG['parent']) / name) == sha
    assert C.sha(PARENT / 'runs/stage_b/last.pt') == CFG['writer_stage_b_sha256']
    assert C.sha(C.ROOT / 'runs/stage_a/best.pt') == U.CFG['stage_a_sha256']
    assert all(CFG[k] == U.CFG[k] for k in ['train_counts', 'formal_counts', 'effective_batch', 'epochs',
                                         'expected_updates', 'prefix_max_new_tokens', 'supervised_positions_per_example_max'])
    prepared = {split: U.rows(split) for split in ['train', 'validation', 'test']}
    for split, rr in prepared.items():
        assert C.digest(rr) == pm['prepared_sha256'][split]
        assert all(U.source_path(r).exists() for r in rr), ('Missing parent Sender cache', split)
    assert dict(Counter(r['task'] for r in prepared['train'])) == CFG['train_counts']
    assert len(prepared['train']) == 2048
    tq = {r['question_key'] for r in prepared['train']}
    vq = {r['question_key'] for r in prepared['validation']}
    eq = {r['question_key'] for r in prepared['test']}
    assert not tq & (vq | eq) and not vq & eq
    cpu = cpu_test()
    metadata = dict(config=CFG, parent_manifest_sha256=U.identity(), writer_stage_a_sha256=U.CFG['stage_a_sha256'],
                    writer_stage_b_sha256=CFG['writer_stage_b_sha256'], parent_code=pm['code'],
                    source_files={split: C.sha(PARENT / f'prepared/{split}.jsonl') for split in prepared},
                    model_configs=pm['model_configs'], reference_hashes=pm['reference_hashes'], versions=pm['versions'],
                    code={p.name: C.sha(p) for p in ROOT.glob('*.py')}, config_sha256=C.sha(ROOT / 'config.json'),
                    exact_parent_data_order_and_protocol=True, native_teacher_reader_disabled=True,
                    imported_mmlu_custom_split_sha256=pm['custom_mmlu_split_sha256'],
                    caveats=['MMLU-Pro remains the approved CUSTOM split and label-scoring extension, not untouched official test.',
                             'Reader changes only continuation computation, never transmitted prefix KV. Future decoded KV can change indirectly.',
                             'Native greedy prefixes and64-token training horizon are unchanged from parent UFD. Formal GSM8K cap remains4096.'])
    path = ROOT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == metadata, 'Immutable Reader definition changed'
    else:
        C.save(path, metadata)
    C.save(ROOT / 'prepared/completed.json', dict(status='completed', manifest_sha256=identity(), cpu_tests=cpu,
                                                counts={s: dict(Counter(r['task'] for r in rr)) for s, rr in prepared.items()},
                                                reused_parent_aligned_sender_states=True))
    status(status='prepared', phase='cpu_audit')
    C.log('READER CPU / DATA / FROZEN WRITER AUDIT PASSED')


def verify():
    m = json.loads((ROOT / 'manifest.json').read_text())
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    for name, sha in m['code'].items():
        assert C.sha(ROOT / name) == sha, ('Reader code changed after audit', name)
    assert U.identity() == m['parent_manifest_sha256']


def load_all():
    C.seed()
    tok, model = C.model('qwen')
    base, writer = C.base_from_best(), U.selected_adapter()
    reader = QOReader(model, CFG['rank'], CFG['alpha'], CFG['layers'])
    assert sum(p.numel() for p in reader.parameters()) == CFG['expected_trainable_parameters']
    reader.assert_trainable(model, base, writer)
    torch.cuda.reset_peak_memory_stats()
    return tok, model, base, writer, reader


def memory():
    return dict(peak_pytorch_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_pytorch_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                scope='This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak')


@torch.no_grad()
def fixed_state(model, base, writer, reader, row):
    with reader.mode(False):
        bk, bv = C.map_base(base, U.loaded_source(row))
        k, v = C.adapted(writer, bk, bv)
        native = U.native_body(model, row)
    return native, k.detach(), v.detach()


@torch.no_grad()
def teacher_prefix(model, tok, reader, row, native):
    path = PARENT / 'native_prefix' / (U.fname(row) + '.json')
    if path.exists():
        value = json.loads(path.read_text())
        assert value['manifest_sha256'] == U.identity()
        return value
    path = ROOT / 'native_prefix' / (U.fname(row) + '.json')
    if path.exists():
        value = json.loads(path.read_text())
        assert value['manifest_sha256'] == identity()
        return value
    with reader.mode(False):
        value = U.rollout(model, tok, row, U.clone_cache(native, model), CFG['prefix_max_new_tokens'])
    C.save(path, dict(value, manifest_sha256=identity()))
    return value


def example_loss(model, tok, base, writer, reader, row, epoch, student=False, validation=False):
    native, k, v = fixed_state(model, base, writer, reader, row)
    with torch.no_grad():
        if student:
            with reader.mode(True):
                trajectory = U.rollout(model, tok, row, U.translated_cache(model, row, native, k, v), CFG['prefix_max_new_tokens'])
        else:
            trajectory = teacher_prefix(model, tok, reader, row, native)
    pp = U.positions_for(trajectory['tokens'], row, epoch, validation)
    prefix = trajectory['tokens'][:max(pp)]
    with reader.mode(False), torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        teacher = U.suffix_logits(model, row, U.clone_cache(native, model), prefix, pp)
    with reader.mode(True), torch.amp.autocast('cuda', dtype=torch.float16):
        student_logits = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), prefix, pp)
    loss = U.full_kl(teacher, student_logits)
    assert torch.isfinite(loss)
    details = dict(task=row['task'], kl=float(loss.detach()), positions=len(pp),
                   top1_matches=int((teacher.argmax(-1) == student_logits.detach().argmax(-1)).sum()),
                   prefix_source='student' if student else 'native', rollout_tokens=len(trajectory['tokens']),
                   rollout_reached_cap=int(trajectory['reached_cap']))
    return loss, details


def smoke():
    if (ROOT / 'smoke/completed.json').exists():
        return
    tok, model, base, writer, reader = load_all()
    rr = U.rows('train', True)
    checks, teacher_before = [], {}
    for row in rr:
        native, k, v = fixed_state(model, base, writer, reader, row)
        with torch.no_grad():
            with reader.mode(False):
                expected = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), [], [0])
                teacher_before[row['id']] = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
            cache = U.translated_cache(model, row, native, k, v)
            original = [(a.clone(), b.clone()) for a, b in [U._cache_layer(cache, i) for i in range(36)]]
            with reader.mode(True):
                actual = U.suffix_logits(model, row, cache, [], [0])
            assert torch.equal(expected, actual), ('Zero-init Writer output changed', row['id'], float((expected-actual).abs().max()))
            length = len(row['frames']['qwen']['body_ids'])
            for i, (oldk, oldv) in enumerate(original):
                newk, newv = U._cache_layer(cache, i)
                assert torch.equal(oldk, newk[:, :, :length]) and torch.equal(oldv, newv[:, :, :length]), 'Transmitted KV rewritten'
            checks.append(dict(id=row['id'], task=row['task'], zero_init_bitwise_logits=True,
                               zero_init_top1_identical=True, transmitted_cache_unchanged=True))
        del original, cache, expected, actual, native, k, v
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    gradients = []
    for online in [False, True]:
        optimizer.zero_grad(set_to_none=True)
        for row in rr:
            loss, _ = example_loss(model, tok, base, writer, reader, row, 2 if online else 1, student=online)
            scaler.scale(loss / len(rr)).backward()
        scaler.unscale_(optimizer)
        norms = reader.gradient_norms()
        assert all(0 < n < float('inf') for n in norms.values()), norms
        reader.assert_trainable(model, base, writer)
        norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG['gradient_clip'])
        assert torch.isfinite(norm) and norm > 0
        scaler.step(optimizer)
        scaler.update()
        gradients.append(dict(source='online_writer_reader' if online else 'native', branches=norms))
        C.log(f'READER SMOKE gradient source={gradients[-1]["source"]} Q/O={norms}')
    with torch.no_grad(), reader.mode(False):
        for row in rr:
            native = U.native_body(model, row)
            after = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
            assert torch.equal(after, teacher_before[row['id']]), 'Native Teacher was modified by Reader training'
    assert all(m.enabled_calls > 0 for m in reader.branches.values())
    result = dict(status='completed', manifest_sha256=identity(), checks=checks, gradients=gradients,
                  trainable_parameters=sum(p.numel() for p in reader.parameters()),
                  native_teacher_bitwise_unchanged_after_updates=True, writer_and_base_frozen=True,
                  reader_only_continuation=True, memory=memory())
    C.save(ROOT / 'smoke/completed.json', result)
    status(status='smoke_passed', phase='smoke')
    C.log('ALL Q/O READER GPU SMOKE CHECKS PASSED')


@torch.no_grad()
def validate(model, tok, base, writer, reader):
    bytask = defaultdict(list)
    for row in U.rows('validation'):
        if len(bytask[row['task']]) >= CFG['monitor_validation_per_task']:
            continue
        _, details = example_loss(model, tok, base, writer, reader, row, 0, validation=True)
        if row['task'] != 'gsm8k':
            native, k, v = fixed_state(model, base, writer, reader, row)
            with reader.mode(False):
                details['qwen_native_correct'] = U.candidate_scores(model, tok, row, native)['correct']
                details['writer_only_correct'] = U.candidate_scores(model, tok, row, native, k, v)['correct']
            with reader.mode(True):
                details['writer_reader_correct'] = U.candidate_scores(model, tok, row, native, k, v)['correct']
        bytask[row['task']].append(details)
    result = {}
    for task, values in bytask.items():
        count = sum(v['positions'] for v in values)
        out = dict(n=len(values), mean_kl=sum(v['kl'] for v in values) / len(values),
                   same_prefix_next_token_agreement=sum(v['top1_matches'] for v in values) / count,
                   supervised_positions=count)
        if task != 'gsm8k':
            out['label_accuracy'] = {a: sum(v[a + '_correct'] for v in values) / len(values)
                                     for a in ['qwen_native', 'writer_only', 'writer_reader']}
        result[task] = out
    return dict(tasks=result, macro_kl=sum(v['mean_kl'] for v in result.values()) / len(result))


def save_checkpoint(path, reader, optimizer, scaler, epoch, step, position, complete, history):
    C.save_tensor(path, dict(state=reader.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, step=step, position=position, epoch_completed=complete,
                            history=history, manifest_sha256=identity()))


def train():
    folder = ROOT / 'runs/reader'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    assert (ROOT / 'smoke/completed.json').exists()
    tok, model, base, writer, reader = load_all()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    start, step, position0, history = 1, 0, 0, []
    if (folder / 'last.pt').exists():
        old = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        assert old['manifest_sha256'] == identity()
        reader.load_state_dict(old['state'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        step, history = old['step'], old['history']
        start = old['epoch'] + int(old['epoch_completed'])
        position0 = 0 if old['epoch_completed'] else old['position']
        del old
        C.log(f'READER RESUME epoch={start} step={step} position={position0}')
    with (folder / 'steps.jsonl').open('a') as stream:
        for epoch in range(start, CFG['epochs'] + 1):
            optimizer.zero_grad(set_to_none=True)
            details = []
            for position, (row, online) in enumerate(U.epoch_order(epoch), 1):
                if epoch == start and position <= position0:
                    continue
                loss, d = example_loss(model, tok, base, writer, reader, row, epoch, student=online)
                scaler.scale(loss / 8).backward()
                details.append(d)
                if position % 8:
                    continue
                scaler.unscale_(optimizer)
                reader.assert_trainable(model, base, writer)
                norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG['gradient_clip'])
                assert torch.isfinite(norm) and norm > 0
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                result = dict(epoch=epoch, step=step, position=position, mean_kl=sum(d['kl'] for d in details) / 8,
                              gradient_norm=float(norm), student_examples=sum(d['prefix_source'] == 'student' for d in details),
                              supervised_positions=sum(d['positions'] for d in details),
                              per_task={t: sum(d['kl'] for d in details if d['task'] == t) / 2 for t in CFG['train_counts']},
                              memory=memory())
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                details = []
                status(status='running', phase='reader_training', **result)
                if step % 4 == 0:
                    C.log(f'READER epoch={epoch}/4 step={step}/1024 KL={result["mean_kl"]:.6f} student={result["student_examples"]}/8')
                if step % 64 == 0 and position < 2048:
                    save_checkpoint(folder / 'last.pt', reader, optimizer, scaler, epoch, step, position, False, history)
            metrics = validate(model, tok, base, writer, reader)
            entry = dict(epoch=epoch, step=step, validation=metrics)
            history = [h for h in history if h['epoch'] != epoch] + [entry]
            best = min(history, key=lambda h: (h['validation']['macro_kl'], h['epoch']))
            C.save_tensor(folder / f'epoch{epoch}_step{step}.pt', dict(state=reader.state_dict(), epoch=epoch, step=step,
                          validation=metrics, manifest_sha256=identity(), diagnostic_only=epoch != 4))
            if entry == best:
                C.save_tensor(folder / 'best.pt', dict(state=reader.state_dict(), epoch=epoch, step=step,
                              validation=metrics, manifest_sha256=identity(), diagnostic_only=True))
            C.save(folder / 'validation.json', dict(primary=CFG['selection'], best_diagnostic=best, history=history))
            save_checkpoint(folder / 'last.pt', reader, optimizer, scaler, epoch, step, 2048, True, history)
            C.log(f'READER EPOCH {epoch} COMPLETED: {metrics}')
    assert step == 1024
    result = dict(status='completed', epochs=4, updates=step, selection=CFG['selection'],
                  checkpoint=str(folder / 'epoch4_step1024.pt'), checkpoint_sha256=C.sha(folder / 'epoch4_step1024.pt'),
                  validation=history, memory=memory(), manifest_sha256=identity())
    C.save(folder / 'completed.json', result)
    shared('reader_training_completed_evaluation_pending', result)
    C.log('READER TRAINING COMPLETED')


@torch.no_grad()
def evaluate(task):
    folder = ROOT / 'evaluation' / task
    if (folder / 'completed.json').exists():
        return
    assert (ROOT / 'runs/reader/completed.json').exists()
    tok, model, base, writer, reader = load_all()
    checkpoint = torch.load(ROOT / 'runs/reader/epoch4_step1024.pt', map_location='cpu', weights_only=True)
    assert checkpoint['manifest_sha256'] == identity() and checkpoint['step'] == 1024
    reader.load_state_dict(checkpoint['state'])
    rt = C.runtime(model, tok)
    rr = [r for r in U.rows('test') if r['task'] == task]
    for i, row in enumerate(rr, 1):
        path = folder / 'per_sample' / (U.fname(row) + '.json')
        if path.exists():
            assert json.loads(path.read_text())['reader_manifest_sha256'] == identity()
            continue
        status(status='running', phase='evaluation', task=task, sample=i, total=len(rr))
        baseline_path = PARENT / 'evaluation' / task / 'per_sample' / (U.fname(row) + '.json')
        baseline = json.loads(baseline_path.read_text())
        assert baseline['manifest_sha256'] == U.identity()
        for field in ['id', 'task', 'prompt', 'receiver_suffix', 'candidates', 'gold']:
            assert baseline[field] == row[field], ('Frozen evaluation input changed', field, row['id'])
        native, k, v = fixed_state(model, base, writer, reader, row)
        with reader.mode(True):
            if task == 'gsm8k':
                cache = U.translated_cache(model, row, native, k, v)
                text, count = C.R.generate(rt, torch.tensor([row['suffix_ids']], device='cuda'), U.CFG['gsm8k_formal_max_new_tokens'],
                                          past_key_values=cache, past_length=len(row['frames']['qwen']['body_ids']), temperature=0.0)
                prediction = C.S.extract_numeric_answer(text)
                value = dict(prediction=prediction, correct=int(prediction == C.S.normalize_number(row['gold'])), generation=text,
                             generated_tokens=count, reached_cap=int(count >= U.CFG['gsm8k_formal_max_new_tokens']))
            else:
                value = U.candidate_scores(model, tok, row, native, k, v)
        result = dict(baseline, conditions=dict(baseline['conditions'], ufd_writer_qo_reader=value),
                      reader_manifest_sha256=identity(), reused_baseline_source=str(baseline_path))
        C.save(path, result)
        if i % 8 == 0 or task == 'gsm8k':
            C.log(f'READER evaluation {task} {i}/{len(rr)}')
    results = [json.loads((folder / 'per_sample' / (U.fname(r) + '.json')).read_text()) for r in rr]
    C.write_jsonl(folder / 'per_sample.jsonl', results)
    C.save(folder / 'completed.json', dict(status='completed', n=len(results), reader_manifest_sha256=identity()))


def summarize():
    records, tasks = [], {}
    for task, expected in CFG['formal_counts'].items():
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
        paired = {}
        a = [r['conditions']['ufd_writer_qo_reader']['correct'] for r in rr]
        for arm in ['qwen_native', 'old_mcq_stage_b', 'new_ufd_stage_b']:
            b = [r['conditions'][arm]['correct'] for r in rr]
            paired[arm] = dict(both_correct=sum(x and y for x, y in zip(a, b)), reader_only=sum(x and not y for x, y in zip(a, b)),
                               other_only=sum(not x and y for x, y in zip(a, b)), both_wrong=sum(not x and not y for x, y in zip(a, b)))
        tasks[task] = dict(metrics=metrics, paired=paired)
    summary = dict(experiment=CFG['name'], status='completed', config=CFG, tasks=tasks,
                   training=json.loads((ROOT / 'runs/reader/completed.json').read_text()),
                   reader_manifest_sha256=identity(), parent_manifest_sha256=U.identity(),
                   caveats=json.loads((ROOT / 'manifest.json').read_text())['caveats'])
    C.write_jsonl(ROOT / 'evaluation/per_sample.jsonl', records)
    C.save(ROOT / 'summary.json', summary)
    shared('completed', summary)
    C.save(ROOT / 'completed.json', dict(status='completed', reader_manifest_sha256=identity()))
    status(status='completed', phase='all')
    C.log('ALL Q/O LORA READER EXPERIMENTS COMPLETED')


if __name__ == '__main__':
    try:
        action = sys.argv[1]
        if action == 'prepare':
            prepare()
        else:
            verify()
            if action == 'smoke': smoke()
            elif action == 'train': train()
            elif action == 'evaluate': evaluate(sys.argv[2])
            elif action == 'summarize': summarize()
            else: raise ValueError(action)
    except Exception as exc:
        error = dict(status='failed_or_incomplete', type=type(exc).__name__, message=str(exc))
        C.save(ROOT / 'failure.json', error)
        status(status='failed', error=error)
        shared('failed_or_incomplete', error)
        raise
