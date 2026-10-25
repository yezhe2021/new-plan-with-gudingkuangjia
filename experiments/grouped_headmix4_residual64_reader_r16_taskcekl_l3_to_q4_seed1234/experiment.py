"""Task-supervised Reader; all historical model/translator artifacts are read-only."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import json
import time
import shutil
import importlib.util
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
REFERENCE = Path(CFG['reference_reader'])
spec = importlib.util.spec_from_file_location('frozen_reference_reader', REFERENCE / 'experiment.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
C, U, torch = R.C, R.U, R.torch
sys.path.insert(0, str(ROOT))
from objectives import answer_blocks, candidate_loss, clean_solution, kl
F = torch.nn.functional


def identity():
    return C.sha(ROOT / 'manifest.json')


def save(path, value):
    C.save(path, value)


def status(**values):
    save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **values))


def mark(path, **values):
    save(path, dict(manifest_sha256=identity(), **values))


def targets():
    return {r['id']: r for r in C.read_jsonl(ROOT / 'prepared/targets.jsonl')}


def prepare():
    assert R.identity() == CFG['reference_reader_manifest_sha256']
    R.verify()
    for key in ['base_stage_a_sha256', 'frozen_stage_b_sha256', 'rank', 'alpha',
                'train_counts', 'formal_counts', 'effective_batch', 'epochs']:
        assert CFG[key] == R.CFG[key], key
    tok = C.AutoTokenizer.from_pretrained(C.CFG['models']['qwen'], local_files_only=True)
    legacy = json.loads((Path(C.CFG['legacy_root']) / 'config.json').read_text())
    raw_path = Path(legacy['datasets']['gsm8k_train'])
    raw = C.read_jsonl(raw_path)
    records, lengths, counts = [], defaultdict(list), {}
    source_hashes = {str(raw_path): C.sha(raw_path)}
    sets = {}
    for split in ['train', 'validation', 'test']:
        rr = U.rows(split)
        counts[split] = dict(Counter(r['task'] for r in rr))
        sets[split] = {r['question_key'] for r in rr}
        source_hashes[str(U.ROOT / f'prepared/{split}.jsonl')] = C.sha(U.ROOT / f'prepared/{split}.jsonl')
        assert all(U.source_path(r).exists() for r in rr)
        if split == 'test':
            continue
        for row in rr:
            rec = dict(id=row['id'], task=row['task'], split=split, row_sha256=C.digest(row))
            if row['task'] == 'gsm8k':
                assert row['id'].startswith('gsm8k_train_')
                d = raw[int(row['id'].rsplit('_', 1)[1])]
                assert C.digest(d) == row['doc_sha256'] and U.key(d['question']) == row['question_key']
                text = clean_solution(d['answer'])
                assert C.S.extract_numeric_answer(text) == C.S.normalize_number(row['gold'])
                token_ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
                assert 1 < len(token_ids) <= CFG['max_target_tokens'], (row['id'], len(token_ids))
                assert tok.eos_token_id not in token_ids[:-1]
                rec.update(text=text, token_ids=token_ids, gold=row['gold'], source_doc_sha256=C.digest(d))
                lengths[split].append(len(token_ids))
            else:
                labels = [C.R.candidate_ids(tok, label) for label in row['candidates']]
                assert all(len(v) == 1 for v in labels), 'Multi-token labels require full-label scoring'
                ids = [v[0] for v in labels]
                assert len(ids) == len(set(ids)) and 0 <= row['gold'] < len(ids)
                rec.update(label_ids=ids, gold=row['gold'])
            records.append(rec)
    assert counts['train'] == CFG['train_counts'] and counts['test'] == CFG['formal_counts']
    assert not sets['train'] & (sets['validation'] | sets['test'])
    assert not sets['validation'] & sets['test']
    assert len(records) == 2304 and len({r['id'] for r in records}) == 2304
    # Pin the old Reader predictions for the same complete test subset.
    for row in U.rows('test'):
        p = REFERENCE / 'evaluation' / row['task'] / 'per_sample' / (U.fname(row) + '.json')
        v = json.loads(p.read_text())
        assert v['reader_manifest_sha256'] == R.identity()
        assert all(v[k] == row[k] for k in ['id', 'task', 'prompt', 'gold'])
        source_hashes[str(p)] = C.sha(p)
    path = ROOT / 'prepared/targets.jsonl'
    if path.exists():
        assert C.read_jsonl(path) == records
    else:
        C.write_jsonl(path, records)
    source_hashes[str(path)] = C.sha(path)
    audit = {s: dict(n=len(v), min=min(v), max=max(v), mean=sum(v)/len(v),
                      over64=sum(x > 64 for x in v), truncated=0, all_include_final_answer_and_eos=True)
             for s, v in lengths.items()}
    meta = dict(config=CFG, config_sha256=C.sha(ROOT / 'config.json'),
                code={p.name: C.sha(p) for p in ROOT.glob('*.py')},
                reference_reader_manifest_sha256=R.identity(), source_hashes=source_hashes,
                counts=counts, target_audit=audit, stage_a_retrained=False, stage_b_retrained=False,
                answer_supervision_from_train_only=True,
                caveats=CFG['caveats'])
    p = ROOT / 'manifest.json'
    if p.exists():
        assert json.loads(p.read_text()) == meta
    else:
        save(p, meta)
    mark(ROOT / 'prepared/completed.json', status='completed', counts=counts, target_audit=audit)
    status(status='prepared', phase='data_audit', target_audit=audit)
    C.log('FROZEN A+B / REFERENCE ANSWERS / SPLITS VERIFIED ' + json.dumps(audit))


def verify():
    m = json.loads((ROOT / 'manifest.json').read_text())
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    assert R.identity() == m['reference_reader_manifest_sha256']
    R.verify()
    for name, digest in m['code'].items():
        assert C.sha(ROOT / name) == digest, name
    for path, digest in m['source_hashes'].items():
        assert C.sha(Path(path)) == digest, path


def block_logits(model, cache, block, length):
    inputs = torch.tensor([block['inputs']], device='cuda')
    output = model.model(inputs, past_key_values=cache,
                         attention_mask=torch.ones((1, length + inputs.shape[1]), device='cuda', dtype=torch.long),
                         position_ids=torch.arange(length, length + inputs.shape[1], device='cuda')[None],
                         use_cache=True)
    hidden = output.last_hidden_state[0, block['offset']:]
    assert hidden.shape[0] == len(block['labels'])
    return model.lm_head(hidden).float(), output.past_key_values


def loss_example(model, tok, base, reader, row, target, backward=None, divisor=1):
    """Backward per bounded block; return detached, example-normalized metrics."""
    assert target['row_sha256'] == C.digest(row)
    native, k, v = R.fixed_state(model, base, reader, row)
    if row['task'] != 'gsm8k':
        with torch.no_grad(), reader.mode(False), torch.amp.autocast('cuda', dtype=torch.float16):
            teacher = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0])
        with reader.mode(True), torch.amp.autocast('cuda', dtype=torch.float16):
            student = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), [], [0])
            loss, ce, auxiliary, pred = candidate_loss(teacher, student, target['label_ids'], target['gold'], CFG['kl_weight'])
            assert torch.isfinite(loss)
            if backward is not None:
                backward(loss / divisor)
        return dict(task=row['task'], loss=float(loss.detach()), ce=float(ce.detach()), kl=float(auxiliary.detach()),
                    positions=1, blocks=1, correct=int(pred == target['gold']))
    tcache = U.clone_cache(native, model)
    scache = U.translated_cache(model, row, native, k, v)
    del native, k, v
    result = dict(task=row['task'], loss=0., ce=0., kl=0., positions=0, blocks=0)
    length = len(row['frames']['qwen']['body_ids'])
    for block in answer_blocks(row['suffix_ids'], target['token_ids'], CFG['block_tokens']):
        with torch.no_grad(), reader.mode(False), torch.amp.autocast('cuda', dtype=torch.float16):
            teacher, tcache = block_logits(model, tcache, block, length)
        with reader.mode(True), torch.amp.autocast('cuda', dtype=torch.float16):
            student, scache = block_logits(model, scache, block, length)
            labels = torch.tensor(block['labels'], device='cuda')
            ce = F.cross_entropy(student.float(), labels)
            auxiliary = kl(teacher, student)
            loss = ce + CFG['kl_weight'] * auxiliary
            assert torch.isfinite(loss), row['id']
            # Keep Reader enabled while any backward/recomputation executes.
            if backward is not None:
                backward(loss * block['weight'] / divisor)
        for name, value in [('loss', loss), ('ce', ce), ('kl', auxiliary)]:
            result[name] += float(value.detach()) * block['weight']
        result['positions'] += len(block['labels'])
        result['blocks'] += 1
        # Forward context is retained, gradients across earlier blocks are truncated.
        scache = U.clone_cache(scache, model)
        length += len(block['inputs'])
        del loss, ce, auxiliary, student, teacher, labels
    assert result['positions'] == len(target['token_ids'])
    return result


@torch.no_grad()
def validate(model, tok, base, reader, target_map):
    bytask = defaultdict(list)
    for row in U.rows('validation'):
        if len(bytask[row['task']]) < CFG['monitor_validation_per_task']:
            bytask[row['task']].append(loss_example(model, tok, base, reader, row, target_map[row['id']]))
    tasks = {}
    for task, vv in bytask.items():
        value = dict(n=len(vv), **{k: sum(v[k] for v in vv)/len(vv) for k in ['loss','ce','kl']},
                     supervised_positions=sum(v['positions'] for v in vv))
        if task != 'gsm8k':
            value['accuracy'] = sum(v['correct'] for v in vv)/len(vv)
        tasks[task] = value
    return dict(tasks=tasks, macro_loss=sum(v['loss'] for v in tasks.values())/len(tasks))


def grad_step(reader, model, base, optimizer, scaler):
    scaler.unscale_(optimizer)
    reader.assert_trainable(model, base)
    norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG['gradient_clip'])
    assert torch.isfinite(norm) and norm > 0
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


@torch.no_grad()
def block_parity(model, tok, base, reader, row, target):
    # Check chunked teacher-forced evaluation against one full causal forward.
    native, k, v = R.fixed_state(model, base, reader, row)
    ids = target['token_ids'][:97]
    with reader.mode(True), torch.amp.autocast('cuda', dtype=torch.float16):
        full = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), ids[:-1], list(range(len(ids))))
        cache = U.translated_cache(model, row, native, k, v)
        length = len(row['frames']['qwen']['body_ids'])
        chunks = []
        for block in answer_blocks(row['suffix_ids'], ids, CFG['block_tokens']):
            out, cache = block_logits(model, cache, block, length)
            chunks.append(out)
            length += len(block['inputs'])
        actual = torch.cat(chunks)
    divergence = float(kl(full, actual))
    assert divergence < 1e-4, divergence
    # Same target at every position; this also checks the boundary shift.
    labels = torch.tensor(ids, device='cuda')
    ce_error = abs(float(F.cross_entropy(full, labels) - F.cross_entropy(actual, labels)))
    assert ce_error < .02, ce_error
    return dict(tokens=len(ids), kl=divergence, ce_error=ce_error,
                max_logit_error=float((full-actual).abs().max()))


def smoke():
    if (ROOT / 'smoke/completed.json').exists():
        assert json.loads((ROOT / 'smoke/completed.json').read_text())['manifest_sha256'] == identity()
        return
    tok, model, base, reader = R.load_all()
    target_map = targets()
    rr = U.rows('train', True)
    longest = max((r for r in U.rows('train') if r['task'] == 'gsm8k'), key=lambda r: len(target_map[r['id']]['token_ids']))
    rr = [longest if r['task'] == 'gsm8k' else r for r in rr]
    frozen = R.base_digest(base)
    teacher_before = {}
    for row in rr:
        native, k, v = R.fixed_state(model, base, reader, row)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            with reader.mode(False):
                teacher_before[row['id']] = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
                expected = U.suffix_logits(model, row, U.translated_cache(model,row,native,k,v), [], [0])
            with reader.mode(True):
                actual = U.suffix_logits(model, row, U.translated_cache(model,row,native,k,v), [], [0])
            assert torch.equal(actual, expected), 'Fresh Reader changed A+B output'
        del native, k, v, expected, actual
    parity_before = block_parity(model, tok, base, reader, longest, target_map[longest['id']])
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    details = []
    for row in rr:
        d = loss_example(model, tok, base, reader, row, target_map[row['id']],
                         backward=lambda value: scaler.scale(value).backward())
        reader.assert_trainable(model, base)
        assert all(m.b.grad is not None and torch.isfinite(m.b.grad).all() and m.b.grad.abs().sum() > 0
                   for m in reader.branches.values()), ('Disconnected Reader branch', row['id'])
        d['gradient_norm'] = grad_step(reader, model, base, optimizer, scaler)
        details.append(d)
        C.log('TASK READER SMOKE ' + json.dumps(d))
    parity_after = block_parity(model, tok, base, reader, longest, target_map[longest['id']])
    with torch.no_grad(), reader.mode(False), torch.amp.autocast('cuda', dtype=torch.float16):
        for row in rr:
            native = U.native_body(model, row)
            after = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
            assert torch.equal(after, teacher_before[row['id']]), 'Teacher changed'
    assert R.base_digest(base) == frozen
    mark(ROOT / 'smoke/completed.json', status='completed', task_checks=details,
         block_parity_before=parity_before, block_parity_after=parity_after,
         all72_reader_branches_receive_gradients=True, frozen_ab_unchanged=True,
         teacher_bitwise_unchanged=True, fresh_reader_zero_init_bitwise=True,
         longest_training_solution_tokens=len(target_map[longest['id']]['token_ids']), memory=R.memory())
    status(status='smoke_passed', phase='smoke')


def save_checkpoint(path, reader, optimizer, scaler, epoch, step, position, complete, history):
    assert shutil.disk_usage(ROOT).free/2**30 > CFG['disk_reserve_gib']
    C.save_tensor(path, dict(state=reader.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, step=step, position=position, epoch_completed=complete, history=history,
                            manifest_sha256=identity(), state_scope='reader_lora_only',
                            frozen_stage_a_sha256=CFG['base_stage_a_sha256'], frozen_stage_b_sha256=CFG['frozen_stage_b_sha256']))


def train():
    folder = ROOT / 'runs/reader'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        assert json.loads((folder/'completed.json').read_text())['manifest_sha256'] == identity()
        return
    assert json.loads((ROOT/'smoke/completed.json').read_text())['manifest_sha256'] == identity()
    tok, model, base, reader = R.load_all()
    target_map, frozen = targets(), R.base_digest(base)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    start, step, position0, history = 1, 0, 0, []
    if (folder/'last.pt').exists():
        old = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
        assert old['manifest_sha256'] == identity()
        assert old['frozen_stage_a_sha256'] == CFG['base_stage_a_sha256'] and old['frozen_stage_b_sha256'] == CFG['frozen_stage_b_sha256']
        reader.load_state_dict(old['state'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        step, history = old['step'], old['history']
        start = old['epoch'] + int(old['epoch_completed'])
        position0 = 0 if old['epoch_completed'] else old['position']
        del old
    with (folder/'steps.jsonl').open('a') as stream:
        for epoch in range(start, CFG['epochs']+1):
            optimizer.zero_grad(set_to_none=True)
            details, tick = [], time.monotonic()
            for position, (row, _) in enumerate(U.epoch_order(epoch), 1):
                if epoch == start and position <= position0:
                    continue
                d = loss_example(model, tok, base, reader, row, target_map[row['id']],
                                 backward=lambda value: scaler.scale(value).backward(), divisor=CFG['effective_batch'])
                details.append(d)
                if position % CFG['effective_batch']:
                    continue
                norm = grad_step(reader, model, base, optimizer, scaler)
                step += 1
                result = dict(epoch=epoch, step=step, position=position,
                              **{k: sum(d[k] for d in details)/len(details) for k in ['loss','ce','kl']},
                              supervised_positions=sum(d['positions'] for d in details), gradient_norm=norm,
                              seconds=time.monotonic()-tick,
                              per_task={t: {k: sum(d[k] for d in details if d['task']==t)/2 for k in ['loss','ce','kl']}
                                        for t in CFG['train_counts']}, memory=R.memory())
                stream.write(json.dumps(result)+'\n'); stream.flush()
                status(status='running', phase='reader_training', **result)
                if step <= 4 or step % 4 == 0:
                    C.log(f'TASK READER epoch={epoch}/4 step={step}/1024 loss={result["loss"]:.5f} CE={result["ce"]:.5f} KL={result["kl"]:.5f} seconds={result["seconds"]:.2f}')
                details, tick = [], time.monotonic()
                if step % CFG['checkpoint_every_steps']==0 and position < 2048:
                    save_checkpoint(folder/'last.pt', reader, optimizer, scaler, epoch, step, position, False, history)
            assert R.base_digest(base) == frozen
            metrics = validate(model, tok, base, reader, target_map)
            entry = dict(epoch=epoch, step=step, validation=metrics)
            history = [v for v in history if v['epoch'] != epoch] + [entry]
            C.save_tensor(folder/f'epoch{epoch}_step{step}.pt', dict(state=reader.state_dict(), manifest_sha256=identity(),
                          epoch=epoch, step=step, diagnostic_only=epoch!=4, state_scope='reader_lora_only',
                          frozen_stage_a_sha256=CFG['base_stage_a_sha256'], frozen_stage_b_sha256=CFG['frozen_stage_b_sha256']))
            mark(folder/'validation.json', history=history, selection=CFG['selection'])
            save_checkpoint(folder/'last.pt', reader, optimizer, scaler, epoch, step, position, True, history)
            C.log('TASK READER EPOCH COMPLETED '+json.dumps(entry))
    assert step == CFG['expected_updates'] and R.base_digest(base)==frozen
    mark(folder/'completed.json', status='completed', updates=step, checkpoint=str(folder/'epoch4_step1024.pt'),
         checkpoint_sha256=C.sha(folder/'epoch4_step1024.pt'), history=history, frozen_ab_unchanged=True, memory=R.memory())


@torch.no_grad()
def evaluate(task):
    folder = ROOT/'evaluation'/task
    if (folder/'completed.json').exists():
        assert json.loads((folder/'completed.json').read_text())['manifest_sha256']==identity()
        return
    assert json.loads((ROOT/'runs/reader/completed.json').read_text())['manifest_sha256']==identity()
    tok, model, base, reader = R.load_all()
    record = torch.load(ROOT/'runs/reader/epoch4_step1024.pt', map_location='cpu', weights_only=True)
    assert record['manifest_sha256']==identity() and record['step']==1024
    assert record['frozen_stage_a_sha256']==CFG['base_stage_a_sha256'] and record['frozen_stage_b_sha256']==CFG['frozen_stage_b_sha256']
    reader.load_state_dict(record['state'])
    rt = C.runtime(model, tok)
    rr = [r for r in U.rows('test') if r['task']==task]
    for i, row in enumerate(rr, 1):
        path = folder/'per_sample'/(U.fname(row)+'.json')
        if path.exists():
            assert json.loads(path.read_text())['manifest_sha256']==identity()
            continue
        prior_path = REFERENCE/'evaluation'/task/'per_sample'/(U.fname(row)+'.json')
        prior = json.loads(prior_path.read_text())
        assert all(prior[k]==row[k] for k in ['id','task','prompt','gold'])
        conditions = dict(prior['conditions'])
        native, k, v = R.fixed_state(model, base, reader, row)
        with reader.mode(True):
            if task=='gsm8k':
                cache = U.translated_cache(model,row,native,k,v)
                text, count = C.R.generate(rt, torch.tensor([row['suffix_ids']],device='cuda'),
                    U.CFG['gsm8k_formal_max_new_tokens'], past_key_values=cache,
                    past_length=len(row['frames']['qwen']['body_ids']),temperature=0.)
                prediction = C.S.extract_numeric_answer(text)
                value = dict(prediction=prediction, correct=int(prediction==C.S.normalize_number(row['gold'])),
                             generation=text, generated_tokens=count, reached_cap=int(count>=U.CFG['gsm8k_formal_max_new_tokens']))
            else:
                value = U.candidate_scores(model,tok,row,native,k,v)
        conditions['task_supervised_reader'] = value
        mark(path, id=row['id'], task=task, prompt=row['prompt'], gold=row['gold'], conditions=conditions,
             baseline_source=str(prior_path))
        status(status='running', phase='evaluation', task=task, sample=i, total=len(rr))
        if i%8==0 or task=='gsm8k': C.log(f'TASK READER evaluation {task} {i}/{len(rr)}')
    values = [json.loads((folder/'per_sample'/(U.fname(r)+'.json')).read_text()) for r in rr]
    C.write_jsonl(folder/'per_sample.jsonl',values)
    mark(folder/'completed.json',status='completed',n=len(values))


def summarize():
    records, tasks = [], {}
    for task, expected in CFG['formal_counts'].items():
        assert (ROOT/f'evaluation/{task}/completed.json').exists()
        rr = C.read_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl')
        assert len(rr)==expected and all(r['manifest_sha256']==identity() for r in rr)
        records.extend(rr)
        metrics, paired = {}, {}
        a = [r['conditions']['task_supervised_reader']['correct'] for r in rr]
        for arm in rr[0]['conditions']:
            vv = [r['conditions'][arm] for r in rr]
            metrics[arm] = dict(n=expected, correct=sum(v['correct'] for v in vv), accuracy=sum(v['correct'] for v in vv)/expected)
            if task=='gsm8k':
                metrics[arm].update(mean_generated_tokens=sum(v['generated_tokens'] for v in vv)/expected,
                                    cap_count=sum(v['reached_cap'] for v in vv))
            if arm in ['qwen_native','grouped_residual','grouped_residual_reader']:
                b = [v['correct'] for v in vv]
                paired[arm] = dict(both_correct=sum(x and y for x,y in zip(a,b)), new_only=sum(x and not y for x,y in zip(a,b)),
                                   baseline_only=sum(not x and y for x,y in zip(a,b)),both_wrong=sum(not x and not y for x,y in zip(a,b)))
        tasks[task] = dict(metrics=metrics,paired=paired)
    assert len(records)==544 and len({r['id'] for r in records})==544
    C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records)
    mark(ROOT/'summary.json',status='completed',config=CFG,tasks=tasks,
         training=json.loads((ROOT/'runs/reader/completed.json').read_text()),caveats=CFG['caveats'])
    mark(ROOT/'completed.json',status='completed')
    status(status='completed',phase='all')
    C.log('TASK-SUPERVISED READER TRAINING AND ALL EVALUATIONS COMPLETED')


if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare': prepare()
        else:
            verify()
            if action=='verify': C.log('TASK READER MANIFEST VERIFIED')
            elif action=='smoke': smoke()
            elif action=='train': train()
            elif action=='evaluate': evaluate(sys.argv[2])
            elif action=='summarize': summarize()
            else: raise ValueError(action)
    except Exception as exc:
        save(ROOT/'failure.json',dict(type=type(exc).__name__,message=str(exc),time=time.strftime('%F %T')))
        status(status='failed',phase=sys.argv[1],error=str(exc))
        raise
