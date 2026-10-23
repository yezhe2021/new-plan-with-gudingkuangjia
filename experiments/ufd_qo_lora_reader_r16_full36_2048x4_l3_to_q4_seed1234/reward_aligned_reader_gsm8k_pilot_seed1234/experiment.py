"""Single-arm OPDVR-inspired Reader pilot; explicit64-token truncated BPTT."""
import os
for env in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[env] = '1'
import sys
import json
import time
import random
import shutil
import fcntl
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
PARENT = Path(CFG['reader_parent'])
spec = importlib.util.spec_from_file_location('existing_reader_pilot', PARENT / 'experiment.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
U, C, torch = P.U, P.C, P.torch
import torch.nn.functional as F


def identity():
    return C.sha(ROOT / 'manifest.json')


def status(**kw):
    C.save(ROOT / 'status.json', dict(time=time.strftime('%F %T'), **kw))


def shared(state, results):
    path = PARENT.parent / 'EXPERIMENT_RESULTS.md'
    record = dict(experiment=CFG['name'], timestamp=time.strftime('%F %T'), status=state,
                  seed=CFG['seed'], config=CFG, checkpoint_selection=CFG['selection'], results=results,
                  raw_results=str(ROOT / 'evaluation/per_sample.jsonl'), log=str(ROOT / 'logs/pipeline.log'))
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write('\n\n## ' + CFG['name'] + '\n\n```json\n' + json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + '\n```\n')
        stream.flush()
        os.fsync(stream.fileno())
    C.save(ROOT / 'shared_summary_updated.json', dict(path=str(path), status=state))


def initial_path():
    return PARENT / 'runs/reader/epoch4_step1024.pt'


def audit_parents():
    P.verify()
    assert P.identity() == CFG['reader_parent_manifest_sha256']
    evidence = {}
    for root in [PARENT, U.ROOT, C.ROOT]:
        m = json.loads((root / 'manifest.json').read_text())
        for name, expected in m['code'].items():
            actual = C.sha(root / name)
            assert actual == expected, ('Unarchived parent code change', root, name)
            evidence[str(root / name)] = actual
        if 'config_sha256' in m:
            assert C.sha(root / 'config.json') == m['config_sha256']
    for name, expected in C.CFG['reference_hashes'].items():
        path = Path(C.CFG['hf_root']) / 'heterofold' / name
        assert C.sha(path) == expected
        evidence[str(path)] = expected
    for path, expected in [(initial_path(), CFG['reader_initial_sha256']),
                           (C.ROOT / 'runs/stage_a/best.pt', CFG['writer_stage_a_sha256']),
                           (U.ROOT / 'runs/stage_b/last.pt', CFG['writer_stage_b_sha256'])]:
        assert C.sha(path) == expected, ('Checkpoint identity changed', path)
    return evidence


def rows(split):
    return C.read_jsonl(ROOT / f'prepared/{split}.jsonl')


def prepare():
    evidence = audit_parents()
    train = [r for r in U.rows('train') if r['task'] == 'gsm8k']
    dev = [r for r in U.rows('validation') if r['task'] == 'gsm8k']
    oldtest = [r for r in U.rows('test') if r['task'] == 'gsm8k']
    assert len(train) == CFG['train_questions'] and len(oldtest) == CFG['historical_questions']
    legacy = json.loads((Path(C.CFG['legacy_root']) / 'config.json').read_text())
    dataset = Path(legacy['datasets']['gsm8k_test'])
    docs = C.read_jsonl(dataset)
    lt, qt = [U.AutoTokenizer.from_pretrained(C.CFG['models'][f], local_files_only=True) for f in ['llama', 'qwen']]
    test = []
    for index, doc in enumerate(docs[:CFG['evaluation_questions']]):
        if index < len(oldtest):
            row = dict(oldtest[index])
            assert row['question_key'] == U.key(doc['question'])
            assert str(row['gold']) == str(C.S.format_item('gsm8k', doc)[2])
        else:
            row = U.make_row('gsm8k_test_' + str(index), 'gsm8k', doc, lt, qt)
            row.update(source_ref=str(ROOT / 'source' / (U.fname(row) + '.pt')), source_identity='PILOT_MANIFEST')
            if CFG.get('source_cache_reuse_manifest_sha256'):
                assert U.source_path(row).exists(), 'Previously captured Sender cache missing; stop repair'
                cached = U.loaded_source(source_row(row))
                del cached
        row['evaluation_subset'] = 'historical32' if index < len(oldtest) else 'new96'
        test.append(row)
    tq, vq, eq = [{r['question_key'] for r in rr} for rr in [train, dev, test]]
    assert len(tq) == len(train) and len(eq) == CFG['evaluation_questions']
    assert not tq & (vq | eq) and not vq & eq
    upstream_train = {U.key(U.qdoc(r['doc'])) for r in C.rows('train')}
    assert not upstream_train & eq
    for split, rr in [('train', train), ('validation', dev), ('test', test)]:
        for row in rr:
            if row.get('source_identity') != 'PILOT_MANIFEST':
                path = U.source_path(row)
                assert path.exists(), ('Existing Sender cache missing', path)
                # Make every reused path explicit; source loader retains parent identity.
                row['source_ref'] = str(path)
                row.setdefault('source_identity', U.identity())
        C.write_jsonl(ROOT / f'prepared/{split}.jsonl', rr)
    estimate = sum(28 * len(r['source_indices']) * 8 * 128 * 4 for r in test if r.get('source_identity') == 'PILOT_MANIFEST') / 2**30
    assert shutil.disk_usage(ROOT).free / 2**30 > estimate + CFG['disk_reserve_gib']
    data = dict(config=CFG, config_sha256=C.sha(ROOT / 'config.json'), parent_code=evidence,
                code={p.name: C.sha(p) for p in ROOT.glob('*.py')},
                parent_manifest_sha256=P.identity(), ufd_manifest_sha256=U.identity(),
                prepared_sha256={s: C.sha(ROOT / f'prepared/{s}.jsonl') for s in ['train', 'validation', 'test']},
                source_test_dataset=dict(path=str(dataset), sha256=C.sha(dataset)),
                counts=dict(train=len(train), protected_validation=len(dev), test=len(test), new_test=96, historical_test=32),
                source_cache_estimate_new_gib=estimate,
                caveats=['Only one trained arm; gain cannot be attributed solely to reward gating.',
                         '64-token detached continuation boundaries approximate full policy gradients.',
                         'Test128 includes historical32; report new96 separately.',
                         'No MCQ retention claim or equal-budget HeteroFold comparison.'])
    path = ROOT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == data, 'Immutable experiment changed'
    else:
        C.save(path, data)
    C.save(ROOT / 'prepared/completed.json', dict(status='completed', manifest_sha256=identity(), counts=data['counts']))
    status(status='prepared', phase='cpu_audit')
    C.log(f'PILOT PREPARED:512train/64protected-dev/128test; new Sender estimate {estimate:.2f} GiB')


def verify():
    m = json.loads((ROOT / 'manifest.json').read_text())
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    for name, expected in m['code'].items():
        assert C.sha(ROOT / name) == expected, ('Pilot code changed', name)
    for path, expected in m['parent_code'].items():
        assert C.sha(path) == expected, ('Parent code changed', path)
    for split, expected in m['prepared_sha256'].items():
        assert C.sha(ROOT / f'prepared/{split}.jsonl') == expected
    assert P.identity() == m['parent_manifest_sha256'] and U.identity() == m['ufd_manifest_sha256']


def source_row(row):
    if row.get('source_identity') == 'PILOT_MANIFEST':
        cache_identity = CFG.get('source_cache_reuse_manifest_sha256') or identity()
        return dict(row, source_identity=cache_identity)
    return row


@torch.no_grad()
def capture():
    todo = [source_row(r) for r in rows('test') if not U.source_path(r).exists()]
    if not todo:
        return
    tok, model = C.model('llama')
    for i, row in enumerate(todo, 1):
        assert shutil.disk_usage(ROOT).free / 2**30 > CFG['disk_reserve_gib']
        ids = row['frames']['llama']['body_ids']
        k, v, _ = C.capture(model, ids, len(ids), [0])
        ix = row['source_indices']
        C.save_tensor(U.source_path(row), dict(key=k[:, ix].contiguous(), value=v[:, ix].contiguous(),
                                             id=row['id'], manifest_sha256=identity()))
        status(status='running', phase='new_test_sender_capture', sample=i, total=len(todo))
        if i % 8 == 0 or i == len(todo):
            C.log(f'PILOT new-test Sender capture {i}/{len(todo)}')


def load_all():
    tok, model, base, writer, reader = P.load_all()
    checkpoint = torch.load(initial_path(), map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 1024 and checkpoint['manifest_sha256'] == P.identity()
    reader.load_state_dict(checkpoint['state'])
    reader.assert_trainable(model, base, writer)
    return tok, model, base, writer, reader


def forward(model, row, cache, ids, offset, grad=False):
    length = len(row['frames']['qwen']['body_ids']) + offset
    tensor = torch.tensor([ids], device='cuda')
    # Keep eval mode (no dropout) for sampling, teacher and differentiable replay.
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model.model(tensor, past_key_values=cache,
                          attention_mask=torch.ones(1, length + len(ids), device='cuda', dtype=torch.long),
                          position_ids=torch.arange(length, length + len(ids), device='cuda')[None], use_cache=True)
    return out


def stops_for(model, tok):
    values = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}
    eos = getattr(model.generation_config, 'eos_token_id', None)
    values.update(eos if isinstance(eos, list) else [eos])
    values.discard(None)
    return values


@torch.no_grad()
def sample(model, tok, reader, row, native, k, v, cap, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokens, logps, stopped = [], [], False
    with reader.mode(True):
        out = forward(model, row, U.translated_cache(model, row, native, k, v), row['suffix_ids'], 0)
        offset = len(row['suffix_ids'])
        stops = stops_for(model, tok)
        for i in range(cap):
            lp = model.lm_head(out.last_hidden_state[0, -1]).float().log_softmax(-1)
            token = int(torch.multinomial(lp.exp(), 1))
            tokens.append(token)
            logps.append(float(lp[token]))
            if token in stops:
                stopped = True
                break
            if i + 1 < cap:
                out = forward(model, row, out.past_key_values, [token], offset)
                offset += 1
    text = tok.decode(tokens, skip_special_tokens=True)
    prediction = C.S.extract_numeric_answer(text)
    return dict(tokens=tokens, rollout_logps=logps, text=text, prediction=prediction,
                correct=int(prediction == C.S.normalize_number(row['gold'])) if stopped else None,
                completed=stopped, reached_cap=not stopped, generated_tokens=len(tokens), seed=seed)


def gate(teacher_logp, student_logp, correct):
    difference = (teacher_logp.detach() - student_logp.detach())
    return difference.clamp_min(0) if correct else difference.clamp_max(0)


def cpu_gate_test():
    for correct, teacher_p, expected_sign in [(True, .8, -1), (False, .2, 1)]:
        logits = torch.tensor([0., 0.], requires_grad=True)
        teacher = torch.tensor(teacher_p, requires_grad=True).log()
        lp = logits.log_softmax(0)[0]
        weight = gate(teacher, lp, correct)
        assert not weight.requires_grad
        (-weight * lp).backward()
        assert expected_sign * float(logits.grad[0]) > 0
    assert float(gate(torch.tensor(-2.), torch.tensor(-1.), True)) == 0
    assert float(gate(torch.tensor(-1.), torch.tensor(-2.), False)) == 0
    return dict(correct_token_pushed_up=True, incorrect_token_pushed_down=True, conflicting_updates_zero=True, gate_detached=True)


def replay(model, reader, row, native, k, v, trajectory, scaler, denominator, audit_cache=False):
    assert trajectory['completed'] and trajectory['correct'] in [0, 1]
    tokens = trajectory['tokens']
    total, loss_sum, active, positive, negative, abs_mass = len(tokens), 0., 0, 0, 0, 0.
    teacher_cache = U.clone_cache(native, model)
    student_cache = U.translated_cache(model, row, native, k, v)
    prefix_length = len(row['frames']['qwen']['body_ids'])
    snapshot = [(a.detach().clone(), b.detach().clone()) for a, b in
                [U._cache_layer(student_cache, i) for i in range(36)]] if audit_cache else []
    offset, differences = 0, []
    for start in range(0, total, CFG['tbptt_tokens']):
        end = min(start + CFG['tbptt_tokens'], total)
        ids = row['suffix_ids'] + tokens[:end - 1] if start == 0 else tokens[start - 1:end - 1]
        target = torch.tensor(tokens[start:end], device='cuda')
        with reader.mode(False), torch.no_grad():
            teacher = forward(model, row, teacher_cache, ids, offset)
            teacher_cache = teacher.past_key_values
            th = teacher.last_hidden_state[0, -(end - start):]
            teacher_lp = -F.cross_entropy(model.lm_head(th).float(), target, reduction='none')
        with reader.mode(True):
            student = forward(model, row, student_cache, ids, offset, grad=True)
            sh = student.last_hidden_state[0, -(end - start):]
            student_lp = -F.cross_entropy(model.lm_head(sh).float(), target, reduction='none')
            weights = gate(teacher_lp, student_lp, trajectory['correct'])
            loss = -(weights * student_lp).sum() / total / denominator
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            # Every block's gradients reach Q/O LoRA, but earlier continuation states
            # are constants in subsequent blocks (explicit truncated BPTT).
            student_cache = U.clone_cache(student.past_key_values, model)
        if audit_cache:
            for i, (oldk, oldv) in enumerate(snapshot):
                newk, newv = U._cache_layer(student_cache, i)
                assert torch.equal(oldk, newk[:, :, :prefix_length]), 'Transmitted K modified'
                assert torch.equal(oldv, newv[:, :, :prefix_length]), 'Transmitted V modified'
        loss_sum += float(loss.detach()) * denominator
        active += int((weights != 0).sum())
        positive += int((weights > 0).sum())
        negative += int((weights < 0).sum())
        abs_mass += float(weights.abs().sum())
        rollout_lp = torch.tensor(trajectory['rollout_logps'][start:end], device='cuda')
        differences.extend((student_lp.detach() - rollout_lp).abs().cpu().tolist())
        offset += len(ids)
        del teacher, student, teacher_lp, student_lp, th, sh, weights, loss
    return dict(surrogate_loss=loss_sum, active_tokens=active, positive_tokens=positive,
                negative_tokens=negative, abs_gate_mass=abs_mass, supervised_tokens=total,
                replay_logp_mean_abs_error=sum(differences) / total, replay_logp_max_abs_error=max(differences))


def smoke():
    if (ROOT / 'smoke/completed.json').exists():
        return
    checks = cpu_gate_test()
    tok, model, base, writer, reader = load_all()
    initial = reader.state_dict()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    cap, records, usable = CFG['initial_train_max_new_tokens'], [], []
    for index, row in enumerate(rows('train')[:CFG['smoke_questions']]):
        native, k, v = P.fixed_state(model, base, writer, reader, row)
        with reader.mode(False), torch.no_grad():
            teacher_before = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
        for replicate in range(CFG['rollouts_per_question']):
            trajectory = sample(model, tok, reader, row, native, k, v, cap, 900000 + index * 2 + replicate)
            if trajectory['reached_cap']:
                cap = CFG['extended_train_max_new_tokens']
                trajectory = sample(model, tok, reader, row, native, k, v, cap, 900000 + index * 2 + replicate)
            record = dict(id=row['id'], **trajectory)
            if trajectory['completed']:
                optimizer.zero_grad(set_to_none=True)
                details = replay(model, reader, row, native, k, v, trajectory, scaler, 1, audit_cache=True)
                scaler.unscale_(optimizer)
                reader.assert_trainable(model, base, writer)
                norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG['gradient_clip'])
                assert torch.isfinite(norm) and norm > 0
                assert details['replay_logp_mean_abs_error'] < .1, details
                record.update(details, gradient_norm=float(norm), gradient_norms=reader.gradient_norms())
                # Smoke update verifies optimizer isolation; the separate training
                # process reloads the unmodified initial Reader checkpoint.
                scaler.step(optimizer)
                scaler.update()
                with reader.mode(False), torch.no_grad():
                    after = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
                assert torch.equal(teacher_before, after), 'Native teacher changed'
                reader.load_state_dict(initial)
                usable.append(row['id'])
            records.append(record)
            C.log(f'PILOT SMOKE row={index + 1} replicate={replicate + 1} tokens={trajectory["generated_tokens"]} complete={trajectory["completed"]}')
        del native, k, v
    assert usable, 'All smoke responses capped; stop before formal training'
    assert all(not m.base.weight.requires_grad for m in reader.branches.values())
    C.write_jsonl(ROOT / 'smoke/trajectories.jsonl', records)
    result = dict(status='completed', checks=checks, train_max_new_tokens=cap,
                  completed_trajectories=len(usable), attempted_trajectories=len(records),
                  native_teacher_unchanged=True, reader_only_gradients=True, transmitted_prefix_unchanged=True,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  gradient_approximation=CFG['gradient_approximation'], manifest_sha256=identity())
    C.save(ROOT / 'smoke/completed.json', result)
    status(status='smoke_passed', phase='smoke', **{k: result[k] for k in ['train_max_new_tokens', 'completed_trajectories']})
    C.log('ALL PILOT SMOKE CHECKS PASSED')


def step_rows(step):
    rr = rows('train')
    per_epoch = len(rr) // CFG['questions_per_update']
    epoch, offset = divmod(step - 1, per_epoch)
    random.Random(CFG['seed'] + epoch).shuffle(rr)
    start = offset * CFG['questions_per_update']
    return rr[start:start + CFG['questions_per_update']]


def save_checkpoint(path, reader, optimizer, scaler, step):
    C.save_tensor(path, dict(state=reader.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            step=step, manifest_sha256=identity(), selection=CFG['selection']))


def train():
    assert (ROOT / 'smoke/completed.json').exists()
    if (ROOT / 'runs/reader/completed.json').exists():
        return
    tok, model, base, writer, reader = load_all()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    checkpoint_path = ROOT / 'runs/reader/last.pt'
    start = 1
    if checkpoint_path.exists():
        old = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert old['manifest_sha256'] == identity()
        reader.load_state_dict(old['state'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        start = old['step'] + 1
    cap = json.loads((ROOT / 'smoke/completed.json').read_text())['train_max_new_tokens']
    folder = ROOT / 'training'
    folder.mkdir(parents=True, exist_ok=True)
    denominator = CFG['questions_per_update'] * CFG['rollouts_per_question']
    for step in range(start, CFG['updates'] + 1):
        optimizer.zero_grad(set_to_none=True)
        records = []
        status(status='running', phase='reader_training', step=step, total=CFG['updates'], question=0)
        for qi, row in enumerate(step_rows(step), 1):
            native, k, v = P.fixed_state(model, base, writer, reader, row)
            for replicate in range(CFG['rollouts_per_question']):
                seed = int(C.digest(dict(seed=CFG['seed'], step=step, id=row['id'], replicate=replicate))[:8], 16)
                trajectory = sample(model, tok, reader, row, native, k, v, cap, seed)
                details = replay(model, reader, row, native, k, v, trajectory, scaler, denominator) if trajectory['completed'] else {}
                assert not trajectory.keys() & details.keys(), 'Trajectory and diagnostic field collision'
                records.append(dict(id=row['id'], replicate=replicate, **trajectory, **details))
            status(status='running', phase='reader_training', step=step, total=CFG['updates'], question=qi,
                   completed_so_far=sum(r['completed'] for r in records))
            del native, k, v
        complete = [r for r in records if r['completed']]
        assert complete and sum(r['active_tokens'] for r in complete) > 0, 'No nonzero gated training signal'
        scaler.unscale_(optimizer)
        reader.assert_trainable(model, base, writer)
        norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG['gradient_clip'])
        assert torch.isfinite(norm) and norm > 0, ('Invalid Reader gradient', step, norm)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= old_scale, 'GradScaler skipped update; do not count it as successful'
        value = dict(step=step, attempted=len(records), completed=len(complete), censored=len(records) - len(complete),
                     correct=sum(r['correct'] for r in complete), generated_tokens=sum(r['generated_tokens'] for r in records),
                     active_tokens=sum(r['active_tokens'] for r in complete), positive_tokens=sum(r['positive_tokens'] for r in complete),
                     negative_tokens=sum(r['negative_tokens'] for r in complete), abs_gate_mass=sum(r['abs_gate_mass'] for r in complete),
                     surrogate_loss=sum(r['surrogate_loss'] for r in complete) / denominator, gradient_norm=float(norm),
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     replay_logp_mean_abs_error=sum(r['replay_logp_mean_abs_error'] for r in complete) / len(complete),
                     manifest_sha256=identity())
        C.write_jsonl(folder / f'step{step:04d}_trajectories.jsonl', records)
        C.save(folder / f'step{step:04d}.json', value)
        if step % CFG['checkpoint_every_steps'] == 0 or step == CFG['updates']:
            save_checkpoint(checkpoint_path, reader, optimizer, scaler, step)
            if step == CFG['updates']:
                save_checkpoint(ROOT / 'runs/reader/step256.pt', reader, optimizer, scaler, step)
        C.log(f'PILOT TRAIN step={step}/256 correct={value["correct"]}/{value["completed"]} cap={value["censored"]}/16 tokens={value["generated_tokens"]} grad={float(norm):.4f}')
    result = dict(status='completed', updates=CFG['updates'], checkpoint=str(ROOT / 'runs/reader/step256.pt'),
                  checkpoint_sha256=C.sha(ROOT / 'runs/reader/step256.pt'), selection=CFG['selection'], manifest_sha256=identity())
    C.save(ROOT / 'runs/reader/completed.json', result)
    shared('reader_training_completed_evaluation_pending', result)


@torch.no_grad()
def evaluate():
    assert (ROOT / 'runs/reader/completed.json').exists()
    tok, model, base, writer, reader = load_all()
    initial = reader.state_dict()
    final = torch.load(ROOT / 'runs/reader/step256.pt', map_location='cpu', weights_only=True)
    assert final['step'] == CFG['updates'] and final['manifest_sha256'] == identity()
    runtime = C.runtime(model, tok)
    old = {r['id']: r for r in C.read_jsonl(PARENT / 'evaluation/gsm8k/per_sample.jsonl')}
    for index, raw in enumerate(rows('test'), 1):
        row = source_row(raw)
        path = ROOT / 'evaluation/per_sample' / (U.fname(row) + '.json')
        if path.exists():
            assert json.loads(path.read_text())['manifest_sha256'] == identity()
            continue
        status(status='running', phase='evaluation', sample=index, total=CFG['evaluation_questions'])
        native, k, v = P.fixed_state(model, base, writer, reader, row)
        conditions = {}
        if row['id'] in old:
            previous = old[row['id']]
            assert previous['reader_manifest_sha256'] == P.identity()
            for field in ['prompt', 'receiver_suffix', 'gold']:
                assert previous[field] == row[field]
            conditions['initial_reader'] = dict(previous['conditions']['ufd_writer_qo_reader'], reused_from=str(PARENT / 'evaluation/gsm8k/per_sample.jsonl'))
        for arm, state in [('initial_reader', initial), ('reward_reader', final['state'])]:
            if arm in conditions:
                continue
            reader.load_state_dict(state)
            with reader.mode(True):
                text, count = C.R.generate(runtime, torch.tensor([row['suffix_ids']], device='cuda'), CFG['formal_max_new_tokens'],
                                         past_key_values=U.translated_cache(model, row, native, k, v),
                                         past_length=len(row['frames']['qwen']['body_ids']), temperature=0.0)
            prediction = C.S.extract_numeric_answer(text)
            conditions[arm] = dict(generation=text, prediction=prediction,
                                   correct=int(prediction == C.S.normalize_number(row['gold'])),
                                   generated_tokens=count, reached_cap=int(count >= CFG['formal_max_new_tokens']))
        C.save(path, dict(id=row['id'], prompt=row['prompt'], gold=row['gold'], subset=row['evaluation_subset'],
                          conditions=conditions, manifest_sha256=identity()))
        C.log(f'PILOT EVAL {index}/128 baseline={conditions["initial_reader"]["correct"]} reward={conditions["reward_reader"]["correct"]}')
        del native, k, v
    records = [json.loads((ROOT / 'evaluation/per_sample' / (U.fname(r) + '.json')).read_text()) for r in rows('test')]
    C.write_jsonl(ROOT / 'evaluation/per_sample.jsonl', records)
    C.save(ROOT / 'evaluation/completed.json', dict(status='completed', n=len(records), manifest_sha256=identity()))


def summarize():
    assert (ROOT / 'evaluation/completed.json').exists()
    records = C.read_jsonl(ROOT / 'evaluation/per_sample.jsonl')
    assert len(records) == CFG['evaluation_questions']
    subsets = {}
    for name in ['all128', 'historical32', 'new96']:
        rr = records if name == 'all128' else [r for r in records if r['subset'] == name]
        metrics = {}
        for arm in ['initial_reader', 'reward_reader']:
            vv = [r['conditions'][arm] for r in rr]
            metrics[arm] = dict(n=len(vv), correct=sum(v['correct'] for v in vv), accuracy=sum(v['correct'] for v in vv) / len(vv),
                                mean_generated_tokens=sum(v['generated_tokens'] for v in vv) / len(vv), cap_count=sum(v['reached_cap'] for v in vv))
        pairs = [(r['conditions']['initial_reader']['correct'], r['conditions']['reward_reader']['correct']) for r in rr]
        subsets[name] = dict(metrics=metrics, paired=dict(both_correct=sum(a and b for a, b in pairs),
                              initial_only=sum(a and not b for a, b in pairs), reward_only=sum(b and not a for a, b in pairs),
                              both_wrong=sum(not a and not b for a, b in pairs)))
    steps = [json.loads((ROOT / f'training/step{s:04d}.json').read_text()) for s in range(1, CFG['updates'] + 1)]
    result = dict(experiment=CFG['name'], status='completed', config=CFG, subsets=subsets,
                  training=dict(updates=len(steps), attempted=sum(s['attempted'] for s in steps),
                                completed=sum(s['completed'] for s in steps), censored=sum(s['censored'] for s in steps),
                                generated_tokens=sum(s['generated_tokens'] for s in steps)),
                  smoke=json.loads((ROOT / 'smoke/completed.json').read_text()),
                  manifest_sha256=identity(), caveats=json.loads((ROOT / 'manifest.json').read_text())['caveats'])
    C.save(ROOT / 'summary.json', result)
    shared('completed', result)
    C.save(ROOT / 'completed.json', dict(status='completed', manifest_sha256=identity()))
    status(status='completed', phase='all')
    C.log('ALL REWARD-ALIGNED READER PILOT STAGES COMPLETED')


if __name__ == '__main__':
    try:
        action = sys.argv[1]
        if action == 'prepare':
            prepare()
        else:
            verify()
            {'capture': capture, 'smoke': smoke, 'train': train, 'evaluate': evaluate, 'summarize': summarize}[action]()
    except Exception as exc:
        error = dict(status='failed_or_incomplete', type=type(exc).__name__, message=str(exc))
        C.save(ROOT / 'failure.json', error)
        status(phase='failure', **error)
        shared('failed_or_incomplete', error)
        raise
