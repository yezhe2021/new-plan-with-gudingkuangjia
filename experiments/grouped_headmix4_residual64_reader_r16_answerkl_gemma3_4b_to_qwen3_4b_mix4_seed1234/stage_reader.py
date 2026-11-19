"""Training functions preserved from the matching Llama experiment; see provenance.json."""
import sys,json,time,random,shutil,hashlib
from collections import defaultdict
from pathlib import Path
from common import U,C,T,torch,ROOT,identity,status,mark,done,shared,stage_config,verify
from grouped_translator import GroupedKVTranslator
from residual_translator import FrozenTranslatorWithResidual
from reader import QOReader
from objective import mcq_positions

CFG = stage_config('reader')
PARENT = ROOT

def base_digest(base):
    digest = hashlib.sha256()
    for name, tensor in base.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_all():
    C.seed()
    tok, model = C.model('qwen')
    record = torch.load(Path(CFG['base_parent']) / 'runs/stage_a/best.pt', map_location='cpu', weights_only=True)
    assert record['manifest_sha256'] == CFG['base_parent_manifest_sha256'] and record['stage'] == 'a'
    base = GroupedKVTranslator(**CFG['architecture'])
    base.load_state_dict(record['state'], strict=True)
    del record
    assert sum(p.numel() for p in base.parameters()) == CFG['expected_translator_parameters']
    # Residual construction does not alter the CPU RNG state used elsewhere.
    with torch.random.fork_rng(devices=[]):
        base = FrozenTranslatorWithResidual(base, CFG['residual_rank'])
    record = torch.load(Path(CFG['residual_parent']) / 'runs/stage_b/last.pt', map_location='cpu', weights_only=True)
    assert record['manifest_sha256'] == CFG['residual_parent_manifest_sha256']
    assert record['state_scope'] == 'cross_head_residual_only' and record['frozen_stage_a_sha256'] == CFG['base_stage_a_sha256']
    assert record['step'] == 1024 and record['epoch'] == 4 and record['epoch_completed']
    base.residual.load_state_dict(record['state'], strict=True)
    del record
    assert sum(p.numel() for p in base.residual.parameters()) == CFG['expected_residual_parameters']
    assert any(p.detach().abs().sum() > 0 for n, p in base.residual.named_parameters() if '.2.weight' in n)
    base = base.cuda().eval().requires_grad_(False)
    assert all(not p.requires_grad for p in base.parameters())
    reader = QOReader(model, CFG['rank'], CFG['alpha'], CFG['layers'])
    assert sum(p.numel() for p in reader.parameters()) == CFG['expected_trainable_parameters']
    reader.assert_trainable(model, base)
    torch.cuda.reset_peak_memory_stats()
    return tok, model, base, reader


def memory():
    return dict(peak_pytorch_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_pytorch_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                scope='This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak')


@torch.no_grad()
def fixed_state(model, base, reader, row):
    with reader.mode(False):
        source = U.loaded_source(row)
        pieces = []
        for start in range(0, source['key'].shape[1], CFG['chunk_tokens']):
            key = source['key'][:, start:start + CFG['chunk_tokens']][None].cuda()
            value = source['value'][:, start:start + CFG['chunk_tokens']][None].cuda()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pieces.append(base(key, value))
        k, v = (torch.cat([pair[i][0] for pair in pieces], dim=1) for i in range(2))
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


def example_loss(model, tok, base, reader, row, epoch, student=False, validation=False):
    native, k, v = fixed_state(model, base, reader, row)
    if row['task'] != 'gsm8k':
        pp = mcq_positions(row, lambda label: C.R.candidate_ids(tok, label))
        prefix, source_name = [], 'answer_boundary'
        trajectory = dict(tokens=[], reached_cap=False)
    else:
        with torch.no_grad():
            if student:
                with reader.mode(True):
                    trajectory = U.rollout(model, tok, row, U.translated_cache(model, row, native, k, v), CFG['prefix_max_new_tokens'])
            else:
                trajectory = teacher_prefix(model, tok, reader, row, native)
        pp = U.positions_for(trajectory['tokens'], row, epoch, validation)
        prefix = trajectory['tokens'][:max(pp)]
        source_name = 'student' if student else 'native'
    with reader.mode(False), torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        teacher = U.suffix_logits(model, row, U.clone_cache(native, model), prefix, pp)
    with reader.mode(True), torch.amp.autocast('cuda', dtype=torch.float16):
        student_logits = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), prefix, pp)
    loss = U.full_kl(teacher, student_logits)
    assert torch.isfinite(loss)
    details = dict(task=row['task'], kl=float(loss.detach()), positions=len(pp),
                   top1_matches=int((teacher.argmax(-1) == student_logits.detach().argmax(-1)).sum()),
                   prefix_source=source_name, rollout_tokens=len(trajectory['tokens']),
                   rollout_reached_cap=int(trajectory['reached_cap']))
    return loss, details


def smoke():
    if (ROOT / 'smoke/reader.json').exists():
        assert json.loads((ROOT / 'smoke/reader.json').read_text())['manifest_sha256'] == identity()
        return
    tok, model, base, reader = load_all()
    frozen_before = base_digest(base)
    rr = U.rows('train', True)
    checks, teacher_before = [], {}
    # Independently check the reused raw-K / single-RoPE cache interface.
    row = rr[0]
    ids = row['frames']['qwen']['body_ids']
    anchor = row['frames']['qwen']['native_prefix_length']
    with reader.mode(False), torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        key, value, _ = C.capture(model, ids, len(ids), [0])
        native = U.native_body(model, row)
        a = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0])
        b = U.suffix_logits(model, row, U.translated_cache(model, row, native, key[:, anchor:].cuda(), value[:, anchor:].cuda()), [], [0])
        parity = float(U.full_kl(a, b))
        assert parity < U.CFG['native_parity_kl_max'] and int(a.argmax()) == int(b.argmax())
    del key, value, native, a, b
    for row in rr:
        native, k, v = fixed_state(model, base, reader, row)
        with torch.no_grad():
            with reader.mode(False):
                expected = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), [], [0])
                teacher_before[row['id']] = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
            cache = U.translated_cache(model, row, native, k, v)
            original = [(a.clone(), b.clone()) for a, b in [U._cache_layer(cache, i) for i in range(36)]]
            with reader.mode(True):
                actual = U.suffix_logits(model, row, cache, [], [0])
            assert torch.equal(expected, actual), ('Zero-init Stage-A+B output changed', row['id'], float((expected-actual).abs().max()))
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
            loss, detail = example_loss(model, tok, base, reader, row, 2 if online else 1, student=online)
            if row['task'] != 'gsm8k':
                assert detail['positions'] == 1 and detail['rollout_tokens'] == 0 and detail['prefix_source'] == 'answer_boundary'
            scaler.scale(loss / len(rr)).backward()
        scaler.unscale_(optimizer)
        norms = reader.gradient_norms()
        assert all(0 < n < float('inf') for n in norms.values()), norms
        reader.assert_trainable(model, base)
        assert all(m.b.grad is not None and torch.isfinite(m.b.grad).all() and m.b.grad.abs().sum() > 0
                   for m in reader.branches.values()), 'A Reader branch has no finite nonzero gradient'
        norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), CFG['gradient_clip'])
        assert torch.isfinite(norm) and norm > 0
        scaler.step(optimizer)
        scaler.update()
        gradients.append(dict(source='online_stage_ab_reader' if online else 'native', branches=norms))
        C.log(f'READER SMOKE gradient source={gradients[-1]["source"]} Q/O={norms}')
    with torch.no_grad(), reader.mode(False):
        for row in rr:
            native = U.native_body(model, row)
            after = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0]).cpu()
            assert torch.equal(after, teacher_before[row['id']]), 'Native Teacher was modified by Reader training'
    assert base_digest(base) == frozen_before, 'Frozen Stage-A/B changed during smoke'
    assert all(m.enabled_calls > 0 for m in reader.branches.values())
    result = dict(status='completed', manifest_sha256=identity(), checks=checks, gradients=gradients,
                  trainable_parameters=sum(p.numel() for p in reader.parameters()),
                  native_teacher_bitwise_unchanged_after_updates=True, stage_a_and_b_bitwise_unchanged=True, residual_adapter_loaded=True,
                  reader_only_continuation=True, native_cache_parity_kl=parity, all72_branches_receive_gradients=True, memory=memory())
    C.save(ROOT / 'smoke/reader.json', result)
    status(status='smoke_passed', phase='smoke')
    C.log('ALL Q/O READER GPU SMOKE CHECKS PASSED')


@torch.no_grad()
def validate(model, tok, base, reader):
    bytask = defaultdict(list)
    for row in U.rows('validation'):
        if len(bytask[row['task']]) >= CFG['monitor_validation_per_task']:
            continue
        _, details = example_loss(model, tok, base, reader, row, 0, validation=True)
        if row['task'] != 'gsm8k':
            native, k, v = fixed_state(model, base, reader, row)
            with reader.mode(False):
                details['qwen_native_correct'] = U.candidate_scores(model, tok, row, native)['correct']
                details['stage_ab_only_correct'] = U.candidate_scores(model, tok, row, native, k, v)['correct']
            with reader.mode(True):
                details['stage_ab_reader_correct'] = U.candidate_scores(model, tok, row, native, k, v)['correct']
        bytask[row['task']].append(details)
    result = {}
    for task, values in bytask.items():
        count = sum(v['positions'] for v in values)
        out = dict(n=len(values), mean_kl=sum(v['kl'] for v in values) / len(values),
                   same_prefix_next_token_agreement=sum(v['top1_matches'] for v in values) / count,
                   supervised_positions=count)
        if task != 'gsm8k':
            out['label_accuracy'] = {a: sum(v[a + '_correct'] for v in values) / len(values)
                                     for a in ['qwen_native', 'stage_ab_only', 'stage_ab_reader']}
        result[task] = out
    return dict(tasks=result, macro_kl=sum(v['mean_kl'] for v in result.values()) / len(result))


def save_checkpoint(path, reader, optimizer, scaler, epoch, step, position, complete, history):
    assert shutil.disk_usage(ROOT).free / 2**30 > CFG['disk_reserve_gib']
    C.save_tensor(path, dict(state=reader.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, step=step, position=position, epoch_completed=complete,
                            history=history, manifest_sha256=identity(), frozen_stage_a_sha256=CFG['base_stage_a_sha256'], frozen_stage_b_sha256=CFG['frozen_stage_b_sha256'], state_scope='reader_lora_only'))


def train():
    folder = ROOT / 'runs/reader'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        assert json.loads((folder / 'completed.json').read_text())['manifest_sha256'] == identity()
        return
    assert json.loads((ROOT / 'smoke/reader.json').read_text())['manifest_sha256'] == identity()
    tok, model, base, reader = load_all()
    frozen_before = base_digest(base)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in reader.parameters()}
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    start, step, position0, history = 1, 0, 0, []
    if (folder / 'last.pt').exists():
        old = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        assert old['manifest_sha256'] == identity()
        assert old['frozen_stage_b_sha256'] == CFG['frozen_stage_b_sha256']
        assert old['state_scope'] == 'reader_lora_only' and old['frozen_stage_a_sha256'] == CFG['base_stage_a_sha256']
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
                loss, d = example_loss(model, tok, base, reader, row, epoch, student=online)
                scaler.scale(loss / 8).backward()
                details.append(d)
                if position % 8:
                    continue
                scaler.unscale_(optimizer)
                reader.assert_trainable(model, base)
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
            assert base_digest(base) == frozen_before, 'Frozen Stage-A/B changed during training'
            metrics = validate(model, tok, base, reader)
            entry = dict(epoch=epoch, step=step, validation=metrics)
            history = [h for h in history if h['epoch'] != epoch] + [entry]
            best = min(history, key=lambda h: (h['validation']['macro_kl'], h['epoch']))
            C.save_tensor(folder / f'epoch{epoch}_step{step}.pt', dict(state=reader.state_dict(), epoch=epoch, step=step,
                          validation=metrics, manifest_sha256=identity(), diagnostic_only=epoch != 4,
                          frozen_stage_a_sha256=CFG['base_stage_a_sha256'], frozen_stage_b_sha256=CFG['frozen_stage_b_sha256'], state_scope='reader_lora_only'))
            if entry == best:
                C.save_tensor(folder / 'best.pt', dict(state=reader.state_dict(), epoch=epoch, step=step,
                              validation=metrics, manifest_sha256=identity(), diagnostic_only=True,
                              frozen_stage_a_sha256=CFG['base_stage_a_sha256'], frozen_stage_b_sha256=CFG['frozen_stage_b_sha256'], state_scope='reader_lora_only'))
            C.save(folder / 'validation.json', dict(primary=CFG['selection'], best_diagnostic=best, history=history))
            save_checkpoint(folder / 'last.pt', reader, optimizer, scaler, epoch, step, 2048, True, history)
            C.log(f'READER EPOCH {epoch} COMPLETED: {metrics}')
    assert step == 1024
    assert base_digest(base) == frozen_before
    result = dict(status='completed', epochs=4, updates=step, selection=CFG['selection'],
                  checkpoint=str(folder / 'epoch4_step1024.pt'), checkpoint_sha256=C.sha(folder / 'epoch4_step1024.pt'),
                  validation=history, memory=memory(), manifest_sha256=identity(), stage_a_and_b_bitwise_unchanged=True)
    C.save(folder / 'completed.json', result)
    shared('reader_training_completed_evaluation_pending', result)
    C.log('READER TRAINING COMPLETED')


if __name__ == "__main__":
    verify()
    action = sys.argv[1]
    {"smoke": smoke, "train": train}[action]()
