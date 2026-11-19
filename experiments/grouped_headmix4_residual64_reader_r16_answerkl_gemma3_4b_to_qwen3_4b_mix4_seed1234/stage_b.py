"""Training functions preserved from the matching Llama experiment; see provenance.json."""
import sys,json,time,random,shutil,hashlib
from collections import defaultdict
from pathlib import Path
from common import U,C,T,torch,ROOT,identity,status,mark,done,shared,stage_config,verify
from grouped_translator import GroupedKVTranslator
from residual_translator import FrozenTranslatorWithResidual
from reader import QOReader
from objective import mcq_positions

CFG = stage_config('b')
PARENT = ROOT

def fresh():
    model = GroupedKVTranslator(**CFG['architecture'])
    assert sum(p.numel() for p in model.parameters()) == CFG['expected_parameters']
    return model


def fresh_stage_b():
    record = torch.load(Path(CFG['base_parent']) / 'runs/stage_a/best.pt', map_location='cpu', weights_only=True)
    assert record['manifest_sha256'] == CFG['base_parent_manifest_sha256'] and record['stage'] == 'a'
    base = fresh()
    base.load_state_dict(record['state'], strict=True)
    del record
    module = FrozenTranslatorWithResidual(base, CFG['residual_rank'])
    assert sum(p.numel() for p in module.parameters() if p.requires_grad) == CFG['expected_trainable_parameters']
    module.assert_frozen()
    return module


def base_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.base.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_weights(path, trainable=False):
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert value['manifest_sha256'] == identity()
    assert value['state_scope'] == 'cross_head_residual_only'
    assert value['frozen_stage_a_sha256'] == CFG['base_stage_a_sha256']
    module = fresh_stage_b()
    module.residual.load_state_dict(value['state'], strict=True)
    del value
    module.residual.requires_grad_(trainable)
    return module.cuda().train(trainable)


def map_source(module, source):
    pieces = []
    for start in range(0, source['key'].shape[1], CFG['chunk_tokens']):
        key = source['key'][:, start:start + CFG['chunk_tokens']][None].cuda()
        value = source['value'][:, start:start + CFG['chunk_tokens']][None].cuda()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # The frozen base already runs under no_grad; only small residual graphs are retained.
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
    module.assert_frozen()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(module.residual.parameters(), CFG['gradient_clip'])
    assert torch.isfinite(norm) and norm > 0, 'Invalid translator gradient'
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def save_checkpoint(path, module, stage, epoch, step, position, complete,
                    history, optimizer=None, scaler=None):
    assert shutil.disk_usage(ROOT).free / 2**30 > CFG['disk_reserve_gib']
    module.assert_frozen()
    record = dict(manifest_sha256=identity(), state={k: v.detach().cpu() for k, v in module.residual.state_dict().items()},
                  state_scope='cross_head_residual_only', frozen_stage_a_sha256=CFG['base_stage_a_sha256'],
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
    assert stage == 'b', 'Stage A is reused and frozen'
    path = ROOT / 'smoke' / (stage + '.json')
    if done(path):
        return
    C.seed()
    module = fresh_stage_b().cuda()
    frozen_before = base_digest(module)
    optimizer = torch.optim.AdamW(module.residual.parameters(), lr=CFG['stage_' + stage + '_learning_rate'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    checks = []
    if stage == 'a':
        for row in C.rows('train', True):
            metrics = T.stage_a_backward(module, C.state('train', row, 'llama'),
                                         C.state('train', row, 'qwen'), scaler=scaler)
            checks.append(dict(id=row['id'], grad_norm=update(module, optimizer, scaler), metrics=metrics))
    else:
        tok, model = C.model('qwen')
        # Identity at initialization is checked on a real aligned Sender cache.
        first = U.rows('train', True)[0]
        source = U.loaded_source(first)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            sk, sv = source['key'][:, :64][None].cuda(), source['value'][:, :64][None].cuda()
            a, b = module.base(sk, sv)
            c, d = module(sk, sv)
            assert torch.equal(a, c) and torch.equal(b, d), 'Residual initialization changed Stage-A outputs'
        del source, sk, sv, a, b, c, d
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
                module.assert_frozen()
                # Zero-up initialization gives zero down-projection gradients on the first pass.
                # Every layer and both roles must nevertheless receive an up-projection gradient.
                assert all(branch[-1].weight.grad is not None and branch[-1].weight.grad.abs().sum() > 0
                           for branch in list(module.residual.k) + list(module.residual.v))
                checks.append(dict(id=row['id'], grad_norm=update(module, optimizer, scaler), metrics=metrics))
        checks.append(dict(native_cache_kl=parity))
    assert base_digest(module) == frozen_before, 'Frozen translator weights changed during smoke'
    mark(path, status='completed', checks=checks, base_bitwise_unchanged=True,
         frozen_parameters=sum(p.numel() for p in module.base.parameters()),
         trainable_parameters=sum(p.numel() for p in module.residual.parameters()))
    C.log('GROUPED SMOKE PASSED ' + stage)


def train(stage):
    assert stage == 'b', 'Stage A is reused and frozen'
    folder = ROOT / ('runs/stage_' + stage)
    folder.mkdir(parents=True, exist_ok=True)
    if done(folder / 'completed.json'):
        return
    assert done(ROOT / 'smoke' / (stage + '.json'))
    C.seed()
    module = fresh_stage_b().cuda()
    frozen_before = base_digest(module)
    tok = model = None
    if stage == 'b':
        tok, model = C.model('qwen')
    optimizer = torch.optim.AdamW(module.residual.parameters(), lr=CFG['stage_' + stage + '_learning_rate'], weight_decay=CFG['weight_decay'])
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in module.residual.parameters()}
    scaler = torch.amp.GradScaler('cuda', init_scale=64., growth_interval=1000000)
    epoch0, position0, step, history = 1, 0, 0, []
    if (folder / 'last.pt').exists():
        record = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        assert record['manifest_sha256'] == identity() and record['stage'] == stage
        assert record['state_scope'] == 'cross_head_residual_only' and record['frozen_stage_a_sha256'] == CFG['base_stage_a_sha256']
        module.residual.load_state_dict(record['state'], strict=True)
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
            module.assert_frozen()
            assert base_digest(module) == frozen_before, 'Frozen translator changed during training'
            metrics = validate_a(module) if stage == 'a' else validate_b(module, model, tok)
            entry = dict(epoch=epoch, step=step, validation=metrics)
            history = [r for r in history if r['epoch'] != epoch] + [entry]
            if stage == 'a' and min(history, key=lambda r: (r['validation']['total'], r['epoch'])) == entry:
                save_checkpoint(folder / 'best.pt', module, stage, epoch, step, len(order), True, history)
            mark(folder / 'validation.json', history=history)
            save_checkpoint(folder / 'last.pt', module, stage, epoch, step, len(order), True, history, optimizer, scaler)
    assert step == epochs * 256
    result = dict(status='completed', stage=stage, steps=step, history=history, base_bitwise_unchanged=True,
                  trainable_parameters=CFG['expected_trainable_parameters'], frozen_stage_a_sha256=CFG['base_stage_a_sha256'],
                  selected='best.pt' if stage == 'a' else 'last.pt')
    mark(folder / 'completed.json', **result)
    shared('stage_' + stage + '_completed', result)


if __name__ == "__main__":
    verify()
    action = sys.argv[1]
    {"smoke": smoke, "train": train}[action]('b')
