"""Freeze all existing weights; change only the reader, retaining the last schedule."""
import argparse
from collections import defaultdict
import json
import random
import time
import torch
from experiment import (HERE, PREVIOUS, CONFIG, previous, base_train, mcq_train,
                        gpu_check, model_bundle, ReaderHFLM, read_rows, write_rows,
                        save, log, digest, verify_manifest, freeze_manifest)
import student_kd as student


def prepare():
    previous.verify_manifest()
    folder = HERE / 'prepared'
    folder.mkdir(exist_ok=True)
    names = ('train_trajectories.jsonl', 'validation_trajectories.jsonl', 'validation_questions.jsonl',
             'validation_attempts.jsonl', 'demos.jsonl', 'validation_student_prefixes.jsonl')
    for name in names:
        source, target = PREVIOUS / 'prepared' / name, folder / name
        if target.exists():
            assert source.read_bytes() == target.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    save(folder / 'summary.json', {'train_references': len(read_rows(folder / names[0])),
                                  'reference_validation': len(read_rows(folder / names[1])),
                                  'fixed_validation_prefixes': len(read_rows(folder / names[-1])),
                                  'same_data_and_objectives_as_previous': True,
                                  'fixed_prefixes_reused_not_regenerated': True})


def frozen_gradients(bundle):
    receiver, sender, base, adapter = bundle[:4]
    assert all(not p.requires_grad and p.grad is None for m in (receiver, sender, base, adapter) for p in m.parameters())


@torch.no_grad()
def validate(bundle, anchors):
    receiver, sender, base, adapter, source_tok, hf, reader = bundle
    refs = read_rows(HERE / 'prepared/validation_trajectories.jsonl')
    ref = base_train.validate(receiver, sender, base, adapter, source_tok, hf, refs, anchors)
    values = []
    for row in read_rows(HERE / 'prepared/validation_student_prefixes.jsonl'):
        item, bk, bv = base_train.prefix(sender, receiver, base, source_tok, hf, row, anchors)
        _, metrics = student.objective(receiver, adapter, row, item, bk, bv)
        values.append(metrics['token_kl'])
    op = sum(values) / len(values)
    return {'total': .5 * ref['total'] + .5 * op, 'reference': ref,
            'fixed_student_prefix_kl': op, 'fixed_prefix_n': len(values)}


def optimizer_for(reader):
    return torch.optim.AdamW(reader.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])


def checkpoint(path, reader, opt, scaler, epoch, step, position, history, manifest):
    p = {'protocol': CONFIG['protocol'], 'frozen_stage_b_sha256': manifest['identities']['frozen_stage_b']['sha256'],
         'state': {k: v.detach().cpu() for k, v in reader.state_dict().items()},
         'epoch': epoch, 'step': step, 'position': position, 'history': history}
    if opt is not None:
        p.update(optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix('.tmp')
    torch.save(p, tmp)
    tmp.replace(path)


def smoke():
    gpu_check()
    bundle = model_bundle(True)
    receiver, sender, base, adapter, source_tok, hf, reader = bundle
    anchors = {}
    rows = read_rows(HERE / 'prepared/train_trajectories.jsonl')
    row = max(rows, key=lambda r: len(r['context_ids']) + len(r['continuation_ids']))
    item, bk, bv = base_train.prefix(sender, receiver, base, source_tok, hf, row, anchors)
    from kv_backend import assemble
    with torch.no_grad():
        k, v = assemble(item, bk, bv, adapter)
        with reader.scope(len(row['context_ids']) - 1, 'disabled'):
            original = base_train.suffix_hidden(receiver, row['context_ids'], row['continuation_ids'], k, v)
        modified = base_train.suffix_hidden(receiver, row['context_ids'], row['continuation_ids'], k, v)
        drift = float((original - modified).abs().max())
        assert torch.equal(original, modified), f'Zero reader parity drift={drift}'
    del k, v, original, modified
    opt = optimizer_for(reader)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=1000000)
    torch.cuda.reset_peak_memory_stats()
    before = {n: p.detach().clone() for n, p in reader.named_parameters()}
    records = []
    for kind in ('reference', 'student', 'mcq'):
        if kind == 'reference':
            loss, metrics = base_train.objective(receiver, adapter, row, item, bk, bv)
            scaler.scale(loss).backward()
            del loss
        elif kind == 'student':
            gen = ReaderHFLM(receiver, 'translated', sender, base, adapter)
            ids, text = student.rollout(gen, adapter, row, item, bk, bv, 128)
            loss, metrics = student.objective(receiver, adapter, dict(row, continuation_ids=ids), item, bk, bv)
            scaler.scale(loss).backward()
            del loss, gen
            metrics.update(sampled_tokens=len(ids), sampled_response=text)
        else:
            del item, bk, bv
            replay = read_rows(previous.parent.PREVIOUS / 'prepared/train_epoch_1.jsonl')
            q = max((r for r in replay if r['view'] == 'text'), key=lambda r: r['qwen_context_tokens'])
            item, bk, bv, pairs = mcq_train.frozen_prefix(sender, receiver, base, source_tok, hf, q, anchors)
            metrics = mcq_train.objective(receiver, hf, adapter, q, item, bk, bv, pairs, 1, scaler)
        frozen_gradients(bundle)
        branch_norms = {part: sum(float(p.grad.float().square().sum()) for n, p in reader.named_parameters()
                                 if part in n and p.grad is not None) ** .5
                        for part in ('.query.', '.output.', '.gate.')}
        assert all(x > 0 and torch.isfinite(torch.tensor(x)) for x in branch_norms.values()), branch_norms
        norm = base_train.update(reader, opt, scaler)
        assert norm > 0
        records.append({'kind': kind, 'metrics': metrics, 'grad_norm': norm, 'branch_grad_norms': branch_norms})
        log('SMOKE ' + json.dumps(records[-1]))
    delta = max(float((p.detach() - before[n]).abs().max()) for n, p in reader.named_parameters())
    assert delta > 0
    save(HERE / 'reports/smoke.json', {'passed': True, 'SMOKE_WEIGHTS_DISCARDED': True,
         'zero_reader_bitwise_parity': True, 'hidden_max_drift': drift,
         'trainable_parameters': sum(p.numel() for p in reader.parameters()), 'records': records,
         'frozen_parameters_received_gradients': False, 'max_parameter_change': delta,
         'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
         'gpu_name': torch.cuda.get_device_name(0)})
    log('READER SMOKE PASSED; formal training reloads zero-initialized reader')


def train():
    manifest = verify_manifest()
    gpu_check()
    bundle = model_bundle(True)
    receiver, sender, base, adapter, source_tok, hf, reader = bundle
    gen = ReaderHFLM(receiver, 'translated', sender, base, adapter)
    opt = optimizer_for(reader)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=1000000)
    refs = read_rows(HERE / 'prepared/train_trajectories.jsonl')
    replay = sum((read_rows(previous.parent.PREVIOUS / f'prepared/train_epoch_{e}.jsonl') for e in (1, 2)), [])
    output = HERE / 'reader'
    output.mkdir(exist_ok=True)
    start_epoch, start_position, step, history = 1, 0, 0, []
    if (output / 'last.pt').exists():
        p = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        assert p['protocol'] == CONFIG['protocol']
        assert p['frozen_stage_b_sha256'] == manifest['identities']['frozen_stage_b']['sha256']
        reader.load_state_dict(p['state'])
        opt.load_state_dict(p['optimizer'])
        scaler.load_state_dict(p['scaler'])
        start_epoch, start_position, step, history = p['epoch'], p['position'], p['step'], p['history']
        torch.set_rng_state(p['torch_rng'])
        torch.cuda.set_rng_state(p['cuda_rng'])
        log(f'RESUME reader epoch={start_epoch} position={start_position} update={step}')
    anchors = {}
    if not history:
        reader.eval()
        metrics = validate(bundle, anchors)
        history = [{'epoch': 0, 'step': 0, 'validation': metrics}]
        checkpoint(output / 'best.pt', reader, None, None, 0, 0, 0, history, manifest)
        save(output / 'selection.json', {'selection': CONFIG['selection'], 'best': history[0], 'epochs': history})
        log('INITIAL VALIDATION ' + json.dumps(history[0]))
    with (output / 'training_steps.jsonl').open('a') as stream, (output / 'student_rollouts.jsonl').open('a', encoding='utf-8') as rollout_log:
        for epoch in range(start_epoch, CONFIG['epochs'] + 1):
            rng = random.Random(CONFIG['seed'] + epoch)
            jobs = student.reference_partition(refs, epoch) + [('mcq', r) for r in rng.sample(replay, CONFIG['mcq_replay_per_epoch'])]
            rng.shuffle(jobs)
            reader.train()
            opt.zero_grad(set_to_none=True)
            accumulated, sums, counts = 0, defaultdict(float), defaultdict(int)
            log(f'START READER TRAIN epoch={epoch}/{CONFIG["epochs"]} draws={len(jobs)}')
            for position, (kind, row) in enumerate(jobs, 1):
                if epoch == start_epoch and position <= start_position:
                    continue
                divisor = min(CONFIG['effective_batch'], len(jobs) - ((position - 1) // CONFIG['effective_batch']) * CONFIG['effective_batch'])
                if kind == 'mcq':
                    item, bk, bv, pairs = mcq_train.frozen_prefix(sender, receiver, base, source_tok, hf, row, anchors)
                    values = mcq_train.objective(receiver, hf, adapter, row, item, bk, bv, pairs, divisor, scaler)
                else:
                    item, bk, bv = base_train.prefix(sender, receiver, base, source_tok, hf, row, anchors)
                    if kind == 'student':
                        ids, text = student.rollout(gen, adapter, row, item, bk, bv, CONFIG['student_rollout_tokens_by_epoch'][epoch - 1])
                        rollout_log.write(json.dumps({'epoch': epoch, 'position': position, 'base_id': row['base_id'],
                            'ids': ids, 'response': text, 'conditioning_only_not_gold_labels': True}, ensure_ascii=False) + '\n')
                        rollout_log.flush()
                        loss, values = student.objective(receiver, adapter, dict(row, continuation_ids=ids), item, bk, bv)
                    else:
                        loss, values = base_train.objective(receiver, adapter, row, item, bk, bv)
                    scaler.scale(loss / divisor).backward()
                    del loss
                del item, bk, bv
                frozen_gradients(bundle)
                accumulated += 1
                counts[kind] += 1
                for k, v in values.items():
                    sums[kind + '_' + k] += v
                if accumulated == divisor:
                    norm = base_train.update(reader, opt, scaler)
                    step += 1
                    entry = {'epoch': epoch, 'step': step, 'position': position,
                             'pre_clip_grad_norm': norm, 'types': dict(counts),
                             **{k: v / counts[k.split('_')[0]] for k, v in sums.items()}}
                    stream.write(json.dumps(entry) + '\n')
                    stream.flush()
                    save(HERE / 'status.json', {'phase': 'training', 'epoch': epoch, 'epochs': CONFIG['epochs'],
                         'step': step, 'position': position, 'epoch_draws': len(jobs)})
                    if step == 1 or step % 4 == 0:
                        log('TRAIN ' + json.dumps(entry))
                    accumulated, sums, counts = 0, defaultdict(float), defaultdict(int)
                    if step == 1 or step % CONFIG['checkpoint_interval_updates'] == 0:
                        checkpoint(output / 'last.pt', reader, opt, scaler, epoch, step, position, history, manifest)
            reader.eval()
            metrics = validate(bundle, anchors)
            candidate = {'epoch': epoch, 'step': step, 'validation': metrics}
            history = [h for h in history if h['epoch'] != epoch] + [candidate]
            best = min(history, key=lambda h: (h['validation']['total'], h['epoch']))
            if best is candidate:
                checkpoint(output / 'best.pt', reader, None, None, epoch, step, len(jobs), history, manifest)
            save(output / 'selection.json', {'selection': CONFIG['selection'], 'best': best, 'epochs': history})
            checkpoint(output / 'last.pt', reader, opt, scaler, epoch + 1, step, 0, history, manifest)
            start_position = 0
            log('EPOCH COMPLETED ' + json.dumps(candidate))
    save(output / 'completed.json', {'epochs': CONFIG['epochs'], 'updates': step,
                                    'trained_reader_only': True, 'frozen_translator': manifest['identities']['frozen_stage_b']})
    log('READER TRAINING COMPLETED')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'smoke', 'freeze', 'train'])
    args = ap.parse_args()
    if args.action == 'freeze':
        freeze_manifest()
    else:
        globals()[args.action]()
