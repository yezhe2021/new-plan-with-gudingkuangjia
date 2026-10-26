"""Hybrid MCQ supervision and original mixed-prefix GSM KL; frozen A+B."""
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
SUPERVISED = Path(CFG['supervised_reference'])
spec = importlib.util.spec_from_file_location('frozen_reference_reader', REFERENCE / 'experiment.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
C, U, torch = R.C, R.U, R.torch
sys.path.insert(0, str(ROOT))
from objectives import candidate_loss, clean_controls
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
    assert C.sha(SUPERVISED/'manifest.json') == CFG['supervised_manifest_sha256']
    for key in ['base_stage_a_sha256','frozen_stage_b_sha256','rank','alpha','train_counts','formal_counts','effective_batch','epochs','prefix_max_new_tokens','supervised_positions_per_example_max']:
        assert CFG[key] == R.CFG[key], key
    tok = C.AutoTokenizer.from_pretrained(C.CFG['models']['qwen'], local_files_only=True)
    paths = {
        REFERENCE/'runs/reader/epoch4_step1024.pt': CFG['old_reader_checkpoint_sha256'],
        SUPERVISED/'runs/reader/epoch4_step1024.pt': CFG['supervised_checkpoint_sha256'],
        SUPERVISED/'manifest.json': CFG['supervised_manifest_sha256'],
    }
    source_hashes = {}
    for p,digest in paths.items():
        assert C.sha(p)==digest
        source_hashes[str(p)] = digest
    records, counts, sets = [], {}, {}
    for split in ['train','validation','test']:
        rr=U.rows(split)
        counts[split]=dict(Counter(r['task'] for r in rr))
        sets[split]={r['question_key'] for r in rr}
        source_hashes[str(U.ROOT/f'prepared/{split}.jsonl')]=C.sha(U.ROOT/f'prepared/{split}.jsonl')
        assert all(U.source_path(r).exists() for r in rr)
        if split=='test': continue
        for row in rr:
            rec=dict(id=row['id'],task=row['task'],split=split,row_sha256=C.digest(row))
            if row['task']=='gsm8k':
                rec['objective']='original_mixed_prefix_kl_no_gold_answer'
            else:
                labels=[C.R.candidate_ids(tok,label) for label in row['candidates']]
                assert all(len(v)==1 for v in labels)
                ids=[v[0] for v in labels]
                assert len(ids)==len(set(ids)) and 0<=row['gold']<len(ids)
                rec.update(label_ids=ids,gold=row['gold'])
            records.append(rec)
    assert counts['train']==CFG['train_counts'] and counts['test']==CFG['formal_counts']
    assert not sets['train']&(sets['validation']|sets['test']) and not sets['validation']&sets['test']
    assert len(records)==len({r['id'] for r in records})==2304
    for row in U.rows('test'):
        p=SUPERVISED/'evaluation'/row['task']/'per_sample'/(U.fname(row)+'.json')
        v=json.loads(p.read_text())
        assert v['manifest_sha256']==CFG['supervised_manifest_sha256']
        assert all(v[k]==row[k] for k in ['id','task','prompt','gold'])
        source_hashes[str(p)]=C.sha(p)
    path=ROOT/'prepared/targets.jsonl'
    if path.exists(): assert C.read_jsonl(path)==records
    else: C.write_jsonl(path,records)
    source_hashes[str(path)]=C.sha(path)
    validation_ids=[r['id'] for r in U.rows('validation') if r['task']=='gsm8k'][:CFG['diagnostic_validation_n']]
    assert len(validation_ids)==16
    meta=dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},
              reference_reader_manifest_sha256=R.identity(),source_hashes=source_hashes,counts=counts,
              diagnostic_validation_ids=validation_ids,stage_a_retrained=False,stage_b_retrained=False,
              gsm_gold_answers_used_by_training_loss=False,caveats=CFG['caveats'])
    path=ROOT/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==meta
    else: save(path,meta)
    mark(ROOT/'prepared/completed.json',status='completed',counts=counts,diagnostic_validation_ids=validation_ids)
    status(status='prepared',phase='data_audit')
    C.log('HYBRID READER FROZEN A+B / CHECKPOINTS / SPLITS VERIFIED')


def verify():
    m = json.loads((ROOT / 'manifest.json').read_text())
    assert C.sha(ROOT / 'config.json') == m['config_sha256']
    assert R.identity() == m['reference_reader_manifest_sha256']
    R.verify()
    for name, digest in m['code'].items():
        assert C.sha(ROOT / name) == digest, name
    for path, digest in m['source_hashes'].items():
        assert C.sha(Path(path)) == digest, path


def local_teacher_prefix(model,tok,reader,row,native):
    # Reuse immutable historical prefixes; write any newly needed prefix locally.
    for root,expected in [(U.ROOT,U.identity()),(REFERENCE,R.identity()),(ROOT,identity())]:
        path=root/'native_prefix'/(U.fname(row)+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            assert record['manifest_sha256']==expected
            return record
    with reader.mode(False):
        record=U.rollout(model,tok,row,U.clone_cache(native,model),CFG['prefix_max_new_tokens'])
    mark(ROOT/'native_prefix'/(U.fname(row)+'.json'),id=row['id'],**record)
    return record


R.teacher_prefix=local_teacher_prefix


def loss_example(model,tok,base,reader,row,target,backward=None,divisor=1,epoch=0,student=False,validation=False):
    assert target['row_sha256']==C.digest(row)
    if row['task']=='gsm8k':
        loss,details=R.example_loss(model,tok,base,reader,row,epoch,student=student,validation=validation)
        assert torch.isfinite(loss)
        if backward is not None: backward(loss/divisor)
        return dict(task=row['task'],loss=float(loss.detach()),ce=0.,kl=float(loss.detach()),
                    positions=details['positions'],prefix_source=details['prefix_source'],
                    rollout_tokens=details['rollout_tokens'],rollout_reached_cap=details['rollout_reached_cap'])
    native,k,v=R.fixed_state(model,base,reader,row)
    with torch.no_grad(),reader.mode(False),torch.amp.autocast('cuda',dtype=torch.float16):
        teacher=U.suffix_logits(model,row,U.clone_cache(native,model),[],[0])
    with reader.mode(True),torch.amp.autocast('cuda',dtype=torch.float16):
        student_logits=U.suffix_logits(model,row,U.translated_cache(model,row,native,k,v),[],[0])
        loss,ce,auxiliary,pred=candidate_loss(teacher,student_logits,target['label_ids'],target['gold'],CFG['kl_weight'])
        assert torch.isfinite(loss)
        if backward is not None: backward(loss/divisor)
    return dict(task=row['task'],loss=float(loss.detach()),ce=float(ce.detach()),kl=float(auxiliary.detach()),
                positions=1,prefix_source='answer_boundary',rollout_tokens=0,correct=int(pred==target['gold']))


@torch.no_grad()
def validate(model, tok, base, reader, target_map):
    bytask = defaultdict(list)
    for row in U.rows('validation'):
        if len(bytask[row['task']]) < CFG['monitor_validation_per_task']:
            bytask[row['task']].append(loss_example(model, tok, base, reader, row, target_map[row['id']],validation=True))
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


def gsm_validation_rows(n):
    rows = [r for r in U.rows('validation') if r['task']=='gsm8k'][:n]
    assert len(rows)==n
    return rows


@torch.no_grad()
def generate_value(model,tok,row,cache):
    text,count=C.R.generate(C.runtime(model,tok),torch.tensor([row['suffix_ids']],device='cuda'),
                            CFG['generation_max_new_tokens'],past_key_values=cache,
                            past_length=len(row['frames']['qwen']['body_ids']),temperature=0.)
    prediction=C.S.extract_numeric_answer(clean_controls(text))
    return dict(prediction=prediction,correct=int(prediction==C.S.normalize_number(row['gold'])),
                generation=text,generated_tokens=count,reached_cap=int(count>=CFG['generation_max_new_tokens']))


@torch.no_grad()
def validation_generate(model,tok,base,reader,epoch,step):
    folder=ROOT/'validation_generation'/f'epoch{epoch}'
    digest=R.base_digest(reader)
    values=[]
    rows=gsm_validation_rows(CFG['validation_generation_n'])
    for i,row in enumerate(rows,1):
        path=folder/(U.fname(row)+'.json')
        if path.exists():
            rec=json.loads(path.read_text())
            assert rec['manifest_sha256']==identity() and rec['reader_state_sha256']==digest
        else:
            native,k,v=R.fixed_state(model,base,reader,row)
            with reader.mode(True):
                value=generate_value(model,tok,row,U.translated_cache(model,row,native,k,v))
            mark(path,id=row['id'],gold=row['gold'],epoch=epoch,step=step,reader_state_sha256=digest,**value)
            rec=json.loads(path.read_text())
            del native,k,v
        values.append(rec)
        status(status='running',phase='validation_generation',epoch=epoch,step=step,sample=i,total=len(rows))
        C.log(f'GSM VALIDATION epoch={epoch} {i}/{len(rows)} correct={sum(v["correct"] for v in values)}')
    metrics=dict(n=len(values),correct=sum(v['correct'] for v in values),
                 accuracy=sum(v['correct'] for v in values)/len(values),
                 mean_generated_tokens=sum(v['generated_tokens'] for v in values)/len(values),
                 cap_count=sum(v['reached_cap'] for v in values),diagnostic_only=True)
    mark(folder/'summary.json',epoch=epoch,step=step,reader_state_sha256=digest,**metrics)
    return metrics


def smoke():
    if (ROOT/'smoke/completed.json').exists():
        assert json.loads((ROOT/'smoke/completed.json').read_text())['manifest_sha256']==identity()
        return
    tok,model,base,reader=R.load_all()
    tm=targets();rr=U.rows('train',True);frozen=R.base_digest(base);teacher_before={}
    for row in rr:
        native,k,v=R.fixed_state(model,base,reader,row)
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            with reader.mode(False):
                teacher_before[row['id']]=U.suffix_logits(model,row,U.clone_cache(native,model),[],[0]).cpu()
                expected=U.suffix_logits(model,row,U.translated_cache(model,row,native,k,v),[],[0])
            with reader.mode(True):
                actual=U.suffix_logits(model,row,U.translated_cache(model,row,native,k,v),[],[0])
            assert torch.equal(actual,expected)
        del native,k,v,expected,actual
    gsm=next(r for r in rr if r['task']=='gsm8k')
    parity=[]
    for online in [False,True]:
        with torch.no_grad():
            old,old_details=R.example_loss(model,tok,base,reader,gsm,2,student=online)
            new=loss_example(model,tok,base,reader,gsm,tm[gsm['id']],epoch=2,student=online)
            assert float(old)==new['loss'] and old_details['positions']==new['positions']
            assert old_details['prefix_source']==new['prefix_source']
            # Labels are not inputs to the restored GSM training objective.
            altered=dict(gsm,gold='UNUSED_GOLD')
            changed_target=dict(tm[gsm['id']],row_sha256=C.digest(altered))
            changed=loss_example(model,tok,base,reader,altered,changed_target,epoch=2,student=online)
            assert changed['loss']==new['loss']
        parity.append(dict(student=online,loss=new['loss'],positions=new['positions'],gold_independent=True))
    optimizer=torch.optim.AdamW(reader.parameters(),lr=CFG['learning_rate'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)
    details=[]
    for row,online in [(r,False) for r in rr]+[(gsm,True)]:
        d=loss_example(model,tok,base,reader,row,tm[row['id']],epoch=2,student=online,
                       backward=lambda value:scaler.scale(value).backward())
        reader.assert_trainable(model,base)
        assert all(m.b.grad is not None and torch.isfinite(m.b.grad).all() and m.b.grad.abs().sum()>0
                   for m in reader.branches.values())
        d['gradient_norm']=grad_step(reader,model,base,optimizer,scaler)
        details.append(d);C.log('HYBRID SMOKE '+json.dumps(d))
    with torch.no_grad(),reader.mode(False),torch.amp.autocast('cuda',dtype=torch.float16):
        for row in rr:
            native=U.native_body(model,row)
            after=U.suffix_logits(model,row,U.clone_cache(native,model),[],[0]).cpu()
            assert torch.equal(after,teacher_before[row['id']])
    assert R.base_digest(base)==frozen
    mark(ROOT/'smoke/completed.json',status='completed',task_checks=details,old_gsm_loss_exact_parity=parity,
         all72_reader_branches_receive_gradients=True,frozen_ab_unchanged=True,
         teacher_bitwise_unchanged=True,fresh_reader_zero_init_bitwise=True,memory=R.memory())
    status(status='smoke_passed',phase='smoke')


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
            for position, (row, online) in enumerate(U.epoch_order(epoch), 1):
                if epoch == start and position <= position0:
                    continue
                d = loss_example(model, tok, base, reader, row, target_map[row['id']],
                                 backward=lambda value: scaler.scale(value).backward(), divisor=CFG['effective_batch'], epoch=epoch, student=online)
                details.append(d)
                if position % CFG['effective_batch']:
                    continue
                norm = grad_step(reader, model, base, optimizer, scaler)
                step += 1
                result = dict(epoch=epoch, step=step, position=position,
                              **{k: sum(d[k] for d in details)/len(details) for k in ['loss','ce','kl']},
                              supervised_positions=sum(d['positions'] for d in details), gradient_norm=norm,
                              student_gsm_examples=sum(d['task']=='gsm8k' and d['prefix_source']=='student' for d in details),
                              seconds=time.monotonic()-tick,
                              per_task={t: {k: sum(d[k] for d in details if d['task']==t)/2 for k in ['loss','ce','kl']}
                                        for t in CFG['train_counts']}, memory=R.memory())
                stream.write(json.dumps(result)+'\n'); stream.flush()
                status(status='running', phase='reader_training', **result)
                if step <= 4 or step % 4 == 0:
                    C.log(f'HYBRID READER epoch={epoch}/4 step={step}/1024 loss={result["loss"]:.5f} CE={result["ce"]:.5f} KL={result["kl"]:.5f} seconds={result["seconds"]:.2f}')
                details, tick = [], time.monotonic()
                if step % CFG['checkpoint_every_steps']==0 and position < 2048:
                    save_checkpoint(folder/'last.pt', reader, optimizer, scaler, epoch, step, position, False, history)
            assert R.base_digest(base) == frozen
            metrics = validate(model, tok, base, reader, target_map)
            metrics['gsm8k_free_generation']=validation_generate(model,tok,base,reader,epoch,step)
            entry = dict(epoch=epoch, step=step, validation=metrics)
            history = [v for v in history if v['epoch'] != epoch] + [entry]
            C.save_tensor(folder/f'epoch{epoch}_step{step}.pt', dict(state=reader.state_dict(), manifest_sha256=identity(),
                          epoch=epoch, step=step, diagnostic_only=epoch!=4, state_scope='reader_lora_only',
                          frozen_stage_a_sha256=CFG['base_stage_a_sha256'], frozen_stage_b_sha256=CFG['frozen_stage_b_sha256']))
            mark(folder/'validation.json', history=history, selection=CFG['selection'])
            save_checkpoint(folder/'last.pt', reader, optimizer, scaler, epoch, step, position, True, history)
            C.log('HYBRID READER EPOCH COMPLETED '+json.dumps(entry))
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
        prior_path = SUPERVISED/'evaluation'/task/'per_sample'/(U.fname(row)+'.json')
        prior = json.loads(prior_path.read_text())
        assert all(prior[k]==row[k] for k in ['id','task','prompt','gold'])
        conditions = json.loads(json.dumps(prior['conditions']))
        if task=='gsm8k':
            for arm, value in conditions.items():
                value['original_prediction'],value['original_correct']=value['prediction'],value['correct']
                value['prediction']=C.S.extract_numeric_answer(clean_controls(value['generation']))
                value['correct']=int(value['prediction']==C.S.normalize_number(row['gold']))
        native, k, v = R.fixed_state(model, base, reader, row)
        with reader.mode(True):
            if task=='gsm8k':
                cache = U.translated_cache(model,row,native,k,v)
                text, count = C.R.generate(rt, torch.tensor([row['suffix_ids']],device='cuda'),
                    U.CFG['gsm8k_formal_max_new_tokens'], past_key_values=cache,
                    past_length=len(row['frames']['qwen']['body_ids']),temperature=0.)
                prediction = C.S.extract_numeric_answer(clean_controls(text))
                value = dict(prediction=prediction, correct=int(prediction==C.S.normalize_number(row['gold'])),
                             generation=text, generated_tokens=count, reached_cap=int(count>=U.CFG['gsm8k_formal_max_new_tokens']))
            else:
                value = U.candidate_scores(model,tok,row,native,k,v)
        conditions['hybrid_reader'] = value
        mark(path, id=row['id'], task=task, prompt=row['prompt'], gold=row['gold'], conditions=conditions,
             baseline_source=str(prior_path))
        status(status='running', phase='evaluation', task=task, sample=i, total=len(rr))
        if i%8==0 or task=='gsm8k': C.log(f'HYBRID READER evaluation {task} {i}/{len(rr)}')
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
        a = [r['conditions']['hybrid_reader']['correct'] for r in rr]
        for arm in rr[0]['conditions']:
            vv = [r['conditions'][arm] for r in rr]
            metrics[arm] = dict(n=expected, correct=sum(v['correct'] for v in vv), accuracy=sum(v['correct'] for v in vv)/expected)
            if task=='gsm8k':
                metrics[arm].update(mean_generated_tokens=sum(v['generated_tokens'] for v in vv)/expected,
                                    cap_count=sum(v['reached_cap'] for v in vv))
            if arm in ['qwen_native','grouped_residual','grouped_residual_reader','task_supervised_reader']:
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
    C.log('HYBRID READER TRAINING AND ALL EVALUATIONS COMPLETED')


if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare': prepare()
        else:
            verify()
            if action=='verify': C.log('HYBRID READER MANIFEST VERIFIED')
            elif action=='smoke': smoke()
            elif action=='train': train()
            elif action=='evaluate': evaluate(sys.argv[2])
            elif action=='summarize': summarize()
            else: raise ValueError(action)
    except Exception as exc:
        save(ROOT/'failure.json',dict(type=type(exc).__name__,message=str(exc),time=time.strftime('%F %T')))
        status(status='failed',phase=sys.argv[1],error=str(exc))
        raise
