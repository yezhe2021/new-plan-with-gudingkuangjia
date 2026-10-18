"""Native-cache generation gate and resumable, teacher-verified training references."""
import argparse
from collections import Counter, defaultdict
import gc
import json
import time

import torch

from trajectory_common import (CONFIG, HERE, PARENT, CFG, FullSyncHFLM, load_model, seed_all,
                               previous_runtime, log, save, read_rows, write_rows, gpu_check, digest, identity)
from trajectory_data import make_question, prompt, acceptance, training_record, rng_for, question
from kv_backend import capture, encoding, forward_continuation


def rows_by_task(split):
    pools = defaultdict(list)
    for row in read_rows(PARENT / CONFIG['parent_' + split + '_data']):
        pools[row['task']].append(row)
    for task in pools:
        pools[task] = sorted(pools[task], key=lambda r: rng_for('order', split, r['id']).random())
    return pools


@torch.no_grad()
def generate(hf, context, cap):
    ids = hf.tok_encode(context)
    tensor = torch.tensor([ids], device='cuda')
    hf._active_contexts = {tuple(ids): context}
    try:
        output = hf._model_generate(tensor, len(ids) + cap, ['Question:'], do_sample=False, temperature=0.0)
    finally:
        hf._active_contexts = {}
    generated = output[0, len(ids):].tolist()
    return generated, hf.tokenizer.decode(generated, skip_special_tokens=True)


def gate():
    previous_runtime.verify_manifest()
    gpu_check()
    seed_all(CONFIG['seed'])
    receiver = load_model(CFG, 'qwen')
    native = FullSyncHFLM(receiver, mode='native')
    oracle = FullSyncHFLM(receiver, mode='oracle')
    pool = rows_by_task('training')
    flat = [r for rows in pool.values() for r in rows]
    qs = [make_question(r, 'gate') for r in flat[:20]]
    contexts = [prompt(qs[0]), prompt(qs[1]), prompt(qs[-1])]
    # Purely an interface probe using training questions and explicit gold demos.
    for i in (1,2):
        count = 5 if i == 1 else 18
        from trajectory_data import render, DESCRIPTION
        contexts[i] = DESCRIPTION + ''.join(render(q) + f' The answer is ({q["gold"]}).\n\n' for q in qs[:count]) + render(qs[-1])
    reports = []
    for context in contexts:
        ids = native.tok_encode(context)
        with torch.no_grad():
            k,v = oracle._prefix(context, ids)
            empty = []
            n = receiver.model(input_ids=torch.tensor([ids],device='cuda'), use_cache=False).last_hidden_state[0,-1]
            from protocol import final_logits
            direct = receiver.lm_head(n).float()
            restored = final_logits(receiver, ids[-1:], k,v, torch.arange(len(ids)-1,device='cuda'),len(ids)-1)
            prob_error = float((direct.softmax(-1)-restored.softmax(-1)).abs().max())
            # Control FP16 prefill-vs-decode matrix-shape rounding independently
            # of pre-RoPE capture/reconstruction: use Qwen's own unmodified cache.
            native_prefix = receiver.model(input_ids=torch.tensor([ids[:-1]],device='cuda'),
                                           attention_mask=torch.ones((1,len(ids)-1),device='cuda',dtype=torch.long),
                                           use_cache=True,return_dict=True).past_key_values
            native_last = receiver.model(input_ids=torch.tensor([ids[-1:]],device='cuda'),
                                         attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),
                                         position_ids=torch.tensor([[len(ids)-1]],device='cuda'),
                                         past_key_values=native_prefix,use_cache=False,return_dict=True)
            control = receiver.lm_head(native_last.last_hidden_state[0,-1]).float()
            control_error = float((control.softmax(-1)-restored.softmax(-1)).abs().max())
            split_rounding = float((direct.softmax(-1)-control.softmax(-1)).abs().max())
            del native_prefix,native_last
            nx,_ = generate(native, context,32)
            ox,_ = generate(oracle, context,32)
        record = {'context_tokens':len(ids),'same_first_token':int(direct.argmax())==int(restored.argmax()),
                  'max_probability_error':prob_error,'same_32_token_generation':nx==ox,
                  'native_split_control_probability_error':control_error,
                  'native_full_vs_split_probability_error':split_rounding,
                  'native_ids':nx,'oracle_ids':ox}
        reports.append(record)
        log('Native/oracle gate: ' + json.dumps(record))
    passed = all(r['same_first_token'] and r['native_split_control_probability_error'] <= 1e-5
                 and r['max_probability_error'] <= r['native_full_vs_split_probability_error'] + 1e-5
                 and r['same_32_token_generation'] for r in reports)
    save(HERE/'preflight.json',{'passed':passed,'reports':reports,'qwen_config':CFG['models']['qwen'],
                              'eos_ids':receiver.generation_config.eos_token_id,'no_benchmark_questions':True})
    assert passed, 'Native KV round-trip generation failed; inspect before training'
    log('NATIVE KV GENERATION GATE PASSED')


def reference(hf, q, demos, verbose=False):
    context = prompt(q,demos,verbose)
    if verbose:
        # Bootstrap high-quality training-only demonstrations with Qwen's native
        # non-thinking chat format. Actual trajectory KD and benchmarks remain
        # raw completions with the same teacher/student context.
        context = hf.tokenizer.apply_chat_template([{'role':'user','content':context}],
                                                   tokenize=False,add_generation_prompt=True,enable_thinking=False)
    n = len(hf.tok_encode(context))
    if n > CONFIG['teacher_prompt_max_tokens']:
        return None, {'reason':'context_budget','context_tokens':n}
    ids,text = generate(hf, context, CONFIG['teacher_max_new_tokens'])
    eos = hf.model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    audit = acceptance(text,ids,eos,q['gold'],CONFIG['teacher_max_answer_matches'])
    raw = {**q,'context':context,'demo_ids':[d['base_id'] for d in demos], 'ids':ids,'text':text,'audit':audit,'context_tokens':n,
           'reference_prompt_policy':CONFIG['demo_reference_prompt'] if verbose else 'raw_five_shot_completion'}
    if not audit['accepted']:
        return None,raw
    return training_record(q,context,demos,ids,text,audit,hf.tokenizer),raw


def append_raw(path, record):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()


def prepare(bootstrap=False):
    previous_runtime.verify_manifest()
    assert json.loads((HERE/'preflight.json').read_text())['passed']
    gpu_check(); seed_all(CONFIG['seed'])
    receiver = load_model(CFG,'qwen')
    hf = FullSyncHFLM(receiver,mode='native')
    folder=HERE/'prepared'; folder.mkdir(exist_ok=True)
    ident = {'config_sha256':digest(HERE/'config.json'),'stage_a_sha256':digest(PARENT/CONFIG['parent_stage_a']),
             'stage_b_sha256':digest(previous_runtime.HERE/'stage_b/best.pt')}
    idpath=folder/'reference_identity.json'
    if idpath.exists():
        assert json.loads(idpath.read_text())==ident, 'Stale reference cache'
    else:
        save(idpath,ident)
    train=rows_by_task('training'); validation=rows_by_task('validation')
    demo_path=folder/'demos.jsonl'
    demos=read_rows(demo_path) if demo_path.exists() else []
    demo_raw=folder/'demo_attempts.jsonl'
    tried={r['base_id'] for r in read_rows(demo_raw)} if demo_raw.exists() else set()
    for task in sorted(train):
        for row in train[task][:CONFIG['demo_candidate_budget_per_task']]:
            if sum(d['task']==task for d in demos)>=CONFIG['demo_pool_per_task']:
                break
            if row['id'] in tried:
                continue
            q=make_question(row,'demo')
            record,raw=reference(hf,q,[],verbose=True)
            append_raw(demo_raw,raw); tried.add(row['id'])
            if record and len(record['continuation_ids'])>=CONFIG['teacher_demo_min_tokens']:
                demos.append(record); write_rows(demo_path,demos)
            log(f'TEACHER DEMOS {task}: accepted={sum(d["task"]==task for d in demos)}/{CONFIG["demo_pool_per_task"]}; last={raw.get("audit",{})}')
        assert sum(d['task']==task for d in demos)==CONFIG['demo_pool_per_task'], 'Insufficient verified complete demo references'
    demo_ids={d['base_id'] for d in demos}
    train_questions=[]
    for task in sorted(train):
        candidates=[r for r in train[task][CONFIG['demo_candidate_budget_per_task']:] if r['id'] not in demo_ids]
        train_questions.extend(candidates)
    fixed_validation=[r for task in sorted(validation) for r in validation[task][:CONFIG['validation_questions_per_task']]]
    write_rows(folder/'validation_questions.jsonl',fixed_validation)
    settings=[('train',train_questions,CONFIG['train_trajectories_per_task']),('validation',fixed_validation,None)]
    for split,pool,target in settings:
        dest=folder/(split+'_trajectories.jsonl')
        accepted=read_rows(dest) if dest.exists() else []
        rawpath=folder/(split+'_attempts.jsonl')
        existing=read_rows(rawpath) if rawpath.exists() else []
        tried={r['base_id'] for r in existing}
        for row in pool:
            if target and sum(a['task']==row['task'] for a in accepted)>=target:
                continue
            if row['id'] in tried:
                continue
            q=make_question(row,split)
            shots=CONFIG['teacher_shots']
            selected=rng_for('demos',split,row['id']).sample([d for d in demos if d['task']==row['task']],shots)
            record,raw=reference(hf,q,selected)
            append_raw(rawpath,raw); tried.add(row['id'])
            if record:
                accepted.append(record); write_rows(dest,accepted)
            counts=Counter(a['task'] for a in accepted)
            save(HERE/'status.json',{'phase':'teacher_references','split':split,'accepted_by_task':dict(counts),'attempted':len(tried),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            log(f'TEACHER {split}: {dict(counts)} accepted; attempted={len(tried)}; tokens={len(raw.get("ids",[]))}; terminal={raw.get("audit",{})}')
            if bootstrap and split=='train' and len(accepted)>=2:
                save(HERE/'bootstrap.json',{'passed':True,'train_references':len(accepted),'demos':len(demos)})
                return
        if target:
            assert all(sum(a['task']==t for a in accepted)==target for t in train), 'Not enough complete correct training references'
        else:
            assert len(accepted)>=8, 'Too few held-out complete correct references'
    all_train=read_rows(folder/'train_trajectories.jsonl'); all_val=read_rows(folder/'validation_trajectories.jsonl')
    train_ids={r['id'] for pool in train.values() for r in pool}; val_ids={r['id'] for r in fixed_validation}
    assert not train_ids & val_ids
    assert not {question(r) for pool in train.values() for r in pool} & {question(r) for r in fixed_validation}
    for row in all_train+all_val:
        assert set(row['demo_ids'])<=train_ids and row['base_id'] not in row['demo_ids']
        assert row['stop']['accepted'] and row['stop']['terminal']!='length_limit'
    summary={'train_trajectories':len(all_train),'validation_trajectories':len(all_val),'fixed_validation_questions':len(fixed_validation),
             'teacher_filter_bias':'Trajectory validation conditions on teacher-correct complete references; independent fixed validation questions remain unfiltered.',
             'train_context_tokens_min_max':[min(len(r['context_ids']) for r in all_train),max(len(r['context_ids']) for r in all_train)],
             'train_output_tokens_min_max':[min(len(r['continuation_ids']) for r in all_train),max(len(r['continuation_ids']) for r in all_train)],
             'terminal_counts':dict(Counter(r['stop']['terminal'] for r in all_train)),
             'gold_counts':dict(Counter(r['gold'] for r in all_train)), 'benchmark_questions_used':False}
    save(folder/'summary.json',summary)
    log('TEACHER PREPARATION COMPLETED: '+json.dumps(summary))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['gate','bootstrap','prepare']); args=ap.parse_args()
    if args.action=='gate': gate()
    else: prepare(args.action=='bootstrap')
