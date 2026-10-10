import hashlib
import json
import random
from pathlib import Path
from runtime import CFG,ROOT,save,tokenizers
from prompt_protocol import paper_text,paper_protocol,render


def prepare():
    st,rt=tokenizers(); rng=random.Random(CFG['seed'])
    data={s:json.loads((Path(CFG['data'])/f'qasper-{s}-v0.3.json').read_text()) for s in ('train','dev','test')}
    ids={s:set(d) for s,d in data.items()}
    assert not(ids['train']&ids['dev'] or ids['train']&ids['test'] or ids['dev']&ids['test'])
    hashes={s:{hashlib.sha256(paper_text(p).encode()).hexdigest() for p in d.values()} for s,d in data.items()}
    train_ids=sorted(ids['train']); rng.shuffle(train_ids)
    hold=set(train_ids[:max(1,len(train_ids)//10)])
    eligible={}; stats={}
    for split in ('train','dev'):
        rows=[]; rejected={}; lengths={1024:0,2048:0,3072:0,4096:0}
        for number,(pid,p) in enumerate(sorted(data[split].items()),1):
            if number%32==0: print(f'PREPARE {split} {number}/{len(data[split])}',flush=True)
            reason=None; body=paper_text(p); digest=hashlib.sha256(body.encode()).hexdigest()
            qs=list({q['question_id']:q for q in p['qas']}.values())
            if len(qs)<CFG['min_queries_per_paper']: reason='fewer_than_minimum_queries'
            elif split=='train' and pid in hold: reason='train_internal_heldout_paper'
            elif split=='train' and digest in (hashes['dev']|hashes['test']): reason='cross_split_text_duplicate'
            if reason is None:
                try:
                    first=render(rt,body,qs[0]['question'])
                    if len(first['spans'])>=CFG['min_tokens']:
                        for limit in lengths:
                            if len(first['spans'])<=limit: lengths[limit]+=1
                    if not CFG['min_tokens']<=len(first['spans'])<=CFG['max_tokens']: reason='length'
                    else: protocol=paper_protocol(st,rt,body,qs)
                except (AssertionError,ValueError) as ex: reason='protocol_boundary_failure'
            if reason is None and not CFG['min_tokens']<=protocol['body_count']<=CFG['max_tokens']: reason='length'
            if reason:
                rejected[reason]=rejected.get(reason,0)+1; continue
            rng.shuffle(qs)
            held=qs[-1:] if split=='train' and len(qs)>=3 else []
            train_q=qs[:-1] if held else qs
            rows.append({'paper_id':pid,'paper':body,'questions':qs,'train_questions':train_q,
                         'heldout_questions':held,'body_tokens':protocol['body_count'],'text_sha256':digest})
        rng.shuffle(rows); eligible[split]=rows
        stats[split]={'official_papers':len(data[split]),'eligible':len(rows),'rejected':rejected,
                      'first_query_token_grid_counts_by_max_length':lengths}
    save(ROOT/'preparation_counts.json',stats)
    save(ROOT/'available_papers.json',eligible)
    actual=min(len(eligible['train']),CFG['train_papers'])
    if actual<CFG['train_papers']:
        if not CFG['use_available_if_insufficient']: raise RuntimeError(f'Insufficient Train: {stats}')
        print(f"EXPLICIT AVAILABLE BUDGET: requested={CFG['train_papers']} actual={actual}; no truncation or Dev/Test borrowing",flush=True)
    if actual==0: raise RuntimeError('No eligible Train papers')
    pilot=json.loads((Path(CFG['fixed_pilot'])/'prepared.json').read_text())
    dev=pilot['dev']
    assert len(dev)==CFG['dev_papers']
    for row in dev:
        assert row['paper_id'] in ids['dev'] and row['paper']==paper_text(data['dev'][row['paper_id']])
    train=eligible['train'][:actual]
    assert not({r['paper_id'] for r in train}&{r['paper_id'] for r in dev})
    # Recheck existing short-paper gates plus one longest selected paper.
    longest=max(train,key=lambda p:p['body_tokens'])
    smoke=[p for p in pilot['smoke'] if p['paper_id']!=longest['paper_id']][:CFG['smoke_papers']-1]+[longest]
    selected={'train':train,'dev':dev,'smoke':smoke}
    save(ROOT/'prepared.json',selected)
    save(ROOT/'data_audit.json',{'stats':stats,'internal_train_holdout_ids':sorted(hold),
        'selected_counts':{k:len(v) for k,v in selected.items()},'test_read_for_overlap_audit_only':True,
        'requested_train_papers':CFG['train_papers'],'actual_train_papers':actual,
        'fixed_dev_source':CFG['fixed_pilot'],'fixed_dev_question_ids':[q['question_id'] for p in dev for q in p['questions']],
        'train_body_length_buckets':dict(__import__('collections').Counter('512-2048' if p['body_tokens']<=2048 else '2049-4096' for p in train)),
        'train_questions':sum(len(p['train_questions']) for p in train),
        'stage_a_planned_updates':__import__('math').ceil(actual/CFG['stage_a_batch_papers'])*CFG['stage_a_epochs'],
        'stage_b_planned_updates':__import__('math').ceil(sum(len(p['train_questions']) for p in train)/CFG['stage_b_batch_queries'])*CFG['stage_b_epochs'],
        'first_answer_type_counts':{split:dict(__import__('collections').Counter(
            'none' if q['answers'][0]['answer']['unanswerable'] else 'answerable' for p in pp for q in p['questions'])) for split,pp in selected.items()},
        'cache_estimate_bytes_per_paper':(28+36)*CFG['max_tokens']*8*128*2*2,
        'bounded_cpu_paper_cache':CFG['cpu_cache_papers']})
    print('PREPARED',stats,flush=True)
    print('SELECTED',{'train':actual,'dev':len(dev),'training_queries':sum(len(p['train_questions']) for p in train)},flush=True)


if __name__=='__main__': prepare()
