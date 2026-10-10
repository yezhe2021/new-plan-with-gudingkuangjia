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
        rows=[]; rejected={}; lengths={1024:0,2048:0,3072:0}
        for number,(pid,p) in enumerate(sorted(data[split].items()),1):
            if number%32==0: print(f'PREPARE {split} {number}/{len(data[split])}',flush=True)
            reason=None; body=paper_text(p); digest=hashlib.sha256(body.encode()).hexdigest()
            qs=list({q['question_id']:q for q in p['qas']}.values())
            if len(qs)<2: reason='fewer_than_two_queries'
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
    for split,count in [('train',CFG['train_papers']),('dev',CFG['dev_papers'])]:
        if len(eligible[split])<count: raise RuntimeError(f'Insufficient {split} papers: {stats}; no silent budget changes')
    smoke=sorted([p for p in eligible['train'] if p['body_tokens']<=CFG['smoke_max_tokens']],key=lambda p:(p['body_tokens'],p['paper_id']))[:CFG['smoke_papers']]
    if len(smoke)<CFG['smoke_papers']: raise RuntimeError(f"Need {CFG['smoke_papers']} complete <={CFG['smoke_max_tokens']}-token smoke papers, found {len(smoke)}; {stats}")
    selected={'train':eligible['train'][:CFG['train_papers']], 'dev':eligible['dev'][:CFG['dev_papers']], 'smoke':smoke}
    save(ROOT/'prepared.json',selected)
    save(ROOT/'data_audit.json',{'stats':stats,'internal_train_holdout_ids':sorted(hold),
        'selected_counts':{k:len(v) for k,v in selected.items()},'test_read_for_overlap_audit_only':True,
        'first_answer_type_counts':{split:dict(__import__('collections').Counter(
            'none' if q['answers'][0]['answer']['unanswerable'] else 'answerable' for p in pp for q in p['questions'])) for split,pp in selected.items()},
        'cache_estimate_bytes_per_paper':(28+36)*CFG['max_tokens']*8*128*2*2,
        'bounded_cpu_paper_cache':CFG['cpu_cache_papers']})
    print('PREPARED',stats,flush=True)


if __name__=='__main__': prepare()
