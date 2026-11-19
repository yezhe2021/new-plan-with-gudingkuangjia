import hashlib
import json
from common_exp import ROOT, CONFIG, BASELINE, CFG, save, tokenizers
from prompt_protocol import paper_text, paper_protocol


def main():
    source = json.loads((BASELINE/'prepared.json').read_text())
    if (ROOT/'prepared.json').exists(): return
    st,rt = tokenizers()
    data = json.loads((__import__('pathlib').Path(CFG['data'])/'qasper-train-v0.3.json').read_text())
    blocked = {r['paper_id'] for rows in source.values() for r in rows}
    blocked_text = {hashlib.sha256(r['paper'].encode()).hexdigest() for rows in source.values() for r in rows}
    for split in ('dev','test'):
        other = json.loads((__import__('pathlib').Path(CFG['data'])/f'qasper-{split}-v0.3.json').read_text())
        blocked.update(other)
        blocked_text.update(hashlib.sha256(paper_text(p).encode()).hexdigest() for p in other.values())
    ids = sorted(data, key=lambda pid: hashlib.sha256(f'{CONFIG["seed"]}:{pid}'.encode()).hexdigest())
    validation = []
    for pid in ids:
        p = data[pid]; body = paper_text(p)
        if pid in blocked or hashlib.sha256(body.encode()).hexdigest() in blocked_text or not p['qas']: continue
        try: proto = paper_protocol(st,rt,body,p['qas'])
        except (AssertionError,ValueError): continue
        if not CFG['min_tokens'] <= proto['body_count'] <= CFG['max_tokens']: continue
        validation.append({'paper_id':pid,'paper':body,'questions':p['qas'],
            'train_questions':[],'heldout_questions':[],'body_tokens':proto['body_count']})
        if len(validation) == CONFIG['internal_validation_papers']: break
    assert len(validation) == CONFIG['internal_validation_papers'], 'Insufficient unused complete papers'
    assert not {r['paper_id'] for r in validation} & {r['paper_id'] for r in source['train']}
    source['validation'] = validation
    save(ROOT/'prepared.json',source)
    save(ROOT/'data_audit.json', {'train_papers':len(source['train']),
        'train_queries':sum(len(r['train_questions']) for r in source['train']),
        'validation_paper_ids':[r['paper_id'] for r in validation],
        'validation_queries':sum(len(r['questions']) for r in validation),
        'selection':'unused official train papers, disjoint from baseline train/dev/smoke and official dev/test by ID/text',
        'dev_report_only':True,'test_not_evaluated':True,'no_truncation':True})
    print('PREPARED INTERNAL VALIDATION',len(validation),flush=True)


if __name__ == '__main__': main()
