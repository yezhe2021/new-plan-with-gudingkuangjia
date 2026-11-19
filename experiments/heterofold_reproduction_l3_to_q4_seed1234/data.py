"""Explicit input manifests: never silently substitute another calibration corpus."""
import hashlib
import json
import random
from pathlib import Path
from common import CFG,ROOT,save
from capture_kv import protocol


def read_manifest(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Supply Open-R1/HotpotQA input-only JSONL manifests in config.json')
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        if row['corpus'] not in CFG.get('calibration_corpora',['open_r1','hotpotqa']): raise ValueError('Unexpected corpus')
        # Reject gold fields, rather than accidentally rendering a solution.
        if set(row)&{'answer','answers','answerKey','solution','reasoning','response','completion'}:
            raise ValueError('Calibration rows must exclude gold answers/solutions')
        s,e=row['question_start'],row['question_end']
        assert 0<=s<e<=len(row['text']) and row['text'][s:e]==row['question']
        row['text_sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
    return rows


def prepare(st,rt,smoke=False,write=True):
    sets={s:read_manifest(CFG[f'{s}_manifest']) for s in ('train','heldout')}
    a,b=sets.values()
    assert not({r['text_sha256'] for r in a}&{r['text_sha256'] for r in b}), 'Text split leakage'
    assert not({(r['corpus'],r['id']) for r in a}&{(r['corpus'],r['id']) for r in b}), 'ID split leakage'
    if CFG.get('evaluation_manifest'):
        evaluation=[json.loads(line) for line in Path(CFG['evaluation_manifest']).read_text().splitlines() if line.strip()]
        eval_texts={hashlib.sha256(r['text'].encode()).hexdigest() for r in evaluation}
        eval_questions={r['question'] for r in evaluation}
        assert not eval_texts&{r['text_sha256'] for r in a+b},'Calibration/evaluation text overlap'
        assert not eval_questions&{r['question'] for r in a+b},'Calibration/evaluation question overlap'
    out={}; token_totals={}
    for split,rows in sets.items():
        selected=[]; totals={}
        for corpus in CFG.get('calibration_corpora',['open_r1','hotpotqa']):
            candidates=[r for r in rows if r['corpus']==corpus]
            random.Random(CFG['seed']).shuffle(candidates)
            count=CFG.get(split+'_counts',{}).get(corpus,CFG[split+'_per_corpus']) if not smoke else CFG['smoke_'+split+'_per_corpus']
            picked=[]; total=0
            for row in candidates:
                p=protocol(st,rt,row)
                if max(len(p['source']['ids']),len(p['target']['ids']))>CFG['max_context_tokens']: continue
                picked.append(row); total+=len(p['target']['offsets'])
                if len(picked)==count: break
            if len(picked)!=count: raise ValueError(f'{split}/{corpus}: need{count}, eligible{len(picked)}; no silent fallback')
            selected+=picked; totals[corpus]=total
        out[split]=selected; token_totals[split]=totals
    out['token_totals']=token_totals; out['smoke_only']=smoke
    if write: save(ROOT/'runs'/('smoke' if smoke else 'formal')/'prepared.json',out)
    return out
