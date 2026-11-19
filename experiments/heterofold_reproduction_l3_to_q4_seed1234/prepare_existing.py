"""User-authorized calibration replacement. Official TRAIN only, no gold rendering."""
import hashlib
import json
import random
import zipfile
from pathlib import Path
from common import CFG,ROOT,tokenizers,save
from capture_kv import protocol


def jsonl(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def input_row(row,corpus,index):
    if corpus=='gsm8k':
        q=row['question'];text='Question:\n'+q+'\n\nAnswer:'
    else:
        q=row['question']['stem'];choices=row['question']['choices']
        text='Question:\n'+q+'\n\nOptions:\n'+'\n'.join(c['label']+'. '+c['text'] for c in choices)+'\n\nAnswer:'
    return {'id':str(row.get('id',index)),'corpus':corpus,'text':text,'question':q,
            'question_start':len('Question:\n'),'question_end':len('Question:\n')+len(q)}


def main():
    # Data paths are inherited read-only from an established server-B experiment.
    source=Path('/hy-tmp/yezhe/异构模型/unified_fullbody_hub_kl4ep_l3_to_q4_seed1234/config.json')
    paths=json.loads(source.read_text())['datasets'];st,rt=tokenizers()
    with zipfile.ZipFile(paths['arc_challenge_zip']) as z:
        members=[n for n in z.namelist() if n.endswith('/ARC-Challenge-Train.jsonl') and '__MACOSX' not in n]
        assert len(members)==1
        arc=[json.loads(s) for s in z.read(members[0]).decode().splitlines() if s.strip()]
        excluded_members=[n for n in z.namelist() if n.endswith(('/ARC-Challenge-Dev.jsonl','/ARC-Challenge-Test.jsonl')) and '__MACOSX' not in n]
        arc_excluded=[json.loads(s) for n in excluded_members for s in z.read(n).decode().splitlines() if s.strip()]
    raw={'openbookqa':jsonl(paths['openbookqa']['train']),'arc_challenge':arc,'gsm8k':jsonl(paths['gsm8k_train'])}
    eval_raw={'openbookqa':jsonl(paths['openbookqa']['validation'])+jsonl(paths['openbookqa']['test']),
              'arc_challenge':arc_excluded,'gsm8k':jsonl(paths['gsm8k_test'])}
    sets={'train':[],'heldout':[]};audit={};global_seen=set()
    for corpus,rows in raw.items():
        excluded={input_row(r,corpus,i)['question'].strip() for i,r in enumerate(eval_raw[corpus])}
        eligible=[];seen=set()
        for i,r in enumerate(rows):
            item=input_row(r,corpus,i);digest=hashlib.sha256(item['text'].encode()).hexdigest()
            if digest in seen or digest in global_seen or item['question'].strip() in excluded: continue
            seen.add(digest)
            p=protocol(st,rt,item)
            if max(len(p['source']['ids']),len(p['target']['ids']))>CFG['max_context_tokens']: continue
            eligible.append(item);global_seen.add(digest)
        random.Random(CFG['seed']).shuffle(eligible)
        n,h=CFG['train_counts'][corpus],CFG['heldout_counts'][corpus]
        assert len(eligible)>=n+h,(corpus,len(eligible),n+h)
        sets['train']+=eligible[:n];sets['heldout']+=eligible[n:n+h]
        audit[corpus]={'source_split':'official train','raw_count':len(rows),'eligible':len(eligible),
            'calibration_train':n,'calibration_heldout':h,'excluded_official_dev_test_questions':len(excluded),
            'gold_fields_rendered':False}
    ROOT.joinpath('manifests').mkdir(exist_ok=True)
    for split,rows in sets.items():
        target=ROOT/'manifests'/f'{split}.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        if target.exists(): assert target.read_text()==content,'Existing split differs; never overwrite silently'
        else: target.write_text(content,encoding='utf-8')
    save(ROOT/'manifests/audit.json',{'sources_config':str(source),'corpora':audit,'QASPER_used_for_calibration':False,
         'statistical_weight':'one third token-weight mass per corpus','calibration_weight':'equal examples'})
    print('EXISTING DATA PREPARED',len(sets['train']),len(sets['heldout']),flush=True)


if __name__=='__main__':main()
