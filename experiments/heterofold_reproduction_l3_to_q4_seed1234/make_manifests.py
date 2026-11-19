"""Input-only calibration adapters. Input exports must come from TRAIN splits."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def records(path):
    path=Path(path)
    if path.suffix=='.jsonl': return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    obj=json.loads(path.read_text())
    if not isinstance(obj,list): raise ValueError('Export dataset training records as a JSON list or JSONL')
    return obj


def adapt(row,corpus,index):
    if corpus=='open_r1':
        q=row.get('problem') or row.get('question')
        if not isinstance(q,str): raise ValueError('Open-R1 export requires problem/question string; never use solution')
        text=q; begin=0
    else:
        q=row['question']; context=row['context']
        if isinstance(context,dict): context=list(zip(context['title'],context['sentences']))
        evidence='\n\n'.join(title+'\n'+''.join(sentences) for title,sentences in context)
        text='Context:\n'+evidence+'\n\nQuestion: '+q; begin=len(text)-len(q)
    return {'id':str(row.get('id',row.get('_id',index))),'corpus':corpus,'text':text,
            'question':q,'question_start':begin,'question_end':begin+len(q)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--open-r1-train',required=True);p.add_argument('--hotpotqa-train',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--train-count',type=int,default=800);p.add_argument('--heldout-count',type=int,default=200)
    p.add_argument('--exclude-texts',help='JSONL calibration/evaluation manifest: prohibit overlapping input texts')
    args=p.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    excluded=set()
    if args.exclude_texts:
        excluded={hashlib.sha256(r['text'].encode()).hexdigest() for r in records(args.exclude_texts)}
    sets={'train':[],'heldout':[]};audit={}
    for corpus,path in [('open_r1',args.open_r1_train),('hotpotqa',args.hotpotqa_train)]:
        rows=[];seen=set();ids=set()
        for i,r in enumerate(records(path)):
            a=adapt(r,corpus,i);digest=hashlib.sha256(a['text'].encode()).hexdigest()
            if digest in seen or digest in excluded or a['id'] in ids: continue
            seen.add(digest);ids.add(a['id']);rows.append(a)
        random.Random(args.seed).shuffle(rows)
        n=args.train_count; h=args.heldout_count
        if len(rows)<n+h: raise ValueError(f'{corpus}: {len(rows)} unique inputs, need {n+h}')
        sets['train']+=rows[:n];sets['heldout']+=rows[n:n+h]
        audit[corpus]={'source':str(Path(path).resolve()),'source_split_asserted_by_operator':'train',
                       'unique_count':len(rows),'train':n,'heldout':h}
    for split,rows in sets.items():
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Manifests written; set train_manifest/heldout_manifest in config.json. Overlength inputs are rejected, not truncated.')


if __name__=='__main__': main()
