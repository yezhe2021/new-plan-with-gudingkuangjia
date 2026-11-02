"""CPU-only source selection; never uses evaluation outcomes or truncates evidence."""
import random
import zipfile
from common import *

def main():
    if (ROOT/'manifest.json').exists():verify();return
    C.verify();assert C.identity()==CFG['parent_manifest_sha256']
    dependency=Path(CFG['dependency']);assert C.sha(dependency/'manifest.json')==CFG['dependency_manifest_sha256']
    cp=PARENT/'runs/C/epoch4_step800.pt';assert C.sha(cp)==CFG['c_checkpoint_sha256']
    prior=C.rows('train')+C.rows('validation')+C.rows('test')
    fresh=C.read_jsonl(dependency/'prepared/test.jsonl')
    excluded={C.norm(r['question']) for r in prior+fresh}
    excluded_ids={str(r[k]) for r in prior+fresh for k in ['source_id','hotpot_id'] if r.get(k)}
    with zipfile.ZipFile(CFG['arc_zip']) as z:
        arc={}
        for split,label in [('train','Train'),('validation','Dev')]:
            name=next(n for n in z.namelist() if n.endswith('/ARC-Challenge-'+label+'.jsonl') and not n.startswith('__MACOSX'))
            data=[json.loads(s) for s in z.read(name).decode().splitlines() if s]
            arc[split]=[dict(id=d['id'],question=d['question']['stem'],answerKey=d['answerKey'],choices={k:[c[k] for c in d['question']['choices']] for k in ['label','text']}) for d in data]
    import pyarrow.parquet as pq
    table=pq.read_table(CFG['hotpot_parquet'],columns=['id','question','answer','context'])
    # Read questions first; materialize long paragraph arrays only for selected rows.
    hids=table.column('id').to_pylist();questions=table.column('question').to_pylist()
    orders={}
    for split in ['train','validation']:
        ix=list(range(len(arc[split])));random.Random(CFG['seed']).shuffle(ix);orders[split]=ix
    hi=list(range(len(hids)));random.Random(CFG['seed']).shuffle(hi)
    tok=C.AutoTokenizer.from_pretrained(C.CFG['models']['receiver'],local_files_only=True)
    m=C.read(PARENT/'manifest.json');assistant=m['assistant'];selected={s:[] for s in ['train','validation','test']}
    used=set(excluded)
    def add(doc,task,split,source_index):
        question=doc['question'];key=C.norm(question)
        if key in used or str(doc['id']) in excluded_ids:return False
        if task=='arc':
            prompt,candidates,gold,kind,_=C.SC.format_item('arc',doc)
            row=dict(candidates=candidates,gold=gold,normalize=False,continuation_suffix=assistant+C.SC.ANSWER_CUE[kind])
            answer=candidates[gold]
        else:
            context=doc['context'];passages=['Passage '+str(i)+':\n'+title+'\n'+''.join(sentences).strip() for i,(title,sentences) in enumerate(zip(context['title'],context['sentences']),1)]
            prompt=('Answer the question using the passages below. Output only the short answer, without an explanation.\n\n'+'\n\n'.join(passages)+'\n\nQuestion: '+question+'\nGive only the answer.')
            answer=doc['answer'];row=dict(gold=answer,paragraph_count=len(passages),context_sha256=C.digest(context),continuation_suffix=assistant)
        row.update(id=task+'_'+split+'_'+str(doc['id']),source_id=str(doc['id']),source_index=source_index,question=question,prompt=prompt,task=task)
        row['content_ids']=tok.encode(prompt,add_special_tokens=False);row['suffix_ids']=tok.encode(row['continuation_suffix'],add_special_tokens=False)
        row['gold_tokens']=tok.encode(answer,add_special_tokens=False)
        if task=='hotpotqa':row['gold_tokens'].append(tok.convert_tokens_to_ids('<|im_end|>'))
        assert row['gold_tokens']
        assert tok.encode(m['prefix']+prompt+row['continuation_suffix'],add_special_tokens=False)==m['prefix_ids']+row['content_ids']+row['suffix_ids']
        assert len(m['prefix_ids'])+len(row['content_ids'])+32+len(row['suffix_ids'])+max(64,len(row['gold_tokens']))<CFG['max_context_tokens'],('No truncation: source exceeds model context',row['id'])
        selected[split].append(row);used.add(key);return True
    for split in ['train','validation']:
        for i in orders[split]:
            add(arc[split][i],'arc',split,i)
            if sum(r['task']=='arc' for r in selected[split])==CFG[split+'_counts']['arc']:break
        assert sum(r['task']=='arc' for r in selected[split])==CFG[split+'_counts']['arc']
    cursor=0
    for split in ['train','validation']:
        while sum(r['task']=='hotpotqa' for r in selected[split])<CFG[split+'_counts']['hotpotqa']:
            i=hi[cursor];cursor+=1
            if C.norm(questions[i]) in used or str(hids[i]) in excluded_ids:continue
            add(table.slice(i,1).to_pylist()[0],'hotpotqa',split,i)
    selected['test']=[r for r in fresh if r['task'] in CFG['test_counts']]
    assert len(selected['test'])==128
    for split,rr in selected.items():C.write_jsonl(ROOT/'prepared'/(split+'.jsonl'),rr)
    refs=[PARENT/f for f in ['manifest.json','core.py','train.py','residuals.py','metrics.py','runs/C/epoch4_step800.pt']]
    refs += [Path(CFG[k]) for k in ['arc_zip','hotpot_parquet']]+[dependency/'manifest.json',dependency/'prepared/test.jsonl']
    refs += [ROOT/'prepared'/(s+'.jsonl') for s in selected]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},
        sources={str(p):C.sha(p) for p in refs},counts={s:len(rr) for s,rr in selected.items()},
        maxima={s:max(len(r['content_ids']) for r in rr) for s,rr in selected.items()},
        trainable_parameters=409600,train_validation_test_question_overlap=0,excluded_parent_calibration=True,
        original_C_checkpoint=True,test_selection='Exact fixed dependency ARC64/Hotpot64, no outcome-based filtering.'))
    mark('prepared/completed.json',status='completed',counts={s:len(rr) for s,rr in selected.items()},maxima=C.read(ROOT/'manifest.json')['maxima'])
    C.log('RNKV PREPARED',C.read(ROOT/'manifest.json')['counts'])

if __name__=='__main__':main()
