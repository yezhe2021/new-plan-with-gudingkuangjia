"""Prediction-independent, document-disjoint train/calibration/test preparation."""
import backend as B
from backend import ROOT,CFG,E,read,save,sha,np
from pilot_metrics import seed_for
from pathlib import Path
from collections import Counter
from scipy.optimize import linear_sum_assignment
import json,random,sys,time


def normal(raw):
    context=raw['context'];facts=raw['supporting_facts']
    context=list(zip(context['title'],context['sentences'])) if isinstance(context,dict) else context
    titles=set(facts['title'] if isinstance(facts,dict) else [x[0] for x in facts])
    docs=[(t,s) for t,s in context if t in titles]
    if raw['type']!='bridge' or len(titles)!=2 or len(docs)!=2:return None
    return dict(id=raw.get('_id',raw.get('id')),question=raw['question'].strip(),answer=raw['answer'].strip(),
        titles=sorted(titles),support='\n\n'.join(f'Document {i+1}: {t}\n'+''.join(s).strip() for i,(t,s) in enumerate(docs)))


def make_row(d,tok):
    qt=tok['qwen'];q=d['question'];evidence='Reference information:\n'+d['support']+'\nEnd of reference'
    r=E.U.make_row('hotpot_'+d['id'],'gsm8k',dict(question=q),tok['llama'],qt,prompt=evidence,candidates=[],gold=d['answer'],normalize=False,cue='')
    frame=r['frames']['qwen'];body=frame['body_ids']
    assert not r['native_tail']
    assert len(body)<=CFG['max_evidence_tokens'] and len(r['frames']['llama']['body_ids'])<=CFG['max_sender_tokens']
    before='\n\nAnswer the question. Use the reference only when relevant. Give only the short answer, without explanation.\n\nQuestion: '
    query=before+q;rendered=frame['rendered_prefix']+evidence+query+frame['rendered_suffix']
    encoded=qt(rendered,add_special_tokens=False,return_offsets_mapping=True);ids=list(encoded['input_ids'])
    assert ids[:len(body)]==body and len(ids)+CFG['max_new_tokens']<=CFG['max_total_tokens']
    qstart=len(frame['rendered_prefix'])+len(evidence)+len(before);qend=qstart+len(q)
    qp=[i-len(body) for i,(a,b) in enumerate(encoded['offset_mapping']) if qstart<=a<b<=qend]
    assert qp and min(qp)>0
    gold=qt.encode(rendered+d['answer'],add_special_tokens=False)
    assert gold[:len(ids)]==ids and len(gold)>len(ids)
    r.update(task='hotpotqa_controlled',source_id=d['id'],question=q,answer=d['answer'],support=d['support'],titles=d['titles'],
        continuation_ids=ids[len(body):],question_positions=qp,gold_ids=gold[len(ids):],
        compact_ids=qt.encode(frame['rendered_prefix']+query.lstrip()+frame['rendered_suffix'],add_special_tokens=False))
    return r


def prepare():
    E.verify();out=ROOT/'prepared/questions.jsonl'
    if out.exists():
        assert sha(out)==read(ROOT/'prepared/data_manifest.json')['prepared_sha256'];return
    import pyarrow.parquet as pq
    prior_dirs=[Path(CFG['previous_hotpot'])/'prepared',Path(CFG['previous_pilot'])/'prepared']
    excluded_ids=set();excluded_titles=set();excluded_questions=set();exclusion_files=[]
    for directory in prior_dirs:
        for p in sorted(directory.glob('*.jsonl')):
            exclusion_files.append(p)
            for line in p.read_text().splitlines():
                r=json.loads(line)
                if r.get('id'):excluded_ids.add(r['id'].removeprefix('hotpot_'))
                excluded_titles.update(r.get('titles',[]))
                if r.get('question'):excluded_questions.add(B.norm(r['question']))
    oldm=read(Path(CFG['previous_pilot'])/'manifest.json');excluded_ids.update(oldm['excluded_question_ids'])
    for split in ['train','validation']:excluded_questions.update(r['question_key'] for r in E.U.rows(split))
    train=pq.read_table(CFG['train_source']).to_pylist();test=read(CFG['source_json'])
    random.Random(CFG['seed']+2701).shuffle(train);random.Random(CFG['seed']+2702).shuffle(test)
    tok={f:E.tok(f) for f in ['llama','qwen']};used_ids=set(excluded_ids);used_q=set(excluded_questions);used_titles=set(excluded_titles)
    selected=[];skips=Counter()
    # Test documents reserved first; official train supplies all fit/calibration questions.
    for split,pool in [('test',test),('calibration',train),('train',train)]:
        group=[]
        for raw in pool:
            d=normal(raw)
            if d is None:skips[split+'/not_bridge_two_docs']+=1;continue
            if d['id'] in used_ids or B.norm(d['question']) in used_q:skips[split+'/used_question']+=1;continue
            if set(d['titles'])&used_titles:skips[split+'/used_document']+=1;continue
            try:r=make_row(d,tok)
            except (AssertionError,ValueError):skips[split+'/length_boundary']+=1;continue
            r['split']=split;group.append(r);used_ids.add(d['id']);used_q.add(B.norm(d['question']));used_titles.update(d['titles'])
            if len(group)==CFG['counts'][split]:break
        assert len(group)==CFG['counts'][split],(split,len(group),dict(skips))
        # Local length-matched donor permutations bound the in-memory KV working set.
        group.sort(key=lambda r:len(r['source_indices']))
        for start in range(0,len(group),32):
            block=group[start:start+32];lengths=np.array([len(r['source_indices']) for r in block])
            cost=np.abs(lengths[:,None]-lengths[None,:]).astype(float)
            cost+=np.random.default_rng(seed_for(CFG['seed'],split,start,'donor')).uniform(0,.5,cost.shape)
            np.fill_diagonal(cost,1e9);ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):assert i!=j;block[i]['donor_id']=block[j]['id']
        # Shuffle blocks, not outcomes; keep donor neighbourhoods together.
        blocks=[group[i:i+32] for i in range(0,len(group),32)];random.Random(seed_for(CFG['seed'],split,'blocks')).shuffle(blocks)
        group=[r for block in blocks for r in block];selected.extend(group);E.log(f'PREPARED {split} {len(group)}')
    ids=[r['id'] for r in selected];titles=[t for r in selected for t in r['titles']]
    assert len(ids)==len(set(ids)) and len(titles)==len(set(titles))
    out.parent.mkdir(exist_ok=True);out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    save(ROOT/'prepared/data_manifest.json',dict(counts=CFG['counts'],prepared_sha256=sha(out),skips=dict(skips),
        source_sha256={p:sha(p) for p in [CFG['train_source'],CFG['source_json']]},
        exclusion_sources={str(p):sha(p) for p in exclusion_files},excluded_ids=len(excluded_ids),excluded_titles=len(excluded_titles),
        all_selected_questions_and_document_titles_unique=True,selection_uses_predictions=False))


def freeze():
    E.verify();records=B.rows();data=read(ROOT/'prepared/data_manifest.json')
    assert data['prepared_sha256']==sha(ROOT/'prepared/questions.jsonl')
    anchors={tuple(r['frames']['qwen']['body_ids'][:r['frames']['qwen']['native_prefix_length']]) for r in records};assert len(anchors)==1
    cp={k:v for k,v in read(E.ROOT/'manifest.json')['checkpoints'].items() if k in ['stage_a','stage_b']}
    m=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),prepared_sha256=data['prepared_sha256'],
        reference_manifest_sha256=E.identity(),code={p.name:sha(p) for p in ROOT.glob('*.py')},checkpoints=cp,
        anchor_ids=list(next(iter(anchors))),data_manifest_sha256=sha(ROOT/'prepared/data_manifest.json'),
        no_reader_installed=True,models_frozen=True,counts=CFG['counts'],frozen_at=time.strftime('%F %T'))
    assert not (ROOT/'manifest.json').exists();save(ROOT/'manifest.json',m)
    save(ROOT/'prepared/completed.json',dict(manifest_sha256=B.ident()))


if __name__=='__main__':
    {'prepare':prepare,'freeze':freeze}[sys.argv[1]]()
