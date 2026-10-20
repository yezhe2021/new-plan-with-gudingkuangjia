"""Keep old document IDs; regenerate only the newly fixed input protocol."""
from core import *
import re


def question(doc):
    q=doc.get('question_stem',doc.get('question'))
    return q['stem'] if isinstance(q,dict) else q


def qkey(doc):return re.sub(r'\s+',' ',question(doc)).strip().casefold()


def raw_mcq(doc):
    d=doc.get('question',doc)
    if not isinstance(d,dict):d=doc
    choices=d['choices']
    if isinstance(choices,list):choices={k:[v[k] for v in choices] for k in ['label','text']}
    return dict(id=str(doc.get('id','')),question=question(doc),choices=choices,answerKey=str(doc['answerKey']))


def prepare_row(doc_id,task,doc,lt,qt):
    prompt,candidates,gold,kind,extra=S.format_item('arc',raw_mcq(doc))
    eligible=set(frame(lt,'llama','chat',prompt))&set(frame(qt,'qwen','chat',prompt))
    eligible=[x for x in eligible if 0<x<len(prompt) and len(prompt[x:])<=8 and not any(c.isalnum() for c in prompt[x:])]
    assert eligible,(doc_id,'No punctuation-only boundary')
    end=max(eligible)
    frames={f:frame(t,f,'chat',prompt,end) for f,t in [('llama',lt),('qwen',qt)]}
    text=prompt[:end]
    aligned=dict(id=doc_id,body=text,encoded={f:dict(body=encode_prefix(t,text),option_token_indices=list(range(1,len(encode_prefix(t,text))))) for f,t in [('llama',lt),('qwen',qt)]})
    target,source,counts=fullsync_map(aligned,lt,qt,'llama')
    assert target==list(range(1,len(frames['qwen']['content_ids'])))
    source=[x-int(lt.bos_token_id is not None)+len(frames['llama']['prefix_ids']) for x in source]
    prefix,assistant=chat_affixes(qt,thinking=False)
    suffix=frames['qwen']['rendered_suffix']+S.ANSWER_CUE[kind]
    expected=list(qt.encode(prefix+prompt+assistant+S.ANSWER_CUE[kind],add_special_tokens=False))
    assert expected==frames['qwen']['body_ids']+list(qt.encode(suffix,add_special_tokens=False))
    assert not any(c.isalnum() for c in prompt[end:])
    assert all(len(frames['llama']['prefix_ids'])<=x<len(frames['llama']['body_ids']) for x in source)
    return dict(id=doc_id,task=task,doc=doc,doc_sha256=digest(doc),prompt=prompt,candidates=candidates,
        gold=gold,normalize=False,cue=S.ANSWER_CUE[kind],frames=frames,source_indices=source,
        target_indices=target,mapping_counts=counts,native_tail=prompt[end:],receiver_suffix=suffix,native_context_ids=expected)


def main():
    legacy=Path(CFG['legacy_root']);oldcfg=json.loads((legacy/'config.json').read_text())
    for name,expected in CFG['reference_hashes'].items():assert sha(Path(CFG['hf_root'])/'heterofold'/name)==expected,name
    original={s:read_jsonl(legacy/f'runs/mcq/{s}.jsonl') for s in ['train','validation']}
    official={}
    ob=Path(oldcfg['datasets']['openbookqa_root'])
    sources=[legacy/'config.json',legacy/'runs/mcq/train.jsonl',legacy/'runs/mcq/validation.jsonl']
    for split,filename in [('train','train.jsonl'),('validation','dev.jsonl'),('test','test.jsonl')]:
        official['openbookqa',split]=[raw_mcq(x) for x in read_jsonl(ob/filename)];sources.append(ob/filename)
    az=Path(oldcfg['datasets']['arc_zip']);sources.append(az)
    with zipfile.ZipFile(az) as z:
        for split,suffix in [('train','Train'),('validation','Dev'),('test','Test')]:
            name=next(x for x in z.namelist() if x.endswith(f'ARC-Challenge-{suffix}.jsonl'))
            official['arc_challenge',split]=[raw_mcq(json.loads(x)) for x in z.read(name).decode().splitlines() if x.strip()]
    blocked={qkey(x) for (task,split),rr in official.items() if split in ['validation','test'] for x in rr}
    trainkeys=[qkey(r['doc']) for r in original['train']]
    collisions=[r['id'] for r in original['train'] if qkey(r['doc']) in blocked]
    assert not collisions,('Official dev/test question overlap',collisions)
    assert len(set(trainkeys))==len(trainkeys),'Duplicate training questions across corpora'
    for split in ['train','validation']:
        wanted=CFG['train_counts'] if split=='train' else CFG['validation_counts']
        assert dict(Counter(r['task'] for r in original[split]))==wanted
        for task in wanted:
            expected={digest(raw_mcq(d)) for d in official[task,split]}
            for r in original[split]:
                if r['task']!=task:continue
                assert digest(raw_mcq(r['doc'])) in expected,('Official document changed',r['id'])
    lt,qt=[AutoTokenizer.from_pretrained(CFG['models'][f],local_files_only=True) for f in ['llama','qwen']]
    prepared={s:[prepare_row(r['id'],r['task'],r['doc'],lt,qt) for r in rr] for s,rr in original.items()}
    previous=Path(CFG['previous_root'])
    previous_rows=json.loads((previous/'previous_prepared.json').read_text())
    # Regenerate the previously audited ARC rows to prove the helper is identical.
    for row in previous_rows:
        if row['task']!='arc':continue
        inp=json.loads((previous/'inputs/arc_rows.json').read_text())['rows'][row['index']]
        fresh=prepare_row(row['id'],'arc',inp,lt,qt)
        for key in ['prompt','candidates','gold','frames','source_indices','target_indices','receiver_suffix','native_context_ids']:
            assert fresh[key]==row[key],('Previous evaluation interface changed',row['id'],key)
    prior_manifest=json.loads((previous/'previous_manifest.json').read_text())
    for f in ['llama','qwen']:
        for key,file in [('config_sha256','config.json'),('tokenizer_sha256','tokenizer_config.json')]:
            assert sha(Path(CFG['models'][f])/file)==prior_manifest['models'][f][key]
    estimate=sum((28+36)*8*128*4*len(r['source_indices'])+36*8*128*4*r['frames']['qwen']['native_prefix_length'] for rr in prepared.values() for r in rr)
    required=estimate/2**30+CFG['disk_reserve_gib']
    free=shutil.disk_usage(ROOT).free/2**30
    assert free>=required,('Insufficient cache+checkpoint disk budget',free,required)
    audit=dict(train_counts=CFG['train_counts'],validation_counts=CFG['validation_counts'],
        same_old_train_document_ids=True,official_split_membership_verified=True,
        train_vs_all_official_mcq_dev_test_question_overlap=0,train_duplicates=0,
        test_not_used_in_training=True,estimated_cache_gib=estimate/2**30,free_disk_gib=free,
        protocol='OpenBookQA adapted to pinned ARC MCQ formatter; no facts added; no answer in prompt',
        translated_tokens={s:dict(min=min(len(r['source_indices']) for r in rr),max=max(len(r['source_indices']) for r in rr),mean=sum(len(r['source_indices']) for r in rr)/len(rr)) for s,rr in prepared.items()})
    manifest=dict(config=CFG,source_files={str(p):sha(p) for p in sources},
        code={p.name:sha(p) for p in ROOT.glob('*.py')},config_sha256=sha(ROOT/'config.json'),
        reference_hashes=CFG['reference_hashes'],versions={k:importlib.metadata.version(k) for k in ['torch','transformers']},
        old_split_ids={s:[r['id'] for r in rr] for s,rr in original.items()},
        prepared_sha256={s:digest(rr) for s,rr in prepared.items()},
        previous_manifest_sha256=sha(previous/'previous_manifest.json'),
        previous_comparison_sha256=sha(previous/'comparison/per_sample.jsonl'))
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==manifest,'Immutable manifest changed'
    else:save(ROOT/'manifest.json',manifest)
    for s,rr in prepared.items():write_jsonl(ROOT/f'prepared/{s}.jsonl',rr)
    save(ROOT/'data_audit.json',audit)
    save(ROOT/'prepared/completed.json',dict(status='completed',audit=audit,manifest_sha256=identity()))
    status(status='prepared',train=2048,validation=256)
    log(f'DATA/PROTOCOL AUDIT PASSED; cache estimate {estimate/2**30:.2f} GiB')


if __name__=='__main__':
    try:main()
    except Exception as exc:fail(exc);raise
