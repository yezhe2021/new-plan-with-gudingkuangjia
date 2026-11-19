"""One shared split for both Gemma arms; exclude upstream calibration from held-out sets."""
import random
import zipfile
from bisect import bisect_right
from collections import Counter
from gemma_common import *
from runtime_gemma import batches

def fast_map(row,gt,qt):
    text=row['body'];spans={f:L.token_spans(t,text,row['encoded'][f]['body']) for f,t in [('gemma',gt),('qwen',qt)]}
    cb=[0]
    for ch in text:cb.append(cb[-1]+len(ch.encode('utf-8')))
    bb={f:[(x.index,cb[x.start],cb[x.end]) for x in ss if x.index!=0 and x.start!=x.end] for f,ss in spans.items()}
    ends={f:[x[2] for x in ss] for f,ss in bb.items()}
    assert all(e==sorted(e) for e in ends.values())
    shared=sorted(set(ends['gemma'])&set(ends['qwen'])|{0});options=row['encoded']['qwen']['option_token_indices'];wanted=set(options);mapping={};counts=Counter()
    for left,right in zip(shared,shared[1:]):
        si=[x[0] for x in bb['gemma'][bisect_right(ends['gemma'],left):bisect_right(ends['gemma'],right)]]
        ti=[x[0] for x in bb['qwen'][bisect_right(ends['qwen'],left):bisect_right(ends['qwen'],right)]]
        if not si or not ti:continue
        touched=[i for i in ti if i in wanted]
        if not touched:continue
        kind='one_to_one' if len(si)==len(ti)==1 else 'copy' if len(si)==1 else 'drop' if len(ti)==1 else 'many_to_many'
        counts[kind+'_units']+=1;counts[kind+'_target_tokens']+=len(touched)
        for rank,index in enumerate(ti):
            if index in wanted:mapping[index]=si[L.source_rank(len(si),len(ti),rank)]
    assert set(mapping)==wanted
    return options,[mapping[i] for i in options],dict(counts)

def align(row,gt,qt):
    r=json.loads(json.dumps(row));text=r['prompt']
    rendered=gt.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
    assert rendered.count(text)==1;start=rendered.index(text);end=start+len(text)
    enc=gt(rendered,add_special_tokens=False,return_offsets_mapping=True)
    assert not any(a<start<b or a<end<b for a,b in enc['offset_mapping'])
    indexes=[i for i,(a,b) in enumerate(enc['offset_mapping']) if start<=a<b<=end]
    assert indexes==list(range(indexes[0],indexes[-1]+1))
    prefix=enc['input_ids'][:indexes[0]];core=enc['input_ids'][indexes[0]:indexes[-1]+1]
    assert core==gt.encode(text,add_special_tokens=False)
    assert qt.encode(text,add_special_tokens=False)==r['content_ids']
    src=L.encode_prefix(gt,text);tar=L.encode_prefix(qt,text)
    assert tar==r['content_ids'],'Qwen BOS/anchor contract changed'
    aligned=dict(id=r['id'],body=text,encoded={'gemma':dict(body=src),'qwen':dict(body=tar,option_token_indices=list(range(1,len(tar))))})
    target,source,counts=fast_map(aligned,gt,qt)
    assert target==list(range(1,len(tar)))
    source=[i-int(gt.bos_token_id is not None)+len(prefix) for i in source]
    assert all(len(prefix)<=i<len(prefix)+len(core) for i in source)
    r.update(gemma_body_ids=prefix+core,gemma_source_indices=source,gemma_mapping_counts=counts)
    return r

def main():
    if (TOP/'manifest.json').exists():verify();return
    N.verify();verify_gemma();assert N.identity()==CFG['data_reference_manifest_sha256']
    data={s:N.rows(s) for s in ['train','validation','test']}
    old={s:C.read_jsonl(GP/f'prepared/{s}.jsonl') for s in ['train','validation']}
    blocked={r['question_key'] for rr in old.values() for r in rr}
    overlap_before={s:{t:sum(C.norm(r['question']) in blocked for r in rr if r['task']==t) for t in CFG[s+'_counts']} for s,rr in data.items()}
    qt=C.AutoTokenizer.from_pretrained(CFG['receiver'],local_files_only=True);gt=C.AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True)
    historical=C.read_jsonl(GP/'prepared/test.jsonl');alignment_checks=[]
    for task in sorted({r['task'] for r in historical}):
        oldrow=next(r for r in historical if r['task']==task)
        text=oldrow['prompt'][:-len(oldrow['native_tail'])] if oldrow['native_tail'] else oldrow['prompt']
        probe=align(dict(id=oldrow['id'],prompt=text,content_ids=oldrow['frames']['qwen']['content_ids']),gt,qt)
        assert probe['gemma_body_ids']==oldrow['frames']['gemma']['body_ids'] and probe['gemma_source_indices']==oldrow['source_indices']
        assert probe['gemma_mapping_counts']==oldrow['mapping_counts']
        exact_input=dict(id=oldrow['id'],body=text,encoded={'gemma':dict(body=L.encode_prefix(gt,text)),'qwen':dict(body=L.encode_prefix(qt,text),option_token_indices=list(range(1,len(oldrow['frames']['qwen']['content_ids']))))})
        assert fast_map(exact_input,gt,qt)==L.fullsync_map(exact_input,gt,qt,'gemma')
        alignment_checks.append(dict(task=task,id=oldrow['id'],historical_alignment_exact=True))
    affix=C.read(C.ROOT/'manifest.json');arc_zip=Path(N.R.CFG['arc_zip'])
    # Gemma's old Stage-A/B saw some ARC Dev questions; redraw all64 validation
    # rows before fitting any new memory parameters.
    with zipfile.ZipFile(arc_zip) as z:
        path=next(n for n in z.namelist() if n.endswith('/ARC-Challenge-Dev.jsonl') and not n.startswith('__MACOSX'))
        raw=[json.loads(s) for s in z.read(path).decode().splitlines() if s]
    used={C.norm(r['question']) for rr in data.values() for r in rr};used|=blocked
    order=list(range(len(raw)));random.Random(CFG['seed']).shuffle(order);fresh=[]
    for i in order:
        d=raw[i];question=d['question']['stem']
        if C.norm(question) in used:continue
        doc=dict(question=question,choices={k:[v[k] for v in d['question']['choices']] for k in ['label','text']},answerKey=d['answerKey'])
        prompt,candidates,gold,kind,_=C.SC.format_item('arc',doc);suffix=affix['assistant']+C.SC.ANSWER_CUE[kind]
        fresh.append(dict(id='arc_gemma_validation_'+d['id'],task='arc',source_id=d['id'],question=question,prompt=prompt,candidates=candidates,gold=gold,
            normalize=False,continuation_suffix=suffix,content_ids=qt.encode(prompt,add_special_tokens=False),suffix_ids=qt.encode(suffix,add_special_tokens=False),gold_tokens=qt.encode(candidates[gold],add_special_tokens=False)))
        used.add(C.norm(question))
        if len(fresh)==64:break
    assert len(fresh)==64
    data['validation']=fresh+[r for r in data['validation'] if r['task']!='arc']
    # Keep the same category quota, replacing only three MMLU-Pro items that
    # overlap upstream calibration. No outcome/length-dependent reselection.
    rawpath=Path(N.ORIGINAL/'cross_dataset5x64_seed1234/config.json')
    mmlu_path=Path(C.read(rawpath)['sources']['mmlu_pro'])
    import pyarrow.parquet as pq
    raw=pq.read_table(mmlu_path).to_pylist();order=list(range(len(raw)));random.Random(CFG['seed']).shuffle(order);replaced=[]
    for index,r in enumerate(data['test']):
        if C.norm(r['question']) not in blocked:continue
        assert r['task']=='mmlu_pro',('Unexpected upstream-test overlap',r['id'])
        d=next(raw[i] for i in order if raw[i]['category']==r['category'] and C.norm(raw[i]['question']) not in used)
        labels=[chr(65+i) for i in range(len(d['options']))];question=d['question'];gold=int(d['answer_index'])
        prompt='Answer the following multiple-choice question about '+d['category'].replace('_',' ')+'. Select the single correct option.\n\nQuestion: '+question+'\n\nChoices:\n'+'\n'.join(a+'. '+str(b).strip() for a,b in zip(labels,d['options']))+'\n\nAnswer with only the option label.'
        new=dict(r,id='mmlu_pro_gemma_'+str(d['question_id']),source_id=str(d['question_id']),question=question,prompt=prompt,gold=gold,candidates=labels,
            content_ids=qt.encode(prompt,add_special_tokens=False),source_doc_sha256=C.digest(d),option_count=len(labels))
        replaced.append(dict(old_id=r['id'],new_id=new['id'],category=r['category']));data['test'][index]=new;used.add(C.norm(question))
    assert len(replaced)==3
    assert all(C.norm(r['question']) not in blocked for s in ['validation','test'] for r in data[s])
    sets={s:{C.norm(r['question']) for r in rr} for s,rr in data.items()};assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    for split,rr in data.items():
        out=[]
        for index,r in enumerate(rr,1):
            assert qt.encode(affix['prefix']+r['prompt']+r['continuation_suffix'],add_special_tokens=False)==affix['prefix_ids']+r['content_ids']+r['suffix_ids']
            out.append(align(r,gt,qt))
            if index%128==0:C.log('GEMMA ALIGN',split,index,len(rr))
        data[split]=out;C.write_jsonl(TOP/f'prepared/{split}.jsonl',out)
        assert {t:sum(r['task']==t for r in out) for t in CFG[split+'_counts']}==CFG[split+'_counts']
    refs=[REFERENCE/'manifest.json',GP/'manifest.json',GP/'config.json',GP/'prepared/train.jsonl',GP/'prepared/validation.jsonl',GP/'prepared/test.jsonl',
        GP/'runs/stage_a/best.pt',GP/'runs/stage_b/last.pt',Path(CFG['legacy_core']),arc_zip,mmlu_path]
    refs+=list(REFERENCE.glob('*.py'))+list(GP.glob('*.py'))+[REFERENCE/f'prepared/{s}.jsonl' for s in data]+[TOP/f'prepared/{s}.jsonl' for s in data]
    refs += [REFERENCE/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json' for r in data['test'] if (REFERENCE/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json').exists()]
    C.save(TOP/'manifest.json',dict(config=CFG,config_sha256=C.sha(TOP/'config.json'),code={f.name:C.sha(f) for f in TOP.glob('*.py')},
        sources={str(f):C.sha(f) for f in refs},counts={s:len(rr) for s,rr in data.items()},upstream_overlap_before=overlap_before,
        heldout_upstream_question_overlap_after=0,arc_validation_redrawn=64,mmlu_test_replacements=replaced,
        historical_alignment_checks=alignment_checks,
        trainable_parameters=409600,train_original_question_ids_unchanged=True,
        batch_schedules={str(e):C.digest([[(r['id'],k) for r,k in b] for b in batches(e)]) for e in range(1,5)},
        maxima={s:max(len(r['gemma_body_ids']) for r in rr) for s,rr in data.items()}))
    mark('prepared/completed.json',status='completed',counts={s:len(rr) for s,rr in data.items()});C.log('GEMMA TWO-ARM PREPARED',C.read(TOP/'manifest.json')['counts'])

if __name__=='__main__':main()
