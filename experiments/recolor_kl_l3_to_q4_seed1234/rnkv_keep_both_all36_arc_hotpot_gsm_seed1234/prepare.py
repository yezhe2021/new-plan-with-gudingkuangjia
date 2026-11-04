"""Retain original ARC/Hotpot rows; add disjoint official-train GSM8K."""
import random
import re
from experiment_common import *
from runtime_new import batches

def main():
    if (ROOT/'manifest.json').exists():verify();return
    R.verify();A.verify();assert R.identity()==CFG['original_rn_manifest_sha256'] and A.identity()==CFG['structural_manifest_sha256']
    assert C.sha(GSM/'manifest.json')==CFG['gsm_experiment_manifest_sha256']
    from datasets import Dataset
    raw=Dataset.from_file(CFG['gsm_train_source']);assert len(raw)==7473
    data={s:R.rows(s) for s in ['train','validation']}
    data['test']=C.read_jsonl(STRUCT/'prepared/test.jsonl')+C.read_jsonl(GSM/'prepared/test.jsonl')
    assert len(data['test'])==512
    excluded=C.rows('train')+C.rows('validation')+C.rows('test')+sum(data.values(),[])
    used={C.norm(r['question']) for r in excluded};order=list(range(len(raw)));random.Random(CFG['seed']).shuffle(order)
    tok=C.AutoTokenizer.from_pretrained(C.CFG['models']['receiver'],local_files_only=True);affix=C.read(C.ROOT/'manifest.json')
    cursor=0;selection={};rejected=0
    for split in ['train','validation']:
        selected=[];n=CFG[split+'_counts']['gsm8k']
        while len(selected)<n:
            i=order[cursor];cursor+=1;doc=raw[i];key=C.norm(doc['question'])
            if key in used:rejected+=1;continue
            prompt,_,gold,_,_=C.SC.format_item('gsm8k',doc)
            rationale,reference=doc['answer'].rsplit('####',1);assert C.SC.normalize_number(reference)==C.SC.normalize_number(gold)
            rationale=re.sub(r'<<[^<>]*>>','',rationale).strip()
            target=rationale+'\n\nFinal answer: \\boxed{'+gold+'}'
            row=dict(id=f'gsm8k_{split}_{i}',task='gsm8k',source_id=f'gsm8k_train_{i}',source_index=i,source_split='official_train',
                source_doc_sha256=C.digest(doc),question=doc['question'],prompt=prompt,gold=gold,continuation_suffix=affix['assistant'],
                gold_target_text=target,content_ids=tok.encode(prompt,add_special_tokens=False),suffix_ids=tok.encode(affix['assistant'],add_special_tokens=False))
            row['gold_tokens']=tok.encode(target,add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]
            assert tok.encode(affix['prefix']+prompt+affix['assistant'],add_special_tokens=False)==affix['prefix_ids']+row['content_ids']+row['suffix_ids']
            assert len(affix['prefix_ids'])+len(row['content_ids'])+32+len(row['suffix_ids'])+max(64,len(row['gold_tokens']))<CFG['max_context_tokens'],('No truncation/resampling',row['id'])
            selected.append(row);used.add(key)
        data[split]+=selected;selection[split]=[r['source_index'] for r in selected]
    for split,rr in data.items():
        assert {t:sum(r['task']==t for r in rr) for t in CFG[split+'_counts']}==CFG[split+'_counts']
        C.write_jsonl(ROOT/f'prepared/{split}.jsonl',rr)
    sets={s:{C.norm(r['question']) for r in rr} for s,rr in data.items()}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    for s in ['train','validation']:
        assert [r for r in data[s] if r['task']!='gsm8k']==R.rows(s)
    for row in data['test']:baseline(row)
    refs=[ORIGINAL/'manifest.json',STRUCT/'manifest.json',GSM/'manifest.json',Path(CFG['gsm_train_source']),GSM/'numeric.py',
        ORIGINAL/'prepared/train.jsonl',ORIGINAL/'prepared/validation.jsonl',STRUCT/'prepared/test.jsonl',GSM/'prepared/test.jsonl']
    refs+=list(ORIGINAL.glob('*.py'))+list(STRUCT.glob('*.py'))+[baseline_path(r) for r in data['test']]
    refs+=[ROOT/f'prepared/{s}.jsonl' for s in data]
    maxima={s:{t:dict(body_tokens=max(len(r['content_ids']) for r in rr if r['task']==t),gold_tokens=max([len(r.get('gold_tokens',[])) for r in rr if r['task']==t] or [0])) for t in CFG[s+'_counts']} for s,rr in data.items()}
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},
        sources={str(f):C.sha(f) for f in refs},counts={s:len(rr) for s,rr in data.items()},gsm_selected_indices=selection,gsm_overlap_candidates_excluded=rejected,
        original_arc_hotpot_rows_identical=True,split_normalized_question_overlap=0,maxima=maxima,trainable_parameters=409600,
        batch_schedules={str(e):C.digest([[(r['id'],k) for r,k in b] for b in batches(e)]) for e in range(1,5)}))
    mark('prepared/completed.json',status='completed',counts={s:len(rr) for s,rr in data.items()},maxima=maxima)
    C.log('ALL36 THREE-TASK PREPARED',{s:len(rr) for s,rr in data.items()},'GSM max target',maxima['train']['gsm8k']['gold_tokens'])

if __name__=='__main__':main()
