"""Keep all existing question IDs and the exact Qwen input; rebuild Gemma alignment."""
from common import *
from transformers import AutoTokenizer,AutoConfig
from objective import mcq_positions
import shutil

def convert(row,gt,qt):
    r=json.loads(json.dumps(row));old=r['frames']['qwen']
    text=r['prompt'][:len(r['prompt'])-len(r['native_tail'])] if r['native_tail'] else r['prompt']
    # Truncating only the punctuation-only historical tail keeps Qwen's anchor and suffix unchanged.
    frame=C.frame(gt,'gemma','chat',text,len(text))
    aligned=dict(id=r['id'],body=text,encoded={f:dict(body=C.encode_prefix(t,text),option_token_indices=list(range(1,len(C.encode_prefix(t,text))))) for f,t in [('gemma',gt),('qwen',qt)]})
    target,source,counts=C.fullsync_map(aligned,gt,qt,'gemma')
    assert target==r['target_indices']==list(range(1,len(old['content_ids'])))
    source=[x-int(gt.bos_token_id is not None)+len(frame['prefix_ids']) for x in source]
    assert all(len(frame['prefix_ids'])<=x<len(frame['body_ids']) for x in source)
    r['frames']={'gemma':frame,'qwen':old};r['source_indices']=source;r['mapping_counts']=counts
    for k in ['source_ref','source_identity']:r.pop(k,None)
    assert len(source)+old['native_prefix_length']==len(old['body_ids'])
    assert old['body_ids']+r['suffix_ids']==r['native_context_ids']
    for k in ['id','task','prompt','candidates','gold','receiver_suffix','suffix_ids','native_context_ids','target_indices','question_key']:
        assert r[k]==row[k],(row['id'],k)
    assert r['frames']['qwen']==row['frames']['qwen']
    if r['task'] in ['openbookqa','arc_challenge','mmlu_pro']:mcq_positions(r,lambda c:C.R.candidate_ids(qt,c))
    return r

def prepare():
    gt,qt=[AutoTokenizer.from_pretrained(CFG[k],local_files_only=True) for k in ['sender','receiver']]
    config=AutoConfig.from_pretrained(CFG['sender'],local_files_only=True).text_config
    assert (config.num_hidden_layers,config.num_key_value_heads,config.head_dim)==(34,4,256)
    prepared={};counts={}
    for split in ['train','validation','test']:
        original=C.read_jsonl(OLD_U_ROOT/f'prepared/{split}.jsonl')
        prior=json.loads((OLD_U_ROOT/'manifest.json').read_text())
        assert C.digest(original)==prior['prepared_sha256'][split]
        prepared[split]=[convert(r,gt,qt) for r in original]
        counts[split]=dict(Counter(r['task'] for r in prepared[split]))
        expected=U.CFG[{'train':'train_counts','validation':'validation_counts','test':'formal_counts'}[split]]
        assert counts[split]==expected
    keys={s:{r['question_key'] for r in rr} for s,rr in prepared.items()}
    assert len(keys['train'])==2048 and not keys['train']&(keys['validation']|keys['test']) and not keys['validation']&keys['test']
    # Stage A and all later stages share exactly the existing four-task train/dev partition.
    tokens=sum(len(r['source_indices']) for s in ['train','validation'] for r in prepared[s])
    source_gib=tokens*34*1024*4/2**30
    assert shutil.disk_usage(ROOT).free/2**30>=source_gib+CFG['disk_reserve_gib']+6
    dependencies={}
    for folder in [OLD_U_ROOT,OLD_C_ROOT,Path(CFG['reference_reader'])]:
        for p in list(folder.glob('*.py'))+[folder/'config.json',folder/'manifest.json']:
            dependencies[str(p)]=C.sha(p)
    for s in prepared:dependencies[str(OLD_U_ROOT/f'prepared/{s}.jsonl')]=C.sha(OLD_U_ROOT/f'prepared/{s}.jsonl')
    for name,digest in C.CFG['reference_hashes'].items():
        p=Path(C.CFG['hf_root'])/'heterofold'/name;assert C.sha(p)==digest;dependencies[str(p)]=digest
    # All reused Native outputs and prefixes are pinned. No transferred Llama output is treated as Gemma.
    for p in (OLD_U_ROOT/'native_prefix').glob('*.json'):dependencies[str(p)]=C.sha(p)
    for row in prepared['test']:
        p=OLD_U_ROOT/'evaluation'/row['task']/'per_sample'/(U.fname(row)+'.json')
        record=json.loads(p.read_text());assert record['manifest_sha256']==OLD_U_ID
        for k in ['id','task','prompt','receiver_suffix','candidates','gold']:assert record[k]==row[k]
        dependencies[str(p)]=C.sha(p)
    metadata={}
    for role in ['sender','receiver']:
        for filename in ['config.json','tokenizer_config.json','model.safetensors.index.json']:
            p=Path(CFG[role])/filename
            if p.exists():metadata[str(p)]=C.sha(p)
    manifest=dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},
        dependencies=dependencies,model_metadata=metadata,prepared_sha256={s:C.digest(rr) for s,rr in prepared.items()},
        counts=counts,source_cache_gib=source_gib,qwen_inputs_unchanged=True,training_inputs_contain_gold_answers=False,
        training_supervision='Post-k_norm pre-RoPE K and V; NMSE+cosine; source prompt has no appended answers or CoT.',
        stage_a_data='Same four-task512each train and64each validation as B/Reader, same IDs; no test data.',
        architecture=dict(source_bounds=[0,8,17,25,34],target_bounds=[0,9,18,27,36]),caveats=CFG['reader']['caveats'])
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==manifest
    else:C.save(ROOT/'manifest.json',manifest)
    for s,rr in prepared.items():C.write_jsonl(ROOT/f'prepared/{s}.jsonl',rr)
    mark(ROOT/'prepared/completed.json',status='completed',counts=counts,source_cache_gib=source_gib)
    status(status='prepared',phase='ready',counts=counts)
    print(json.dumps(dict(counts=counts,source_gib=source_gib,manifest_sha256=identity()),ensure_ascii=False))

if __name__=='__main__':prepare()
