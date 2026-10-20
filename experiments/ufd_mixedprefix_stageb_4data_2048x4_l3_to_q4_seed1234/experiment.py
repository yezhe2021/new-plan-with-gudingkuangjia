"""Single frozen-base Mixed-Prefix UFD Stage-B, 2048 contexts ×4 /8=1024 updates."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,math,random,re,zipfile,shutil,fcntl,importlib.util
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CFG['parent'])
import core as C
import prepare as OLD_P
import evaluate as OLD_E
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,DynamicCache
from heterofold.transfer import _cache_layer


def identity():return C.sha(ROOT/'manifest.json')
def status(**kw):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def key(q):return re.sub(r'\s+',' ',q).strip().casefold()
def qdoc(d):return OLD_P.question(d)
def rows(split,smoke=False):
    rr=C.read_jsonl(ROOT/f'prepared/{split}.jsonl')
    if not smoke:return rr
    seen=set();out=[]
    for r in rr:
        if r['task'] not in seen:out.append(r);seen.add(r['task'])
    return out
def fname(r):return r['id'].replace(':','_')
def own_source(r):return ROOT/'source'/(fname(r)+'.pt')
def source_path(r):return Path(r['source_ref']) if r.get('source_ref') else own_source(r)
def shared(state,results):
    shared_path=ROOT.parent/'EXPERIMENT_RESULTS.md'
    record=dict(experiment=CFG['name'],time=time.strftime('%F %T'),status=state,config=CFG,results=results,
        manifest=str(ROOT/'manifest.json'),raw_results=str(ROOT/'evaluation/per_sample.jsonl'),log=str(ROOT/'logs/pipeline.log'))
    with shared_path.open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    C.save(ROOT/'shared_summary_updated.json',dict(status=state,path=str(shared_path)))


def make_row(rid,task,doc,lt,qt,prompt=None,candidates=None,gold=None,normalize=False,cue=None):
    if prompt is None:
        if task=='gsm8k':prompt,candidates,gold,kind,_=C.S.format_item('gsm8k',doc)
        else:prompt,candidates,gold,kind,_=C.S.format_item('arc',OLD_P.raw_mcq(doc))
        cue='' if task=='gsm8k' else C.S.ANSWER_CUE[kind]
    eligible=C.frame(lt,'llama','chat',prompt)&C.frame(qt,'qwen','chat',prompt)
    choices=[x for x in eligible if 0<x<len(prompt) and len(prompt[x:])<=8 and not any(c.isalnum() for c in prompt[x:])]
    end=max(choices) if choices else len(prompt)
    frames={f:C.frame(t,f,'chat',prompt,end) for f,t in [('llama',lt),('qwen',qt)]}
    text=prompt[:end]
    aligned=dict(id=rid,body=text,encoded={f:dict(body=C.encode_prefix(t,text),option_token_indices=list(range(1,len(C.encode_prefix(t,text))))) for f,t in [('llama',lt),('qwen',qt)]})
    target,source,counts=C.fullsync_map(aligned,lt,qt,'llama')
    assert target==list(range(1,len(frames['qwen']['content_ids'])))
    source=[i-int(lt.bos_token_id is not None)+len(frames['llama']['prefix_ids']) for i in source]
    prefix,assistant=C.chat_affixes(qt,thinking=False);suffix=frames['qwen']['rendered_suffix']+cue
    ids=qt.encode(prefix+prompt+assistant+cue,add_special_tokens=False)
    suffix_ids=qt.encode(suffix,add_special_tokens=False)
    assert frames['qwen']['body_ids']+suffix_ids==ids
    assert not any(c.isalnum() for c in prompt[end:])
    assert frames['qwen']['native_prefix_length']+len(source)==len(frames['qwen']['body_ids'])
    return dict(id=rid,task=task,doc_sha256=C.digest(doc),prompt=prompt,candidates=candidates,gold=gold,normalize=normalize,cue=cue,
        question_key=key(doc['question'] if task=='gsm8k' else qdoc(doc)),frames=frames,source_indices=source,target_indices=target,
        mapping_counts=counts,native_tail=prompt[end:],receiver_suffix=suffix,suffix_ids=suffix_ids,native_context_ids=ids)


def reuse_parent(r,split):
    r=dict(r);r['suffix_ids']=AutoTokenizer.from_pretrained(C.CFG['models']['qwen'],local_files_only=True).encode(r['receiver_suffix'],add_special_tokens=False) if 'suffix_ids' not in r else r['suffix_ids']
    r['question_key']=key(qdoc(r['doc']))
    p=C.cache_path(split,r,'llama')
    if p.exists():r.update(source_ref=str(p),source_identity=C.identity())
    return r


def stratified(pool,count,seed):
    groups=defaultdict(list)
    for d in pool:groups[d['category']].append(d)
    rng=random.Random(seed)
    for gg in groups.values():rng.shuffle(gg)
    result=[]
    while len(result)<count:
        before=len(result)
        for category in sorted(groups):
            if groups[category] and len(result)<count:result.append(groups[category].pop())
        assert len(result)>before,'Insufficient unique stratified examples'
    return result


def prepare():
    import pyarrow.parquet as pq
    parent=Path(CFG['parent']);assert C.sha(parent/'runs/stage_a/best.pt')==CFG['stage_a_sha256']
    assert C.sha(parent/'runs/stage_b/best.pt')==CFG['old_stage_b_sha256']
    for file,sha in C.CFG['reference_hashes'].items():assert C.sha(Path(C.CFG['hf_root'])/'heterofold'/file)==sha
    legacy=json.loads((Path(C.CFG['legacy_root'])/'config.json').read_text())
    lt,qt=[AutoTokenizer.from_pretrained(C.CFG['models'][f],local_files_only=True) for f in ['llama','qwen']]
    oldtrain=C.rows('train');oldval=C.rows('validation')
    rr=dict(train=[],validation=[],test=[]);sources=[parent/'prepared/train.jsonl',parent/'prepared/validation.jsonl']
    for task in ['openbookqa','arc_challenge']:
        for split,pool in [('train',oldtrain),('validation',oldval)]:
            choices=[r for r in pool if r['task']==task];random.Random(CFG['seed']+int(split=='validation')).shuffle(choices)
            for row in choices[:CFG[split+'_counts'][task]]:
                row=dict(row,question_key=key(qdoc(row['doc'])),suffix_ids=qt.encode(row['receiver_suffix'],add_special_tokens=False))
                p=C.cache_path(split,row,'llama')
                if p.exists():row.update(source_ref=str(p),source_identity=C.identity())
                rr[split].append(row)
    ob=Path(legacy['datasets']['openbookqa_root']);obtest=C.read_jsonl(ob/'test.jsonl');sources.append(ob/'test.jsonl')
    for i,d in enumerate(obtest[:CFG['formal_counts']['openbookqa']]):rr['test'].append(make_row('obqa_test_'+str(d['id']),'openbookqa',d,lt,qt))
    # Preserve the exact ARC128 and HellaSwag64 snapshots, including continuation scoring.
    arc_snapshot=Path(C.CFG['previous_root'])/'inputs/arc_rows.json'
    arc_docs=json.loads(arc_snapshot.read_text())['rows'];sources.append(arc_snapshot)
    for old in OLD_E.test_rows():
        r=dict(old);r['task']='arc_challenge' if old['task']=='arc' else 'hellaswag'
        q=key(qdoc(arc_docs[old['index']])) if old['task']=='arc' else key(old['prompt'])
        r.update(question_key=q,suffix_ids=qt.encode(old['receiver_suffix'],add_special_tokens=False))
        p=C.ROOT/'evaluation/source'/(old['id']+'.pt')
        if p.exists():r.update(source_ref=str(p),source_identity=C.identity())
        rr['test'].append(r)
    sciq=parent.parent/'decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64/prepared/test.jsonl'
    assert sciq.exists(),'Previously fixed SciQ64 snapshot missing';sources.append(sciq)
    for old in C.read_jsonl(sciq):
        doc=dict(question=old['question'],choices=dict(label=list('ABCD'),text=old['choice_texts']),answerKey='ABCD'[old['gold']])
        rr['test'].append(make_row(old['id'],'sciq',doc,lt,qt))
    gsmtrain=Path(legacy['datasets']['gsm8k_train']);gsmtest=Path(legacy['datasets']['gsm8k_test']);sources.extend([gsmtrain,gsmtest])
    gg=C.read_jsonl(gsmtrain);gt=C.read_jsonl(gsmtest)
    protected={key(d['question']) for d in gt}|{key(qdoc(d)) for d in obtest}
    protected.update(key(qdoc(r['doc'])) for r in oldval)
    old_train_q={key(qdoc(r['doc'])) for r in oldtrain}
    pool=[];seen=set()
    for i,d in enumerate(gg):
        q=key(d['question'])
        if q not in protected and q not in old_train_q and q not in seen:pool.append((i,d));seen.add(q)
    random.Random(CFG['seed']).shuffle(pool)
    for split,start,count in [('train',0,512),('validation',512,64)]:
        for i,d in pool[start:start+count]:rr[split].append(make_row('gsm8k_train_'+str(i),'gsm8k',d,lt,qt))
    for i,d in enumerate(gt[:CFG['formal_counts']['gsm8k']]):rr['test'].append(make_row('gsm8k_test_'+str(i),'gsm8k',d,lt,qt))
    mpath=Path(legacy['datasets']['mmlu_pro_test']);sources.append(mpath);mm=pq.read_table(mpath).to_pylist()
    assert len(mm)==12032
    blocked=old_train_q|{r['question_key'] for s in rr.values() for r in s}|{key(d['question']) for d in gt}
    groups=defaultdict(list)
    for i,d in enumerate(mm):groups[key(d['question'])].append(dict(d,original_index=i))
    historical={key(d['question']) for d in mm[:128]}
    eligible=[ds[0] for q,ds in groups.items() if q not in blocked and q not in historical]
    mt=stratified(eligible,512,CFG['seed']);tkeys={key(d['question']) for d in mt}
    mv=stratified([d for d in eligible if key(d['question']) not in tkeys],64,CFG['seed']+1);vkeys={key(d['question']) for d in mv}
    held=[ds[0] for q,ds in groups.items() if q not in blocked|tkeys|vkeys]
    held.sort(key=lambda d:d['original_index']);assert len(held)>=128
    assigned={}
    for split,dd in [('train',mt),('validation',mv),('test',held[:128])]:
        assigned[split]=[]
        for d in dd:
            labels=list('ABCDEFGHIJ')[:len(d['options'])];assert len(labels)==len(d['options']) and d['answer'] in labels
            doc=dict(id=str(d['question_id']),question=d['question'],choices=dict(label=labels,text=[s.strip() for s in d['options']]),answerKey=d['answer'])
            r=make_row('mmlu_pro_'+str(d['question_id']),'mmlu_pro',doc,lt,qt)
            r.update(category=d['category'],official_question_id=int(d['question_id']),original_index=d['original_index'])
            rr[split].append(r);assigned[split].append(int(d['question_id']))
    split_record=dict(source=str(mpath),sha256=C.sha(mpath),seed=CFG['seed'],official_split='test',reporting='CUSTOM SPLIT ONLY; never official full-test unseen score',
        sampled_question_ids=assigned,all_remaining_heldout_question_ids=[int(d['question_id']) for d in held],
        training_question_group_ids={str(d['question_id']):[int(x['question_id']) for x in groups[key(d['question'])]] for d in mt},
        validation_question_group_ids={str(d['question_id']):[int(x['question_id']) for x in groups[key(d['question'])]] for d in mv},
        blocked_cross_dataset_groups=sum(q in blocked for q in groups),protected_historical_first128=True,
        categories={s:dict(Counter(r['category'] for r in rr[s] if r['task']=='mmlu_pro')) for s in rr})
    C.save(ROOT/'mmlu_pro_custom_split.json',split_record)
    tq=[r['question_key'] for r in rr['train']];vq={r['question_key'] for r in rr['validation']};eq={r['question_key'] for r in rr['test']}
    assert len(tq)==len(set(tq))==2048
    assert not set(tq)&(vq|eq),'Train/validation/test normalized question overlap'
    assert not vq&eq,'Validation/test normalized question overlap'
    assert not old_train_q&eq,'Upstream Stage-A train question leaked into evaluation'
    for split in ['train','validation']:assert dict(Counter(r['task'] for r in rr[split]))==CFG[split+'_counts']
    assert dict(Counter(r['task'] for r in rr['test']))==CFG['formal_counts']
    estimate=sum(28*8*128*4*len(r['source_indices']) for ss in rr.values() for r in ss if not r.get('source_ref'))/2**30
    assert shutil.disk_usage(ROOT).free/2**30>estimate+CFG['disk_reserve_gib'],('Source cache exceeds disk budget',estimate)
    metadata=dict(config=CFG,source_files={str(p):C.sha(p) for p in sources},parent_manifest_sha256=C.identity(),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},config_sha256=C.sha(ROOT/'config.json'),parent_code={f:C.sha(parent/f) for f in ['core.py','translator.py','prepare.py','evaluate.py']},
        model_configs={f:dict(config=C.sha(Path(p)/'config.json'),tokenizer=C.sha(Path(p)/'tokenizer_config.json')) for f,p in C.CFG['models'].items()},
        reference_hashes=C.CFG['reference_hashes'],versions={k:C.importlib.metadata.version(k) for k in ['torch','transformers','pyarrow']},
        prepared_sha256={s:C.digest(v) for s,v in rr.items()},custom_mmlu_split_sha256=C.sha(ROOT/'mmlu_pro_custom_split.json'),
        source_cache_estimate_new_gib=estimate,train_vs_dev_test_question_overlap=0,validation_vs_test_question_overlap=0,
        stage_a_retrained=False,old_stage_b_initialization=False,reader=False,training_answer_labels_used_by_loss=False,
        caveats=['MMLU-Pro custom split of official test, not leaderboard evaluation. No gold COT used.',
        'Native prefix policy is greedy, not an unbiased sample from the full Native sequence distribution.',
        'Mixed data and loss both change; not a single-variable loss ablation. Exact normalized overlap is not a semantic/pretraining contamination audit.'])
    p=ROOT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==metadata,'Immutable experiment definition changed'
    else:C.save(p,metadata)
    for s,ss in rr.items():C.write_jsonl(ROOT/f'prepared/{s}.jsonl',ss)
    C.save(ROOT/'prepared/completed.json',dict(status='completed',manifest_sha256=identity(),counts={s:dict(Counter(r['task'] for r in ss)) for s,ss in rr.items()}))
    status(status='prepared',phase='data',new_cache_gib=estimate);C.log(f'DATA AUDIT PASSED; train2048/dev256/test{len(rr["test"])}; new Sender cache {estimate:.2f} GiB')


def loaded_source(r):
    x=torch.load(source_path(r),map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==r.get('source_identity',identity()),('Source identity mismatch',r['id'])
    if 'id' in x:assert x['id']==r['id']
    assert x['key'].shape==x['value'].shape==(28,len(r['source_indices']),8,128)
    return x


@torch.no_grad()
def capture(smoke=False):
    rr=[r for s in ['train','validation','test'] for r in rows(s,smoke)] if not smoke else rows('train',True)
    todo=[r for r in rr if not source_path(r).exists()]
    if not todo:C.log('All requested aligned Sender states available');return
    tok,model=C.model('llama')
    for i,r in enumerate(todo,1):
        assert shutil.disk_usage(ROOT).free/2**30>CFG['disk_reserve_gib'],'Disk reserve reached'
        ids=r['frames']['llama']['body_ids'];k,v,_=C.capture(model,ids,len(ids),[0]);ix=r['source_indices']
        C.save_tensor(own_source(r),dict(key=k[:,ix].contiguous(),value=v[:,ix].contiguous(),id=r['id'],manifest_sha256=identity()))
        status(status='running',phase='sender_capture',sample=i,total=len(todo),id=r['id'])
        if i%16==0 or smoke or i==len(todo):C.log(f'Aligned Sender capture {i}/{len(todo)} {r["id"]}')
    C.log('SENDER CAPTURE COMPLETED smoke='+str(smoke))


def clone_cache(cache,model):
    return DynamicCache(ddp_cache_data=[(k.detach().clone(),v.detach().clone()) for k,v in [_cache_layer(cache,i) for i in range(36)]],config=model.config)


@torch.no_grad()
def native_body(model,r):
    ids=torch.tensor([r['frames']['qwen']['body_ids']],device='cuda')
    return model.model(ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1],device='cuda')[None],use_cache=True).past_key_values


def translated_cache(model,r,native,k,v):
    frame=r['frames']['qwen'];anchor=frame['native_prefix_length'];length=len(frame['body_ids'])
    assert anchor+k.shape[1]==length and k.shape==v.shape
    positions=torch.arange(anchor,length,device='cuda');data=[]
    for layer in range(36):
        nk,nv=_cache_layer(native,layer)
        rotated=C.apply_rope(model,k[layer],positions,layer).transpose(0,1)[None]
        value=v[layer].transpose(0,1)[None]
        data.append((torch.cat((nk[:,:,:anchor].detach(),rotated),2),torch.cat((nv[:,:,:anchor].detach(),value),2)))
    return DynamicCache(ddp_cache_data=data,config=model.config)


def suffix_logits(model,r,cache,prefix_ids,positions):
    ids=r['suffix_ids']+list(prefix_ids)
    tensor=torch.tensor([ids],device='cuda');length=len(r['frames']['qwen']['body_ids'])
    output=model.model(tensor,past_key_values=cache,attention_mask=torch.ones(1,length+len(ids),device='cuda',dtype=torch.long),
        position_ids=torch.arange(length,length+len(ids),device='cuda')[None],use_cache=False)
    ix=torch.tensor([len(r['suffix_ids'])-1+p for p in positions],device='cuda')
    return model.lm_head(output.last_hidden_state[0,ix]).float()


def full_kl(native,student):
    n=native.float().log_softmax(-1).detach();s=student.float().log_softmax(-1)
    return F.kl_div(s,n,reduction='none',log_target=True).sum(-1).mean()


@torch.no_grad()
def rollout(model,tok,r,cache,max_new):
    length=len(r['frames']['qwen']['body_ids']);suffix=torch.tensor([r['suffix_ids']],device='cuda')
    output=model.model(suffix,past_key_values=cache,attention_mask=torch.ones(1,length+suffix.shape[1],device='cuda',dtype=torch.long),
        position_ids=torch.arange(length,length+suffix.shape[1],device='cuda')[None],use_cache=True)
    cache=output.past_key_values;total=length+suffix.shape[1]
    stops={tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')};eos=getattr(model.generation_config,'eos_token_id',None)
    stops.update(eos if isinstance(eos,list) else [eos]);stops.discard(None)
    tokens=[];stopped=False
    for i in range(max_new):
        token=int(model.lm_head(output.last_hidden_state[:,-1]).float().argmax(-1));tokens.append(token)
        if token in stops:stopped=True;break
        if i+1<max_new:
            input_ids=torch.tensor([[token]],device='cuda')
            output=model.model(input_ids,past_key_values=cache,attention_mask=torch.ones(1,total+1,device='cuda',dtype=torch.long),
                position_ids=torch.tensor([[total]],device='cuda'),use_cache=True);cache=output.past_key_values;total+=1
    assert tokens
    return dict(tokens=tokens,stopped_eos=stopped,reached_cap=not stopped)


def native_prefix(model,tok,r,native):
    p=ROOT/'native_prefix'/(fname(r)+'.json')
    if p.exists():
        x=json.loads(p.read_text());assert x['manifest_sha256']==identity();return x
    x=rollout(model,tok,r,clone_cache(native,model),CFG['prefix_max_new_tokens'])
    x.update(id=r['id'],manifest_sha256=identity());C.save(p,x);return x


def positions_for(tokens,r,epoch,validation=False):
    count=min(len(tokens),CFG['supervised_positions_per_example_max']);selected={0,len(tokens)-1}
    if count==1:return [0]
    rng=random.Random(int(C.digest(dict(id=r['id'],epoch=0 if validation else epoch,seed=CFG['seed']))[:16],16))
    candidates=[i for i in range(len(tokens)) if i not in selected];rng.shuffle(candidates)
    selected.update(candidates[:count-len(selected)])
    return sorted(selected)


def example_loss(model,tok,base,adapter,r,epoch,student=False,validation=False):
    source=loaded_source(r)
    with torch.no_grad():
        bk,bv=C.map_base(base,source);native=native_body(model,r)
        if student:
            k,v=C.adapted(adapter,bk,bv)
            trajectory=rollout(model,tok,r,translated_cache(model,r,native,k,v),CFG['prefix_max_new_tokens'])
        else:trajectory=native_prefix(model,tok,r,native)
    pp=positions_for(trajectory['tokens'],r,epoch,validation);prefix=trajectory['tokens'][:max(pp)]
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
        teacher=suffix_logits(model,r,clone_cache(native,model),prefix,pp)
    with torch.amp.autocast('cuda',dtype=torch.float16):
        k,v=C.adapted(adapter,bk,bv)
        student_logits=suffix_logits(model,r,translated_cache(model,r,native,k,v),prefix,pp)
    loss=full_kl(teacher,student_logits)
    assert torch.isfinite(loss),'Nonfinite full-vocabulary KL'
    details=dict(kl=float(loss.detach()),positions=len(pp),prefix_length=max(pp),rollout_tokens=len(trajectory['tokens']),
        prefix_source='student' if student else 'native',rollout_reached_cap=int(trajectory['reached_cap']))
    return loss,details


def fresh_adapter():return C.ResidualKVAdapter(rank=CFG['adapter_rank']).cuda()


def smoke():
    if (ROOT/'smoke/completed.json').exists():return
    C.seed();tok,model=C.model('qwen');base=C.base_from_best();adapter=fresh_adapter()
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=CFG['learning_rate'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000);checks=[]
    rr=rows('train',True)
    for r in rr:
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            source=loaded_source(r);bk,bv=C.map_base(base,source);k,v=C.adapted(adapter,bk,bv)
            # Adapter is still zero-up until the later optimizer loop.
            assert torch.equal(k,bk) and torch.equal(v,bv),'Fresh residual is not identity'
            native=native_body(model,r);trajectory=native_prefix(model,tok,r,native)
            pp=positions_for(trajectory['tokens'],r,0,True);pref=trajectory['tokens'][:max(pp)]
            cached=suffix_logits(model,r,clone_cache(native,model),pref,pp)
            fullids=r['frames']['qwen']['body_ids']+r['suffix_ids']+pref
            output=model.model(torch.tensor([fullids],device='cuda'),use_cache=False)
            ix=torch.tensor([len(r['native_context_ids'])-1+p for p in pp],device='cuda')
            full=model.lm_head(output.last_hidden_state[0,ix]).float();kl=float(full_kl(full,cached))
            agreement=bool(torch.equal(full.argmax(-1),cached.argmax(-1)))
            assert kl<CFG['native_parity_kl_max'] and agreement,('Native multi-prefix parity failed',r['id'],kl,agreement)
            # Compare final-cache assembly with the existing Full-Sync pre-RoPE builder.
            n=r['frames']['qwen']['native_prefix_length'];body=r['frames']['qwen']['body_ids']
            nk,nv,_=C.capture(model,body[:n],n,[0])
            legacy=C.make_cache(model,torch.cat((nk.cuda(),bk),1),torch.cat((nv.cuda(),bv),1),torch.arange(len(body),device='cuda'))
            expected=suffix_logits(model,r,legacy,[],[0]);actual=suffix_logits(model,r,translated_cache(model,r,native,bk,bv),[],[0])
            interface_kl=float(full_kl(expected,actual));assert interface_kl<CFG['native_parity_kl_max'],('Full-Sync interface changed',r['id'],interface_kl)
            checks.append(dict(id=r['id'],task=r['task'],native_cache_full_vocab_kl=kl,native_top1_identical=agreement,
                fullsync_builder_kl=interface_kl,valid_positions=len(pp),zero_up_identity=True))
    for round0 in range(2):
        optimizer.zero_grad(set_to_none=True);count=0
        for r in rr:
            loss,detail=example_loss(model,tok,base,adapter,r,round0+1,student=round0==1)
            scaler.scale(loss/len(rr)).backward();count+=detail['positions']
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),CFG['gradient_clip'])
        assert torch.isfinite(norm) and norm>0,'No valid Stage-B gradient'
        assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in base.parameters()),'Frozen parameters received gradients'
        scaler.step(optimizer);scaler.update();C.log(f'SMOKE backward source={"student" if round0 else "native"} grad={float(norm):.6f} positions={count}')
    C.save(ROOT/'smoke/completed.json',dict(status='completed',manifest_sha256=identity(),checks=checks,
        native_and_online_student_backward=True,trainable_parameters=sum(p.numel() for p in adapter.parameters()),teacher_detached=True,stage_a_qwen_frozen=True))
    C.log('MIXED-PREFIX UFD SMOKE PASSED')


def epoch_order(epoch):
    rr=rows('train');bytask={t:[r for r in rr if r['task']==t] for t in CFG['train_counts']}
    for t,ss in bytask.items():random.Random(CFG['seed']+epoch+list(bytask).index(t)*100).shuffle(ss)
    out=[]
    for batch in range(256):
        for t,ss in bytask.items():
            for offset in [0,1]:out.append((ss[2*batch+offset],epoch>CFG['warmup_native_epochs'] and (offset+epoch)%2==0))
    assert len(out)==2048
    return out


@torch.no_grad()
def validate(model,tok,base,adapter):
    values=defaultdict(list)
    for r in rows('validation'):
        if len(values[r['task']])>=CFG['monitor_validation_per_task']:continue
        loss,d=example_loss(model,tok,base,adapter,r,0,validation=True);values[r['task']].append(d)
    result={t:dict(n=len(v),mean_kl=sum(d['kl'] for d in v)/len(v),supervised_positions=sum(d['positions'] for d in v),
        reached_cap=sum(d['rollout_reached_cap'] for d in v)) for t,v in values.items()}
    return dict(tasks=result,macro_kl=sum(x['mean_kl'] for x in result.values())/len(result))


def checkpoint(path,adapter,optimizer,scaler,epoch,step,position,complete,history):
    C.save_tensor(path,dict(state={k:v.detach().cpu() for k,v in adapter.state_dict().items()},optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        manifest_sha256=identity(),epoch=epoch,step=step,position=position,epoch_completed=complete,history=history))


def train():
    folder=ROOT/'runs/stage_b';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return
    assert (ROOT/'smoke/completed.json').exists()
    C.seed();tok,model=C.model('qwen');base=C.base_from_best();adapter=fresh_adapter()
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)
    start=1;step=position0=0;history=[]
    if (folder/'last.pt').exists():
        x=torch.load(folder/'last.pt',map_location='cpu',weights_only=True);assert x['manifest_sha256']==identity()
        adapter.load_state_dict(x['state'],strict=True);optimizer.load_state_dict(x['optimizer']);scaler.load_state_dict(x['scaler'])
        step=x['step'];start=x['epoch']+int(x['epoch_completed']);position0=0 if x['epoch_completed'] else x['position'];history=x['history'];del x
        C.log(f'RESUME epoch={start} step={step} position={position0}')
    with (folder/'steps.jsonl').open('a') as stream:
        for epoch in range(start,CFG['epochs']+1):
            adapter.train();order=epoch_order(epoch);optimizer.zero_grad(set_to_none=True);details=[]
            for position,(r,student) in enumerate(order,1):
                if epoch==start and position<=position0:continue
                loss,d=example_loss(model,tok,base,adapter,r,epoch,student=student)
                scaler.scale(loss/CFG['effective_batch']).backward();details.append(dict(d,task=r['task']))
                if position%CFG['effective_batch']==0:
                    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),CFG['gradient_clip'])
                    assert torch.isfinite(norm) and norm>0,'Invalid Stage-B gradient'
                    assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in base.parameters())
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);step+=1
                    record=dict(epoch=epoch,step=step,position=position,total_contexts=2048,mean_kl=sum(d['kl'] for d in details)/8,
                        gradient_norm=float(norm),supervised_positions=sum(d['positions'] for d in details),student_examples=sum(d['prefix_source']=='student' for d in details),
                        per_task={t:sum(d['kl'] for d in details if d['task']==t)/2 for t in CFG['train_counts']},
                        rollout_cap_examples=sum(d['rollout_reached_cap'] for d in details),max_prefix_length=max(d['prefix_length'] for d in details))
                    stream.write(json.dumps(record)+'\n');stream.flush();details=[];status(status='running',phase='stage_b_training',**record)
                    if step%4==0:C.log(f'UFD epoch={epoch}/4 step={step}/1024 KL={record["mean_kl"]:.6f} positions={record["supervised_positions"]} student={record["student_examples"]}/8')
                    if step%CFG['checkpoint_every_steps']==0 and position<len(order):checkpoint(folder/'last.pt',adapter,optimizer,scaler,epoch,step,position,False,history)
            adapter.eval();metrics=validate(model,tok,base,adapter);entry=dict(epoch=epoch,step=step,validation=metrics);history=[h for h in history if h['epoch']!=epoch]+[entry]
            best=min(history,key=lambda h:(h['validation']['macro_kl'],h['epoch']))
            if entry==best:C.save_tensor(folder/'best.pt',dict(state={k:v.detach().cpu() for k,v in adapter.state_dict().items()},manifest_sha256=identity(),epoch=epoch,step=step,validation=metrics,diagnostic_only=True))
            C.save(folder/'validation.json',dict(primary='Fixed epoch4/step1024',best_diagnostic=best,history=history))
            checkpoint(folder/'last.pt',adapter,optimizer,scaler,epoch,step,len(order),True,history)
            C.log(f'UFD EPOCH {epoch}/4 COMPLETED: {metrics}')
    assert step==CFG['expected_updates']==1024
    result=dict(status='completed',epochs=4,updates=step,primary_checkpoint=str(folder/'last.pt'),primary_sha256=C.sha(folder/'last.pt'),
        selection=CFG['selection'],validation=history,manifest_sha256=identity())
    C.save(folder/'completed.json',result);shared('stage_b_training_completed_evaluation_pending',result);C.log('UFD STAGE-B TRAINING COMPLETED')


def selected_adapter():
    x=torch.load(ROOT/'runs/stage_b/last.pt',map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==identity() and x['step']==1024 and x['epoch_completed'] and x['epoch']==4
    a=fresh_adapter();a.load_state_dict(x['state'],strict=True);return a.eval().requires_grad_(False)


@torch.no_grad()
def candidate_scores(model,tok,r,native,k=None,v=None):
    out=[]
    for candidate in r['candidates']:
        target=C.R.candidate_ids(tok,candidate);pref=target
        cache=clone_cache(native,model) if k is None else translated_cache(model,r,native,k,v)
        logits=suffix_logits(model,r,cache,pref,list(range(len(target))))
        out.append(C.R._candidate_log_likelihood(logits[None],torch.tensor([target],device='cuda'),r['normalize']))
    return C.record(out,r['gold'])


@torch.no_grad()
def evaluate(task):
    out=ROOT/'evaluation'/task
    if (out/'completed.json').exists():return
    assert (ROOT/'runs/stage_b/completed.json').exists()
    tok,model=C.model('qwen');rt=C.runtime(model,tok);base=C.base_from_best();old=C.adapter_from_best();new=selected_adapter()
    rr=[r for r in rows('test') if r['task']==task]
    for i,r in enumerate(rr,1):
        path=out/'per_sample'/(fname(r)+'.json')
        if path.exists():assert json.loads(path.read_text())['manifest_sha256']==identity();continue
        status(status='running',phase='evaluation',task=task,sample=i,total=len(rr))
        native=native_body(model,r);bk,bv=C.map_base(base,loaded_source(r));conditions={}
        for arm,adapter in [('qwen_native',None),('old_mcq_stage_b',old),('new_ufd_stage_b',new)]:
            if adapter is None:k=v=None
            else:k,v=C.adapted(adapter,bk,bv)
            if r['task']=='gsm8k':
                if adapter is None:text,generated=C.R.greedy_decode(rt,torch.tensor([r['native_context_ids']],device='cuda'),CFG['gsm8k_formal_max_new_tokens'])
                else:
                    cache=translated_cache(model,r,native,k,v)
                    text,generated=C.R.generate(rt,torch.tensor([r['suffix_ids']],device='cuda'),CFG['gsm8k_formal_max_new_tokens'],past_key_values=cache,
                        past_length=len(r['frames']['qwen']['body_ids']),temperature=0.0)
                prediction=C.S.extract_numeric_answer(text);gold=C.S.normalize_number(r['gold'])
                value=dict(prediction=prediction,correct=int(prediction==gold),generation=text,generated_tokens=generated,reached_cap=int(generated>=CFG['gsm8k_formal_max_new_tokens']))
            else:value=candidate_scores(model,tok,r,native,k,v)
            conditions[arm]=value
        C.save(path,dict(id=r['id'],task=task,prompt=r['prompt'],receiver_suffix=r['receiver_suffix'],candidates=r['candidates'],gold=r['gold'],conditions=conditions,
            manifest_sha256=identity(),mmlu_pro_custom_split=task=='mmlu_pro'))
        if i%8==0 or task=='gsm8k':C.log(f'UFD evaluation {task} {i}/{len(rr)}')
    records=[json.loads((out/'per_sample'/(fname(r)+'.json')).read_text()) for r in rr]
    C.write_jsonl(out/'per_sample.jsonl',records);C.save(out/'completed.json',dict(status='completed',samples=len(rr),manifest_sha256=identity()))


def summarize():
    records=[];tasks={}
    for task in CFG['formal_counts']:
        assert (ROOT/f'evaluation/{task}/completed.json').exists()
        ss=C.read_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl');records.extend(ss)
        metrics={arm:dict(n=len(ss),correct=sum(r['conditions'][arm]['correct'] for r in ss),accuracy=sum(r['conditions'][arm]['correct'] for r in ss)/len(ss)) for arm in ss[0]['conditions']}
        if task=='gsm8k':
            for arm,m in metrics.items():m.update(mean_generated_tokens=sum(r['conditions'][arm]['generated_tokens'] for r in ss)/len(ss),cap_count=sum(r['conditions'][arm]['reached_cap'] for r in ss))
        paired={}
        for other in ['qwen_native','old_mcq_stage_b']:
            a=[r['conditions']['new_ufd_stage_b']['correct'] for r in ss];b=[r['conditions'][other]['correct'] for r in ss]
            paired[other]=dict(both_correct=sum(x and y for x,y in zip(a,b)),new_only=sum(x and not y for x,y in zip(a,b)),other_only=sum(not x and y for x,y in zip(a,b)),both_wrong=sum(not x and not y for x,y in zip(a,b)))
        tasks[task]=dict(metrics=metrics,paired=paired)
    summary=dict(experiment=CFG['name'],status='completed',config=CFG,tasks=tasks,training=json.loads((ROOT/'runs/stage_b/completed.json').read_text()),
        caveats=['MMLU-Pro is custom heldout split with explicit ARC-style A-J extension, NOT untouched official test/CoT benchmark.',
        'New loss and training mixture both changed; comparisons are frozen checkpoints, not same-data loss ablations.',
        'Old MCQ best checkpoint was validation-choice-KL selected; new primary is fixed epoch4. Report checkpoint identity, not a hidden common selection claim.'])
    C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records);C.save(ROOT/'summary.json',summary);shared('completed',summary)
    C.save(ROOT/'completed.json',dict(status='completed',manifest_sha256=identity()));status(status='completed',phase='all');C.log('ALL MIXED-PREFIX UFD EXPERIMENTS COMPLETED')


if __name__=='__main__':
    try:
        C.seed();action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='capture':capture(len(sys.argv)>2 and sys.argv[2]=='smoke')
        elif action=='smoke':smoke()
        elif action=='train':train()
        elif action=='evaluate':evaluate(sys.argv[2])
        elif action=='summarize':summarize()
        else:raise ValueError(action)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc));C.save(ROOT/'failure.json',dict(status='failed_or_incomplete',**error));status(status='failed',**error)
        shared('failed_or_incomplete',error);raise
