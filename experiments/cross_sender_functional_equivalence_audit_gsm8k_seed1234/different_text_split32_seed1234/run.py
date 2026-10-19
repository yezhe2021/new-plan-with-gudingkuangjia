"""Frozen, complementary-fragment KV composition pilot. Server A only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,re,time,hashlib,importlib.util,itertools,gc
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_fragment_capture',ROOT/'capture_driver.py')
D=importlib.util.module_from_spec(spec);sys.modules[spec.name]=D;spec.loader.exec_module(D)
A,C,CFG=D.A,D.C,D.CFG
def save(name,x):A.save_json(ROOT/name,x)
def log(x):D.log(x)
def stamp(**x):save('status.json',dict(timestamp=time.strftime('%F %T'),**x))
def items(mode):
    result=json.loads((ROOT/'splits.json').read_text())
    return result[:CFG['smoke_samples']] if mode=='smoke' else result
def temp(mode):return ROOT/'temporary_states'/mode
def references():return {r['id']:r for r in A.read_jsonl(ROOT/'reference_baselines.jsonl')}

def split_text(q):
    def eligible(cut,kind):
        a,b=q[:cut],q[cut:]
        if len(a.split())<4 or len(b.split())<4:return False
        if not a[-1:].isspace() or not b[:1] or b[0].isspace():return False
        # Conservative: do not cut immediately beside ANY token containing a digit.
        # A full sentence boundary cannot split a number/unit within that sentence.
        # Permit e.g. "ratio of 7:11. | If their total age...".
        if kind!='sentence' and any(c.isdigit() for c in a.split()[-1]+b.split()[0]):return False
        return True
    sentence=[]
    for m in re.finditer(r'(?<=[.!?])\s+',q):
        if not re.search(r'\b(?:Mr|Mrs|Ms|Dr|St|vs|etc)\.$',q[:m.start()],re.I):sentence.append(m.end())
    tiers=[('sentence',sentence),('punctuation_clause',[m.end() for m in re.finditer(r'(?<=[,;:])\s+',q)]),
        ('conjunction_or_question',[m.end() for m in re.finditer(r'\s+(?=(?:and|but|then|while|after|before|when|if|how|what)\b)',q,re.I)]),
        ('word_boundary',[m.end() for m in re.finditer(r'\s+(?=[A-Za-z])',q)])]
    for kind,candidates in tiers:
        candidates=[c for c in candidates if eligible(c,kind)]
        if candidates:
            cut=min(candidates,key=lambda c:(abs(len(q[:c].split())-len(q[c:].split())),c))
            raw_a,raw_b=q[:cut],q[cut:]
            assert raw_a+raw_b==q
            assert not (q[cut-1].isalnum() and q[cut].isalnum())
            return dict(A=raw_a.strip(),B=raw_b.strip(),raw_A=raw_a,raw_B=raw_b,offset=cut,boundary_type=kind)
    raise ValueError('No valid split in fixed sample; do not silently replace question: '+q)

def prepare():
    assert torch.cuda.is_available()
    source=A.rows()[:CFG['sample_count']]
    baseline=A.read_jsonl(Path(CFG['reference_run'])/'formal/per_sample.jsonl')[:CFG['sample_count']]
    assert len(source)==len(baseline)==32
    selected=[]
    for i,(r,b) in enumerate(zip(source,baseline)):
        assert b['id']==f'test_{i}' and b['question']==r['question'] and b['gold_answer']==r['answer']
        s=split_text(r['question'])
        assert s['raw_A']+s['raw_B']==r['question']
        for value in b['conditions'].values():
            assert C.generation_record(value['generated_text'],value['generated_tokens'],value['stop_reason'],r['answer'])==value
        selected.append(dict(id=f'test_{i}',question=r['question'],answer=r['answer'],**s,
                             both_fragments_required='not assumed; inspect omission controls and human review'))
    # Unit cases include a decimal, comma-separated number and abbreviation.
    for q in ['A worker bought 1.5 pounds of apples. She later sold 12 pounds and asks how many remain now?',
              'Mr. Smith owns 1,000 red marbles and buys more green marbles. How many marbles does he own after buying ten more?']:
        s=split_text(q);assert s['raw_A']+s['raw_B']==q
    files=[Path(__file__),ROOT/'config.json',ROOT/'capture_driver.py',ROOT.parent/'equivalence_audit.py',Path(CFG['generation_common'])]
    for family in ('gemma','llama'):files += [Path(A.CFG[family][k]) for k in ('stage_a','adapter','translator')]
    manifest=dict(config=CFG,parent_config=A.CFG,files={str(p):D.sha(p) for p in files},
        split_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
        baseline_sha256=D.sha(Path(CFG['reference_run'])/'formal/per_sample.jsonl'))
    old=json.loads((Path(CFG['reference_run'])/'manifest.json').read_text())
    for f in ('gemma','llama'):
        for k in ('stage_a','adapter','translator'):
            path=A.CFG[f][k];assert old['files'][path]==manifest['files'][path]
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==manifest,'Manifest changed'
    else:save('manifest.json',manifest)
    save('splits.json',selected);A.write_jsonl(ROOT/'reference_baselines.jsonl',baseline)
    lines=['# Fixed first32 GSM8K question splits','',
        'No gold answer or prior correctness used by the split rule. A+B raw strings reconstruct the original.','']
    for s in selected:lines += [f'## {s["id"]}: {s["boundary_type"]}','', '**A:** '+s['A'],'','**B:** '+s['B'],'']
    (ROOT/'splits.md').write_text('\n'.join(lines))
    save('resource_policy.json',dict(torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),server='A',gemma='FP32',llama_qwen='FP16'))
    log('PREPARE PASSED: first32 fixed IDs, lossless splits, frozen weights, parser identity')

@torch.no_grad()
def capture(family,mode):
    rows=items(mode);root=temp(mode)/family
    sr,tr=root/'source',root/'translated';sr.mkdir(parents=True,exist_ok=True);tr.mkdir(parents=True,exist_ok=True)
    pairs=[(i,part) for i in range(len(rows)) for part in ('A','B') if not (tr/f'{i:04d}_{part}.pt').exists()]
    if not pairs:return
    stok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models'][family],local_files_only=True)
    qtok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models']['qwen'],local_files_only=True)
    missing=[(i,p) for i,p in pairs if not (sr/f'{i:04d}_{p}.pt').exists()]
    if missing:
        D.guard(family);stamp(status='running',phase='fragment_capture',family=family,mode=mode)
        sender=A.ref.load_model(family)
        for n,(i,part) in enumerate(missing,1):
            row=rows[i];text=f"Question: {row[part]}\n"
            assert row[part] in row['question']
            aligned=A.aligned_row(row['id']+'_'+part,text,stok,qtok,family)
            target,source,counts=A.ref.fullsync_map(aligned,stok,qtok,family)
            sbody=aligned['encoded'][family]['body'];qbody=aligned['encoded']['qwen']['body']
            assert target==list(range(1,len(qbody)))
            k,v,_=A.ref.capture(sender,sbody,len(sbody),[0])
            torch.save(dict(id=row['id'],part=part,fragment=row[part],source_text=text,qwen_body=qbody,
                target_indices=target,source_k=k[:,source].half(),source_v=v[:,source].half(),mapping_counts=counts),sr/f'{i:04d}_{part}.pt')
            del k,v
            if n%8==0 or n==len(missing):log(f'{mode} {family} fragment capture {n}/{len(missing)}')
        del sender;gc.collect();torch.cuda.empty_cache()
    base,adapter=A.make_models(family)
    for n,(i,part) in enumerate(pairs,1):
        p=sr/f'{i:04d}_{part}.pt';row=torch.load(p,map_location='cpu',weights_only=True);ks,vs=[],[]
        for begin in range(0,row['source_k'].shape[1],A.CFG['chunk_tokens']):
            end=begin+A.CFG['chunk_tokens']
            with torch.amp.autocast('cuda',dtype=torch.float16):
                k,v=base(row['source_k'][:,begin:end][None].cuda(),row['source_v'][:,begin:end][None].cuda())
                k,v,_,_=adapter(k,v)
            ks.append(k[0].cpu().half());vs.append(v[0].cpu().half())
        row.pop('source_k');row.pop('source_v');row.update(key=torch.cat(ks,1),value=torch.cat(vs,1))
        assert torch.isfinite(row['key']).all() and torch.isfinite(row['value']).all()
        torch.save(row,tr/f'{i:04d}_{part}.pt');p.unlink()
        if n%8==0 or n==len(pairs):log(f'{mode} {family} fragment translated {n}/{len(pairs)}')

@torch.no_grad()
def masked_generate(model,tok,suffix,key,value,memory_mask):
    # Exact old greedy loop, except for preserving a predefined memory visibility mask.
    assert len(memory_mask)==key.shape[1]
    past=A.ref.make_cache(model,key,value,torch.arange(key.shape[1],device='cuda'))
    current=torch.tensor([suffix],device='cuda',dtype=torch.long)
    eos=model.generation_config.eos_token_id
    eos={int(eos)} if isinstance(eos,int) else {int(x) for x in eos}
    generated,reason,text=[],'max_tokens',''
    visible=torch.tensor([memory_mask],device='cuda',dtype=torch.long)
    for _ in range(CFG['max_new_tokens']):
        prefix=int(past.get_seq_length());total=prefix+current.shape[1]
        mask=torch.cat((visible,torch.ones(1,total-len(memory_mask),device='cuda',dtype=torch.long)),1)
        output=A.ref.backbone(model)(input_ids=current,attention_mask=mask,
            position_ids=torch.arange(prefix,total,device='cuda')[None],past_key_values=past,use_cache=True,return_dict=True)
        token=int(model.lm_head(output.last_hidden_state[:,-1])[0].argmax().item());past=output.past_key_values
        if token in eos:reason='eos';break
        generated.append(token);text=tok.decode(generated,skip_special_tokens=True)
        hits=[(text.find(marker),marker) for marker in ('Question:','</s>','<|im_end|>') if text.find(marker)>=0]
        if hits:
            first,marker=min(hits,key=lambda x:x[0]);text,reason=text[:first],f'until:{marker}';break
        current=torch.tensor([[token]],device='cuda',dtype=torch.long)
    return text,len(generated),reason

@torch.no_grad()
def native_gate(qwen,suffix,body,native):
    cache_logits=A.ref.readout(qwen,suffix,*native)
    ids=torch.tensor([body+suffix],device='cuda')
    output=A.ref.backbone(qwen)(input_ids=ids,attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(ids.shape[1],device='cuda')[None],use_cache=False)
    full=qwen.lm_head(output.last_hidden_state[:,-1])[0].float()
    lf,lc=F.log_softmax(full,-1),F.log_softmax(cache_logits,-1)
    kl=float((lf.exp()*(lf-lc)).sum());mae=float((full-cache_logits).abs().mean())
    same=int(full.argmax())==int(cache_logits.argmax())
    assert same and kl<.001,(same,kl,mae)
    return dict(native_cache_first_argmax_same=same,first_logit_kl=kl,first_logit_mae=mae)

@torch.no_grad()
def summarize(records):
    names=list(records[0]['conditions'])
    metrics={n:C.summarize_generations([r['conditions'][n] for r in records]) for n in names}
    scores=torch.tensor([[int(r['conditions'][n]['flexible_correct']) for n in names] for r in records],device='cuda',dtype=torch.float32)
    rng=torch.Generator(device='cuda');rng.manual_seed(CFG['seed'])
    boot=scores[torch.randint(len(records),(CFG['bootstrap_resamples'],len(records)),generator=rng,device='cuda')].mean(1)
    comparisons=[]
    for h in ('llamaA_gemmaB','gemmaA_llamaB'):
        for control in ('llamaA_llamaB','gemmaA_gemmaB','nativeA_nativeB',h+'_maskA',h+'_maskB'):comparisons.append((control,h))
    comparisons.append(('llamaA_gemmaB','gemmaA_llamaB'))
    paired={}
    for a,b in comparisons:
        ia,ib=names.index(a),names.index(b);x,y=scores[:,ia].bool(),scores[:,ib].bool()
        paired[a+'_vs_'+b]=dict(a=a,b=b,both_correct=int((x&y).sum()),only_a=int((x&~y).sum()),only_b=int((~x&y).sum()),both_wrong=int((~x&~y).sum()),
            accuracy_b_minus_a=float((scores[:,ib]-scores[:,ia]).mean()),paired_bootstrap_ci95=torch.quantile(boot[:,ib]-boot[:,ia],torch.tensor([.025,.975],device='cuda')).tolist())
    groups={}
    for r in records:
        c=r['conditions'];l,g=c['full_llama']['flexible_correct'],c['full_gemma']['flexible_correct']
        name='both' if l and g else 'llama_only' if l else 'gemma_only' if g else 'neither'
        group=groups.setdefault(name,dict(count=0,correct={n:0 for n in names}));group['count']+=1
        for n in names:group['correct'][n]+=int(c[n]['flexible_correct'])
    return dict(count=len(records),planned_count=32,metrics=metrics,paired=paired,full_question_endpoint_groups=groups,
        baseline_provenance='full_* references reused from prior full-question mix128 run, first32 exact IDs',
        caveats=['Automatic splits do not prove every sample requires both parts','Frozen full-question Writers face fragment distribution shift',
                 'Native-split is independently encoded fragments, not native complete-question prefill','Masked controls retain lengths/positions'])

def report(s):
    lines=['# Different-text KV composition, fixed32 GSM8K questions','',
        'Frozen GSM8K Stage-A+Hybrid Writers. Old greedy256/stops/Flexible. Text order A→B fixed.',
        'full_* references reused, nine fragment conditions newly generated.','',
        '| Condition | Correct | Flexible accuracy | Mean generated tokens | Max-token rate |','|---|---:|---:|---:|---:|']
    for n,m in s['metrics'].items():lines.append(f'| {n} | {round(m["flexible_accuracy"]*s["count"])} | {m["flexible_accuracy"]*100:.2f}% | {m["mean_generated_tokens"]:.2f} | {m["max_token_rate"]*100:.2f}% |')
    lines+=['','Pairwise counts/intervals and endpoint groups are in formal/summary.json.',
        'Inspect splits.md before semantic conclusions; no new parameters were trained.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

@torch.no_grad()
def evaluate(mode):
    D.guard('qwen');qwen=A.ref.load_model('qwen');tok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models']['qwen'],local_files_only=True)
    suffix=list(tok.encode('Answer:',add_special_tokens=False));previous=references();records=[];checks=[]
    for i,row in enumerate(items(mode)):
        p=ROOT/mode/'per_sample'/f'{i:04d}.json'
        if p.exists():records.append(json.loads(p.read_text()));continue
        saved={f+'_'+part:torch.load(temp(mode)/f/'translated'/f'{i:04d}_{part}.pt',map_location='cpu',weights_only=True) for f in ('llama','gemma') for part in ('A','B')}
        for part in ('A','B'):
            a,b=saved['llama_'+part],saved['gemma_'+part]
            assert a['qwen_body']==b['qwen_body'] and a['target_indices']==b['target_indices']
            assert a['fragment']==b['fragment']==row[part]
        states={n:(s['key'].cuda(),s['value'].cuda()) for n,s in saved.items()}
        for part in ('A','B'):
            body=saved['llama_'+part]['qwen_body']
            k,v,_=A.ref.capture(qwen,body,len(body),[0]);k,v=k.cuda().half(),v.cuda().half()
            if mode=='smoke':checks.append(dict(id=row['id'],part=part,**native_gate(qwen,suffix,body,(k,v))))
            if part=='A':anchor=(k[:,:1],v[:,:1])
            else:assert float(F.cosine_similarity(anchor[0].float(),k[:,:1].float(),dim=-1).mean())>.9999
            states['native_'+part]=(k[:,1:],v[:,1:])
        na,nb=states['llama_A'][0].shape[1],states['llama_B'][0].shape[1]
        assert states['gemma_A'][0].shape[1]==states['native_A'][0].shape[1]==na
        assert states['gemma_B'][0].shape[1]==states['native_B'][0].shape[1]==nb
        record=dict(id=row['id'],question=row['question'],gold_answer=row['answer'],A=row['A'],B=row['B'],boundary_type=row['boundary_type'],
            block_A_slots=na,block_B_slots=nb,cache_tokens=1+na+nb,conditions={'full_'+n:r for n,r in previous[row['id']]['conditions'].items()})
        for name,(left,right,visibility) in CFG['conditions'].items():
            stamp(status='running',phase='generation',mode=mode,sample=i+1,total=len(items(mode)),condition=name)
            k,v=tuple(torch.cat((anchor[j],states[left][j],states[right][j]),1) for j in (0,1))
            assert k.shape[1]==v.shape[1]==1+na+nb
            mask=[1]*(1+na+nb)
            if visibility=='maskA':mask[1:1+na]=[0]*na
            if visibility=='maskB':mask[1+na:]=[0]*nb
            assert mask[0]==1 and sum(mask)==(1+nb if visibility=='maskA' else 1+na if visibility=='maskB' else 1+na+nb)
            if mode=='smoke':
                assert torch.equal(k[:,:1],anchor[0]) and torch.equal(k[:,1:1+na],states[left][0]) and torch.equal(k[:,1+na:],states[right][0])
                cache=A.ref.make_cache(qwen,k,v,torch.arange(1+na+nb,device='cuda'))
                assert cache.get_seq_length()==1+na+nb
                del cache
            if visibility=='both':result=C.greedy_generate(qwen,tok,suffix,CFG['max_new_tokens'],k,v)
            else:result=masked_generate(qwen,tok,suffix,k,v,mask)
            if mode=='smoke' and name=='llamaA_gemmaB':
                assert result==masked_generate(qwen,tok,suffix,k,v,[1]*(1+na+nb)),'All-ones masked decoder differs from old decoder'
                checks.append(dict(id=row['id'],all_ones_decoder_exact=True,block_A_slots=na,block_B_slots=nb))
            parsed=C.generation_record(*result,row['answer']);record['conditions'][name]=parsed
            log(f'{mode} {i+1}/{len(items(mode))} {name}: flexible={parsed["flexible_correct"]} tokens={result[1]} stop={result[2]}')
        save(f'{mode}/per_sample/{i:04d}.json',record);records.append(record)
        save(f'{mode}/progress.json',dict(completed_samples=len(records),last_id=row['id']))
        if (i+1)%4==0:save(f'{mode}/partial_summary.json',summarize(records))
        del saved,states,k,v;torch.cuda.empty_cache()
    A.write_jsonl(ROOT/mode/'per_sample.jsonl',records)
    s=summarize(records);save(f'{mode}/summary.json',s);save(f'{mode}/completed.json',dict(status='completed',count=len(records),checks=checks))
    log(mode.upper()+' COMPLETED')
    if mode=='formal':
        report(s);stamp(status='completed',phase='generation',samples=len(records))
        base=(ROOT/'temporary_states').resolve();assert base.parent==ROOT and base.name=='temporary_states'
        for p in base.rglob('*.pt'):
            assert p.resolve().is_relative_to(base);p.unlink()
        log('Own temporary KV tensors removed after results saved')

if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='capture':capture(sys.argv[2],sys.argv[3])
        elif action=='evaluate':evaluate(sys.argv[2])
        else:raise ValueError(action)
    except Exception as exc:
        save('failure.json',dict(type=type(exc).__name__,message=str(exc),timestamp=time.strftime('%F %T')))
        stamp(status='failed',reason=str(exc));raise
