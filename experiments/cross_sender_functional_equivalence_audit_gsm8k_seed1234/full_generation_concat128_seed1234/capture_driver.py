import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,gc,hashlib,time,importlib.util
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
A=module('original_cross_sender_audit',ROOT.parent/'equivalence_audit.py')
C=module('original_generation_common_mix',CFG['generation_common'])
torch.set_num_threads(1)
def save(path,x): A.save_json(ROOT/path,x)
def log(x): print(f'[{time.strftime("%F %T")}] {x}',flush=True)
def status(**x): save('status.json',dict(time=time.strftime('%F %T'),**x))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def samples(mode):return A.rows()[:CFG['smoke_samples'] if mode=='smoke' else CFG['sample_count']]
def state_dir(mode):return ROOT/'temporary_states'/mode
def prepare():
    A.preflight()
    assert torch.cuda.is_available()
    rows=samples('formal');assert len(rows)==128
    old={family:A.read_jsonl(A.CFG[family]['per_sample'])[:128] for family in ('gemma','llama')}
    for family in old:
        assert [r['id'] for r in old[family]]==[f'test_{i}' for i in range(128)]
        assert [r['question'] for r in old[family]]==[r['question'] for r in rows]
    files=[ROOT/'config.json',Path(__file__),ROOT.parent/'equivalence_audit.py',Path(CFG['generation_common'])]
    for f in ('llama','gemma'):
        files += [Path(A.CFG[f][k]) for k in ('translator','stage_a','adapter','per_sample')]
    manifest={'config':CFG,'parent_config':A.CFG,'rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
              'files':{str(p):sha(p) for p in files},'old_correctness':A.weak_correctness(128)}
    p=ROOT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Manifest changed; preserve old run first'
    else:save('manifest.json',manifest)
    save('resource_policy.json',dict(torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                                   gemma_dtype='float32',llama_qwen_dtype='float16',server='A'))
    log('PREPARE PASSED: exact old 128 test questions and frozen checkpoint identities')
def guard(family):
    assert torch.cuda.is_available()
    free,_=torch.cuda.mem_get_info()
    assert free/2**30>=CFG['minimum_free_gpu_gib'][family],f'{family}: only {free/2**30:.2f} GiB free; no preemption'
@torch.no_grad()
def capture(family,mode):
    selected=samples(mode);root=state_dir(mode)/family
    sr=root/'source';tr=root/'translated';sr.mkdir(parents=True,exist_ok=True);tr.mkdir(parents=True,exist_ok=True)
    todo=[i for i in range(len(selected)) if not (tr/f'{i:04d}.pt').exists()]
    if not todo:log(f'{mode} {family} captures already complete');return
    source_tok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models'][family],local_files_only=True)
    qtok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models']['qwen'],local_files_only=True)
    missing=[i for i in todo if not (sr/f'{i:04d}.pt').exists()]
    if missing:
        guard(family);status(phase='sender_capture',family=family,mode=mode,status='running')
        sender=A.ref.load_model(family)
        for n,i in enumerate(missing,1):
            row=selected[i];text=f"Question: {row['question']}\n"
            aligned=A.aligned_row(f'test_{i}',text,source_tok,qtok,family)
            target,source,counts=A.ref.fullsync_map(aligned,source_tok,qtok,family)
            body=aligned['encoded'][family]['body'];qbody=aligned['encoded']['qwen']['body']
            assert target==list(range(1,len(qbody))), 'Body slots must cover every non-token0 position'
            k,v,_=A.ref.capture(sender,body,len(body),[0])
            torch.save(dict(id=f'test_{i}',question=row['question'],answer=row['answer'],source_k=k[:,source].half(),
                source_v=v[:,source].half(),target_indices=target,qwen_body=qbody,mapping_counts=counts),sr/f'{i:04d}.pt')
            del k,v
            if n%8==0 or n==len(missing):log(f'{mode} {family} sender {n}/{len(missing)}')
        del sender;gc.collect();torch.cuda.empty_cache()
    base,adapter=A.make_models(family)
    for n,i in enumerate(todo,1):
        item=torch.load(sr/f'{i:04d}.pt',map_location='cpu',weights_only=True)
        outk,outv=[],[]
        for begin in range(0,item['source_k'].shape[1],A.CFG['chunk_tokens']):
            end=begin+A.CFG['chunk_tokens']
            with torch.amp.autocast('cuda',dtype=torch.float16):
                k,v=base(item['source_k'][:,begin:end][None].cuda(),item['source_v'][:,begin:end][None].cuda())
                k,v,_,_=adapter(k,v)
            outk.append(k[0].cpu().half());outv.append(v[0].cpu().half())
        item.pop('source_k');item.pop('source_v');item.update(key=torch.cat(outk,1),value=torch.cat(outv,1))
        assert torch.isfinite(item['key']).all() and torch.isfinite(item['value']).all()
        torch.save(item,tr/f'{i:04d}.pt')
        # Only this run's source intermediate, after its translated output is saved.
        (sr/f'{i:04d}.pt').unlink()
        if n%8==0 or n==len(todo):log(f'{mode} {family} translated {n}/{len(todo)}')
    del base,adapter;gc.collect();torch.cuda.empty_cache()
def summary(records):
    names=list(CFG['alphas']);metrics={n:C.summarize_generations([r['conditions'][n] for r in records]) for n in names}
    paired={}
    for a,b in (('gemma','llama'),('gemma','mixed50'),('llama','mixed50')):
        x=[r['conditions'][a]['flexible_correct'] for r in records];y=[r['conditions'][b]['flexible_correct'] for r in records]
        paired[f'{a}_vs_{b}']={'both_correct':sum(i and j for i,j in zip(x,y)),f'only_{a}':sum(i and not j for i,j in zip(x,y)),
                              f'only_{b}':sum(not i and j for i,j in zip(x,y)),'both_wrong':sum(not i and not j for i,j in zip(x,y))}
    groups={}
    for r in records:
        c=r['conditions'];l=c['llama']['flexible_correct'];g=c['gemma']['flexible_correct']
        name='both' if l and g else 'llama_only' if l else 'gemma_only' if g else 'neither'
        s=groups.setdefault(name,{'count':0,'mixed_correct':0});s['count']+=1;s['mixed_correct']+=int(c['mixed50']['flexible_correct'])
    return dict(count=len(records),metrics=metrics,paired=paired,endpoint_groups=groups,
        endpoint_oracle_union=sum(r['conditions']['llama']['flexible_correct'] or r['conditions']['gemma']['flexible_correct'] for r in records),
        historical_reproduction={family:{'answer_correctness_disagreements':sum(not r['historical_match'][family]['correctness'] for r in records),
            'generated_text_disagreements':sum(not r['historical_match'][family]['text'] for r in records)} for family in ('gemma','llama')})
@torch.no_grad()
def evaluate(mode):
    guard('qwen');qwen=A.ref.load_model('qwen');tok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models']['qwen'],local_files_only=True)
    suffix=list(tok.encode('Answer:',add_special_tokens=False));selected=samples(mode);records=[]
    old={f:{r['id']:r for r in A.read_jsonl(A.CFG[f]['per_sample'])[:128]} for f in ('llama','gemma')}
    probes={r['id']:r for r in A.read_jsonl(ROOT.parent/'results/per_sample.jsonl')}
    checks=[]
    for i,row in enumerate(selected):
        path=ROOT/mode/'per_sample'/f'{i:04d}.json'
        if path.exists():records.append(json.loads(path.read_text()));continue
        status(status='running',phase='full_generation',mode=mode,sample=i+1,total=len(selected))
        states={f:torch.load(state_dir(mode)/f/'translated'/f'{i:04d}.pt',map_location='cpu',weights_only=True) for f in ('gemma','llama')}
        g,l=states['gemma'],states['llama'];assert g['id']==l['id']==f'test_{i}' and g['qwen_body']==l['qwen_body']
        assert g['target_indices']==l['target_indices']
        qk,qv,_=A.ref.capture(qwen,g['qwen_body'],len(g['qwen_body']),[0])
        token0=(qk[:,:1].cuda().half(),qv[:,:1].cuda().half());del qk,qv
        k0,v0=g['key'].cuda(),g['value'].cuda();k1,v1=l['key'].cuda(),l['value'].cuda()
        record={'id':f'test_{i}','question':row['question'],'gold_answer':row['answer'],'conditions':{},'historical_match':{}}
        for name,alpha in CFG['alphas'].items():
            k=(1-alpha)*k0+alpha*k1;v=(1-alpha)*v0+alpha*v1
            if alpha in (0.,1.):
                expected=(k0,v0) if alpha==0 else (k1,v1)
                assert torch.equal(k,expected[0]) and torch.equal(v,expected[1]),'Interpolation endpoint identity failed'
            k=torch.cat((token0[0],k),1);v=torch.cat((token0[1],v),1)
            if mode=='smoke' and name!='mixed50':
                # Same fresh-cache eight-token routine as the completed historical audit.
                ids=A.rollout(qwen,(k,v),len(g['qwen_body']),suffix,
                    {int(x) for x in ([qwen.generation_config.eos_token_id] if isinstance(qwen.generation_config.eos_token_id,int) else qwen.generation_config.eos_token_id)},8)
                expected=probes[record['id']]['rollout_token_ids'][name]
                checks.append({'id':record['id'],'endpoint':name,'probe_exact_match':ids==expected})
                assert ids==expected,checks[-1]
            text,count,reason=C.greedy_generate(qwen,tok,suffix,CFG['max_new_tokens'],k,v)
            result=C.generation_record(text,count,reason,row['answer']);record['conditions'][name]=result
            if mode=='smoke':
                again=C.greedy_generate(qwen,tok,suffix,CFG['max_new_tokens'],k,v)
                assert (text,count,reason)==again,'Fresh-cache repeat generation changed'
            if name!='mixed50':
                historical=old[name][record['id']]['conditions'][A.CFG[name]['condition']]
                record['historical_match'][name]={'correctness':result['flexible_correct']==historical['flexible_correct'],
                    'text':result['generated_text']==historical['generated_text']}
            log(f'{mode} {i+1}/{len(selected)} {name}: flexible={result["flexible_correct"]} tokens={count} stop={reason}')
        save(f'{mode}/per_sample/{i:04d}.json',record);records.append(record)
        save(f'{mode}/partial_summary.json',summary(records))
        del states,k0,v0,k1,v1,k,v;torch.cuda.empty_cache()
    A.write_jsonl(ROOT/mode/'per_sample.jsonl',records)
    save(f'{mode}/summary.json',summary(records))
    save(f'{mode}/completed.json',dict(status='completed',count=len(records),checks=checks))
    log(f'{mode.upper()} COMPLETED '+str(summary(records)))
    if mode=='formal':
        status(status='completed',phase='full_generation',total=len(records))
        # Remove only validated own temporary capture tensors, never historical assets.
        base=(ROOT/'temporary_states').resolve()
        assert base.parent==ROOT and base.name=='temporary_states'
        for p in base.rglob('*.pt'):
            assert p.resolve().is_relative_to(base)
            p.unlink()
        log('Own temporary KV tensors removed; scripts/results/checkpoints retained')
if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='capture':capture(sys.argv[2],sys.argv[3])
        elif action=='evaluate':evaluate(sys.argv[2])
        else:raise ValueError(action)
    except Exception as exc:
        save('failure.json',{'type':type(exc).__name__,'message':str(exc),'time':time.strftime('%F %T')})
        status(status='failed',reason=str(exc));raise
