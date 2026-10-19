"""Same-question heterogeneous KV concatenation, no training or monitoring."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,importlib.util,itertools
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
D=load('frozen_concat_capture_driver',ROOT/'capture_driver.py')
A,C,CFG=D.A,D.C,D.CFG
REFERENCE=Path(CFG['reference_run'])
def save(name,data):A.save_json(ROOT/name,data)
def log(x):D.log(x)
def stamp(**x):save('status.json',dict(timestamp=time.strftime('%F %T'),**x))
def references():
    rows=A.read_jsonl(ROOT/'reference_baselines.jsonl')
    return {r['id']:r for r in rows}

def prepare():
    D.prepare()
    assert (REFERENCE/'formal/completed.json').is_file()
    baseline=A.read_jsonl(REFERENCE/'formal/per_sample.jsonl')
    selected=D.samples('formal')
    assert len(baseline)==len(selected)==128
    previous=json.loads((REFERENCE/'manifest.json').read_text())
    current=json.loads((ROOT/'manifest.json').read_text())
    assert previous['rows_sha256']==current['rows_sha256']
    for family in ('llama','gemma'):
        for kind in ('stage_a','adapter','translator'):
            path=A.CFG[family][kind]
            assert previous['files'][path]==current['files'][path],(family,kind)
    assert previous['files'][CFG['generation_common']]==current['files'][CFG['generation_common']]
    for i,(r,b) in enumerate(zip(selected,baseline)):
        assert b['id']==f'test_{i}' and r['question']==b['question']
        assert r['answer']==b['gold_answer']
        for condition,result in b['conditions'].items():
            reconstructed=C.generation_record(result['generated_text'],result['generated_tokens'],result['stop_reason'],r['answer'])
            assert reconstructed==result,(i,condition,'Legacy parser changed')
    manifest={'config':CFG,'new_script_sha256':D.sha(__file__),
        'capture_manifest_sha256':D.sha(ROOT/'manifest.json'),
        'baseline_sha256':D.sha(REFERENCE/'formal/per_sample.jsonl'),
        'baseline_completed_sha256':D.sha(REFERENCE/'formal/completed.json')}
    p=ROOT/'concat_manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Concatenation protocol changed'
    else:save('concat_manifest.json',manifest)
    A.write_jsonl(ROOT/'reference_baselines.jsonl',baseline)
    log('CONCAT PREPARE PASSED: exact questions, checkpoint hashes and original parser')

def assemble(anchor,states,order):
    # Only unrotated post-k_norm K is handled here. The unchanged cache builder applies RoPE.
    left,right=order
    return tuple(torch.cat((anchor[j],states[left][j],states[right][j]),dim=1) for j in (0,1))

def cache_layer(cache,i):
    if hasattr(cache,'layers'):return cache.layers[i].keys,cache.layers[i].values
    return cache.key_cache[i],cache.value_cache[i]

@torch.no_grad()
def protocol_gate(qwen,anchor,states,condition,combined):
    order=CFG['conditions'][condition];n=states['gemma'][0].shape[1]
    k,v=combined
    assert k.shape[1]==v.shape[1]==1+2*n
    for j,x in enumerate((k,v)):
        assert torch.equal(x[:,:1],anchor[j])
        assert torch.equal(x[:,1:1+n],states[order[0]][j])
        assert torch.equal(x[:,1+n:],states[order[1]][j])
    cache=A.ref.make_cache(qwen,k,v,torch.arange(1+2*n,device='cuda'))
    assert cache.get_seq_length()==1+2*n
    single=tuple(torch.cat((anchor[j],states[order[0]][j]),1) for j in (0,1))
    first=A.ref.make_cache(qwen,*single,torch.arange(n+1,device='cuda'))
    error=0.;shift=0.
    for layer in range(k.shape[0]):
        ck,cv=cache_layer(cache,layer);fk,fv=cache_layer(first,layer)
        error=max(error,float((ck[...,:n+1,:]-fk).abs().max()))
        assert torch.allclose(ck[...,:n+1,:],fk,atol=.002,rtol=.001)
        assert torch.equal(cv,value_view(v[layer]))
        # Compare the same second block at original vs shifted positions.
        original=A.ref.apply_rope(qwen,k[layer,1+n:],torch.arange(1,n+1,device='cuda'),layer)
        shifted=ck[0,:,n+1:,:].transpose(0,1)
        shift+=float((shifted-original).abs().mean())/k.shape[0]
    assert shift>1e-5,'Second block did not receive shifted RoPE positions'
    return dict(condition=condition,body_tokens_per_block=n,cache_tokens=1+2*n,
                first_block_rope_max_error=error,second_block_rope_mean_shift=shift,passed=True)

def value_view(v):return v.permute(1,0,2).unsqueeze(0)

@torch.no_grad()
def summarize(records):
    names=list(records[0]['conditions'])
    metrics={name:C.summarize_generations([r['conditions'][name] for r in records]) for name in names}
    scores=torch.tensor([[int(r['conditions'][name]['flexible_correct']) for name in names] for r in records],device='cuda',dtype=torch.float32)
    rng=torch.Generator(device='cuda');rng.manual_seed(CFG['seed'])
    indices=torch.randint(len(records),(CFG['bootstrap_resamples'],len(records)),generator=rng,device='cuda')
    boot=scores[indices].mean(1)
    paired={}
    for ia,ib in itertools.combinations(range(len(names)),2):
        a,b=names[ia],names[ib];x=scores[:,ia].bool();y=scores[:,ib].bool()
        interval=torch.quantile(boot[:,ib]-boot[:,ia],torch.tensor([.025,.975],device='cuda'))
        paired[f'{a}_vs_{b}']=dict(both_correct=int((x&y).sum()),only_a=int((x&~y).sum()),
            only_b=int((~x&y).sum()),both_wrong=int((~x&~y).sum()),a=a,b=b,
            accuracy_b_minus_a=float((scores[:,ib]-scores[:,ia]).mean()),paired_bootstrap_ci95=interval.tolist())
    groups={}
    for r in records:
        c=r['conditions'];l=c['llama']['flexible_correct'];g=c['gemma']['flexible_correct']
        group='both' if l and g else 'llama_only' if l else 'gemma_only' if g else 'neither'
        item=groups.setdefault(group,dict(count=0,correct={name:0 for name in names}))
        item['count']+=1
        for name in names:item['correct'][name]+=int(c[name]['flexible_correct'])
    return dict(count=len(records),planned_count=128,metrics=metrics,paired=paired,endpoint_groups=groups,
        baseline_provenance='gemma/mixed50/llama reused from immediately preceding completed run; four concat arms newly generated',
        protocol='Legacy GSM8K free-generation/Flexible, no training; same-question memories, doubled length controls')

def report(s):
    labels={'gemma':'G (reused)','llama':'L (reused)','mixed50':'Mean50 (reused)',
            'llama_llama':'L+L','gemma_gemma':'G+G','llama_gemma':'L+G','gemma_llama':'G+L'}
    lines=['# Same-question heterogeneous KV concatenation','',f'Completed {s["count"]}/128 examples. No training. Legacy Flexible accuracy.',
        'Single-source/Mean50 baselines reused from the preceding run, not rerun formal arms.','',
        '| Arm | Correct | Accuracy | Mean generated tokens | Max-token rate |','|---|---:|---:|---:|---:|']
    for name,m in s['metrics'].items():
        lines.append(f'| {labels[name]} | {round(m["flexible_accuracy"]*s["count"])} | {m["flexible_accuracy"]*100:.2f}% | {m["mean_generated_tokens"]:.2f} | {m["max_token_rate"]*100:.2f}% |')
    lines+=['','## Endpoint-group retention','','| Group | n | L+L | G+G | L+G | G+L |','|---|---:|---:|---:|---:|---:|']
    for name,m in s['endpoint_groups'].items():
        lines.append('| '+name+' | '+str(m['count'])+' | '+' | '.join(str(m['correct'][a]) for a in CFG['conditions'])+' |')
    lines+=['','Paired counts and paired bootstrap intervals for all arm pairs are in summary.json.',
            'Interpret heterogeneous arms against SAME-LENGTH G+G/L+L before claiming multi-source benefit.',
            'No distinct evidence fusion claim; each source saw the same question. Order effects are reported, not optimized on test.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

@torch.no_grad()
def evaluate(mode):
    D.guard('qwen');qwen=A.ref.load_model('qwen')
    tok=A.AutoTokenizer.from_pretrained(A.ref.CFG['models']['qwen'],local_files_only=True)
    suffix=list(tok.encode('Answer:',add_special_tokens=False))
    previous=references();records=[];checks=[]
    for i,row in enumerate(D.samples(mode)):
        p=ROOT/mode/'per_sample'/f'{i:04d}.json'
        if p.exists():records.append(json.loads(p.read_text()));continue
        saved={family:torch.load(D.state_dir(mode)/family/'translated'/f'{i:04d}.pt',map_location='cpu',weights_only=True) for family in ('llama','gemma')}
        l,g=saved['llama'],saved['gemma'];assert l['id']==g['id']==f'test_{i}'
        assert l['qwen_body']==g['qwen_body'] and l['target_indices']==g['target_indices']
        assert l['question']==g['question']==row['question']
        qk,qv,_=A.ref.capture(qwen,g['qwen_body'],len(g['qwen_body']),[0])
        anchor=(qk[:,:1].cuda().half(),qv[:,:1].cuda().half());del qk,qv
        states={name:(s['key'].cuda(),s['value'].cuda()) for name,s in saved.items()}
        baseline=previous[f'test_{i}']
        if mode=='smoke':
            # Test the exact three old baseline generators on the same freshly recaptured states.
            for name,alpha in (('gemma',0.),('mixed50',.5),('llama',1.)):
                k=torch.cat((anchor[0],(1-alpha)*states['gemma'][0]+alpha*states['llama'][0]),1)
                v=torch.cat((anchor[1],(1-alpha)*states['gemma'][1]+alpha*states['llama'][1]),1)
                text,count,reason=C.greedy_generate(qwen,tok,suffix,CFG['max_new_tokens'],k,v)
                reproduced=C.generation_record(text,count,reason,row['answer'])
                assert reproduced==baseline['conditions'][name],(i,name,'Smoke baseline reproduction failed')
                checks.append(dict(id=f'test_{i}',baseline=name,exact_generation_reproduction=True))
        record=dict(id=f'test_{i}',question=row['question'],gold_answer=row['answer'],
            body_tokens_per_block=g['key'].shape[1],baseline_conditions_reused=['gemma','mixed50','llama'],
            conditions=dict(baseline['conditions']))
        for name,order in CFG['conditions'].items():
            stamp(status='running',phase='generation',mode=mode,sample=i+1,total=len(D.samples(mode)),condition=name)
            k,v=assemble(anchor,states,order)
            assert k.shape[1]==1+2*record['body_tokens_per_block']
            if mode=='smoke':checks.append(dict(id=f'test_{i}',**protocol_gate(qwen,anchor,states,name,(k,v))))
            text,count,reason=C.greedy_generate(qwen,tok,suffix,CFG['max_new_tokens'],k,v)
            result=C.generation_record(text,count,reason,row['answer'])
            if mode=='smoke' and name=='llama_gemma':
                assert (text,count,reason)==C.greedy_generate(qwen,tok,suffix,CFG['max_new_tokens'],k,v),'Fresh cache repeat failed'
            record['conditions'][name]=result
            log(f'{mode} {i+1}/{len(D.samples(mode))} {name}: flexible={result["flexible_correct"]} tokens={count} stop={reason}')
        save(f'{mode}/per_sample/{i:04d}.json',record);records.append(record)
        save(f'{mode}/progress.json',dict(completed_samples=len(records),last_id=record['id']))
        if (i+1)%8==0:save(f'{mode}/partial_summary.json',summarize(records))
        del saved,states,k,v;torch.cuda.empty_cache()
    A.write_jsonl(ROOT/mode/'per_sample.jsonl',records)
    s=summarize(records);save(f'{mode}/summary.json',s)
    save(f'{mode}/completed.json',dict(status='completed',samples=len(records),checks=checks))
    log(f'{mode.upper()} COMPLETED')
    if mode=='formal':
        report(s);stamp(status='completed',phase='generation',samples=len(records))
        base=D.state_dir(mode).parent.resolve()
        assert base==ROOT/'temporary_states'
        for tensor in base.rglob('*.pt'):
            assert tensor.resolve().is_relative_to(base)
            tensor.unlink()
        log('Own temporary KV tensors removed after result saving')

if __name__=='__main__':
    try:
        action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='capture':D.capture(sys.argv[2],sys.argv[3])
        elif action=='evaluate':evaluate(sys.argv[2])
        else:raise ValueError(action)
    except Exception as exc:
        save('failure.json',dict(type=type(exc).__name__,message=str(exc),timestamp=time.strftime('%F %T')))
        stamp(status='failed',reason=str(exc));raise
