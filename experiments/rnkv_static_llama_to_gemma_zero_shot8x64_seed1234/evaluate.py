"""Frozen Llama-trained static weights attached to Gemma Recolor+C KV."""
from experiment_common import *
from score import score

def same_prediction(value,reference,task,condition):
    for k in ['prediction','correct','em','f1','legacy_prediction','legacy_correct','generation','token_ids','generated_tokens','reached_cap']:
        if k in reference:assert value[k]==reference[k],(task,condition,k)
    if 'scores' in reference:
        assert len(value['scores'])==len(reference['scores'])
        assert max(abs(a-b) for a,b in zip(value['scores'],reference['scores']))<1e-4,(task,condition,'scores')

@torch.no_grad()
def check_interfaces(engine,module):
    saved=C.read(ROOT/'export/completed.json')
    assert C.sha(ROOT/'export/canonical.pt')==saved['file_sha256']
    source=torch.load(ROOT/'export/canonical.pt',map_location='cpu',weights_only=True)
    assert source['manifest_sha256']==identity() and source['source_checkpoint_sha256']==CFG['static_checkpoint_sha256']
    assert source['prefix_ids']==engine.prefix_ids and source['receiver_config_sha256']==C.digest(engine.model.config.to_dict())
    actual=module.canonical()
    assert all(torch.equal(x.cpu(),y) for x,y in zip(K.flat(engine.header),source['header_values'])),'Public header changed'
    assert len(actual)==len(source['canonical_memory'])==72
    assert all(torch.equal(x.cpu(),y) for x,y in zip(actual,source['canonical_memory'])),'Canonical static KV must be unchanged across Sender engines'
    row=next(r for r in rows() if r['task']=='arc');base=engine.base(row);body=engine.cache(row,base,G.load_c());cache=module.construct(row,body)
    L=body.get_seq_length();H=len(engine.prefix_ids);q=torch.tensor([L+32,L+33],device='cuda')
    for layer in range(1,37):
        allowed=module.routing.allowed('answer',layer,H,L,32,q,L+34)
        assert allowed[:,:L+32].all() and not allowed[0,L+33] and allowed[1,L+33]
        for x,y in zip(C._cache_layer(cache,layer-1),C._cache_layer(body,layer-1)):assert torch.equal(x[:,:,:L],y)
        assert torch.equal(C._cache_layer(cache,layer-1)[1][:,:,L:L+32].cpu(),source['canonical_memory'][2*layer-1])
    engine.unchanged()
    mark('interfaces.json',status='completed',canonical_memory_bitwise_equal=True,public_header_bitwise_equal=True,all36_answer_masks_identical=True,translated_body_unchanged=True,values_unchanged=True,optimizer_updates=0,source_checkpoint_sha256=CFG['static_checkpoint_sha256'])
    C.log('ZERO-SHOT STATIC INTERFACE CHECKS PASSED')

@torch.no_grad()
def check_endpoints(engine,correction):
    if done('parity/completed.json'):return
    dynamic=GR.GemmaMemory(engine)
    path=GEMMA/'runs/epoch4_step1536.pt';assert C.sha(path)==CFG['gemma_memory_checkpoint_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['manifest_sha256']==G.identity()
    dynamic.load_state_dict(cp['state'],strict=True);dynamic.eval().requires_grad_(False)
    checks=[]
    for task in CFG['test_counts']:
        if paused():raise SystemExit(75)
        row=next(r for r in rows() if r['task']==task);reference=baseline(row)
        base=engine.base(row);body=engine.cache(row,base,correction)
        native=engine.native_cache(row,baseline=True)
        values={'recolor_c':score(dynamic,row,body,rn=False),'native':score(dynamic,row,native,rn=False),DYNAMIC:score(dynamic,row,dynamic.construct(row,body),rn=True)}
        for name,result in values.items():same_prediction(result,reference[name],task,name)
        checks.append(dict(task=task,id=row['id'],C_native_gemma_dynamic_prediction_parity=True))
        engine.unchanged();C.log('STATIC TRANSFER ENDPOINT PARITY',task)
        del base,body,native
    dynamic.routing.close()
    mark('parity/completed.json',status='completed',n=8,checks=checks)

@torch.no_grad()
def main():
    verify();assert done('export/completed.json')
    if done('completed.json'):return
    engine=GE.Engine();correction=G.load_c();module=load_static(engine)
    check_interfaces(engine,module);check_endpoints(engine,correction)
    engine.mode='test';before=[C.versions(correction),C.versions(module)]
    names=['recolor_c',DYNAMIC,ARM,'native'];tasks={}
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows() if r['task']==task],1):
            if paused():status(status='paused',phase='evaluation',task=task,sample=i);raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);body=engine.cache(row,base,correction);cache=module.construct(row,body)
            result=score(module,row,cache,rn=True)
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions={**baseline(row),ARM:result},static_checkpoint_sha256=CFG['static_checkpoint_sha256'],baseline_source=str(baseline_path(row)))
            records.append(C.read(path));del base,body,cache
            engine.unchanged();assert before==[C.versions(correction),C.versions(module)] and engine.student_test_native_prefills==0
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
            status(status='running',phase='evaluation',task=task,sample=i,total=64,completed_questions=sum(len(list((ROOT/f'evaluation/{t}/per_sample').glob('*.json'))) for t in CFG['test_counts']),total_questions=512)
            if i==1 or i%8==0:C.log('STATIC LLAMA TO GEMMA EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={a:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in ['recolor_c',DYNAMIC,'native']}
        changes={a:{k:{label:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions'][a][k])*sign>0] for label,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys} for a in ['recolor_c',DYNAMIC]}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,transfer_minus=differences,rescued_harmed=changes,cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json')
    verify();assert C.sha(SOURCE/'runs/epoch4_step1536.pt')==CFG['static_checkpoint_sha256']
    assert before==[C.versions(correction),C.versions(module)];engine.unchanged()
    mark('summary.json',status='completed',tasks=tasks,config=CFG,optimizer_updates=0,trainable_parameters=0)
    lines=['# Llama-trained Static Memory transferred to Gemma: zero updates','','| Task | Metric | Gemma C | Gemma Dynamic Memory | Transferred Static Memory | Native |','|---|---|---:|---:|---:|---:|']
    for task,t in tasks.items():
        for k in t['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{t["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'),optimizer_updates=0,weights_unchanged=True);status(status='completed');C.log('STATIC LLAMA TO GEMMA COMPLETE')

if __name__=='__main__':main()
