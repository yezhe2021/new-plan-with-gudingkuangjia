"""Zero-training attachment of fixed native Memory KV to the final DST cache."""
from experiment_common import *
from score import score

def same_prediction(value,reference,task,condition):
    for k in ['prediction','correct','em','f1','legacy_prediction','legacy_correct','generation','token_ids','generated_tokens','reached_cap']:
        if k in reference:assert value[k]==reference[k],(task,condition,k)
    if 'scores' in reference:
        assert len(value['scores'])==len(reference['scores'])
        assert max(abs(a-b) for a,b in zip(value['scores'],reference['scores']))<1e-4,(task,condition,'scores')

@torch.no_grad()
def check_interfaces(engine,translator,module):
    file=EXPORT/'export/canonical.pt';assert C.sha(file)==CFG['canonical_export_sha256']
    reference=torch.load(file,map_location='cpu',weights_only=True)
    assert reference['manifest_sha256']==CFG['export_manifest_sha256'] and reference['source_checkpoint_sha256']==CFG['static_checkpoint_sha256']
    assert reference['prefix_ids']==engine.prefix_ids and reference['receiver_config_sha256']==C.digest(engine.model.config.to_dict())
    canonical=module.canonical()
    assert len(canonical)==len(reference['canonical_memory'])==72
    assert all(torch.equal(a.cpu(),b) for a,b in zip(canonical,reference['canonical_memory'])),'Static Memory canonical KV changed'
    assert all(torch.equal(a.cpu(),b) for a,b in zip(K.flat(engine.header),reference['header_values'])),'Native public header changed'
    row=next(r for r in rows() if r['task']=='arc');body=engine.translated(row,translator,checkpointed=False)
    engine.norm_counts=[0]*36;memory=module.construct(row,body)
    assert engine.norm_counts==[0]*36,'Appending post-normalized static KV must not call k_norm again'
    engine.norm_counts=None;L=body.get_seq_length();H=len(engine.prefix_ids)
    assert L==H+len(row['content_ids']) and memory.get_seq_length()==L+32
    q=torch.tensor([L+32,L+33],device='cuda')
    for layer in range(1,37):
        allowed=module.routing.allowed('answer',layer,H,L,32,q,L+34)
        assert allowed[:,:L+32].all() and not allowed[0,L+33] and allowed[1,L+33]
        for a,b in zip(C._cache_layer(memory,layer-1),C._cache_layer(body,layer-1)):assert torch.equal(a[:,:,:L],b)
        assert torch.equal(C._cache_layer(memory,layer-1)[1][:,:,L:L+32].cpu(),reference['canonical_memory'][2*layer-1])
    engine.unchanged()
    mark('interfaces.json',status='completed',canonical_memory_bitwise_equal=True,public_header_bitwise_equal=True,body_bitwise_unchanged=True,all36_keep_both=True,memory_append_k_norm_calls=0,answer_start_position='L+32',values_unchanged=True,optimizer_updates=0)
    C.log('DST PLUS STATIC INTERFACE CHECKS PASSED')

@torch.no_grad()
def check_endpoints(engine,translator,reader):
    if done('parity/completed.json'):return
    checks=[]
    for task in CFG['test_counts']:
        if paused():raise SystemExit(75)
        row=next(r for r in rows() if r['task']==task);old=baseline(row)
        body=engine.translated(row,translator,checkpointed=False)
        result=score(reader,row,body,rn=False);same_prediction(result,old[BASE],task,BASE)
        native=engine.native_cache(row,baseline=True)
        same_prediction(score(reader,row,native,rn=False),old['native'],task,'native')
        checks.append(dict(task=task,id=row['id'],DST_stage2_prediction_parity=True,native_prediction_parity=True));del body,native
        engine.unchanged();C.log('DST PLUS STATIC ENDPOINT PARITY',task)
    mark('parity/completed.json',status='completed',n=8,checks=checks)

@torch.no_grad()
def main():
    verify()
    if done('completed.json'):return
    engine=DE.Engine();translator,module=load_frozen(engine);reader=DE.BodyReader(engine);engine.mode='test'
    check_interfaces(engine,translator,module);check_endpoints(engine,translator,reader)
    before=[C.versions(translator),C.versions(module)];names=['C','C_static',BASE,ARM,'native'];tasks={}
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows() if r['task']==task],1):
            if paused():status(status='paused',phase='evaluation',task=task,sample=i);raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                value=C.read(path);assert value['manifest_sha256']==identity();records.append(value);continue
            body=engine.translated(row,translator,checkpointed=False);memory=module.construct(row,body)
            result=score(module,row,memory,rn=True)
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions={**baseline(row),ARM:result},dst_checkpoint_sha256=CFG['dst_checkpoint_sha256'],static_checkpoint_sha256=CFG['static_checkpoint_sha256'])
            records.append(C.read(path));del body,memory
            engine.unchanged();assert before==[C.versions(translator),C.versions(module)] and engine.student_test_native_prefills==0
            assert all(not p.requires_grad and p.grad is None for m in [translator,module] for p in m.parameters())
            status(status='running',phase='evaluation',task=task,sample=i,total=64,completed_questions=sum(len(list((ROOT/f'evaluation/{t}/per_sample').glob('*.json'))) for t in CFG['test_counts']),total_questions=512)
            if i==1 or i%8==0:C.log('DST PLUS STATIC EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={a:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in [BASE,'C_static','C','native']}
        changes={a:{k:{label:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions'][a][k])*sign>0] for label,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys} for a in [BASE,'C_static']}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,combined_minus=differences,rescued_harmed=changes,cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json')
    verify();assert before==[C.versions(translator),C.versions(module)];engine.unchanged()
    mark('summary.json',status='completed',tasks=tasks,config=CFG,optimizer_updates=0,weights_unchanged=True)
    lines=['# Frozen DST Stage2 + existing Static Memory, zero training','','| Task | Metric | Recolor+C | Recolor+C+Static | DST Stage2 | DST+Static | Native |','|---|---|---:|---:|---:|---:|---:|']
    for task,t in tasks.items():
        for k in t['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{t["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'),optimizer_updates=0,weights_unchanged=True);status(status='completed');C.log('DST PLUS STATIC COMPLETE')

if __name__=='__main__':main()
