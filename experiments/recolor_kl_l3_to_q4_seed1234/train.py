"""Matched Stage-2 budgets; mathematical/interface checks before full training."""
from core import *
from core import _cache_layer


def optimizer_for(module):return torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
def scaler_for():return torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)


def save_checkpoint(arm,module,optimizer,scaler,epoch,step,position,complete,history,*,best=False):
    import shutil
    assert shutil.disk_usage(ROOT).free/2**30>CFG['disk_reserve_gib']
    record=dict(manifest_sha256=identity(),arm=arm,recolor_sha256=CFG['recolor_sha256'],state=module.state_dict(),
        epoch=epoch,step=step,position=position,epoch_completed=complete,history=history,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    folder=ROOT/'runs'/arm
    if complete:atomic_save(record,folder/f'epoch{epoch}_step{step}.pt')
    if best:atomic_save(record,folder/'best_validation.pt')
    record.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict());atomic_save(record,folder/'last.pt')


def load_final(arm):
    if arm=='A':return None
    path=ROOT/'runs'/arm/'epoch4_step800.pt';cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==identity() and cp['arm']==arm and cp['step']==800 and cp['recolor_sha256']==CFG['recolor_sha256']
    module=fresh(arm);module.load_state_dict(cp['state'],strict=True);return module.eval()


def smoke():
    verify();engine=Engine();checks=[]
    # All R64 factors start identically; only E introduces GELU.
    cd=fresh('C');dd=fresh('D');ee=fresh('E')
    assert all(torch.equal(a,b) for a,b in zip(cd.state_dict().values(),dd.state_dict().values()))
    assert all(torch.equal(a,b) for a,b in zip(cd.state_dict().values(),ee.state_dict().values()))
    del cd,dd,ee
    selected=[min([r for r in rows('train') if r['dataset']==dataset],key=lambda r:len(r['content_ids'])) for dataset in ['Open-R1 prompt-only','HotpotQA distractor train full-context']]
    for row in selected:
        base=engine.base(row);native=engine.native_cache(row)
        with torch.no_grad():
            zero=engine.cache(row,base,None)
            # Literal official RAW transfer parity with the same FP32 Recolor constants.
            offsets,k,v=capture_sender(engine.sender_tok,engine.sender,engine.mapping,row['prompt'],device='cpu')
            official,length=build_receiver_cache(engine.tok,engine.model,engine.mapping,offsets,k,v,row['prompt'],engine.prefix_ids)
            assert length==len(engine.prefix_ids)+len(row['content_ids'])
            assert all(torch.equal(a,b) for l in range(36) for a,b in zip(_cache_layer(zero,l),_cache_layer(official,l))), 'Recolor-only differs from official transfer'
            del official,k,v
            for arm in ['B','C','D','E']:
                module=fresh(arm);engine.norm_counts=[0]*36
                student=engine.cache(row,base,module);assert engine.norm_counts==[1]*36,engine.norm_counts
                engine.norm_counts=None
                assert all(torch.equal(a,b) for l in range(36) for a,b in zip(_cache_layer(zero,l),_cache_layer(student,l)))
                del module,student
            before=engine.logits(row,clone_cache(native,engine.model)).detach()
            full=torch.tensor([engine.prefix_ids+row['content_ids']+row['suffix_ids']],device='cuda')
            out=decoder(engine.model)(full,use_cache=False)
            direct=engine.model.lm_head(out.last_hidden_state[0,-1:]).float()
            parity=float(vocab_kl(direct,before));assert abs(parity)<.001 and int(direct.argmax())==int(before.argmax())
            tokens=engine.rollout(row,clone_cache(native,engine.model),cap=16)
            reference,count=RT.generate(engine.runtime,torch.tensor([row['suffix_ids']],device='cuda'),16,past_key_values=clone_cache(native,engine.model),past_length=len(engine.prefix_ids)+len(row['content_ids']),seed=CFG['seed'])
            assert engine.tok.decode(tokens,skip_special_tokens=False)==reference and len(tokens)==count
            del direct,out,full
        example=engine.attention_example(row,base)
        module=fresh('C')
        with torch.no_grad():
            own=attention_objective(engine,module,example)
            original=HC.example_objective(engine.model,module,example,engine.prefix_keys,engine.prefix_values,lambda_k_output=CFG['lambda_k_output'],backward=False)
            errors={name:abs(own[name]-original[name]) for name in ['routing_kl','k_post_wo_mse','post_wo_nmse']}
            assert all(errors[name]<.002+.02*abs(original[name]) for name in errors),('Attention formula parity',own,original)
        optimizer=optimizer_for(module);scaler=scaler_for()
        attention_objective(engine,module,example,scaler=scaler)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for name,p in module.named_parameters() if 'right_' in name)
        attention_grad=update(module,optimizer,scaler);engine.unchanged()
        del module,optimizer,scaler,example
        gradients={}
        for arm in ['D','E']:
            module=fresh(arm);optimizer=optimizer_for(module);scaler=scaler_for();values=[]
            for online in [False,True]:
                loss,detail=kl_loss(engine,module,row,2 if online else 1,online=online,base=base,native=native)
                scaler.scale(loss).backward();del loss
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for name,p in module.named_parameters() if 'right_' in name)
                values.append(dict(grad=update(module,optimizer,scaler),**detail));engine.unchanged()
            gradients[arm]=values;del module,optimizer,scaler
        with torch.no_grad():after=engine.logits(row,clone_cache(native,engine.model))
        assert torch.equal(before,after),'Frozen native teacher changed'
        checks.append(dict(id=row['id'],dataset=row['dataset'],official_raw_cache_exact=True,all_initial_arms_exact=True,
            k_norm_once_per_layer=True,native_cached_full_kl=parity,attention_formula_abs_error=errors,
            attention_gradient=attention_grad,fullkl_gradients=gradients,teacher_unchanged=True))
        del native,base,zero,before,after
        log('SMOKE',row['dataset'],'PASSED')
    engine.unchanged()
    mark('smoke/completed.json',status='completed',checks=checks,rank64_initial_factors_identical=True,
         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,disposable_smoke_weights=True)


@torch.no_grad()
def validate(engine,module,arm):
    values=[];attention=CFG['arms'][arm]['objective']=='attention'
    for i,row in enumerate(rows('validation'),1):
        base=engine.base(row)
        if attention:
            ex=engine.attention_example(row,base);detail=attention_objective(engine,module,ex);del ex
        else:
            _,detail=kl_loss(engine,module,row,0,validation=True,base=base)
        values.append(detail);del base
        if i%50==0:status(status='running',phase='validation',arm=arm,sample=i,total=400)
    keys=values[0].keys();metrics={k:sum(v[k] for v in values)/len(values) for k in keys}
    metrics['selection_loss']=metrics['attention_loss' if attention else 'full_kl'];return metrics


def train(arm):
    verify();assert arm in ['B','C','D','E']
    assert read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    if (ROOT/'runs'/arm/'completed.json').exists():return
    engine=Engine();module=fresh(arm);optimizer=optimizer_for(module);scaler=scaler_for()
    folder=ROOT/'runs'/arm;folder.mkdir(parents=True,exist_ok=True)
    step,epoch0,pos0,history,best=0,1,0,[],None
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==identity() and cp['arm']==arm and cp['recolor_sha256']==CFG['recolor_sha256']
        module.load_state_dict(cp['state']);optimizer.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        step,history=cp['step'],cp['history'];epoch0=cp['epoch']+int(cp['epoch_completed']);pos0=0 if cp['epoch_completed'] else cp['position']
        torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);del cp
        best=min((v['validation']['selection_loss'] for v in history),default=None)
    attention=CFG['arms'][arm]['objective']=='attention'
    with (folder/'steps.jsonl').open('a') as stream:
        for epoch in range(epoch0,CFG['epochs']+1):
            module.train();order=rows('train');random.Random(CFG['seed']+epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True);details=[]
            for pos,row in enumerate(order,1):
                if epoch==epoch0 and pos<=pos0:continue
                base=engine.base(row)
                if attention:
                    example=engine.attention_example(row,base)
                    detail=attention_objective(engine,module,example,scaler=scaler,divisor=CFG['effective_batch']);del example
                else:
                    online=epoch>1 and ((pos-1)%CFG['effective_batch']+epoch)%2==0
                    loss,detail=kl_loss(engine,module,row,epoch,online=online,base=base)
                    scaler.scale(loss/CFG['effective_batch']).backward();del loss
                details.append(detail);del base
                if pos%CFG['effective_batch']:continue
                grad=update(module,optimizer,scaler);step+=1;engine.unchanged()
                metrics={k:sum(v[k] for v in details)/len(details) for k in details[0]};details=[]
                record=dict(arm=arm,epoch=epoch,step=step,total_steps=800,position=pos,metrics=metrics,gradient_norm=grad,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                stream.write(json.dumps(record)+'\n');stream.flush();status(status='running',phase='training',**record)
                if step==1 or step%4==0:log('TRAIN',json.dumps(record))
                if step%CFG['checkpoint_every']==0 and pos<len(order):save_checkpoint(arm,module,optimizer,scaler,epoch,step,pos,False,history)
            module.eval();metrics=validate(engine,module,arm);history.append(dict(epoch=epoch,step=step,validation=metrics))
            improved=best is None or metrics['selection_loss']<best
            if improved:best=metrics['selection_loss']
            save_checkpoint(arm,module,optimizer,scaler,epoch,step,len(order),True,history,best=improved)
            mark(f'runs/{arm}/validation.json',history=history,primary_selection='fixed800',best_own_validation_loss=best)
            engine.unchanged();log('VALIDATION',arm,json.dumps(history[-1]))
    assert step==800;engine.unchanged();verify()
    mark(f'runs/{arm}/completed.json',status='completed',arm=arm,updates=800,history=history,
         checkpoint_sha256=sha(folder/'epoch4_step800.pt'),trainable_parameters=sum(p.numel() for p in module.parameters()))


@torch.no_grad()
def diagnostics(label,arms):
    verify();engine=Engine();modules={a:load_final(a) for a in arms};records=[]
    for i,row in enumerate(rows('validation'),1):
        path=ROOT/f'diagnostics/{label}/per_sample/{row["id"]}.json'
        if path.exists():
            result=read(path);assert result['manifest_sha256']==identity();records.append(result);continue
        base=engine.base(row);native=engine.native_cache(row);example=engine.attention_example(row,base);values={}
        for a,m in modules.items():
            metrics=attention_objective(engine,m,example)
            _,full=kl_loss(engine,m,row,0,validation=True,base=base,native=native);metrics.update(full);values[a]=metrics
        result=dict(id=row['id'],dataset=row['dataset'],metrics=values)
        mark(str(path.relative_to(ROOT)),**result);records.append(result)
        del base,native,example;engine.unchanged()
        status(status='running',phase='cross_validation',label=label,sample=i,total=400)
        if i%20==0:log('CROSS VALIDATION',label,f'{i}/400')
    metrics={a:{k:sum(r['metrics'][a][k] for r in records)/400 for k in records[0]['metrics'][a]} for a in arms}
    write_jsonl(ROOT/f'diagnostics/{label}/per_sample.jsonl',records)
    mark(f'diagnostics/{label}/completed.json',status='completed',n=400,arms=arms,metrics=metrics,
         selection='correctness/finite-metric gate only; no test or performance-based continuation selection')
    log('DIAGNOSTICS COMPLETED',label,json.dumps(metrics))


if __name__=='__main__':
    if sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='train':train(sys.argv[2])
    elif sys.argv[1]=='diagnostics':diagnostics(sys.argv[2],sys.argv[3:])
    else:raise ValueError(sys.argv[1])
