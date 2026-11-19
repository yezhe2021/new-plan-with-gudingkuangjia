"""Train only the new concatenation projection with the established full-vocabulary KL objective."""
from runtime import *

def update(module,optimizer,scaler):
    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(module.parameters(),CFG['gradient_clip'])
    assert torch.isfinite(norm) and norm>0
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);return float(norm)

def checkpoint(name,module,optimizer,scaler,epoch,step,position,complete,history):
    assert shutil.disk_usage(ROOT).free/2**30>CFG['disk_reserve_gib']
    record=dict(state={n:v.detach().cpu() for n,v in module.state_dict().items()},manifest_sha256=identity(),frozen_frontends=E.identity(),state_scope='concat_projection_only',epoch=epoch,step=step,position=position,epoch_completed=complete,history=history)
    if optimizer is not None:record.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict())
    C.save_tensor(ROOT/name,record)

def frozen_digest(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def validate(module,model,tok):
    bytask=defaultdict(list)
    for item in rows('validation'):
        task=item['gemma_row']['task']
        if len(bytask[task])>=CFG['monitor_validation_per_task']:continue
        _,detail=example_loss(module,model,tok,item,0,validation=True);bytask[task].append(detail['kl'])
    tasks={t:dict(n=len(v),kl=sum(v)/len(v)) for t,v in bytask.items()}
    return dict(tasks=tasks,macro_kl=sum(v['kl'] for v in tasks.values())/len(tasks))

def smoke():
    if done('smoke/completed.json'):return
    verify();C.seed();module=fresh();tok,model=C.model('qwen')
    assert all('LoRA' not in type(m).__name__ for m in model.modules())
    teacher_before={};checks=[];selected=[];seen=set()
    for x in rows('train'):
        task=x['gemma_row']['task']
        if task not in seen:selected.append(x);seen.add(task)
    with torch.no_grad():
        for item in selected:
            row=item['gemma_row'];features=load_features(item)
            native=U.native_body(model,row)
            expected_k,expected_v=E.fuse(*features)
            actual_k,actual_v=project(module,features)
            err=max(float((actual_k-expected_k).abs().max()),float((actual_v-expected_v).abs().max()))
            assert err<0.002,('Equal-mean initialization KV mismatch',err)
            with torch.amp.autocast('cuda',dtype=torch.float16):
                a=U.suffix_logits(model,row,U.translated_cache(model,row,native,expected_k,expected_v),[],[0])
                b=U.suffix_logits(model,row,U.translated_cache(model,row,native,actual_k,actual_v),[],[0])
                before=U.suffix_logits(model,row,U.clone_cache(native,model),[],[0])
            kl=float(U.full_kl(a,b));assert kl<.001 and int(a.argmax())==int(b.argmax())
            teacher_before[row['id']]=before.cpu()
            checks.append(dict(task=row['task'],equal_mean_kv_max_error=err,equal_mean_logits_kl=kl,initial_top1_identical=True))
    optimizer=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=0)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in module.parameters()}
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)
    gradients=[]
    for online in [False,True]:
        optimizer.zero_grad(set_to_none=True)
        for item in selected:
            loss,_=example_loss(module,model,tok,item,2 if online else 1,student=online);scaler.scale(loss/len(selected)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()),'A K/V projection branch has no finite gradient'
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        norm=update(module,optimizer,scaler);gradients.append(dict(online=online,norm=norm,all72_branches_nonzero=True))
    with torch.no_grad():
        for item in selected:
            row=item['gemma_row'];native=U.native_body(model,row)
            with torch.amp.autocast('cuda',dtype=torch.float16):after=U.suffix_logits(model,row,U.clone_cache(native,model),[],[0]).cpu()
            assert torch.equal(after,teacher_before[row['id']]),'Native teacher modified'
    mark('smoke/completed.json',status='completed',checks=checks,gradients=gradients,trainable_parameters=CFG['expected_parameters'],teacher_bitwise_unchanged=True,frontends_offloaded_and_frozen=True,reader_absent=True)
    C.log('CONCAT PROJECTION SMOKE PASSED')

def train():
    verify()
    if done('runs/projection/completed.json'):return
    assert done('smoke/completed.json') and done('features_trainval_completed.json')
    C.seed();module=fresh();tok,model=C.model('qwen')
    teacher_hash=frozen_digest(model)
    optimizer=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)
    epoch0,position0,step,history=1,0,0,[]
    folder=ROOT/'runs/projection';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'last.pt').exists():
        r=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        assert r['manifest_sha256']==identity() and r['frozen_frontends']==E.identity() and r['state_scope']=='concat_projection_only'
        module.load_state_dict(r['state'],strict=True);optimizer.load_state_dict(r['optimizer']);scaler.load_state_dict(r['scaler'])
        step,history=r['step'],r['history'];epoch0=r['epoch']+int(r['epoch_completed']);position0=0 if r['epoch_completed'] else r['position'];del r
    with (folder/'steps.jsonl').open('a') as stream:
        for epoch in range(epoch0,CFG['epochs']+1):
            order=epoch_order(epoch);optimizer.zero_grad(set_to_none=True);details=[]
            for position,(item,online) in enumerate(order,1):
                if epoch==epoch0 and position<=position0:continue
                loss,d=example_loss(module,model,tok,item,epoch,student=online);scaler.scale(loss/8).backward();details.append(d);del loss
                if position%8:continue
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
                norm=update(module,optimizer,scaler);step+=1
                record=dict(epoch=epoch,step=step,position=position,mean_kl=sum(d['kl'] for d in details)/8,supervised_positions=sum(d['positions'] for d in details),student_examples=sum(d['prefix_source']=='student' for d in details),grad_norm=norm,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                stream.write(json.dumps(record)+'\n');stream.flush();status(status='running',phase='projection_training',**record);details=[]
                if step%4==0:C.log(f'Concat projection epoch={epoch}/4 step={step}/1024 KL={record["mean_kl"]:.6f}')
                if step%CFG['checkpoint_every_steps']==0 and position<len(order):checkpoint('runs/projection/last.pt',module,optimizer,scaler,epoch,step,position,False,history)
            assert frozen_digest(model)==teacher_hash
            metrics=validate(module,model,tok);entry=dict(epoch=epoch,step=step,validation=metrics);history=[x for x in history if x['epoch']!=epoch]+[entry]
            checkpoint(f'runs/projection/epoch{epoch}_step{step}.pt',module,None,None,epoch,step,len(order),True,history)
            checkpoint('runs/projection/last.pt',module,optimizer,scaler,epoch,step,len(order),True,history)
            mark('runs/projection/validation.json',status='completed',history=history,selection=CFG['selection']);C.log(f'Concat projection epoch {epoch} completed {metrics}')
    assert step==1024 and frozen_digest(model)==teacher_hash
    mark('runs/projection/completed.json',status='completed',epochs=4,steps=1024,history=history,checkpoint=str(folder/'epoch4_step1024.pt'),checkpoint_sha256=C.sha(folder/'epoch4_step1024.pt'),teacher_bitwise_unchanged=True)

if __name__=='__main__':
    if sys.argv[1]=='smoke':smoke()
    else:train()
