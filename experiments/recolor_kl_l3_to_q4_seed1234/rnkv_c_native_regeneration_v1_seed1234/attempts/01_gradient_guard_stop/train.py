"""Full1024-update supervised+KL training of memory parameters only."""
from common import *
from model import load,losses,batches

def save_checkpoint(module,opt,scaler,epoch,step,batch,complete=False):
    import shutil
    assert shutil.disk_usage(ROOT).free>4*2**30
    value=dict(manifest_sha256=identity(),state=module.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
        epoch=epoch,step=step,batch=batch,epoch_completed=complete,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    C.atomic_save(value,ROOT/'runs/last.pt')
    if complete:C.atomic_save({k:v for k,v in value.items() if k not in ['optimizer','scaler']},ROOT/f'runs/epoch{epoch}_step{step}.pt')

@torch.no_grad()
def validate(engine,correction,module,epoch,step):
    values=[];module.eval()
    for i,row in enumerate(rows('validation')):
        if paused():sys.exit(75)
        path=ROOT/f'validation/epoch{epoch}/per_sample/{row["id"]}.json'
        if path.exists():
            item=C.read(path);assert item['manifest_sha256']==identity();values.append(item['metrics']);continue
        base=engine.base(row);translated=engine.cache(row,base,correction);native=engine.native_cache(row)
        kind='native' if i%2 else 'gold'
        _,detail=losses(engine,module,row,translated,native,kind,0)
        mark(str(path.relative_to(ROOT)),id=row['id'],task=row['task'],metrics=detail);values.append(detail)
        del base,translated,native;engine.unchanged()
        status(status='running',phase='validation',epoch=epoch,step=step,sample=i+1,total=128)
        if (i+1)%32==0:C.log('RNKV VALIDATE',epoch,i+1,128)
    fields=['loss','task_loss','full_kl']
    metrics={k:sum(x[k] for x in values)/len(values) for k in fields}
    mark(f'validation/epoch{epoch}/completed.json',status='completed',epoch=epoch,step=step,metrics=metrics,selection='diagnostic only; fixed final epoch4')
    module.train();return metrics

def main():
    verify();assert done('smoke/completed.json')
    if done('runs/completed.json'):return
    engine,correction,module=load();correction_versions=C.versions(correction)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=T.scaler_for();step=0;epoch0=1;batch0=0;history=[]
    if (ROOT/'runs/last.pt').exists():
        cp=torch.load(ROOT/'runs/last.pt',map_location='cpu',weights_only=True);assert cp['manifest_sha256']==identity()
        module.load_state_dict(cp['state']);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        step=cp['step'];epoch0=cp['epoch'];batch0=cp['batch'];torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);del cp
        # An epoch boundary always re-enters diagnostic validation (resumable per row).
    (ROOT/'runs').mkdir(exist_ok=True);opt.zero_grad(set_to_none=True);started=time.time()
    with (ROOT/'runs/steps.jsonl').open('a') as stream:
        for epoch in range(epoch0,5):
            module.train()
            for bi,batch in enumerate(batches(epoch),1):
                if epoch==epoch0 and bi<=batch0:continue
                if paused():
                    save_checkpoint(module,opt,scaler,epoch,step,bi-1);sys.exit(75)
                details=[]
                for row,kind in batch:
                    with torch.no_grad():
                        base=engine.base(row);translated=engine.cache(row,base,correction);native=engine.native_cache(row)
                    loss,detail=losses(engine,module,row,translated,native,kind,epoch)
                    scaler.scale(loss/8).backward();del loss,base,translated,native
                    details.append(detail)
                scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(module.parameters(),CFG['gradient_clip'])
                assert torch.isfinite(grad) and grad>0
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);step+=1
                engine.unchanged();assert C.versions(correction)==correction_versions
                metrics={k:sum(d[k] for d in details)/8 for k in ['loss','task_loss','full_kl','positions','prefix_tokens']}
                quota={k:sum(d['prefix_kind']==k for d in details) for k in ['gold','native','student']}
                assert quota==CFG['prefix_quotas_per_batch'][epoch-1]
                record=dict(epoch=epoch,step=step,total_steps=1024,batch=bi,metrics=metrics,prefix_counts=quota,gradient_norm=float(grad),
                    elapsed_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                stream.write(json.dumps(record)+'\n');stream.flush();status(status='running',phase='training',**record)
                if step==1 or step%4==0:C.log('RNKV TRAIN',json.dumps(record))
                if step%CFG['checkpoint_every']==0 or paused():save_checkpoint(module,opt,scaler,epoch,step,bi)
                if paused():sys.exit(75)
            assert step==epoch*256
            save_checkpoint(module,opt,scaler,epoch,step,256,complete=True)
            metrics=validate(engine,correction,module,epoch,step);history.append(dict(epoch=epoch,step=step,metrics=metrics))
            if paused():sys.exit(75)
    assert step==1024;verify();engine.unchanged();assert C.versions(correction)==correction_versions
    mark('runs/completed.json',status='completed',updates=1024,history=history,checkpoint_sha256=C.sha(ROOT/'runs/epoch4_step1024.pt'),
        trainable_parameters=sum(p.numel() for p in module.parameters()),finished_at=time.strftime('%F %T'))

if __name__=='__main__':main()
