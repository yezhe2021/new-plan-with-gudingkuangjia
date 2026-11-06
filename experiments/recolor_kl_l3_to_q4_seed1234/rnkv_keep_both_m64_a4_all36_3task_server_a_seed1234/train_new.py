"""Full1536-update all-layer three-task supervised+KL training of memory parameters only."""
from experiment_common import *
from runtime_new import load,losses,batches
from runtime_new import backward_batch

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
        status(status='running',phase='validation',epoch=epoch,step=step,sample=i+1,total=sum(CFG['validation_counts'].values()))
        if (i+1)%32==0:C.log('RNKV VALIDATE',epoch,i+1,sum(CFG['validation_counts'].values()))
    fields=['loss','task_loss','full_kl']
    metrics={k:sum(x[k] for x in values)/len(values) for k in fields}
    metrics['by_task']={t:{k:sum(x[k] for x in values if x['task']==t)/sum(x['task']==t for x in values) for k in fields} for t in CFG['validation_counts']}
    mark(f'validation/epoch{epoch}/completed.json',status='completed',epoch=epoch,step=step,metrics=metrics,selection='diagnostic only; fixed final epoch4')
    module.train();return metrics

def main():
    verify();assert done('smoke/completed.json') and done('base_cache/completed.json')
    if done('runs/completed.json'):return
    engine,correction,module=load();correction_versions=C.versions(correction)
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=T.scaler_for();step=0;epoch0=1;batch0=0;history=[]
    if (ROOT/'runs/last.pt').exists():
        cp=torch.load(ROOT/'runs/last.pt',map_location='cpu',weights_only=True);assert cp['manifest_sha256']==identity()
        module.load_state_dict(cp['state']);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        step=cp['step'];epoch0=cp['epoch'];batch0=cp['batch'];torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);del cp
        # Roll back only uncheckpointed optimizer updates after an interrupted process.
        log=ROOT/'runs/steps.jsonl'
        if log.exists():
            records=C.read_jsonl(log);rolled=[r for r in records if r['step']>step]
            if rolled:
                C.write_jsonl(ROOT/f'attempts/rollback_{time.time_ns()}.jsonl',rolled)
                C.write_jsonl(log,[r for r in records if r['step']<=step])
    (ROOT/'runs').mkdir(exist_ok=True);opt.zero_grad(set_to_none=True);started=time.time()
    if not (ROOT/'runs/last.pt').exists():save_checkpoint(module,opt,scaler,1,0,0)
    with (ROOT/'runs/steps.jsonl').open('a') as stream:
        for epoch in range(epoch0,5):
            module.train()
            for bi,batch in enumerate(batches(epoch),1):
                if epoch==epoch0 and bi<=batch0:continue
                if paused():
                    save_checkpoint(module,opt,scaler,epoch,step,bi-1);sys.exit(75)
                details,outcome,retries=backward_batch(engine,correction,module,opt,scaler,batch,epoch,bi,scope='train')
                grad=outcome['gradient_norm'];step+=1
                engine.unchanged();assert C.versions(correction)==correction_versions
                metrics={k:sum(d[k] for d in details)/8 for k in ['loss','task_loss','full_kl','positions','prefix_tokens']}
                quota={k:sum(d['prefix_kind']==k for d in details) for k in ['gold','native','student']}
                assert quota==CFG['prefix_quotas_per_batch'][epoch-1]
                task_metrics={t:{k:sum(d[k] for d in details if d['task']==t)/sum(d['task']==t for d in details) for k in ['loss','task_loss','full_kl']} for t in CFG['train_counts']}
                record=dict(task_metrics=task_metrics,task_counts={t:sum(d['task']==t for d in details) for t in CFG['train_counts']},epoch=epoch,step=step,total_steps=CFG['updates'],batch=bi,metrics=metrics,prefix_counts=quota,gradient_norm=float(grad),amp_scale=outcome['new_scale'],amp_retries=retries,
                    elapsed_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                stream.write(json.dumps(record)+'\n');stream.flush();status(status='running',phase='training',**record)
                if step==1 or step%4==0:C.log('RNKV TRAIN',json.dumps(record))
                if step<=4 or step%CFG['checkpoint_every']==0 or paused():save_checkpoint(module,opt,scaler,epoch,step,bi)
                if paused():sys.exit(75)
            assert step==epoch*CFG['updates_per_epoch']
            save_checkpoint(module,opt,scaler,epoch,step,CFG['updates_per_epoch'],complete=True)
            metrics=validate(engine,correction,module,epoch,step);history.append(dict(epoch=epoch,step=step,metrics=metrics))
            if paused():sys.exit(75)
    assert step==CFG['updates'];verify();engine.unchanged();assert C.versions(correction)==correction_versions
    mark('runs/completed.json',status='completed',updates=CFG['updates'],history=history,checkpoint_sha256=C.sha(ROOT/'runs/epoch4_step1536.pt'),
        trainable_parameters=sum(p.numel() for p in module.parameters()),finished_at=time.strftime('%F %T'))

if __name__=='__main__':main()
