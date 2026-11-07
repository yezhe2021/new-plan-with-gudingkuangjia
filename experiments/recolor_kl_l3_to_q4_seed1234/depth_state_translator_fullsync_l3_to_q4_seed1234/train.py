"""Resumable two-stage DST training with exact accepted-update accounting."""
import shutil
from experiment_common import *
from runtime import make,load_stage,batches,backward_batch,reconstruction,functional,checkpoint_path
from engine import Engine,BodyReader

def save(module,opt,scaler,stage,epoch,step,batch,complete=False):
    reserve=CFG['disk_reserve_gib']*2**30;size=sum(p.numel()*4 for p in module.parameters())
    if shutil.disk_usage(ROOT).free<reserve+3*size:raise RuntimeError('Insufficient space for atomic DST checkpoint; no caches deleted')
    v=dict(manifest_sha256=identity(),state=module.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),stage=stage,epoch=epoch,step=step,batch=batch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    path=ROOT/f'runs/{stage}/last.pt';path.parent.mkdir(parents=True,exist_ok=True);C.atomic_save(v,path)
    if complete:C.atomic_save({k:v for k,v in v.items() if k not in ['optimizer','scaler']},ROOT/f'runs/{stage}/epoch{epoch}_step{step}.pt')

@torch.no_grad()
def validate(engine,reader,module,stage,epoch,step):
    module.eval();metrics=[]
    for i,row in enumerate(rows('validation'),1):
        if paused():raise SystemExit(75)
        path=ROOT/f'validation/{stage}/epoch{epoch}/per_sample/{row["id"]}.json'
        if path.exists():
            v=C.read(path);assert v['manifest_sha256']==identity();metrics.append(v['metrics']);continue
        source=engine.source(row)
        if stage=='stage1':
            tk,tv,native=engine.teacher(row);detail=reconstruction(module,source,row,tk,tv);del tk,tv,native
        else:
            native=engine.native_cache(row);_,detail=functional(engine,reader,module,row,source,native,'native' if i%2 else 'gold',0);del native
        detail['task']=row['task'];mark(str(path.relative_to(ROOT)),id=row['id'],metrics=detail);metrics.append(detail);del source
        engine.unchanged();status(status='running',phase='validation',stage=stage,epoch=epoch,step=step,sample=i,total=len(rows('validation')))
    keys=['loss','kv_nmse'] if stage=='stage1' else ['loss','task_loss','full_kl']
    result={k:sum(v[k] for v in metrics)/len(metrics) for k in keys}
    mark(f'validation/{stage}/epoch{epoch}/completed.json',status='completed',metrics=result,selection='diagnostic only')
    module.train();return result

def main(stage):
    verify();assert done('checks/gpu_completed.json')
    if done(f'runs/{stage}/completed.json'):return
    if stage=='stage2':assert done('runs/stage1/completed.json')
    engine=Engine();reader=BodyReader(engine);module=make()
    if stage=='stage2':load_stage(module,'stage1')
    opt=torch.optim.AdamW(module.parameters(),lr=CFG[stage]['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000);epoch0=1;batch0=0;step=0
    last=ROOT/f'runs/{stage}/last.pt'
    if last.exists():
        cp=torch.load(last,map_location='cpu',weights_only=True);assert cp['manifest_sha256']==identity() and cp['stage']==stage
        module.load_state_dict(cp['state'],strict=True);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        epoch0=cp['epoch'];batch0=cp['batch'];step=cp['step'];torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);del cp
    path=ROOT/f'runs/{stage}/steps.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        rr=C.read_jsonl(path);good=[r for r in rr if r['step']<=step]
        if len(good)!=len(rr):
            C.write_jsonl(ROOT/f'attempts/{stage}_resume_{time.strftime("%Y%m%d_%H%M%S")}/uncommitted.jsonl',[r for r in rr if r['step']>step]);C.write_jsonl(path,good)
        assert [r['step'] for r in good]==list(range(1,step+1))
    if not last.exists():save(module,opt,scaler,stage,1,0,0)
    started=time.time();history=[]
    with path.open('a') as stream:
        for epoch in range(epoch0,CFG[stage]['epochs']+1):
            module.train()
            for bi,batch in enumerate(batches(epoch,stage),1):
                if epoch==epoch0 and bi<=batch0:continue
                if paused():save(module,opt,scaler,stage,epoch,step,bi-1);raise SystemExit(75)
                details,outcome=backward_batch(engine,reader,module,opt,scaler,batch,stage,epoch);step+=1;engine.unchanged()
                metrics={k:sum(x[k] for x in details)/len(details) for k in (['loss','kv_nmse'] if stage=='stage1' else ['loss','task_loss','full_kl'])}
                v=dict(stage=stage,epoch=epoch,step=step,total_steps=CFG[stage]['epochs']*CFG['updates_per_epoch'],batch=bi,metrics=metrics,elapsed_seconds=time.time()-started,**outcome,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                stream.write(json.dumps(v)+'\n');stream.flush();status(status='running',phase='training',**v)
                if step==1 or step%4==0:C.log('DST TRAIN',json.dumps(v))
                if step<=2 or step%CFG['checkpoint_every']==0 or paused():save(module,opt,scaler,stage,epoch,step,bi)
                if paused():raise SystemExit(75)
            assert step==epoch*CFG['updates_per_epoch'];save(module,opt,scaler,stage,epoch,step,CFG['updates_per_epoch'],complete=True)
            history.append(dict(epoch=epoch,step=step,metrics=validate(engine,reader,module,stage,epoch,step)))
    verify();engine.unchanged();mark(f'runs/{stage}/completed.json',status='completed',updates=step,history=history,checkpoint_sha256=C.sha(checkpoint_path(stage)),finished_at=time.strftime('%F %T'))

if __name__=='__main__':main(sys.argv[1])
