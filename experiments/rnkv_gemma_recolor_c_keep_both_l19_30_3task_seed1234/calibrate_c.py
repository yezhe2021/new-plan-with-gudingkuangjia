"""Fresh linearR64 C: same Attention KL+key-output MSE+detached-routing value NMSE, 800 updates."""
import random
from experiment_common import *
from engine_new import Engine,attention_objective

def save_checkpoint(module,opt,scaler,epoch,step,batch,complete=False):
    value=dict(manifest_sha256=identity(),recolor_sha256=map_sha(),state=module.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
        epoch=epoch,step=step,batch=batch,epoch_completed=complete,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    C.atomic_save(value,ROOT/'runs/C/last.pt')
    if complete:C.atomic_save({k:v for k,v in value.items() if k not in ['optimizer','scaler']},ROOT/f'runs/C/epoch{epoch}_step{step}.pt')

def update(engine,module,opt,scaler,batch,epoch,bi,scope='calibration'):
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True);details=[]
        for row in batch:
            base=engine.base(row);example=engine.attention_example(row,base)
            details.append(attention_objective(engine,module,example,scaler=scaler,divisor=len(batch)))
            del example,base
        outcome=O.finish_backward(module,opt,scaler)
        if outcome['accepted']:return {k:sum(v[k] for v in details)/len(details) for k in details[0]},outcome,retry
        p=ROOT/f'diagnostics/{scope}_amp_retries.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as f:f.write(json.dumps(dict(epoch=epoch,batch=bi,retry=retry,**outcome))+'\n')
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    raise RuntimeError('Persistent C calibration overflow; same batch retained, no skipped rows')

@torch.no_grad()
def validate(engine,module,epoch,step):
    values=[]
    for i,row in enumerate(cal_rows('validation'),1):
        if paused():raise SystemExit(75)
        path=ROOT/f'validation_C/epoch{epoch}/per_sample/{row["id"]}.json'
        if path.exists():
            v=C.read(path);assert v['manifest_sha256']==identity();values.append(v['metrics']);continue
        base=engine.base(row);ex=engine.attention_example(row,base);detail=attention_objective(engine,module,ex)
        mark(str(path.relative_to(ROOT)),id=row['id'],dataset=row['dataset'],metrics=detail);values.append(detail)
        del base,ex;engine.unchanged();status(status='running',phase='C_validation',epoch=epoch,step=step,sample=i,total=400)
    metrics={k:sum(v[k] for v in values)/400 for k in values[0]};mark(f'validation_C/epoch{epoch}/completed.json',status='completed',metrics=metrics,fixed_final_selection=True)
    return metrics

def main():
    verify();assert done('smoke_C/completed.json')
    if done('runs/C/completed.json'):return
    engine=Engine();module=fresh_c();cfg=CFG['calibration_c'];opt=torch.optim.AdamW(module.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);scaler=T.scaler_for()
    step=0;epoch0=1;batch0=0;history=[]
    if (ROOT/'runs/C/last.pt').exists():
        cp=torch.load(ROOT/'runs/C/last.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==identity() and cp['recolor_sha256']==map_sha()
        module.load_state_dict(cp['state']);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler']);step=cp['step'];epoch0=cp['epoch'];batch0=cp['batch']
        torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);del cp
    else:save_checkpoint(module,opt,scaler,1,0,0)
    started=time.time();(ROOT/'runs/C').mkdir(parents=True,exist_ok=True)
    with (ROOT/'runs/C/steps.jsonl').open('a') as stream:
        for epoch in range(epoch0,5):
            order=cal_rows('train');random.Random(CFG['seed']+epoch).shuffle(order);assert len(order)==1600
            for bi in range(1,201):
                if epoch==epoch0 and bi<=batch0:continue
                if paused():save_checkpoint(module,opt,scaler,epoch,step,bi-1);raise SystemExit(75)
                metrics,outcome,retries=update(engine,module,opt,scaler,order[(bi-1)*8:bi*8],epoch,bi);step+=1;engine.unchanged()
                rec=dict(epoch=epoch,step=step,total_steps=800,batch=bi,metrics=metrics,amp_retries=retries,**outcome,elapsed_seconds=time.time()-started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cache_hits=engine.cache_hits,cache_misses=engine.cache_misses)
                stream.write(json.dumps(rec)+'\n');stream.flush();status(status='running',phase='C_training',**rec)
                if step==1 or step%4==0:C.log('GEMMA C TRAIN',json.dumps(rec))
                if step<=4 or step%32==0 or paused():save_checkpoint(module,opt,scaler,epoch,step,bi)
                if paused():raise SystemExit(75)
            assert step==epoch*200;save_checkpoint(module,opt,scaler,epoch,step,200,complete=True)
            history.append(dict(epoch=epoch,step=step,metrics=validate(engine,module,epoch,step)))
    assert step==800;verify();engine.unchanged();mark('runs/C/completed.json',status='completed',updates=800,history=history,recolor_sha256=map_sha(),checkpoint_sha256=C.sha(ROOT/'runs/C/epoch4_step800.pt'),finished_at=time.strftime('%F %T'))
if __name__=='__main__':main()
