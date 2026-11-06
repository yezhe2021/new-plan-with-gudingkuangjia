"""Disposable one-update parity and longest-sample gradients before exact resume."""
from pathlib import Path
import sys,time,math,copy
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import gpu_resident_entry as X
E=X.E;RT=X.RT;C=X.C;torch=X.torch;ROOT=E.ROOT;CFG=E.CFG

def main():
    E.verify();assert E.paused(),'Switch requires an accepted-update checkpoint and stopped training'
    cp=torch.load(ROOT/'runs/last.pt',map_location='cpu',weights_only=True);assert cp['manifest_sha256']==E.identity()
    source_sha=C.sha(ROOT/'runs/last.pt');engine,correction,module=RT.load()
    opt=torch.optim.AdamW(module.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay']);scaler=E.T.scaler_for()
    def restore():
        module.load_state_dict(cp['state'],strict=True);opt.load_state_dict(copy.deepcopy(cp['optimizer']));scaler.load_state_dict(copy.deepcopy(cp['scaler']))
        torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state(cp['cuda_rng']);opt.zero_grad(set_to_none=True)
    epoch=cp['epoch'];bi=cp['batch']+1
    if bi>CFG['updates_per_epoch']:epoch+=1;bi=1
    assert epoch<=CFG['epochs'],'Training already complete'
    batch=RT.batches(epoch)[bi-1];reference=None;comparisons=[]
    for resident in [False,True]:
        X.install(resident);restore();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.time();details,outcome,retries=RT.backward_batch(engine,correction,module,opt,scaler,batch,epoch,bi,scope='resident_probe_'+str(resident))
        current={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
        metrics={k:sum(x[k] for x in details)/len(details) for k in ['loss','task_loss','full_kl']}
        record=dict(mode='gpu_resident' if resident else 'cpu_offload',seconds=time.time()-started,metrics=metrics,amp_retries=retries,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**outcome)
        if not resident:reference=current;refmetrics=metrics;refoutcome=outcome
        else:
            abs_error=max(float((current[k]-reference[k]).abs().max()) for k in current)
            numerator=sum(float((current[k]-reference[k]).square().sum()) for k in current);denominator=sum(float(reference[k].square().sum()) for k in current)
            relative=(numerator/max(denominator,1e-30))**.5
            assert abs_error<1e-6 and relative<1e-4,('Execution mode changed optimizer update',abs_error,relative)
            assert max(abs(metrics[k]-refmetrics[k]) for k in metrics)<1e-5,(metrics,refmetrics)
            assert outcome['new_scale']==refoutcome['new_scale'] and retries==comparisons[0]['amp_retries']
            record.update(parameter_max_abs_error=abs_error,parameter_relative_l2=relative)
        comparisons.append(record);engine.unchanged();C.log('GPU RESIDENT UPDATE PARITY',record)
    X.install(True);long=[];chosen={}
    for task in CFG['train_counts']:
        rr=[r for r in E.rows('train') if r['task']==task]
        for name,key in [('body',lambda r:len(r['content_ids'])),('body_plus_gold',lambda r:len(r['content_ids'])+len(r['gold_tokens'])),('body_times_answer',lambda r:len(r['content_ids'])*max(64,len(r['gold_tokens'])))]:
            r=max(rr,key=key);chosen[r['id']]=(r,name)
    for row,criterion in chosen.values():
        restore();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.time()
        details,outcome,retries=RT.backward_batch(engine,correction,module,opt,scaler,[(row,'gold')],epoch,0,scope='resident_probe_long_'+row['task'])
        long.append(dict(id=row['id'],task=row['task'],criterion=criterion,body_tokens=len(row['content_ids']),gold_tokens=len(row['gold_tokens']),seconds=time.time()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,amp_retries=retries,**outcome))
        engine.unchanged();C.log('GPU RESIDENT LONG BACKWARD PASSED',long[-1])
    assert C.sha(ROOT/'runs/last.pt')==source_sha,'Probe must never alter formal checkpoint'
    assert all(not p.requires_grad and p.grad is None for p in correction.parameters())
    peak=max(x['peak_reserved_gib'] for x in comparisons+long)
    E.mark('controls/gpu_resident_probe_completed.json',status='completed',checkpoint_step=cp['step'],checkpoint_epoch=cp['epoch'],checkpoint_batch=cp['batch'],checkpoint_sha256=source_sha,
        parity_batch=bi,comparisons=comparisons,long_backwards=long,minimum_free_mib=math.ceil(peak*1024)+CFG['gpu_reserve_mib'],
        disposable_updates=True,formal_checkpoint_unchanged=True,finished_at=time.strftime('%F %T'))
    C.log('GPU RESIDENT PROBE COMPLETED',cp['step'],peak)
if __name__=='__main__':main()
