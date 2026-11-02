"""AMP overflow is a rejected transaction, not a lost example or optimizer step."""
from common import *
from model import losses

def finish_backward(module,opt,scaler):
    scaler.unscale_(opt)
    gradients={n:p.grad for n,p in module.named_parameters()}
    assert all(g is not None for g in gradients.values()),'Disconnected trainable memory parameter'
    bad={n:dict(nan=int(torch.isnan(g).sum()),inf=int(torch.isinf(g).sum())) for n,g in gradients.items() if not torch.isfinite(g).all()}
    scale=float(scaler.get_scale())
    if bad:
        assert scaler.is_enabled(),'Nonfinite gradients without AMP cannot be recovered by scaling'
        # unscale_ registered found_inf. Public GradScaler.step skips the entire
        # optimizer transaction and update backs off its scale, before clipping.
        scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        new_scale=float(scaler.get_scale());assert new_scale<scale
        return dict(accepted=False,scale=scale,new_scale=new_scale,nonfinite_parameters=bad)
    grad=torch.nn.utils.clip_grad_norm_(module.parameters(),CFG['gradient_clip'])
    assert torch.isfinite(grad) and grad>0,('Invalid unscaled gradient norm',float(grad))
    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    return dict(accepted=True,gradient_norm=float(grad),scale=scale,new_scale=float(scaler.get_scale()))

def backward_batch(engine,correction,module,opt,scaler,batch,epoch,batch_index,*,scope):
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True);details=[]
        for row,kind in batch:
            with torch.no_grad():
                base=engine.base(row);translated=engine.cache(row,base,correction);native=engine.native_cache(row)
            loss,detail=losses(engine,module,row,translated,native,kind,epoch)
            scaler.scale(loss/len(batch)).backward();del loss,base,translated,native
            details.append(detail)
        outcome=finish_backward(module,opt,scaler)
        if outcome['accepted']:return details,outcome,retry
        record=dict(scope=scope,epoch=epoch,batch=batch_index,retry=retry,**outcome,time=time.strftime('%F %T'))
        path=ROOT/'diagnostics'/f'{scope}_amp_retries.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(record)+'\n')
        C.log('AMP RETRY',json.dumps(record))
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    raise RuntimeError('AMP overflow persists after8 scale reductions; batch retained, training stopped')
