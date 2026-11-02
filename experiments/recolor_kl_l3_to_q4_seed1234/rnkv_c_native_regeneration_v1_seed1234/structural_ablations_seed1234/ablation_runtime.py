"""Original RN computation with just one predeclared structural change per arm."""
from ablation_common import *
from attention import Routing
from torch.utils.checkpoint import checkpoint

class KeepBothRouting(Routing):
    def allowed(self,phase,layer,header,L,M,q,key_length):
        value=torch.arange(key_length,device=q.device)[None,:]<=q[:,None]
        if phase=='answer':
            if layer not in self.replace:value[:,L:L+M]=False
        else:assert phase=='construct'
        return value

class AblationModel(M.RNKV):
    def __init__(self,engine,arm,config=None):
        assert arm in CFG['arms'];self.arm=arm
        config=dict(R.CFG if config is None else config)
        if not CFG['arms'][arm]['adapters']:config['adapter_after_layers']=[]
        super().__init__(engine,config)
        if not CFG['arms'][arm]['replace_body']:
            self.routing.close();self.routing=KeepBothRouting(engine,self.memory,config['replace_layers'])

def load(arm,final=False):
    engine=R.Engine();correction=T.load_final('C').requires_grad_(False)
    module=AblationModel(engine,arm)
    assert sum(p.numel() for p in module.parameters())==CFG['arms'][arm]['parameters']
    if final:
        if CFG['arms'][arm]['train']:
            cp=torch.load(ROOT/f'arms/{arm}/runs/epoch4_step1024.pt',map_location='cpu',weights_only=True)
            assert cp['manifest_sha256']==identity() and cp['step']==1024 and cp['arm']==arm
        else:
            assert arm=='keep_both';cp=torch.load(PARENT/'runs/epoch4_step1024.pt',map_location='cpu',weights_only=True)
            assert cp['manifest_sha256']==R.identity() and cp['step']==1024
        module.load_state_dict(cp['state'],strict=True)
    return engine,correction,module

@torch.no_grad()
def native_prefix(module,row,native):
    path=ROOT/'native_prefix'/(row['id']+'.json')
    if path.exists():
        value=C.read(path);assert value['manifest_sha256']==identity();return value['tokens']
    tokens=module.rollout(row,native,rn=False)
    mark(str(path.relative_to(ROOT)),id=row['id'],tokens=tokens,teacher='unchanged native Qwen; shared across arms')
    return tokens

def losses(engine,module,row,translated,native,prefix_kind,epoch):
    regenerated=module.construct(row,translated)
    target=row['gold_tokens'];gold=module.token_logits(row,regenerated,target,rn=True)
    task_loss=torch.nn.functional.cross_entropy(gold,torch.tensor(target,device=gold.device))
    with torch.no_grad():
        tokens=target if prefix_kind=='gold' else native_prefix(module,row,native) if prefix_kind=='native' else module.rollout(row,regenerated,rn=True)
    assert tokens
    positions=C.positions_for(tokens,row,epoch,validation=epoch==0);prefix=tokens[:max(positions)]
    with torch.no_grad():teacher=module.logits(row,native,prefix,positions,rn=False)
    student=module.logits(row,regenerated,prefix,positions,rn=True)
    kl=C.vocab_kl(teacher,student);loss=task_loss+CFG['lambda_kl']*kl;assert torch.isfinite(loss)
    return loss,dict(task_loss=float(task_loss.detach()),full_kl=float(kl.detach()),loss=float(loss.detach()),
        prefix_kind=prefix_kind,positions=len(positions),prefix_tokens=len(tokens),gold_tokens=len(target))

def batches(epoch):return M.batches(epoch)

def backward_batch(engine,correction,module,opt,scaler,batch,epoch,batch_index,*,scope):
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True);details=[]
        for row,kind in batch:
            with torch.no_grad():
                base=engine.base(row);translated=engine.cache(row,base,correction);native=engine.native_cache(row)
            loss,detail=losses(engine,module,row,translated,native,kind,epoch)
            scaler.scale(loss/len(batch)).backward();del loss,base,translated,native;details.append(detail)
        outcome=O.finish_backward(module,opt,scaler)
        if outcome['accepted']:return details,outcome,retry
        value=dict(arm=module.arm,scope=scope,epoch=epoch,batch=batch_index,retry=retry,**outcome,time=time.strftime('%F %T'))
        path=ROOT/f'diagnostics/{module.arm}_{scope}_amp_retries.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(value)+'\n')
        C.log('ABLATION AMP RETRY',json.dumps(value));torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    raise RuntimeError('Persistent AMP overflow after8 reductions; no example skipped')
