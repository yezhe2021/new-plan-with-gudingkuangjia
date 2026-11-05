"""Fresh all-layer Keep-both; original memory architecture and AMP recovery."""
import random
from experiment_common import *

class GemmaMemory(M.RNKV):
    def __init__(self,engine,config=None):
        cfg=dict(R.CFG if config is None else config);cfg['replace_layers']=CFG['replace_layers']
        super().__init__(engine,cfg)
        self.routing.close();self.routing=AR.KeepBothRouting(engine,self.memory,cfg['replace_layers'])

def load(final=False):
    from engine_new import Engine
    engine=Engine();correction=load_c();module=GemmaMemory(engine)
    assert sum(p.numel() for p in module.parameters())==409600
    if final:
        cp=torch.load(ROOT/'runs/epoch4_step1536.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==identity() and cp['step']==1536 and cp['recolor_sha256']==map_sha() and cp['c_checkpoint_sha256']==C.sha(ROOT/'runs/C/epoch4_step800.pt');module.load_state_dict(cp['state'],strict=True)
    return engine,correction,module

def batches(epoch):
    tasks=list(CFG['train_counts']);data={t:[r for r in rows('train') if r['task']==t] for t in tasks};cursor={t:0 for t in tasks}
    for i,t in enumerate(tasks):random.Random(CFG['seed']+epoch*17+i).shuffle(data[t])
    out=[];quota=CFG['prefix_quotas_per_batch'][epoch-1]
    for bi in range(CFG['updates_per_epoch']):
        batch=[]
        for i,t in enumerate(tasks):
            n=2 if i==bi%3 else 3;start=cursor[t];batch+=data[t][start:start+n];cursor[t]+=n
        assert len(batch)==8
        labels=[k for k,n in quota.items() for _ in range(n)]
        random.Random(CFG['seed']+epoch*10000+bi).shuffle(labels);out.append(list(zip(batch,labels)))
    assert cursor==CFG['train_counts'] and len({r['id'] for b in out for r,k in b})==3072
    return out

@torch.no_grad()
def native_prefix(module,row,native):
    path=ROOT/'native_prefix'/(row['id']+'.json')
    if path.exists():
        v=C.read(path);assert v['manifest_sha256']==identity();return v['tokens']
    tokens=module.rollout(row,native,rn=False,cap=CFG['prefix_max_new_tokens'])
    mark(str(path.relative_to(ROOT)),id=row['id'],tokens=tokens);return tokens

def losses(engine,module,row,translated,native,prefix_kind,epoch):
    memory=module.construct(row,translated);target=row['gold_tokens']
    # Full rationale CE is intentionally not limited by the KL rollout budget.
    gold=module.token_logits(row,memory,target,rn=True)
    task_loss=torch.nn.functional.cross_entropy(gold,torch.tensor(target,device=gold.device))
    del gold
    with torch.no_grad():
        tokens=target if prefix_kind=='gold' else native_prefix(module,row,native) if prefix_kind=='native' else module.rollout(row,memory,rn=True,cap=CFG['prefix_max_new_tokens'])
    assert tokens;positions=C.positions_for(tokens,row,epoch,validation=epoch==0);prefix=tokens[:max(positions)]
    with torch.no_grad():teacher=module.logits(row,native,prefix,positions,rn=False)
    student=module.logits(row,memory,prefix,positions,rn=True);kl=C.vocab_kl(teacher,student)
    loss=task_loss+CFG['lambda_kl']*kl;assert torch.isfinite(loss)
    return loss,dict(loss=float(loss.detach()),task_loss=float(task_loss.detach()),full_kl=float(kl.detach()),prefix_kind=prefix_kind,
        positions=len(positions),prefix_tokens=len(tokens),gold_tokens=len(target),task=row['task'])

def backward_batch(engine,correction,module,opt,scaler,batch,epoch,batch_index,*,scope):
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True);details=[]
        for row,kind in batch:
            with torch.no_grad():base=engine.base(row);translated=engine.cache(row,base,correction);native=engine.native_cache(row)
            loss,detail=losses(engine,module,row,translated,native,kind,epoch)
            scaler.scale(loss/len(batch)).backward();details.append(detail);del loss,base,translated,native
        outcome=O.finish_backward(module,opt,scaler)
        if outcome['accepted']:return details,outcome,retry
        value=dict(scope=scope,epoch=epoch,batch=batch_index,retry=retry,**outcome,time=time.strftime('%F %T'))
        path=ROOT/f'diagnostics/{scope}_amp_retries.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(value)+'\n')
        C.log('GEMMA RECOLOR MEMORY AMP RETRY',json.dumps(value));torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    raise RuntimeError('Persistent AMP overflow after8 reductions; no example skipped')
