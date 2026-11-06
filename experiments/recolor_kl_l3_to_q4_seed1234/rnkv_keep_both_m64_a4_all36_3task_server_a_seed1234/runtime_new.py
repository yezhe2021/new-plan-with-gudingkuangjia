"""M64, shared fresh M32 initialization, identical losses with host offloading."""
import random
from contextlib import nullcontext
from experiment_common import *
from engine_offload import Engine,to_cache

class Memory64(M.RNKV):
    def __init__(self,engine,config=None):
        cfg=dict(CFG if config is None else config);cfg['replace_layers']=list(range(1,37))
        super().__init__(engine,cfg);self.routing.close();self.routing=AR.KeepBothRouting(engine,self.memory,cfg['replace_layers'])
        if config is None:
            # Exactly the original baseline's INITIAL weights, never its learned weights.
            master=M.Memory(engine.model.config.hidden_size,32,CFG['adapter_rank'],CFG['adapter_after_layers'],CFG['seed'])
            with torch.no_grad(),torch.random.fork_rng(devices=[]):
                torch.manual_seed(CFG['seed']+6400);extra=torch.randn(32,engine.model.config.hidden_size)*.02
                self.memory.prompt.copy_(torch.cat([master.prompt.detach(),extra],0).to(self.memory.prompt.device))
                self.memory.adapters.load_state_dict(master.adapters.state_dict(),strict=True)

def load(final=False):
    engine=Engine();correction=T.load_final('C').requires_grad_(False);module=Memory64(engine)
    assert sum(p.numel() for p in module.parameters())==491520
    if final:
        cp=torch.load(ROOT/'runs/epoch4_step1536.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==identity() and cp['step']==1536;module.load_state_dict(cp['state'],strict=True)
    return engine,correction,module

def batches(epoch):
    tasks=list(CFG['train_counts']);data={t:[r for r in rows('train') if r['task']==t] for t in tasks};cursor={t:0 for t in tasks};out=[]
    for i,t in enumerate(tasks):random.Random(CFG['seed']+epoch*17+i).shuffle(data[t])
    for bi in range(CFG['updates_per_epoch']):
        batch=[]
        for i,t in enumerate(tasks):
            n=2 if i==bi%3 else 3;start=cursor[t];batch+=data[t][start:start+n];cursor[t]+=n
        labels=[k for k,n in CFG['prefix_quotas_per_batch'][epoch-1].items() for _ in range(n)]
        random.Random(CFG['seed']+epoch*10000+bi).shuffle(labels);out.append(list(zip(batch,labels)))
    assert cursor==CFG['train_counts'] and len({r['id'] for b in out for r,k in b})==3072;return out

def offload_context(enabled=True):
    return torch.autograd.graph.save_on_cpu(pin_memory=True) if enabled and torch.cuda.is_available() else nullcontext()

@torch.no_grad()
def native_prefix(module,row,native):
    local=ROOT/'native_prefix'/(row['id']+'.json');reference=REFERENCE/'native_prefix'/(row['id']+'.json')
    if local.exists():
        v=C.read(local);assert v['manifest_sha256']==identity();return v['tokens']
    if reference.exists():
        v=C.read(reference);assert v['manifest_sha256']==CFG['reference_manifest_sha256'];tokens=v['tokens']
    else:
        cuda=to_cache(native,module.engine.model,next(module.engine.model.parameters()).device)
        tokens=module.rollout(row,cuda,rn=False,cap=CFG['prefix_max_new_tokens']);del cuda
    mark(str(local.relative_to(ROOT)),id=row['id'],tokens=tokens);return tokens

def losses(engine,module,row,translated,native,prefix_kind,epoch,*,offload=True):
    target=row['gold_tokens']
    # Checkpoint inputs saved to host. Native teacher KV is not kept on CUDA
    # across student CE/KL; its values and semantic prefixes are unchanged.
    with offload_context(offload):
        memory=module.construct(row,translated)
        with torch.no_grad():
            tokens=target if prefix_kind=='gold' else native_prefix(module,row,native) if prefix_kind=='native' else module.rollout(row,memory,rn=True,cap=CFG['prefix_max_new_tokens'])
            assert tokens;positions=C.positions_for(tokens,row,epoch,validation=epoch==0);prefix=tokens[:max(positions)]
            cuda=to_cache(native,engine.model,next(engine.model.parameters()).device)
            teacher=module.logits(row,cuda,prefix,positions,rn=False);del cuda
        gold=module.token_logits(row,memory,target,rn=True);task_loss=torch.nn.functional.cross_entropy(gold,torch.tensor(target,device=gold.device));del gold
        student=module.logits(row,memory,prefix,positions,rn=True);kl=C.vocab_kl(teacher,student)
        loss=task_loss+CFG['lambda_kl']*kl;assert torch.isfinite(loss)
    return loss,dict(loss=float(loss.detach()),task_loss=float(task_loss.detach()),full_kl=float(kl.detach()),prefix_kind=prefix_kind,positions=len(positions),prefix_tokens=len(tokens),gold_tokens=len(target),task=row['task'])

def backward_batch(engine,correction,module,opt,scaler,batch,epoch,batch_index,*,scope):
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True);details=[]
        for row,kind in batch:
            with torch.no_grad():
                base=engine.base(row);translated=engine.cache(row,base,correction);del base
                native=engine.native_cache(row)
            with offload_context():
                loss,detail=losses(engine,module,row,translated,native,kind,epoch)
                scaler.scale(loss/len(batch)).backward()
            details.append(detail);del loss,translated,native
        outcome=O.finish_backward(module,opt,scaler)
        if outcome['accepted']:return details,outcome,retry
        p=ROOT/f'diagnostics/{scope}_amp_retries.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as f:f.write(json.dumps(dict(epoch=epoch,batch=batch_index,retry=retry,**outcome))+'\n')
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    raise RuntimeError('Persistent AMP overflow after8 reductions; no sample skipped')
