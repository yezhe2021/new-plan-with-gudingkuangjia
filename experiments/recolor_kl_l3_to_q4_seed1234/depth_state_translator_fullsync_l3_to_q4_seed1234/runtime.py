"""Full-token Native K/V distillation, then task CE plus mixed-prefix full KL."""
import random
from experiment_common import *
from translator import DepthStateTranslator,DSTConfig
from engine import Engine,BodyReader

def make():return DepthStateTranslator(DSTConfig(**CFG['model']),CFG['seed']).cuda().float()
def checkpoint_path(stage):return ROOT/f'runs/{stage}/epoch{CFG[stage]["epochs"]}_step{CFG[stage]["epochs"]*CFG["updates_per_epoch"]}.pt'
def load_stage(module,stage):
    path=checkpoint_path(stage);cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==identity() and cp['stage']==stage and cp['epoch']==CFG[stage]['epochs']
    module.load_state_dict(cp['state'],strict=True);return cp

def batches(epoch,stage):
    data={t:[r for r in rows('train') if r['task']==t] for t in CFG['train_counts']};cursor={t:0 for t in data}
    for i,t in enumerate(data):random.Random(CFG['seed']+epoch*17+i).shuffle(data[t])
    out=[]
    for bi in range(CFG['updates_per_epoch']):
        batch=[]
        for i,t in enumerate(data):
            n=2 if i==bi%3 else 3;start=cursor[t];batch+=data[t][start:start+n];cursor[t]+=n
        labels=[k for k,n in CFG['prefix_quotas_per_batch'][epoch-1].items() for _ in range(n)] if stage=='stage2' else ['gold']*8
        random.Random(CFG['seed']+epoch*10000+bi).shuffle(labels);assert len(batch)==8;out.append(list(zip(batch,labels)))
    assert cursor==CFG['train_counts'];return out

@torch.no_grad()
def native_prefix(reader,row,native):
    original=REFERENCE/'native_prefix'/(row['id']+'.json')
    if original.exists():
        v=C.read(original);assert v['manifest_sha256']==N.identity() and v['id']==row['id'];return v['tokens']
    path=ROOT/'native_prefix'/(row['id']+'.json')
    if path.exists():
        v=C.read(path);assert v['manifest_sha256']==identity();return v['tokens']
    tokens=reader.rollout(row,native,rn=False,cap=CFG['prefix_max_new_tokens'])
    mark(str(path.relative_to(ROOT)),id=row['id'],tokens=tokens);return tokens

def functional(engine,reader,module,row,source,native,kind,epoch):
    cache=engine.translated(row,module,source);target=row['gold_tokens']
    logits=reader.token_logits(row,cache,target,rn=False)
    ce=torch.nn.functional.cross_entropy(logits,torch.tensor(target,device=logits.device));del logits
    with torch.no_grad():
        tokens=target if kind=='gold' else native_prefix(reader,row,native) if kind=='native' else reader.rollout(row,cache,rn=False,cap=CFG['prefix_max_new_tokens'])
        positions=C.positions_for(tokens,row,epoch,validation=epoch==0);prefix=tokens[:max(positions)]
        teacher=reader.logits(row,native,prefix,positions,rn=False)
    student=reader.logits(row,cache,prefix,positions,rn=False);kl=C.vocab_kl(teacher,student)
    loss=ce+CFG['stage2']['lambda_kl']*kl
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite functional loss')
    return loss,dict(loss=float(loss.detach()),task_loss=float(ce.detach()),full_kl=float(kl.detach()),gold_tokens=len(target),prefix_tokens=len(tokens),positions=len(positions))

def reconstruction(module,source,row,target_k,target_v,*,backward=None):
    n=len(row['sync_source_indices']);energy_k=target_k.float().square().mean((1,2)).clamp_min(1e-6).cuda();energy_v=target_v.float().square().mean((1,2)).clamp_min(1e-6).cuda();total=0.
    for begin in range(0,n,CFG['token_chunk']):
        end=min(n,begin+CFG['token_chunk']);k,v=module(*source,row['sync_source_indices'][begin:end])
        kt=target_k[:,begin:end].to(k.device).float();vt=target_v[:,begin:end].to(v.device).float()
        loss=.5*(((k-kt).square().mean((1,2))/energy_k).mean()+((v-vt).square().mean((1,2))/energy_v).mean())*(end-begin)/n
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Native K/V reconstruction loss')
        total+=float(loss.detach())
        if backward is not None:backward(loss)
    return dict(loss=total,kv_nmse=total,body_tokens=n)

def backward_batch(engine,reader,module,opt,scaler,batch,stage,epoch):
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
    for retry in range(9):
        opt.zero_grad(set_to_none=True);details=[]
        for row,kind in batch:
            source=engine.source(row)
            if stage=='stage1':
                tk,tv,native=engine.teacher(row);del native
                detail=reconstruction(module,source,row,tk,tv,backward=lambda loss:scaler.scale(loss/len(batch)).backward());del tk,tv
            else:
                with torch.no_grad():native=engine.native_cache(row)
                loss,detail=functional(engine,reader,module,row,source,native,kind,epoch)
                scaler.scale(loss/len(batch)).backward();del loss,native
            detail.update(task=row['task'],prefix_kind=kind);details.append(detail);del source
        scaler.unscale_(opt);gradients=[p.grad for p in module.parameters()]
        assert all(g is not None for g in gradients),'Disconnected DST parameter'
        scale=float(scaler.get_scale())
        if any(not torch.isfinite(g).all() for g in gradients):
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if float(scaler.get_scale())>=scale:raise FloatingPointError('Unrecoverable DST gradients')
            C.log('DST REJECT SAME BATCH',stage,epoch,retry,scale,float(scaler.get_scale()))
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);continue
        norm=torch.nn.utils.clip_grad_norm_(module.parameters(),CFG['gradient_clip'])
        assert torch.isfinite(norm) and norm>0;scaler.step(opt);scaler.update()
        return details,dict(accepted=True,gradient_norm=float(norm),amp_retries=retry,amp_scale=float(scaler.get_scale()))
    raise FloatingPointError('DST same-batch gradient recovery exhausted; no example skipped')
