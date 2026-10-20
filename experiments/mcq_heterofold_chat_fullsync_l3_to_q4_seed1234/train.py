"""Fresh two-stage Writer training; base frozen during candidate-KL calibration."""
import sys
from collections import defaultdict
from core import *


def stage_a_backward(base,source,target,divisor=1,backward=True,scaler=None):
    tk,tv=target['key'].cuda().float(),target['value'].cuda().float()
    dk,dv=tk.square().sum().clamp_min(1e-8),tv.square().sum().clamp_min(1e-8)
    n=tk.shape[1];values=defaultdict(float)
    for start in range(0,n,CFG['chunk_tokens']):
        stop=min(n,start+CFG['chunk_tokens']);weight=(stop-start)/n
        with torch.amp.autocast('cuda',dtype=torch.float16):
            k,v=base(source['key'][:,start:stop][None].cuda(),source['value'][:,start:stop][None].cuda())
        k,v=k[0].float(),v[0].float()
        kn,vn=(k-tk[:,start:stop]).square().sum()/dk,(v-tv[:,start:stop]).square().sum()/dv
        kc=F.cosine_similarity(k,tk[:,start:stop],-1).mean()
        vc=F.cosine_similarity(v,tv[:,start:stop],-1).mean()
        loss=kn+vn+weight*(2-kc-vc)
        assert torch.isfinite(loss),'Nonfinite reconstruction loss'
        if backward:scaler.scale(loss/divisor).backward()
        for name,value in dict(total=loss,k_nmse=kn,v_nmse=vn,k_cosine=weight*kc,v_cosine=weight*vc).items():values[name]+=float(value.detach())
    return dict(values)


def stage_b_backward(receiver,tok,row,target,bk,bv,adapter,divisor=1,backward=True,scaler=None):
    teacher=target['teacher_scores'].cuda()
    with torch.no_grad():
        k,v=adapted(adapter,bk,bv);k,v=join(target,k,v)
        student=tensor_scores(receiver,tok,row,k,v)
        loss=choice_loss(teacher,student)
        weights=(student.softmax(-1)-teacher.softmax(-1)).detach()
    assert torch.isfinite(loss),'Nonfinite choice KL'
    if backward:
        # Exact derivative of KL(teacher || softmax(scores)), sequential graphs.
        for i in range(len(row['candidates'])):
            k,v=adapted(adapter,bk,bv);k,v=join(target,k,v)
            score=candidate_tensor(receiver,tok,row,k,v,i)
            scaler.scale(weights[i]*score/divisor).backward()
    return dict(total=float(loss),choice_kl=float(loss),native_correct=int(teacher.argmax())==row['gold'],writer_correct=int(student.argmax())==row['gold'])


def update(model,optimizer,scaler):
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_clip'])
    assert torch.isfinite(norm),'Nonfinite gradients'
    assert float(norm)>0,'No gradient reached trainable Writer'
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    return float(norm)


def checkpoint(path,model,stage,epoch,step,metrics,optimizer=None,scaler=None,position=0,completed=True):
    payload=dict(manifest_sha256=identity(),stage=stage,epoch=epoch,step=step,metrics=metrics,
        state={k:v.detach().cpu() for k,v in model.state_dict().items()},sample_position=position,epoch_completed=completed)
    if optimizer is not None:payload['optimizer']=optimizer.state_dict();payload['scaler']=scaler.state_dict()
    save_tensor(path,payload)


@torch.no_grad()
def validate(stage,base,adapter=None,receiver=None,tok=None):
    totals=defaultdict(float);rr=rows('validation')
    for i,row in enumerate(rr,1):
        source,target=state('validation',row,'llama'),state('validation',row,'qwen')
        if stage=='a':value=stage_a_backward(base,source,target,backward=False)
        else:
            bk,bv=map_base(base,source)
            value=stage_b_backward(receiver,tok,row,target,bk,bv,adapter,backward=False)
        for name,v in value.items():totals[name]+=v
        if i%32==0 or i==len(rr):log(f'Stage-{stage.upper()} validation {i}/{len(rr)}')
    return {k:v/len(rr) for k,v in totals.items()}


def train(stage):
    seed();guard('stage_a' if stage=='a' else 'stage_b')
    folder=ROOT/f'runs/stage_{stage}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():log(f'Stage-{stage.upper()} already completed');return
    assert (ROOT/'smoke/completed.json').exists(),'Smoke not passed'
    for family in ['llama','qwen']:assert (ROOT/f'cache/{family}/completed.json').exists()
    receiver=tok=adapter=None
    if stage=='a':base=fresh_base().cuda();trainable=base
    else:
        base=base_from_best();adapter=ResidualKVAdapter(rank=CFG['adapter_rank']).cuda();trainable=adapter
        tok,receiver=model('qwen')
        log(f'Frozen Stage-A SHA256={sha(ROOT/"runs/stage_a/best.pt")}')
    optimizer=torch.optim.AdamW(trainable.parameters(),lr=CFG[f'stage_{stage}_learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    start_epoch=1;step=0;resume=0;history=[]
    if (folder/'last.pt').exists():
        payload=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        assert payload['manifest_sha256']==identity() and payload['stage']==stage
        trainable.load_state_dict(payload['state'],strict=True);optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
        step=payload['step'];start_epoch=payload['epoch']+int(payload['epoch_completed'])
        resume=0 if payload['epoch_completed'] else payload['sample_position']
        if (folder/'selection.json').exists():history=json.loads((folder/'selection.json').read_text())['epochs']
        del payload
        log(f'RESUME Stage-{stage.upper()} epoch={start_epoch} step={step} position={resume}')
    rr=rows('train');batch=CFG['effective_batch'];epochs=CFG[f'stage_{stage}_epochs']
    with (folder/'training_steps.jsonl').open('a') as stream:
        for epoch in range(start_epoch,epochs+1):
            trainable.train();order=list(range(len(rr)));random.Random(CFG['seed']+epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True);totals=defaultdict(float);acc=0
            for position,index in enumerate(order,1):
                if epoch==start_epoch and position<=resume:continue
                row=rr[index];divisor=min(batch,len(order)-((position-1)//batch)*batch)
                source,target=state('train',row,'llama'),state('train',row,'qwen')
                if stage=='a':values=stage_a_backward(base,source,target,divisor,scaler=scaler)
                else:
                    bk,bv=map_base(base,source)
                    values=stage_b_backward(receiver,tok,row,target,bk,bv,adapter,divisor,scaler=scaler)
                    del bk,bv
                del source,target
                for name,v in values.items():totals[name]+=v
                acc+=1
                if acc==batch or position==len(order):
                    norm=update(trainable,optimizer,scaler);step+=1
                    rec=dict(stage=stage,epoch=epoch,step=step,samples=position,total_samples=len(rr),grad_norm=norm,**{k:v/acc for k,v in totals.items()})
                    stream.write(json.dumps(rec)+'\n');stream.flush()
                    status(status='running',phase=f'stage_{stage}',**rec)
                    if step%8==0:log(f'Stage-{stage.upper()} epoch={epoch}/{epochs} step={step} samples={position}/{len(rr)} loss={rec["total"]:.6f} grad={norm:.4f}')
                    acc=0;totals=defaultdict(float)
                    if step%CFG['checkpoint_every_steps']==0 and position<len(order):
                        checkpoint(folder/'last.pt',trainable,stage,epoch,step,{},optimizer,scaler,position,False)
            trainable.eval();metrics=validate(stage,base,adapter,receiver,tok)
            candidate=dict(epoch=epoch,step=step,validation=metrics)
            history=[x for x in history if x['epoch']!=epoch]+[candidate]
            best=min(history,key=lambda x:(x['validation']['total'],x['epoch']))
            if best==candidate:checkpoint(folder/'best.pt',trainable,stage,epoch,step,metrics)
            save(folder/'selection.json',dict(selection=CFG['checkpoint_selection'],best=best,epochs=history))
            checkpoint(folder/'last.pt',trainable,stage,epoch,step,metrics,optimizer,scaler)
            log(f'Stage-{stage.upper()} EPOCH COMPLETED {epoch}/{epochs}: {metrics}')
    assert step==len(rr)//batch*epochs,(stage,step)
    result=dict(status='completed',epochs=epochs,steps=step,best=json.loads((folder/'selection.json').read_text())['best'],best_sha256=sha(folder/'best.pt'),manifest_sha256=identity())
    save(folder/'completed.json',result);shared(f'stage_{stage}_completed_evaluation_pending',result)
    log(f'Stage-{stage.upper()} TRAINING COMPLETED')


if __name__=='__main__':
    try:train(sys.argv[1])
    except Exception as exc:fail(exc);raise
