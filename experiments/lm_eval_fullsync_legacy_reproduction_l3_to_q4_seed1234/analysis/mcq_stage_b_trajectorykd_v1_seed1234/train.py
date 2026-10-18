"""Full-trajectory teacher-forcing KD, final-answer/terminal CE, and MCQ replay."""
import argparse
from collections import defaultdict
import gc
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute

from trajectory_common import (CONFIG,HERE,PARENT,PREVIOUS,CFG,previous_train,previous_runtime,
                               log,save,read_rows,gpu_check,model_bundle,verify_manifest,freeze_manifest,
                               load_initial_adapter,digest)
from kv_backend import capture,source_map,map_base,assemble
from protocol import make_cache


@torch.no_grad()
def prefix(sender,receiver,base,source_tok,hf,row,anchors):
    ids=row['context_ids']
    assert hf.tok_encode(row['context'])==ids
    source_ids,selected,counts=source_map(row['context'],source_tok,hf.tokenizer,ids)
    sk,sv=capture(sender,source_ids)
    if ids[0] not in anchors:
        anchors[ids[0]]=capture(receiver,ids[:1])
    ak,av=anchors[ids[0]]
    item={'source_k':sk[:,selected],'source_v':sv[:,selected], 'anchor_k':ak,'anchor_v':av,
          'tokens':len(selected),'counts':counts}
    k,v=map_base(base,item,CFG['chunk_tokens'])
    return {'anchor_k':ak,'anchor_v':av,'tokens':len(selected)},k,v


def suffix_hidden(receiver,context,continuation,key=None,value=None):
    prefix=0 if key is None else key.shape[1]
    if key is not None:
        assert prefix==len(context)-1
    ids=(context if key is None else context[-1:])+continuation[:-1]
    cache=None if key is None else make_cache(receiver,key,value,torch.arange(prefix,device='cuda'))
    output=receiver.model(input_ids=torch.tensor([ids],device='cuda'),
                          attention_mask=torch.ones((1,prefix+len(ids)),device='cuda',dtype=torch.long),
                          position_ids=torch.arange(prefix,prefix+len(ids),device='cuda')[None],
                          past_key_values=cache,use_cache=False,return_dict=True)
    return output.last_hidden_state[0,-len(continuation):]


def projection_terms(student,teacher,head,targets,answer_mask,terminal_mask):
    slp=head(student).float().log_softmax(-1)
    with torch.no_grad():
        tlp=head(teacher).float().log_softmax(-1)
    kd=F.kl_div(slp,tlp,reduction='sum',log_target=True)
    ce=-slp.gather(-1,targets[:,None]).squeeze(-1)
    return torch.stack((kd,(ce*answer_mask).sum(),(ce*terminal_mask).sum()))


def objective(receiver,adapter,row,item,bk,bv):
    ctx,ids=row['context_ids'],row['continuation_ids']
    with torch.no_grad():
        teacher=suffix_hidden(receiver,ctx,ids)
    k,v=assemble(item,bk,bv,adapter)
    student=recompute(lambda kk,vv:suffix_hidden(receiver,ctx,ids,kk,vv),k,v,use_reentrant=False)
    targets=torch.tensor(ids,device='cuda')
    am=torch.zeros(len(ids),device='cuda'); am[row['answer_positions']]=1
    em=torch.zeros_like(am); em[row['terminal_positions']]=1
    terms=student.new_zeros(3,dtype=torch.float32)
    for start in range(0,len(ids),CONFIG['projection_chunk_tokens']):
        end=start+CONFIG['projection_chunk_tokens']
        def project(s,t,y,a,e):
            return projection_terms(s,t,receiver.lm_head,y,a,e)
        terms=terms+recompute(project,student[start:end],teacher[start:end],targets[start:end],am[start:end],em[start:end],use_reentrant=False)
    kd=terms[0]/len(ids); answer=terms[1]/am.sum(); terminal=terms[2]/em.sum()
    loss=CONFIG['token_kl_weight']*kd+CONFIG['gold_ce_weight']*answer+CONFIG['terminal_ce_weight']*terminal
    assert bool(torch.isfinite(loss)), 'Non-finite full-trajectory objective'
    metrics={'total':float(loss.detach()),'token_kl':float(kd.detach()),'answer_ce':float(answer.detach()),
             'terminal_ce':float(terminal.detach()),'tokens':len(ids),'context_tokens':len(ctx)}
    return loss,metrics


def optimizer_for(adapter):
    return torch.optim.AdamW(adapter.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])


def update(adapter,opt,scaler):
    scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),CONFIG['gradient_clip'])
    assert torch.isfinite(norm), 'Non-finite gradient norm'
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return float(norm)


def smoke():
    previous_runtime.verify_manifest(); gpu_check()
    receiver,sender,base,adapter,source_tok,hf=model_bundle(True)
    refs=read_rows(HERE/'prepared/train_trajectories.jsonl')
    row=max(refs,key=lambda r:len(r['context_ids'])+len(r['continuation_ids']))
    torch.cuda.reset_peak_memory_stats()
    anchors={}; item,bk,bv=prefix(sender,receiver,base,source_tok,hf,row,anchors)
    # No native or parent parameter may receive a gradient.
    before=next(adapter.parameters()).detach().clone()
    opt=optimizer_for(adapter); scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    loss,metrics=objective(receiver,adapter,row,item,bk,bv)
    scaler.scale(loss).backward()
    assert all(p.grad is None for m in (receiver,sender,base) for p in m.parameters())
    norm=update(adapter,opt,scaler)
    delta=float((next(adapter.parameters()).detach()-before).abs().max())
    assert norm>0 and delta>0
    # Exercise the independent replay path with the same optimizer and frozen models.
    del loss,item,bk,bv
    replay=read_rows(PREVIOUS/'prepared/train_epoch_1.jsonl')[0]
    item,bk,bv,pairs=previous_train.frozen_prefix(sender,receiver,base,source_tok,hf,replay,anchors)
    replay_metrics=previous_train.objective(receiver,hf,adapter,replay,item,bk,bv,pairs,1,scaler)
    replay_norm=update(adapter,opt,scaler)
    save(HERE/'smoke.json',{'passed':True,'SMOKE_ONLY_NOT_FORMAL_TRAINING':True,'trajectory':metrics,
                          'trajectory_grad_norm':norm,'mcq_replay':replay_metrics,'replay_grad_norm':replay_norm,
                          'parameter_max_change':delta,'frozen_parameter_gradients':False,
                          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                          'initial_stage_b_sha256':digest(PREVIOUS/'stage_b/best.pt')})
    log('TRAJECTORY BACKWARD + MCQ REPLAY SMOKE PASSED; smoke-updated adapter discarded')


@torch.no_grad()
def validate(receiver,sender,base,adapter,source_tok,hf,rows,anchors):
    sums=defaultdict(float)
    for i,row in enumerate(rows,1):
        item,bk,bv=prefix(sender,receiver,base,source_tok,hf,row,anchors)
        _,metrics=objective(receiver,adapter,row,item,bk,bv)
        for k,v in metrics.items(): sums[k]+=v
        del item,bk,bv
    result={k:v/len(rows) for k,v in sums.items()}
    result['n']=len(rows)
    return result


def checkpoint(path,adapter,opt,scaler,epoch,step,position,validation,history,manifest):
    payload={'protocol':CONFIG['protocol'],'stage_a_sha256':manifest['identities']['stage_a']['sha256'],
             'initial_stage_b_sha256':manifest['identities']['initial_stage_b']['sha256'],
             'state':{k:v.detach().cpu() for k,v in adapter.state_dict().items()},
             'epoch':epoch,'step':step,'position':position,'validation':validation,'history':history}
    if opt is not None: payload.update(optimizer=opt.state_dict(),scaler=scaler.state_dict())
    path.parent.mkdir(exist_ok=True,parents=True); temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)


def train():
    manifest=verify_manifest(); gpu_check()
    receiver,sender,base,adapter,source_tok,hf=model_bundle(True)
    opt=optimizer_for(adapter); scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    train_rows=read_rows(HERE/'prepared/train_trajectories.jsonl')
    val_rows=read_rows(HERE/'prepared/validation_trajectories.jsonl')
    replay=read_rows(PREVIOUS/'prepared/train_epoch_1.jsonl')+read_rows(PREVIOUS/'prepared/train_epoch_2.jsonl')
    output=HERE/'stage_b'; output.mkdir(exist_ok=True)
    epoch_start=1; position_start=0; step=0; history=[]
    if (output/'last.pt').exists():
        p=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        assert p['protocol']==CONFIG['protocol'] and p['initial_stage_b_sha256']==manifest['identities']['initial_stage_b']['sha256']
        adapter.load_state_dict(p['state']); opt.load_state_dict(p['optimizer']); scaler.load_state_dict(p['scaler'])
        epoch_start=p['epoch']; position_start=p['position']; step=p['step']; history=p['history']
        log(f'RESUME trajectory Stage B epoch={epoch_start}, position={position_start}, updates={step}')
    anchors={}
    if not history:
        adapter.eval()
        baseline=validate(receiver,sender,base,adapter,source_tok,hf,val_rows,anchors)
        initial={'epoch':0,'step':0,'validation':baseline}
        history=[initial]
        save(output/'initial_validation.json',initial)
        checkpoint(output/'best.pt',adapter,None,None,0,0,0,baseline,history,manifest)
        save(output/'selection.json',{'selection':CONFIG['selection'],'best':initial,'epochs':history})
        log('INITIAL STAGE B VALIDATION '+json.dumps(initial))
    with (output/'training_steps.jsonl').open('a',encoding='utf-8') as stream:
        for epoch in range(epoch_start,CONFIG['epochs']+1):
            rng=random.Random(CONFIG['seed']+epoch)
            jobs=[('trajectory',r) for r in train_rows]+[('mcq',r) for r in rng.sample(replay,CONFIG['mcq_replay_per_epoch'])]
            rng.shuffle(jobs); opt.zero_grad(set_to_none=True)
            accumulated=0; sums=defaultdict(float); types=defaultdict(int)
            adapter.train()
            log(f'START TRAJECTORY TRAIN epoch={epoch}/{CONFIG["epochs"]}; trajectory={len(train_rows)}; replay={CONFIG["mcq_replay_per_epoch"]}')
            for position,(kind,row) in enumerate(jobs,1):
                if epoch==epoch_start and position<=position_start: continue
                divisor=min(CONFIG['effective_batch'],len(jobs)-((position-1)//CONFIG['effective_batch'])*CONFIG['effective_batch'])
                if kind=='trajectory':
                    item,bk,bv=prefix(sender,receiver,base,source_tok,hf,row,anchors)
                    loss,metrics=objective(receiver,adapter,row,item,bk,bv)
                    scaler.scale(loss/divisor).backward(); del loss
                else:
                    item,bk,bv,pairs=previous_train.frozen_prefix(sender,receiver,base,source_tok,hf,row,anchors)
                    metrics=previous_train.objective(receiver,hf,adapter,row,item,bk,bv,pairs,divisor,scaler)
                del item,bk,bv
                types[kind]+=1; accumulated+=1
                for k,v in metrics.items(): sums[kind+'_'+k]+=v
                if accumulated==divisor:
                    norm=update(adapter,opt,scaler); step+=1
                    entry={'epoch':epoch,'step':step,'position':position,'pre_clip_grad_norm':norm,'types':dict(types),
                           **{k:v/types[k.split('_')[0]] for k,v in sums.items()}}
                    stream.write(json.dumps(entry)+'\n'); stream.flush()
                    save(HERE/'status.json',{'phase':'training','epoch':epoch,'epochs':CONFIG['epochs'],'step':step,'position':position,'epoch_draws':len(jobs)})
                    if step==1 or step%4==0: log('TRAIN '+json.dumps(entry))
                    accumulated=0; sums=defaultdict(float); types=defaultdict(int)
                    if step==1 or step%CONFIG['checkpoint_interval_updates']==0:
                        checkpoint(output/'last.pt',adapter,opt,scaler,epoch,step,position,{},history,manifest)
            adapter.eval()
            metrics=validate(receiver,sender,base,adapter,source_tok,hf,val_rows,anchors)
            candidate={'epoch':epoch,'step':step,'validation':metrics}
            history=[h for h in history if h['epoch']!=epoch]+[candidate]
            best=min(history,key=lambda h:(h['validation']['total'],h['epoch']))
            if best is candidate:
                checkpoint(output/'best.pt',adapter,None,None,epoch,step,len(jobs),metrics,history,manifest)
            save(output/'selection.json',{'selection':CONFIG['selection'],'best':best,'epochs':history})
            checkpoint(output/'last.pt',adapter,opt,scaler,epoch+1,step,0,metrics,history,manifest)
            position_start=0
            log('EPOCH COMPLETED '+json.dumps(candidate))
            gc.collect(); torch.cuda.empty_cache()
    save(output/'completed.json',{'epochs':CONFIG['epochs'],'updates':step,'continued_from_previous_stage_b':True})
    log('TRAJECTORY STAGE B TRAINING COMPLETED')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['smoke','freeze','train']); args=ap.parse_args()
    if args.action=='freeze': freeze_manifest()
    else: globals()[args.action]()
