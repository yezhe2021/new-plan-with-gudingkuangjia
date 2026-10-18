"""One controlled continuation: replace half reference draws by live student prefixes."""
import argparse
from collections import defaultdict
import json
import random
import time

import torch

from followup_common import (CONFIG,HERE,TRAJ,parent,base_train,FullSyncHFLM,model_bundle,gpu_check,
                            log,save,read_rows,verify_manifest,freeze_manifest,digest)
from student_kd import rollout,objective as student_objective,reference_partition

# Same frozen helper implementation, with an explicit child checkpoint protocol.
# No parent source, manifest or tensor is changed.
base_train.CONFIG=CONFIG


@torch.no_grad()
def validate(receiver,sender,base,adapter,source_tok,hf,anchors):
    refs=read_rows(HERE/'prepared/validation_trajectories.jsonl')
    ref=base_train.validate(receiver,sender,base,adapter,source_tok,hf,refs,anchors)
    values=[]
    for row in read_rows(HERE/'prepared/validation_student_prefixes.jsonl'):
        item,bk,bv=base_train.prefix(sender,receiver,base,source_tok,hf,row,anchors)
        _,m=student_objective(receiver,adapter,row,item,bk,bv)
        values.append(m['token_kl']);del item,bk,bv
    op=sum(values)/len(values)
    return {'total':.5*ref['total']+.5*op,'reference':ref,'fixed_student_prefix_kl':op,'fixed_prefix_n':len(values)}


def smoke():
    parent.verify_manifest();gpu_check()
    receiver,sender,base,adapter,source_tok,hf=model_bundle(True)
    gen=FullSyncHFLM(receiver,'translated',sender,base,adapter)
    refs=read_rows(HERE/'prepared/train_trajectories.jsonl');row=max(refs,key=lambda r:len(r['continuation_ids']))
    anchors={};opt=base_train.optimizer_for(adapter);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    before=next(adapter.parameters()).detach().clone();torch.cuda.reset_peak_memory_stats()
    item,bk,bv=base_train.prefix(sender,receiver,base,source_tok,hf,row,anchors)
    sampled,text=rollout(gen,adapter,row,item,bk,bv,128)
    # Student-generated IDs condition the distributions; they are never CE labels.
    loss,op=student_objective(receiver,adapter,dict(row,continuation_ids=sampled),item,bk,bv)
    scaler.scale(loss).backward();del loss
    assert all(p.grad is None for m in (receiver,sender,base) for p in m.parameters())
    op_norm=base_train.update(adapter,opt,scaler)
    ref_loss,ref=base_train.objective(receiver,adapter,row,item,bk,bv)
    scaler.scale(ref_loss).backward();del ref_loss,item,bk,bv
    ref_norm=base_train.update(adapter,opt,scaler)
    replay=read_rows(parent.PREVIOUS/'prepared/train_epoch_1.jsonl')[0]
    item,bk,bv,pairs=parent.previous_train.frozen_prefix(sender,receiver,base,source_tok,hf,replay,anchors)
    mcq=parent.previous_train.objective(receiver,hf,adapter,replay,item,bk,bv,pairs,1,scaler)
    mcq_norm=base_train.update(adapter,opt,scaler)
    assert all(p.grad is None for m in (receiver,sender,base) for p in m.parameters())
    delta=float((next(adapter.parameters()).detach()-before).abs().max())
    assert op_norm>0 and ref_norm>0 and mcq_norm>0 and delta>0
    save(HERE/'reports/smoke.json',{'passed':True,'SMOKE_ONLY_ADAPTER_DISCARDED':True,'student':op,'reference':ref,'mcq':mcq,
                                 'student_grad_norm':op_norm,'reference_grad_norm':ref_norm,'mcq_grad_norm':mcq_norm,
                                 'parameter_max_change':delta,'frozen_parameter_gradients':False,
                                 'sampled_tokens':len(sampled),'sampled_response':text,
                                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    log('STUDENT PREFIX + TRUSTED REFERENCE + MCQ BACKWARD SMOKE PASSED')


def train():
    m=verify_manifest();gpu_check()
    receiver,sender,base,adapter,source_tok,hf=model_bundle(True)
    gen=FullSyncHFLM(receiver,'translated',sender,base,adapter)
    opt=base_train.optimizer_for(adapter);scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    refs=read_rows(HERE/'prepared/train_trajectories.jsonl')
    replay=read_rows(parent.PREVIOUS/'prepared/train_epoch_1.jsonl')+read_rows(parent.PREVIOUS/'prepared/train_epoch_2.jsonl')
    output=HERE/'stage_b';output.mkdir(exist_ok=True)
    epoch_start=1;position_start=0;step=0;history=[];anchors={}
    if (output/'last.pt').exists():
        p=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        assert p['protocol']==CONFIG['protocol'] and p['initial_stage_b_sha256']==m['identities']['initial_stage_b']['sha256']
        adapter.load_state_dict(p['state']);opt.load_state_dict(p['optimizer']);scaler.load_state_dict(p['scaler'])
        epoch_start=p['epoch'];position_start=p['position'];step=p['step'];history=p['history']
    if not history:
        adapter.eval();metrics=validate(receiver,sender,base,adapter,source_tok,hf,anchors)
        initial={'epoch':0,'step':0,'validation':metrics};history=[initial]
        base_train.checkpoint(output/'best.pt',adapter,None,None,0,0,0,metrics,history,m)
        save(output/'selection.json',{'selection':CONFIG['selection'],'best':initial,'epochs':history})
        log('INITIAL VALIDATION '+json.dumps(initial))
    with (output/'training_steps.jsonl').open('a') as stream, (output/'student_rollouts.jsonl').open('a',encoding='utf-8') as rollout_log:
        for epoch in range(epoch_start,CONFIG['epochs']+1):
            rng=random.Random(CONFIG['seed']+epoch)
            jobs=reference_partition(refs,epoch)+[('mcq',r) for r in rng.sample(replay,CONFIG['mcq_replay_per_epoch'])]
            rng.shuffle(jobs);adapter.train();opt.zero_grad(set_to_none=True)
            accumulated=0;sums=defaultdict(float);counts=defaultdict(int)
            log(f'START STUDENT-PREFIX TRAIN epoch={epoch}/2; draws={len(jobs)}; cap={CONFIG["student_rollout_tokens_by_epoch"][epoch-1]}')
            for position,(kind,row) in enumerate(jobs,1):
                if epoch==epoch_start and position<=position_start:continue
                divisor=min(CONFIG['effective_batch'],len(jobs)-((position-1)//CONFIG['effective_batch'])*CONFIG['effective_batch'])
                if kind=='mcq':
                    item,bk,bv,pairs=parent.previous_train.frozen_prefix(sender,receiver,base,source_tok,hf,row,anchors)
                    values=parent.previous_train.objective(receiver,hf,adapter,row,item,bk,bv,pairs,divisor,scaler)
                else:
                    item,bk,bv=base_train.prefix(sender,receiver,base,source_tok,hf,row,anchors)
                    if kind=='student':
                        ids,text=rollout(gen,adapter,row,item,bk,bv,CONFIG['student_rollout_tokens_by_epoch'][epoch-1])
                        sample={'epoch':epoch,'position':position,'optimizer_updates_before_sample':step,'base_id':row['base_id'],
                                'continuation_ids':ids,'response':text,'sampling_gradients':False,'hard_target_ce':False}
                        rollout_log.write(json.dumps(sample,ensure_ascii=False)+'\n');rollout_log.flush()
                        loss,values=student_objective(receiver,adapter,dict(row,continuation_ids=ids),item,bk,bv)
                    else:loss,values=base_train.objective(receiver,adapter,row,item,bk,bv)
                    scaler.scale(loss/divisor).backward();del loss
                del item,bk,bv
                counts[kind]+=1;accumulated+=1
                for k,v in values.items():sums[kind+'_'+k]+=v
                if accumulated==divisor:
                    norm=base_train.update(adapter,opt,scaler);step+=1
                    entry={'epoch':epoch,'step':step,'position':position,'pre_clip_grad_norm':norm,'kinds':dict(counts),
                           **{k:v/counts[k.split('_')[0]] for k,v in sums.items()}}
                    stream.write(json.dumps(entry)+'\n');stream.flush()
                    save(HERE/'status.json',{'phase':'training','epoch':epoch,'epochs':2,'step':step,'position':position,'epoch_draws':len(jobs)})
                    if step==1 or step%4==0:log('TRAIN '+json.dumps(entry))
                    accumulated=0;sums=defaultdict(float);counts=defaultdict(int)
                    if step==1 or step%CONFIG['checkpoint_interval_updates']==0:
                        base_train.checkpoint(output/'last.pt',adapter,opt,scaler,epoch,step,position,{},history,m)
            adapter.eval();metrics=validate(receiver,sender,base,adapter,source_tok,hf,anchors)
            candidate={'epoch':epoch,'step':step,'validation':metrics};history=[h for h in history if h['epoch']!=epoch]+[candidate]
            best=min(history,key=lambda h:(h['validation']['total'],h['epoch']))
            if best is candidate:base_train.checkpoint(output/'best.pt',adapter,None,None,epoch,step,len(jobs),metrics,history,m)
            save(output/'selection.json',{'selection':CONFIG['selection'],'best':best,'epochs':history})
            base_train.checkpoint(output/'last.pt',adapter,opt,scaler,epoch+1,step,0,metrics,history,m)
            position_start=0;log('EPOCH COMPLETED '+json.dumps(candidate))
    save(output/'completed.json',{'epochs':2,'updates':step,'on_policy':True,'initial_stage_b_sha256':m['identities']['initial_stage_b']['sha256']})
    log('STUDENT-PREFIX STAGE B TRAINING COMPLETED')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['smoke','freeze','train']);args=ap.parse_args()
    if args.action=='freeze':freeze_manifest()
    else:globals()[args.action]()
