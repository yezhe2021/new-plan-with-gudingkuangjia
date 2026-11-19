"""Fresh single-source baselines. Only one training/calibration stage runs at once."""
import argparse
import json
import random
from pathlib import Path
import time
import torch
from common import (ROOT, CFG, rows, read, save, append, log, store, scratch, publish,
                    seed, verify, signature, release, digest)
from models import Pool, activate, offload, generate, backbone
from algorithms import (Engine, our_model, our_loss, our_map, residual, token_kl,
    hf_features, exact_end_pairs, hf_objective, LayerCorrection, fold)
from vendor.heterofold.recolor import fit_affine
from vendor.heterofold.maps import materialize


def folder(family, stage):
    p = ROOT / 'training' / family / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def batches(values):
    size = CFG['capture_batch_questions']
    return [values[i:i+size] for i in range(0,len(values),size)]


def context(family, stage):
    return {'family':family,'stage':stage,'manifest_sha':signature(verify())}


def resume_state(identity):
    p = scratch() / 'resume_latest.pt'
    if not p.exists():
        return None
    value = torch.load(p,map_location='cpu',weights_only=True)
    return value if value.get('identity') == identity else None


def save_state(identity, model, opt, epoch, batch, step, history):
    store(scratch()/'resume_latest.pt', {'identity':identity,'state':model.state_dict(),
        'optimizer':opt.state_dict(),'epoch':epoch,'batch':batch,'step':step,'history':history})


def save_best(identity, model, entry):
    store(scratch()/'best.pt',{'identity':identity,'state':model.state_dict(),'selection':entry})


def select(history, candidate, model, identity):
    history = [x for x in history if x['epoch']!=candidate['epoch']] + [candidate]
    best = min(history,key=lambda x:(x['validation_loss'],x['epoch']))
    if best is candidate:
        save_best(identity,model,candidate)
    return history,best


def our_a(family):
    out = folder(family,'our_a')
    if (out/'completed.json').exists():return
    identity = context(family,'our_a');cfg=CFG['our_stage_a'];pool=Pool();engine=Engine(pool)
    learner=our_model(family).cuda();opt=torch.optim.AdamW(learner.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    offload(learner,opt)
    train=rows(ROOT/'prepared/train.jsonl');val=rows(ROOT/'prepared/validation.jsonl')
    state=resume_state(identity);start_epoch=1;start_batch=0;step=0;history=[]
    if state:
        learner.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
        start_epoch,start_batch,step,history=state['epoch'],state['batch'],state['step'],state['history']
        del state;release()
    @torch.no_grad()
    def validation():
        values=[]
        for group in batches(val):
            pairs=engine.pairs(group,family);pool.cpu();activate(learner)
            values += [our_loss(learner,p) for p in pairs]
            offload(learner);del pairs
        return sum(values)/len(values)
    if not history:
        v=validation();history=[{'epoch':0,'validation_loss':v,'step':0}];save_best(identity,learner,history[0])
        save(out/'selection.json',{'best':history[0],'epochs':history})
    for epoch in range(start_epoch,cfg['epochs']+1):
        order=list(train);random.Random(CFG['seed']+epoch).shuffle(order)
        jobs=batches(order);accumulated=0;opt.zero_grad(set_to_none=True)
        for bi,group in enumerate(jobs,1):
            if epoch==start_epoch and bi<=start_batch:continue
            pairs=engine.pairs(group,family);pool.cpu();activate(learner,opt);learner.train()
            losses=[]
            for pair in pairs:
                losses.append(our_loss(learner,pair,True,cfg['effective_batch']));accumulated+=1
                if accumulated==cfg['effective_batch']:
                    norm=torch.nn.utils.clip_grad_norm_(learner.parameters(),cfg['gradient_clip']);assert torch.isfinite(norm)
                    opt.step();opt.zero_grad(set_to_none=True);step+=1;accumulated=0
            offload(learner,opt);del pairs
            entry={'epoch':epoch,'batch':bi,'batches':len(jobs),'step':step,'loss':sum(losses)/len(losses)}
            append(out/'steps.jsonl',entry);save(ROOT/'status.json',{'phase':'training','family':family,'stage':'our_a',**entry})
            if bi==1 or bi%4==0:log('OUR A '+json.dumps(entry))
            if bi%CFG['checkpoint_interval_batches']==0:
                assert accumulated==0
                save_state(identity,learner,opt,epoch,bi,step,history)
        assert accumulated==0
        learner.eval();value=validation();candidate={'epoch':epoch,'validation_loss':value,'step':step}
        history,best=select(history,candidate,learner,identity)
        save(out/'selection.json',{'best':best,'epochs':history});save_state(identity,learner,opt,epoch+1,0,step,history)
        start_batch=0;log('OUR A EPOCH '+json.dumps(candidate))
    selected=publish(scratch()/'best.pt',f'our_{family}_stage_a.pt',True)
    save(out/'completed.json',{'completed':True,'checkpoint':selected,'selection':read(out/'selection.json'),'updates':step})


def load_a(family):
    info=read(folder(family,'our_a')/'completed.json')['checkpoint']
    assert digest(Path(info['path']))==info['sha256'],'Stage A checkpoint identity changed'
    model=our_model(family).eval().requires_grad_(False)
    p=torch.load(info['path'],map_location='cpu',weights_only=True);model.load_state_dict(p['state']);del p;release()
    return model


def our_b(family):
    out=folder(family,'our_b')
    if (out/'completed.json').exists():return
    identity=context(family,'our_b');cfg=CFG['our_stage_b'];pool=Pool();engine=Engine(pool)
    base=load_a(family);adapter=residual().cuda();opt=torch.optim.AdamW(adapter.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    state=resume_state(identity);start_epoch=1;start_batch=0;step=0;history=[]
    if state:
        adapter.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
        start_epoch,start_batch,step,history=state['epoch'],state['batch'],state['step'],state['history'];del state
    teacher_path=ROOT/'training/teacher_trajectories.jsonl'
    teachers={r['prompt_sha256']:r for r in rows(teacher_path)} if teacher_path.exists() else {}
    def teacher_ids(receiver,pair):
        key=signature(pair['proto']['target']['ids'])
        if key not in teachers:
            ids,stop=generate(receiver,pair['proto']['target']['ids'],limit=cfg['teacher_max_new_tokens'])
            item={'id':pair['row']['id'],'prompt_sha256':key,'ids':ids,'stop':stop,
                  'response':pool.tokenizers['qwen'].decode(ids,skip_special_tokens=True),'gold_answer_filtering':False}
            append(teacher_path,item);teachers[key]=item
        return teachers[key]['ids']
    def prepared(group):
        pairs=engine.pairs(group,family,('joint',));pool.cpu();activate(base)
        mapped=[our_map(base,p) for p in pairs];offload(base)
        receiver=pool.gpu('qwen')
        return pairs,mapped,receiver
    @torch.no_grad()
    def validation():
        result=[]
        for group in batches(rows(ROOT/'prepared/validation.jsonl')):
            pairs,mapped,receiver=prepared(group)
            for pair,(k,v) in zip(pairs,mapped):
                result.append(float(token_kl(receiver,adapter,pair,k,v,teacher_ids(receiver,pair))))
            del receiver;pool.cpu();del pairs,mapped
        return sum(result)/len(result)
    if not history:
        value=validation();history=[{'epoch':0,'validation_loss':value,'step':0}];offload(adapter,opt);save_best(identity,adapter,history[0]);activate(adapter,opt)
        save(out/'selection.json',{'best':history[0],'epochs':history})
    for epoch in range(start_epoch,cfg['epochs']+1):
        order=rows(ROOT/'prepared/train.jsonl');random.Random(CFG['seed']+epoch).shuffle(order)
        jobs=batches(order);adapter.train();opt.zero_grad(set_to_none=True)
        for bi,group in enumerate(jobs,1):
            if epoch==start_epoch and bi<=start_batch:continue
            pairs,mapped,receiver=prepared(group);losses=[]
            assert len(pairs)<=cfg['effective_batch']
            for pair,(k,v) in zip(pairs,mapped):
                loss=token_kl(receiver,adapter,pair,k,v,teacher_ids(receiver,pair));(loss/len(pairs)).backward()
                losses.append(float(loss.detach()));del loss
            assert all(p.grad is None for p in receiver.parameters())
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg['gradient_clip']);assert torch.isfinite(norm)
            opt.step();opt.zero_grad(set_to_none=True);step+=1
            del receiver;pool.cpu();del pairs,mapped
            entry={'epoch':epoch,'batch':bi,'batches':len(jobs),'step':step,'loss':sum(losses)/len(losses),'pre_clip_grad_norm':float(norm)}
            append(out/'steps.jsonl',entry);save(ROOT/'status.json',{'phase':'training','family':family,'stage':'our_b',**entry})
            if bi==1 or bi%4==0:log('OUR B '+json.dumps(entry))
            if bi%CFG['checkpoint_interval_batches']==0:
                offload(adapter,opt);save_state(identity,adapter,opt,epoch,bi,step,history);activate(adapter,opt)
        adapter.eval();value=validation();candidate={'epoch':epoch,'step':step,'validation_loss':value}
        offload(adapter,opt);history,best=select(history,candidate,adapter,identity)
        save(out/'selection.json',{'best':best,'epochs':history});save_state(identity,adapter,opt,epoch+1,0,step,history);activate(adapter,opt)
        start_batch=0;log('OUR B EPOCH '+json.dumps(candidate))
    selected=publish(scratch()/'best.pt',f'our_{family}_stage_b.pt')
    save(out/'completed.json',{'completed':True,'checkpoint':selected,'selection':read(out/'selection.json'),'updates':step,
                             'gold_answer_supervision':False,'loss':'full_vocab_teacher_KL_only'})


def blank_stats():
    return {role:[{'sX':torch.zeros(3072,dtype=torch.float64),'sY':torch.zeros(1024,dtype=torch.float64),
             'XtX':torch.zeros(3072,3072,dtype=torch.float64),'XtY':torch.zeros(3072,1024,dtype=torch.float64),
             'YtY':torch.zeros(1024,1024,dtype=torch.float64)} for _ in range(36)] for role in 'kv'}


def hf_fit(family):
    out=folder(family,'hf_fit')
    if (out/'completed.json').exists() and read(out/'completed.json').get('scratch_map_exists') and (scratch()/'recolor.pt').exists():
        p=torch.load(scratch()/'recolor.pt',map_location='cpu',weights_only=True)
        if p.get('family')==family:return
    identity=context(family,'hf_fit');pool=Pool();engine=Engine(pool);stats=None;start=0;count=0
    path=scratch()/'statistics_latest.pt'
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['identity']==identity:
            stats=state['stats'];start=state['batch'];count=state['count']
        del state
    if stats is None:stats=blank_stats()
    jobs=batches(rows(ROOT/'prepared/train.jsonl'))
    for bi,group in enumerate(jobs,1):
        if bi<=start:continue
        pairs=engine.pairs(group,family);pool.cpu()
        indices=[]
        for pair in pairs:
            anchors=exact_end_pairs(pair['proto']['source'][family]['spans'],pair['proto']['target']['spans'])
            indices.append(([a for a,b in anchors],[b for a,b in anchors]))
        for role in 'kv':
            for layer in range(36):
                moment={k:v.cuda() for k,v in stats[role][layer].items()}
                for pair,(si,ti) in zip(pairs,indices):
                    x=hf_features(pair,layer,role,si).cuda().double()
                    target=pair['target']['raw_k' if role=='k' else 'v'][layer,pair['proto']['target']['begin']+torch.tensor(ti)].flatten(1)
                    y=target.cuda().double()
                    moment['sX'].add_(x.sum(0));moment['sY'].add_(y.sum(0));moment['XtX'].add_(x.T@x)
                    moment['XtY'].add_(x.T@y);moment['YtY'].add_(y.T@y)
                stats[role][layer]={k:v.cpu() for k,v in moment.items()};del moment,x,y
        count+=sum(len(si) for si,ti in indices);del pairs
        if bi%CFG['checkpoint_interval_batches']==0 or bi==len(jobs):
            store(path,{'identity':identity,'batch':bi,'count':count,'stats':stats})
        entry={'phase':'recolor_statistics','family':family,'batch':bi,'batches':len(jobs),'exact_anchor_pairs':count}
        save(ROOT/'status.json',entry)
        if bi==1 or bi%4==0:log(json.dumps(entry))
    mapping={'family':family,'Ls':28 if family=='llama' else 34,'Lr':36,'kv_width':1024,
             'depth_stencil':{'span':1,'stride':4,'taps_k':__import__('algorithms').stencil_taps(28 if family=='llama' else 34,36),
                              'taps_v':__import__('algorithms').stencil_taps(28 if family=='llama' else 34,36)},
             'source_space':'kv_prerope_joint','target_space':'kv_pre_norm','map':'kv_depth_stencil_recolor',
             'protocol':CFG['protocol'],'native_sender':True,'receiver_context':'native_chat_evidence_only'}
    for role in 'kv':
        aa,bb,rr=[],[],[]
        for layer in range(36):
            q=scratch()/f'recolor_layer_{role}_{layer}.pt'
            value=torch.load(q,map_location='cpu',weights_only=True) if q.exists() else None
            if value is None or value['identity']!=identity:
                t=time.monotonic();s=stats[role][layer]
                a,b,r=fit_affine(s['sX'],s['sY'],s['XtX'],s['XtY'],s['YtY'],count,device='cpu')
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                value={'identity':identity,'A':a.float(),'b':b.float(),'rms':r};store(q,value)
                log(f'RECOLOR FIT {family} {role} layer={layer+1}/36 seconds={time.monotonic()-t:.1f}')
            aa.append(value['A']);bb.append(value['b']);rr.append(value['rms'])
            stats[role][layer]=None
        mapping[f'A_{role}'],mapping[f'b_{role}'],mapping[f'rms_{role}']=aa,bb,rr
    store(scratch()/'recolor.pt',mapping)
    save(out/'completed.json',{'completed':True,'scratch_map_exists':True,'family':family,'exact_anchor_pairs':count,
         'statistics_dtype':'float64','affine_dtype':'float32','eigenvalue_floor':'unchanged official fit_affine'})


def hf_calib(family):
    out=folder(family,'hf_calib')
    if (out/'completed.json').exists():return
    identity=context(family,'hf_calib');cfg=CFG['heterofold'];pool=Pool();engine=Engine(pool)
    mapping=torch.load(scratch()/'recolor.pt',map_location='cpu',weights_only=True);assert mapping['family']==family
    mapping=materialize(mapping,device='cuda',dtype=torch.float32)
    corrections=torch.nn.ModuleList([LayerCorrection(1024,cfg['rank']) for _ in range(36)]).cuda()
    opt=torch.optim.AdamW(corrections.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    state=resume_state(identity);start_epoch=1;start_batch=0;step=0;history=[]
    if state:
        corrections.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
        start_epoch,start_batch,step,history=state['epoch'],state['batch'],state['step'],state['history'];del state
    @torch.no_grad()
    def validation():
        values=[]
        for group in batches(rows(ROOT/'prepared/validation.jsonl')):
            pairs=engine.pairs(group,family);receiver=pool.gpu('qwen')
            values += [hf_objective(receiver,mapping,corrections,p)['loss'] for p in pairs]
            del receiver;pool.cpu();del pairs
        return sum(values)/len(values)
    if not history:
        value=validation();history=[{'epoch':0,'step':0,'validation_loss':value}];save_best(identity,corrections,history[0])
        save(out/'selection.json',{'best':history[0],'epochs':history})
    for epoch in range(start_epoch,cfg['epochs']+1):
        order=rows(ROOT/'prepared/train.jsonl');random.Random(CFG['seed']+epoch).shuffle(order)
        jobs=batches(order);accumulated=0;corrections.train();opt.zero_grad(set_to_none=True)
        for bi,group in enumerate(jobs,1):
            if epoch==start_epoch and bi<=start_batch:continue
            pairs=engine.pairs(group,family);receiver=pool.gpu('qwen');losses=[]
            for pair in pairs:
                m=hf_objective(receiver,mapping,corrections,pair,True,cfg['effective_batch']);losses.append(m['loss']);accumulated+=1
                if accumulated==cfg['effective_batch']:
                    norm=torch.nn.utils.clip_grad_norm_(corrections.parameters(),cfg['gradient_clip']);assert torch.isfinite(norm)
                    opt.step();opt.zero_grad(set_to_none=True);step+=1;accumulated=0
            assert all(p.grad is None for p in receiver.parameters())
            del receiver;pool.cpu();del pairs
            entry={'epoch':epoch,'batch':bi,'batches':len(jobs),'step':step,'loss':sum(losses)/len(losses)}
            append(out/'steps.jsonl',entry);save(ROOT/'status.json',{'phase':'training','family':family,'stage':'hf_calib',**entry})
            if bi==1 or bi%4==0:log('HF CALIB '+json.dumps(entry))
            if bi%CFG['checkpoint_interval_batches']==0:
                assert accumulated==0
                save_state(identity,corrections,opt,epoch,bi,step,history)
        assert accumulated==0
        corrections.eval();value=validation();candidate={'epoch':epoch,'step':step,'validation_loss':value}
        history,best=select(history,candidate,corrections,identity)
        save(out/'selection.json',{'best':best,'epochs':history});save_state(identity,corrections,opt,epoch+1,0,step,history)
        start_batch=0;log('HF CALIB EPOCH '+json.dumps(candidate))
    best=torch.load(scratch()/'best.pt',map_location='cpu',weights_only=True);corrections.load_state_dict(best['state'])
    cpu_map=materialize(mapping,device='cpu',dtype=torch.float32)
    result=fold(cpu_map,corrections,{'rank':cfg['rank'],'protocol':CFG['protocol'],'selection':read(out/'selection.json')})
    store(scratch()/'folded.pt',result)
    selected=publish(scratch()/'folded.pt',f'heterofold_{family}.pt')
    save(out/'completed.json',{'completed':True,'checkpoint':selected,'selection':read(out/'selection.json'),'updates':step,
                             'inference_residual_modules':0,'evidence_only_protocol_adaptation':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=['llama','gemma']);ap.add_argument('stage',choices=['our_a','our_b','hf_fit','hf_calib'])
    args=ap.parse_args();seed();globals()[args.stage](args.family)
