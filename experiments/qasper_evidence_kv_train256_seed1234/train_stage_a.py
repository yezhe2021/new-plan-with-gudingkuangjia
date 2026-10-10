import argparse
import json
import random
import torch
import torch.nn.functional as F
from runtime import CFG,ROOT,save,seed,load_models,tokenizers,base,checkpoint
from kv_backend import PaperCache,mapped
from evaluate import run_eval


def chunk_losses(model,item):
    count=item['tokens']
    den_k=item['target_k'].float().square().sum().clamp_min(1e-8).item()
    den_v=item['target_v'].float().square().sum().clamp_min(1e-8).item()
    for start in range(0,count,CFG['chunk_tokens']):
        stop=min(count,start+CFG['chunk_tokens'])
        with torch.amp.autocast('cuda',dtype=torch.float16):
            pk,pv=model(item['source_k'][:,start:stop][None].cuda(),item['source_v'][:,start:stop][None].cuda())
        tk,tv=item['target_k'][:,start:stop][None].cuda().float(),item['target_v'][:,start:stop][None].cuda().float()
        loss=(pk.float()-tk).square().sum()/den_k+(pv.float()-tv).square().sum()/den_v
        loss+=(stop-start)/count*(2-F.cosine_similarity(pk.float(),tk,-1).mean()-F.cosine_similarity(pv.float(),tv,-1).mean())
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite Stage A loss')
        yield loss


@torch.no_grad()
def validate(model,cache,rows):
    metrics=[]
    for row in rows:
        item=cache.get(row); k,v=mapped(model,item); m={}
        for name,p,t in [('k',k,item['target_k']),('v',v,item['target_v'])]:
            p,t=p.float(),t.cuda().float()
            m[name+'_nmse']=float((p-t).square().mean()/t.square().mean().clamp_min(1e-8))
            m[name+'_cosine']=float(F.cosine_similarity(p,t,-1).mean())
        m['total']=m['k_nmse']+m['v_nmse']+2-m['k_cosine']-m['v_cosine']; metrics.append(m)
    return {k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}


def train(initialization):
    seed(); prepared=json.loads((ROOT/'prepared.json').read_text())
    rows,dev=prepared['train'],prepared['dev']; sender,receiver=load_models(); st,rt=tokenizers()
    cache=PaperCache(sender,receiver,st,rt)
    old=initialization=='old_gsm8k'
    model=base(__import__('pathlib').Path(CFG['legacy'])/'runs/gsm8k/stage_a/best.pt' if old else None,old=old)
    folder=ROOT/f'runs/stage_a_{initialization}'; folder.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['stage_a_lr'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    start_epoch,step,best=1,0,float('inf'); history=[]
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True); assert last['protocol']==CFG['protocol']
        model.load_state_dict(last['state']); optimizer.load_state_dict(last['optimizer']); scaler.load_state_dict(last['scaler'])
        start_epoch,step=last['epoch']+1,last['step']
        history=json.loads((folder/'selection.json').read_text()); best=min(x['validation']['total'] for x in history)
    for epoch in range(start_epoch,CFG['stage_a_epochs']+1):
        model.train(); order=list(rows); random.Random(CFG['seed']+epoch).shuffle(order)
        for begin in range(0,len(order),CFG['stage_a_batch_papers']):
            batch=order[begin:begin+CFG['stage_a_batch_papers']]; optimizer.zero_grad(set_to_none=True); total=0.
            for row in batch:
                item=cache.get(row)
                for loss in chunk_losses(model,item):
                    total+=float(loss.detach())/len(batch); scaler.scale(loss/len(batch)).backward()
            scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['stage_a_clip'])
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite Stage A gradient')
            scaler.step(optimizer); scaler.update(); step+=1
            record={'epoch':epoch,'step':step,'loss':total,'gradient_norm':float(norm),'papers_seen':begin+len(batch)}
            with (folder/'steps.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(f"STAGE A {initialization} epoch={epoch}/{CFG['stage_a_epochs']} step={step} papers={begin+len(batch)}/{len(rows)} loss={total:.5f}",flush=True)
        model.eval(); metric=validate(model,cache,dev)
        answers=run_eval(cache,receiver,rt,dev,{'stage_a':(model,None)},folder/f'dev_epoch_{epoch}')
        history.append({'epoch':epoch,'step':step,'validation':metric,'answer_metrics':answers})
        if metric['total']<best:
            best=metric['total']; checkpoint(folder/'best.pt',model,epoch,step,metric)
        save(folder/'selection.json',history)
        checkpoint(folder/'last.pt',model,epoch,step,metric,optimizer,scaler)
        print('STAGE A EPOCH COMPLETED',initialization,epoch,metric,flush=True)
    save(folder/'completed.json',{'epochs':CFG['stage_a_epochs'],'steps':step,'initialization':initialization,
        'selection':'lowest Dev representation loss, not Test or answer F1'})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--initialization',choices=('fresh','old_gsm8k'),default='fresh')
    train(p.parse_args().initialization)
