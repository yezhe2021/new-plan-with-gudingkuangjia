import collections
import json
import random
import torch
from runtime import CFG,ROOT,save,seed,load_models,tokenizers,base,adapter,checkpoint
from kv_backend import PaperCache,mapped
from attention_probe import collect,losses
from evaluate import run_eval


@torch.no_grad()
def validate(mapper,calibrator,receiver,cache,rows):
    papers=[]
    for row in rows:
        item=cache.get(row); bk,bv=mapped(mapper,item); qs=[]
        for q in item['protocol']['queries']:
            probe=collect(receiver,item,q); metrics=collections.Counter()
            for layer in range(36):
                lk,lv,stats=losses(receiver,item,probe,layer,bk[layer],bv[layer],calibrator)
                metrics['total']+=float(lk+lv)/36
                for k,v in stats.items(): metrics[k]+=v/36
            qs.append(metrics)
        papers.append({k:sum(q[k] for q in qs)/len(qs) for k in qs[0]})
    return {k:sum(p[k] for p in papers)/len(papers) for k in papers[0]}


def train():
    seed(); prepared=json.loads((ROOT/'prepared.json').read_text())
    sender,receiver=load_models(); st,rt=tokenizers(); cache=PaperCache(sender,receiver,st,rt)
    mapper=base(ROOT/'runs/stage_a_fresh/best.pt').eval().requires_grad_(False)
    corpus_query_mean=sum(len(p['train_questions']) for p in prepared['train'])/len(prepared['train'])
    calibrator=adapter(); folder=ROOT/'runs/stage_b'; folder.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(calibrator.parameters(),lr=CFG['stage_b_lr'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    epoch_start,step,best=1,0,float('inf'); history=[]
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True); assert last['protocol']==CFG['protocol']
        calibrator.load_state_dict(last['state']); optimizer.load_state_dict(last['optimizer']); scaler.load_state_dict(last['scaler'])
        epoch_start,step=last['epoch']+1,last['step']; history=json.loads((folder/'selection.json').read_text())
        best=min(h['validation']['total'] for h in history)
    for epoch in range(epoch_start,CFG['stage_b_epochs']+1):
        calibrator.train(); order=list(prepared['train']); random.Random(CFG['seed']+epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); accumulated,weight,total=0,0.,0.; paper_index=0
        def update():
            nonlocal accumulated,weight,total,step
            scaler.unscale_(optimizer)
            for param in calibrator.parameters():
                if param.grad is not None: param.grad.div_(accumulated)
            norm=torch.nn.utils.clip_grad_norm_(calibrator.parameters(),CFG['stage_b_clip'])
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite Stage B gradient')
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); step+=1
            record={'epoch':epoch,'step':step,'weighted_loss':total/accumulated,'queries':accumulated,
                    'paper_normalization_weight':weight,'gradient_norm':float(norm)}
            with (folder/'steps.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(f"STAGE B epoch={epoch}/{CFG['stage_b_epochs']} step={step} paper={paper_index}/{len(prepared['train'])} loss={total/accumulated:.6f}",flush=True)
            accumulated,weight,total=0,0.,0.
        for paper_index,row in enumerate(order,1):
            item=cache.get(row); bk,bv=mapped(mapper,item)
            wanted={q['question_id'] for q in row['train_questions']}
            queries=[q for q in item['protocol']['queries'] if q['question_id'] in wanted]
            assert queries
            for q in queries:
                probe=collect(receiver,item,q); w=corpus_query_mean/len(queries)
                for layer in range(36):
                    lk,lv,_=losses(receiver,item,probe,layer,bk[layer],bv[layer],calibrator)
                    loss=(lk+lv)*w/36
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite Attention Calibration')
                    total+=float(loss.detach()); scaler.scale(loss).backward()
                accumulated+=1; weight+=w
                if accumulated==CFG['stage_b_batch_queries']: update()
            del bk,bv
        if accumulated: update()
        calibrator.eval(); metric=validate(mapper,calibrator,receiver,cache,prepared['dev'])
        answers=run_eval(cache,receiver,rt,prepared['dev'],{'stage_a':(mapper,None),'stage_b':(mapper,calibrator)},folder/f'dev_epoch_{epoch}')
        history.append({'epoch':epoch,'step':step,'validation':metric,'answer_metrics':answers})
        if metric['total']<best:
            best=metric['total']; checkpoint(folder/'best.pt',calibrator,epoch,step,metric)
        save(folder/'selection.json',history); checkpoint(folder/'last.pt',calibrator,epoch,step,metric,optimizer,scaler)
        assert all(p.grad is None for p in mapper.parameters())
        assert all(p.grad is None for p in receiver.parameters()) and all(p.grad is None for p in sender.parameters())
        print('STAGE B EPOCH COMPLETED',epoch,metric,flush=True)
    save(folder/'completed.json',{'epochs':CFG['stage_b_epochs'],'steps':step,
        'selection':'lowest paper-normalized Dev Attention Functional Loss only',
        'supervision':'Native query probes only, no Gold answer or teacher-forced trajectory'})


if __name__=='__main__': train()
