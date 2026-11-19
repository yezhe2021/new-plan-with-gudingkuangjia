import argparse
import json
import math
import random
import torch
import torch.nn.functional as F
from common_exp import ROOT, CONFIG, CFG, save, prepared, setup, old_base, checkpoint, load_state, seed
from cross_role import RoleResidual, AugmentedMapper
from train_stage_a import validate


def train(mode):
    folder=ROOT/f'runs/A_{mode}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists(): return
    seed(); sender,receiver,rt,cache=setup(); data=prepared()
    frozen=old_base(); residual=RoleResidual(mode,CONFIG['cross_role_rank']).cuda()
    model=AugmentedMapper(frozen,residual)
    optimizer=torch.optim.AdamW(residual.parameters(),lr=CONFIG['stage_a_lr'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    start,best,history=0,float('inf'),[]
    if (folder/'last.pt').exists():
        last=load_state(residual,folder/'last.pt'); optimizer.load_state_dict(last['optimizer']); scaler.load_state_dict(last['scaler'])
        start=last['step']; history=json.loads((folder/'selection.json').read_text()); best=min(r['validation']['total'] for r in history)
        del last
    batch_size=CONFIG['stage_a_batch_papers']; batches=math.ceil(len(data['train'])/batch_size)
    for step0 in range(start,CONFIG['stage_a_steps']):
        epoch=step0//batches+1; index=step0%batches
        order=list(data['train']); random.Random(CONFIG['seed']+epoch).shuffle(order)
        rows=order[index*batch_size:(index+1)*batch_size]
        residual.train(); optimizer.zero_grad(set_to_none=True); total=0.
        for row in rows:
            item=cache.get(row); count=item['tokens']
            den=[item[k].float().square().sum().clamp_min(1e-8).item() for k in ('target_k','target_v')]
            for begin in range(0,count,CFG['chunk_tokens']):
                end=min(count,begin+CFG['chunk_tokens'])
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    k,v=model(item['source_k'][:,begin:end][None].cuda(),item['source_v'][:,begin:end][None].cuda())
                targets=[item[k][:,begin:end][None].cuda().float() for k in ('target_k','target_v')]
                loss=sum((p.float()-t).square().sum()/d+(end-begin)/count*(1-F.cosine_similarity(p.float(),t,-1).mean())
                    for p,t,d in zip((k,v),targets,den))/len(rows)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward(); total+=float(loss.detach())
        scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(residual.parameters(),30.)
        if not torch.isfinite(norm) or norm <= 0: raise RuntimeError('Missing/nonfinite A gradient')
        scaler.step(optimizer); scaler.update(); step=step0+1
        assert all(p.grad is None for p in frozen.parameters())
        assert all(p.grad is None for p in sender.parameters()) and all(p.grad is None for p in receiver.parameters())
        with (folder/'steps.jsonl').open('a') as f: f.write(json.dumps({'step':step,'epoch':epoch,'loss':total,'gradient_norm':float(norm)})+'\n')
        print(f'A_{mode} step={step}/{CONFIG["stage_a_steps"]} epoch={epoch} loss={total:.6f} gradient={float(norm):.5f}',flush=True)
        if step%CONFIG['validation_interval']==0 or step==CONFIG['stage_a_steps']:
            model.eval(); metric=validate(model,cache,data['validation'])
            history.append({'step':step,'validation':metric})
            if metric['total']<best:
                best=metric['total']; checkpoint(folder/'best.pt',residual,step,metric,branch=mode)
            save(folder/'selection.json',history)
            checkpoint(folder/'last.pt',residual,step,metric,optimizer,scaler,mode)
            print('A INTERNAL VALIDATION',mode,step,metric,flush=True)
    save(folder/'completed.json',{'step':CONFIG['stage_a_steps'],'new_parameters':sum(p.numel() for p in residual.parameters()),
        'frozen_base':True,'selection':'lowest unseen internal-train paper reconstruction loss; Dev/Test not used'})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('self','cross')); train(p.parse_args().mode)
