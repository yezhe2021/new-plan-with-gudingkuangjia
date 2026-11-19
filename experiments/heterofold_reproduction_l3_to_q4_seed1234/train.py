import gc
import random
import torch
from torch import nn
from common import CFG,load_model,store,save
from collect_statistics import paired
from layer_alignment import features
from fold_mapper import Correction,apply_map,fold_correction
from calibration import objectives


def load_maps(path): return torch.load(path,map_location='cpu',weights_only=True)


def evaluate_loss(prepared,maps,corrections,st,rt,split='heldout',training=False,optimizer=None,branches='kv'):
    rows=list(prepared[split]); totals=[]
    if training: random.shuffle(rows)
    for number,row in enumerate(rows,1):
        proto,source,native,receiver=paired(row,st,rt,probes=True,keep_receiver=True)
        total=0.
        # Accumulate examples equally, not tokens/corpus lengths.
        for layer in range(36):
            k=apply_map(maps['k'][layer],features(source['k'],layer,36),'cuda').detach()
            v=apply_map(maps['v'][layer],features(source['v'],layer,36),'cuda').detach()
            kc,vc=corrections['k'][layer].cuda(),corrections['v'][layer].cuda()
            context=torch.enable_grad() if training else torch.no_grad()
            with context:
                lk,lv,_=objectives(receiver,layer,native,proto['target'],k,v,kc,vc,CFG['epsilon'])
                loss=(lk+lv if branches=='kv' else lk)/36
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite calibration loss')
                if training: (loss/CFG['batch_examples']).backward()
                total+=float(loss.detach())
        totals.append(total)
        if training and (number%CFG['batch_examples']==0 or number==len(rows)):
            remainder=number%CFG['batch_examples']
            if remainder:
                for p in corrections.parameters():
                    if p.grad is not None: p.grad.mul_(CFG['batch_examples']/remainder)
            norm=nn.utils.clip_grad_norm_(corrections.parameters(),CFG['clip'])
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite correction gradient')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in receiver.parameters())
        print('CALIBRATION',split,number,len(rows),total,flush=True)
        del receiver,source,native,k,v; gc.collect(); torch.cuda.empty_cache()
    return sum(totals)/len(totals)


def train_output_aware_calibration(prepared,maps,st,rt,folder,branches='kv'):
    corr=nn.ModuleDict({role:nn.ModuleList([Correction(1024,CFG['rank']).cuda() for _ in range(36)]) for role in ('k','v')})
    if branches=='k': corr['v'].requires_grad_(False)
    opt=torch.optim.AdamW([p for p in corr.parameters() if p.requires_grad],lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    best=float('inf'); history=[]; start=1
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',weights_only=True); corr.load_state_dict(last['state']);opt.load_state_dict(last['optimizer'])
        start=last['epoch']+1; best=last['best']; history=last['history']
    for epoch in range(start,CFG['epochs']+1):
        print('CALIBRATION EPOCH START',branches,epoch,CFG['epochs'],flush=True)
        opt.zero_grad(set_to_none=True)
        random.seed(CFG['seed']+epoch)
        loss=evaluate_loss(prepared,maps,corr,st,rt,'train',True,opt,branches)
        val=evaluate_loss(prepared,maps,corr,st,rt,branches=branches)
        history.append({'epoch':epoch,'train':loss,'heldout':val})
        state={k:v.detach().cpu() for k,v in corr.state_dict().items()}
        if val<best: best=val;store(folder/'best_correction.pt',state)
        store(folder/'last.pt',{'state':state,'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'history':history})
        save(folder/'history.json',history)
        print('CALIBRATION EPOCH COMPLETED',branches,epoch,'train',loss,'heldout',val,flush=True)
    corr.load_state_dict(torch.load(folder/'best_correction.pt',map_location='cpu',weights_only=True))
    folded={r:[fold_correction(maps[r][l],corr[r][l]) for l in range(36)] for r in ('k','v')}
    if branches=='k': folded['v']=maps['v']
    store(folder/'folded.pt',folded)
    save(folder/'completed.json',{'epochs':CFG['epochs'],'branches':branches,'best_heldout_loss':best,'inference':'folded affine only'})
