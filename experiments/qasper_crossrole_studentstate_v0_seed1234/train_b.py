import argparse
import collections
import json
import math
import random
import torch
from common_exp import ROOT, CONFIG, save, prepared, setup, old_base, checkpoint, load_state, seed, adapter, mapped, collect
from student_probe import collect_student, local_loss


@torch.no_grad()
def validate(mapper,calibrator,receiver,cache,rows,mode):
    papers=[]
    for row in rows:
        item=cache.get(row); bk,bv=mapped(mapper,item); queries=[]
        for query in item['protocol']['queries']:
            probe=collect(receiver,item,query) if mode=='native_matched' else collect_student(receiver,item,query,bk,bv,calibrator)
            stats=collections.Counter()
            for layer in range(36):
                loss,record=local_loss(receiver,item,probe,layer,bk[layer],bv[layer],calibrator)
                stats['total']+=float(loss)/36
                for key,val in record.items(): stats[key]+=val/36
            queries.append(stats)
        papers.append({k:sum(q[k] for q in queries)/len(queries) for k in queries[0]})
    return {k:sum(p[k] for p in papers)/len(papers) for k in papers[0]}


def train(mode):
    folder=ROOT/f'runs/B_{mode}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists(): return
    seed(); sender,receiver,rt,cache=setup(); data=prepared(); mapper=old_base()
    calibrator=adapter(); optimizer=torch.optim.AdamW(calibrator.parameters(),lr=CONFIG['stage_b_lr'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000000)
    nqueries=sum(len(p['train_questions']) for p in data['train']); mean=nqueries/len(data['train'])
    size=CONFIG['stage_b_batch_queries']; per_epoch=math.ceil(nqueries/size); steps=per_epoch*CONFIG['stage_b_epochs']
    done,step,best,history=0,0,float('inf'),[]
    if (folder/'last.pt').exists():
        last=load_state(calibrator,folder/'last.pt'); optimizer.load_state_dict(last['optimizer']); scaler.load_state_dict(last['scaler'])
        done=step=last['step']; history=json.loads((folder/'selection.json').read_text()); best=min(r['validation']['total'] for r in history)
        del last
    print(f'START B_{mode} fresh rank64; {steps} updates; frozen Stage-A512; detached query/suffix',flush=True)
    for epoch in range(done//per_epoch+1,CONFIG['stage_b_epochs']+1):
        papers=list(data['train']); random.Random(CONFIG['seed']+epoch).shuffle(papers)
        entries=[]
        for row in papers:
            entries.extend((row,q['question_id'],mean/len(row['train_questions'])) for q in row['train_questions'])
        first_batch=done%per_epoch if epoch==done//per_epoch+1 else 0
        previous_id=None; bk=bv=None
        for batch_index in range(first_batch,per_epoch):
            batch=entries[batch_index*size:(batch_index+1)*size]
            calibrator.train(); optimizer.zero_grad(set_to_none=True); total=0.
            for row,qid,weight in batch:
                item=cache.get(row)
                if row['paper_id']!=previous_id:
                    bk,bv=mapped(mapper,item); previous_id=row['paper_id']
                query=next(q for q in item['protocol']['queries'] if q['question_id']==qid)
                probe=collect(receiver,item,query) if mode=='native_matched' else collect_student(receiver,item,query,bk,bv,calibrator)
                for layer in range(36):
                    loss,_=local_loss(receiver,item,probe,layer,bk[layer],bv[layer],calibrator)
                    loss=loss*weight/(36*len(batch))
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite student-state calibration')
                    scaler.scale(loss).backward(); total+=float(loss.detach())
            scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(calibrator.parameters(),1.)
            if not torch.isfinite(norm) or norm<=0: raise RuntimeError('Missing/nonfinite B gradient')
            scaler.step(optimizer); scaler.update(); step= (epoch-1)*per_epoch+batch_index+1
            assert all(p.grad is None for p in mapper.parameters())
            assert all(p.grad is None for p in sender.parameters()) and all(p.grad is None for p in receiver.parameters())
            with (folder/'steps.jsonl').open('a') as f: f.write(json.dumps({'step':step,'epoch':epoch,'loss':total,'gradient_norm':float(norm)})+'\n')
            print(f'B_{mode} step={step}/{steps} epoch={epoch} loss={total:.6f} gradient={float(norm):.5f}',flush=True)
            if step%CONFIG['validation_interval']==0 or batch_index==per_epoch-1:
                calibrator.eval(); metric=validate(mapper,calibrator,receiver,cache,data['validation'],mode)
                # Validation can evict the active paper; next iteration explicitly remaps it.
                previous_id=None; bk=bv=None
                history.append({'step':step,'epoch':epoch,'validation':metric})
                if metric['total']<best:
                    best=metric['total']; checkpoint(folder/'best.pt',calibrator,step,metric,branch=mode)
                save(folder/'selection.json',history)
                checkpoint(folder/'last.pt',calibrator,step,metric,optimizer,scaler,mode)
                print('B INTERNAL VALIDATION',mode,step,metric,flush=True)
    save(folder/'completed.json',{'steps':step,'epochs':CONFIG['stage_b_epochs'],
        'selection':'lowest internal-validation functional loss, no answer/test tuning',
        'query_source':mode,'native_vs_student_matched_loss':True,'gold_answer_supervision':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('native_matched','student')); train(p.parse_args().mode)
