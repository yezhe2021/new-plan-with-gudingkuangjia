"""Verify exact data/checkpoint provenance and a real backward pass before resuming."""
import hashlib
import json
from pathlib import Path
import torch
from runtime import CFG,ROOT,seed,save,load_models,tokenizers,base,adapter
from kv_backend import PaperCache,mapped
from train_stage_a import chunk_losses
from attention_probe import collect,losses

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    seed(); assert torch.cuda.is_available()
    origin=Path(CFG['resume_from']).parents[2]
    assert sha(ROOT/'prepared.json')==sha(origin/'prepared.json'), 'Dataset changed'
    prepared=json.loads((ROOT/'prepared.json').read_text())
    assert len(prepared['train'])==256
    assert sum(len(r['train_questions']) for r in prepared['train'])==630
    done=json.loads((origin/'smoke/completed.json').read_text())
    assert (origin/'smoke/native_precision_completed.json').exists()
    last=ROOT/'runs/stage_a_fresh/last.pt'
    assert sha(last)==sha(Path(CFG['resume_from'])), 'Resume file differs'
    payload=torch.load(last,map_location='cpu',weights_only=True)
    assert payload['protocol']==CFG['protocol'] and payload['step']==64 and payload['epoch']==2
    assert payload['optimizer']['state'] and payload['scaler']
    save(ROOT/'resume_smoke/provenance.json',{'prepared_sha256':sha(ROOT/'prepared.json'),
        'resume_sha256':sha(last),'resume_epoch':2,'resume_step':64,'previous_full_protocol_smoke':done,
        'cuda':True,'papers':256,'queries':630,'fresh_stage_b':True})
    print('PROVENANCE PASSED: exact dataset, checkpoint64, optimizer/scaler, previous protocol gates',flush=True)
    sender,receiver=load_models();st,rt=tokenizers();cache=PaperCache(sender,receiver,st,rt)
    model=base();model.load_state_dict(payload['state']);del payload
    row=max(prepared['train'],key=lambda r:r['body_tokens'])
    item=cache.get(row);model.train();model.zero_grad(set_to_none=True)
    first=next(chunk_losses(model,item));assert torch.isfinite(first);first.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
    bk,bv=mapped(model,item);cal=adapter()
    # Real K/V functional backward on the same long-paper input, no weight update.
    q=item['protocol']['queries'][0];probe=collect(receiver,item,q)
    lk,lv,_=losses(receiver,item,probe,0,bk[0],bv[0],cal)
    assert torch.isfinite(lk+lv);(lk+lv).backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in cal.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cal.parameters())
    assert all(p.grad is None for p in sender.parameters()) and all(p.grad is None for p in receiver.parameters())
    assert all(p.grad is None for p in model.parameters())
    save(ROOT/'resume_smoke/completed.json',{'protocol':CFG['protocol'],'paper_id':row['paper_id'],
        'stage_a_loss':float(first),'stage_b_probe_loss':float(lk+lv),'finite_gradients':True,
        'frozen_models':True,'checkpoint_updated':False})
    print('RESUME GPU SMOKE PASSED',flush=True)

if __name__=='__main__':main()
