"""Live detached student prefixes; teacher feedback is soft KL only."""
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute
from followup_common import CONFIG,base_train,parent_prepare
from kv_backend import assemble


@torch.no_grad()
def rollout(hf,adapter,row,item,bk,bv,cap):
    context=row['context']
    hf._cached_context=hashlib.sha256(('translated'+context.rstrip()).encode()).hexdigest()
    hf._cached_kv=assemble(item,bk,bv,adapter)
    try: ids,text=parent_prepare.generate(hf,context,cap)
    finally:
        hf._cached_context=None;hf._cached_kv=None
    assert ids and len(ids)<=cap
    return ids,text


def full_vocab_kl(student_logits,teacher_logits):
    return F.kl_div(student_logits.float().log_softmax(-1),teacher_logits.detach().float().log_softmax(-1),reduction='sum',log_target=True)


def objective(receiver,adapter,row,item,bk,bv):
    ids=row['continuation_ids'];ctx=row['context_ids']
    with torch.no_grad():teacher=base_train.suffix_hidden(receiver,ctx,ids)
    k,v=assemble(item,bk,bv,adapter)
    student=recompute(lambda kk,vv:base_train.suffix_hidden(receiver,ctx,ids,kk,vv),k,v,use_reentrant=False)
    loss=student.new_zeros((),dtype=torch.float32)
    for start in range(0,len(ids),CONFIG['projection_chunk_tokens']):
        end=start+CONFIG['projection_chunk_tokens']
        def project(s,t):return full_vocab_kl(receiver.lm_head(s),receiver.lm_head(t))
        loss=loss+recompute(project,student[start:end],teacher[start:end],use_reentrant=False)
    loss=loss/len(ids)
    assert torch.isfinite(loss)
    return loss,{'total':float(loss.detach()),'token_kl':float(loss.detach()),'tokens':len(ids),
                 'hard_student_token_ce':0.0,'forced_eos_targets':0.0}


def reference_partition(rows,epoch):
    import random
    order=list(range(len(rows)));random.Random(CONFIG['seed']).shuffle(order)
    half=len(order)//2
    refs=set(order[:half] if epoch==1 else order[half:])
    return [('reference' if i in refs else 'student',row) for i,row in enumerate(rows)]
