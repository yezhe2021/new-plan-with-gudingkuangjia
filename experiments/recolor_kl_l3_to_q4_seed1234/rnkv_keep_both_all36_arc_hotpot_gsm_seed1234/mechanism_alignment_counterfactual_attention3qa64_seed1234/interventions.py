"""Memory-only interventions preserve translated body, suffix positions and weights."""
from contextlib import contextmanager
from common_diag import *

@contextmanager
def access(module,mode='keep',blocked=(),header_only=False):
    original=module.routing.allowed;blocked=set(blocked)
    def allowed(phase,layer,header,L,M,q,key_length):
        value=original(phase,layer,header,L,M,q,key_length)
        if phase=='answer' and (mode=='masked' or layer in blocked):value[:,L:L+M]=False
        if phase=='construct' and header_only:value[:,header:L]=False
        return value
    module.routing.allowed=allowed
    try:yield
    finally:module.routing.allowed=original

def memory_pairs(cache,L):return [(a[:,:,L:L+32].detach().clone(),b[:,:,L:L+32].detach().clone()) for a,b in K.pairs(cache)]
def assemble(engine,body,memory):
    return K.append(body,tuple(x.to(next(engine.model.parameters()).device) for pair in memory for x in pair),engine.model)
def padded(engine,body):
    memory=[tuple(torch.zeros(x.shape[0],x.shape[1],32,x.shape[-1],dtype=x.dtype,device=x.device) for x in pair) for pair in K.pairs(body)]
    return assemble(engine,body,memory)

def rephase_tensor(key,source_cos,source_sin,target_cos,target_sin):
    def rotate(x):return torch.cat([-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],-1)
    sc,ss,tc,ts=[v.float()[None,None] for v in [source_cos,source_sin,target_cos,target_sin]]
    raw=(key.float()*sc-rotate(key.float())*ss)/(sc.square()+ss.square()).clamp_min(1e-12)
    return (raw*tc+rotate(raw)*ts).to(key.dtype)

def donor_memory(engine,memory,donor_L,recipient_L):
    if donor_L==recipient_L:return [(a.clone(),b.clone()) for a,b in memory]
    from heterofold.attention import rotary_cos_sin
    dev=next(engine.model.parameters()).device
    src=torch.arange(donor_L,donor_L+32,device=dev);dst=torch.arange(recipient_L,recipient_L+32,device=dev)
    out=[];max_error=0.
    for l,(key,value) in enumerate(memory):
        key=key.to(dev);value=value.to(dev)
        sc,ss=rotary_cos_sin(engine.model,src,l,engine.model.dtype);tc,ts=rotary_cos_sin(engine.model,dst,l,engine.model.dtype)
        moved=rephase_tensor(key,sc,ss,tc,ts);back=rephase_tensor(moved,tc,ts,sc,ss)
        error=D.relative_l2(back,key);assert error<.003,('Memory RoPE roundtrip',l,error);max_error=max(max_error,error)
        assert torch.equal(value,value.clone());out.append((moved,value.clone()))
    return out

def gold_tokens(engine,row):
    answer=engine.tok.encode(str(row['gold']),add_special_tokens=False);assert answer
    return answer+[engine.tok.convert_tokens_to_ids('<|im_end|>')]

@torch.no_grad()
def logits(module,row,cache,tokens,rn,mode='keep',blocked=()):
    with access(module,mode,blocked):return module.logits(row,cache,tokens[:-1],range(len(tokens)),rn=rn)

def functional(value,ref,ref_shift,tokens):
    lp=value.float().log_softmax(-1);target=torch.tensor(tokens,device=lp.device);ix=torch.arange(len(tokens),device=lp.device)
    nll=-lp[ix,target];result=dict(answer_nll=float(nll[:-1].mean()),answer_logprob=float(-nll[:-1].mean()),eos_nll=float(nll[-1]),
        gold_token_probability=float((-nll[:-1]).exp().mean()),top1_gold_match=float((lp.argmax(-1)[:-1]==target[:-1]).float().mean()),
        per_token_nll=nll.cpu().tolist(),answer_tokens=len(tokens)-1)
    for name,teacher in [('native',ref),('native_shifted',ref_shift)]:
        t=teacher.float().log_softmax(-1);kl=(t.exp()*(t-lp)).sum(-1).clamp_min(0)
        result['kl_to_'+name]=float(kl[:-1].mean());result['top1_match_'+name]=float((lp.argmax(-1)[:-1]==t.argmax(-1)[:-1]).float().mean())
    return result

@torch.no_grad()
def score(module,row,cache,rn,mode='keep'):
    with access(module,mode):
        if row['task']=='hotpotqa':return module.score(row,cache,rn=rn)
        return E.X.score(module,row,cache,rn=rn)

@torch.no_grad()
def trace(engine,module,row,cache,tokens,mode='keep'):
    engine._diagnostic_suffix=row['suffix_ids'];engine._diagnostic_candidate_ids=None;engine._diagnostic_gold_index=None
    L=len(engine.prefix_ids)+len(row['content_ids']);past=cache.get_seq_length();ids=len(row['suffix_ids'])+len(tokens)-1
    q=torch.arange(past,past+ids,device=next(engine.model.parameters()).device)
    with access(module,mode),module.routing.context('answer',len(engine.prefix_ids),L,32,q):
        return D.forward_trace(C,engine,cache,tokens,list(range(len(tokens))),list(range(36)))
