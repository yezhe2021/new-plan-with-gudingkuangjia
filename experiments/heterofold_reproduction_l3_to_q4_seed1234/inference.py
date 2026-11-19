import hashlib
import torch
from transformers import DynamicCache
from build_cache import build_cache
from layer_alignment import features
from fold_mapper import apply_map
from common import CFG


def freeze(cache):
    return [(l.keys.detach().cpu().clone(),l.values.detach().cpu().clone()) for l in cache.layers]


def restore(model,state):
    device=next(model.parameters()).device
    return DynamicCache(ddp_cache_data=[(k.to(device).clone(),v.to(device).clone()) for k,v in state],config=model.config)


def digest(state):
    h=hashlib.sha256()
    for pair in state:
        for t in pair: h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def prefill(model,ids,cache=None):
    n=0 if cache is None else cache.get_seq_length()
    x=torch.tensor([ids],device=next(model.parameters()).device)
    out=model(input_ids=x,past_key_values=cache,use_cache=True,
              position_ids=torch.arange(n,n+len(ids),device=x.device)[None],
              attention_mask=torch.ones((1,n+len(ids)),dtype=torch.long,device=x.device))
    return out.logits[0,-1].float(),out.past_key_values


@torch.no_grad()
def mapped_memory(model,source,head,maps):
    """No native evidence input: only independently captured special-token head."""
    device=next(model.parameters()).device; kk=[];vv=[]
    for layer in range(len(maps['k'])):
        for role,out in [('k',kk),('v',vv)]:
            x=apply_map(maps[role][layer],features(source[role],layer,len(maps[role])),device).to(getattr(torch,CFG['mapped_feature_dtype']))
            out.append(torch.cat((head[role][layer].to(device),x.reshape(-1,8,128)),0))
    k,v=torch.stack(kk),torch.stack(vv)
    cache=build_cache(model,k,v,torch.arange(k.shape[1],device=device))
    return freeze(cache)


@torch.no_grad()
def generate(model,tok,ids,state=None,max_tokens=128,sampling=None):
    logits,cache=prefill(model,ids,restore(model,state) if state is not None else None)
    first=logits.cpu();tokens=[];eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    stop='max_tokens'
    for _ in range(max_tokens):
        if sampling and sampling['do_sample']:
            scores=logits/sampling['temperature']; topk=sampling['top_k']
            if topk: scores=scores.masked_fill(scores<scores.topk(topk).values[-1],-torch.inf)
            sorted_scores,order=scores.sort(descending=True)
            drop=sorted_scores.softmax(-1).cumsum(-1)>sampling['top_p']
            drop[1:]=drop[:-1].clone();drop[0]=False
            scores[order[drop]]=-torch.inf
            token=int(torch.multinomial(scores.softmax(-1),1))
        else: token=int(logits.argmax())
        if token in eos: stop='eos';break
        tokens.append(token);logits,cache=prefill(model,[token],cache)
    return {'text':tok.decode(tokens,skip_special_tokens=True).strip(),'ids':tokens,'stop':stop},first
