from collections import OrderedDict
import torch
from runtime import CFG,make_cache,freeze
from prompt_protocol import paper_protocol


@torch.no_grad()
def capture(model,ids):
    keys,values,handles={},{},[]
    def hook(i):
        def call(module,args,kwargs):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            shape=(*hidden.shape[:-1],-1,module.head_dim)
            k=module.k_proj(hidden).view(shape)
            if hasattr(module,'k_norm'): k=module.k_norm(k)
            keys[i]=k[0].cpu(); values[i]=module.v_proj(hidden).view(shape)[0].cpu()
        return call
    for i,l in enumerate(model.model.layers): handles.append(l.self_attn.register_forward_pre_hook(hook(i),with_kwargs=True))
    try:
        ids=torch.tensor([ids],device='cuda',dtype=torch.long)
        model.model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1],device='cuda')[None],use_cache=False)
    finally:
        for h in handles: h.remove()
    return torch.stack([keys[i] for i in range(len(keys))]),torch.stack([values[i] for i in range(len(values))])


class PaperCache:
    """Bounded CPU LRU. Recapture after eviction is explicit, never per Query."""
    def __init__(self,sender,receiver,st,rt):
        self.sender,self.receiver,self.st,self.rt=sender,receiver,st,rt
        self.items=OrderedDict(); self.wrappers={}; self.capture_counts={}
    def add_teacher(self,item):
        if 'native_k' not in item:
            tk,tv=capture(self.receiver,item['protocol']['prefix_ids'])
            n=len(item['protocol']['head_ids'])
            item.update(native_k=tk,native_v=tv,target_k=tk[:,n:],target_v=tv[:,n:])
        return item
    def get(self,row,teacher=True):
        pid=row['paper_id']
        if pid in self.items:
            self.items.move_to_end(pid)
            return self.add_teacher(self.items[pid]) if teacher else self.items[pid]
        proto=paper_protocol(self.st,self.rt,row['paper'],row['questions'])
        sk,sv=capture(self.sender,proto['source_ids'])
        head=tuple(proto['head_ids'])
        if head not in self.wrappers: self.wrappers[head]=capture(self.receiver,list(head))
        item={'protocol':proto,'source_k':sk[:,proto['source_positions']], 'source_v':sv[:,proto['source_positions']],
              'wrapper_k':self.wrappers[head][0],'wrapper_v':self.wrappers[head][1],
              'tokens':proto['body_count'],'probes':{}}
        self.capture_counts[pid]=self.capture_counts.get(pid,0)+1
        self.items[pid]=item
        while len(self.items)>CFG['cpu_cache_papers']: self.items.popitem(last=False)
        return self.add_teacher(item) if teacher else item


@torch.no_grad()
def mapped(base,item):
    kk,vv=[],[]
    for start in range(0,item['tokens'],CFG['chunk_tokens']):
        with torch.amp.autocast('cuda',dtype=torch.float16):
            k,v=base(item['source_k'][:,start:start+CFG['chunk_tokens']][None].cuda(),item['source_v'][:,start:start+CFG['chunk_tokens']][None].cuda())
        kk.append(k[0]); vv.append(v[0])
    return torch.cat(kk,1).detach(),torch.cat(vv,1).detach()


@torch.no_grad()
def memory(receiver,item,k,v,adapter=None):
    if adapter is not None:
        with torch.amp.autocast('cuda',dtype=torch.float16): k,v,_,_=adapter(k[None],v[None])
        k,v=k[0],v[0]
    # Never copy native evidence KV. Only independently computed public wrapper.
    k=torch.cat((item['wrapper_k'].cuda(),k),1)
    v=torch.cat((item['wrapper_v'].cuda(),v),1)
    assert k.shape[1]==len(item['protocol']['prefix_ids'])
    return freeze(make_cache(receiver,k,v,torch.arange(k.shape[1],device='cuda')))
