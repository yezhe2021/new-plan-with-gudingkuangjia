"""RAW Full-Body source features and exactly-once Qwen cache finalization."""
from experiment_common import *
from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint
from heterofold.capture import capture_kv

def finalize_raw(model,key,value,positions):
    tables=C._rotary_tables(model,positions);pairs=[]
    for l in range(model.config.num_hidden_layers):
        cosine,sine=tables(l);pairs.append(C.finalize_layer(model,l,key[l],value[l],cosine,sine))
    return torch.stack([p[0] for p in pairs]),torch.stack([p[1] for p in pairs])

def raw_cache(model,header,key,value):
    H=header.get_seq_length();positions=torch.arange(H,H+key.shape[1],device=key.device)
    keys,values=finalize_raw(model,key,value,positions)
    return C.DynamicCache(ddp_cache_data=[(torch.cat([a.detach(),keys[l][None]],2),torch.cat([b.detach(),values[l][None]],2)) for l,(a,b) in enumerate(K.pairs(header))],config=model.config)

class Engine(R.Engine):
    def __init__(self):
        self.tok,self.model=C.load_model(C.CFG['models']['receiver'],'cuda',dtype=torch.float16)
        self.sender_tok,self.sender=C.load_model(C.CFG['models']['sender'],'cuda',dtype=torch.float16)
        self.prefix_ids=C.read(C.ROOT/'manifest.json')['prefix_ids']
        with torch.no_grad():self.header=C.decoder(self.model)(torch.tensor([self.prefix_ids],device='cuda'),use_cache=True).past_key_values
        self._versions=[C.versions(x) for x in [self.model,self.sender]]
        self.mode='train';self.student_test_native_prefills=0;self.norm_counts=None;self.norm_hooks=[]
        for l,block in enumerate(C.layers_of(self.model)):
            def count(module,args,index=l):
                if self.norm_counts is not None:self.norm_counts[index]+=1
            self.norm_hooks.append(block.self_attn.k_norm.register_forward_pre_hook(count))

    def unchanged(self):
        assert self._versions==[C.versions(x) for x in [self.model,self.sender]]
        assert all(not p.requires_grad and p.grad is None for m in [self.model,self.sender] for p in m.parameters())

    @torch.no_grad()
    def source(self,row):
        assert self.sender_tok.encode(row['prompt'],add_special_tokens=False)==row['sender_ids']
        _,keys,values=capture_kv(self.sender_tok,self.sender,row['prompt'],range(self.sender.config.num_hidden_layers),device='cpu',dtype=torch.float16)
        kk=tuple(keys[l] for l in range(len(keys)));vv=tuple(values[l] for l in range(len(values)))
        assert all(x.shape==(len(row['sender_ids']),CFG['model']['source_width']) and torch.isfinite(x).all() for x in kk+vv)
        return kk,vv

    @torch.no_grad()
    def teacher(self,row):
        if self.mode=='test':raise RuntimeError('Native K/V distillation targets forbidden in student evaluation')
        H=len(self.prefix_ids);keys={};values={};handles=[]
        for l,block in enumerate(C.layers_of(self.model)):
            for name,store in [('k_proj',keys),('v_proj',values)]:
                def capture(module,args,output,store=store,index=l):store[index]=output[0,H:].detach().cpu().contiguous()
                handles.append(getattr(block.self_attn,name).register_forward_hook(capture))
        try:native=self.native_cache(row)
        finally:
            for h in handles:h.remove()
        k=torch.stack([keys[l] for l in range(len(keys))]);v=torch.stack([values[l] for l in range(len(values))])
        assert k.shape==v.shape==(36,len(row['content_ids']),1024) and torch.isfinite(k).all() and torch.isfinite(v).all()
        return k,v,native

    def translated(self,row,translator,source=None,*,checkpointed=True):
        source=self.source(row) if source is None else source;index=row['sync_source_indices'];H=len(self.prefix_ids);chunks=[]
        def build(begin,end):
            key,value=translator(*source,index[begin:end])
            positions=torch.arange(H+begin,H+end,device=key.device)
            return finalize_raw(self.model,key,value,positions)
        for begin in range(0,len(index),CFG['token_chunk']):
            end=min(len(index),begin+CFG['token_chunk'])
            values=checkpoint(build,begin,end,use_reentrant=False,preserve_rng_state=True) if checkpointed and torch.is_grad_enabled() else build(begin,end)
            chunks.append(values)
        return C.DynamicCache(ddp_cache_data=[(torch.cat([a.detach()]+[x[0][l][None] for x in chunks],2),torch.cat([b.detach()]+[x[1][l][None] for x in chunks],2)) for l,(a,b) in enumerate(K.pairs(self.header))],config=self.model.config)

class PlainRouting:
    @contextmanager
    def context(self,*args,**kwargs):yield

class BodyReader:
    """Exact existing readout methods, no Memory tokens or construction hooks."""
    def __init__(self,engine):self.engine=engine;self.activation_checkpointing=True;self.routing=PlainRouting()
    logits=M.RNKV.logits
    token_logits=M.RNKV.token_logits
    rollout=M.RNKV.rollout
    score=M.RNKV.score
    M=0
