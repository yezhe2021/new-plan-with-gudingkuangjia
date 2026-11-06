"""Qwen-only GPU engine; frozen sender outputs are exact CPU disk tensors."""
from experiment_common import *

def to_cache(cache,model,device):
    return C.DynamicCache(ddp_cache_data=[(k.to(device),v.to(device)) for k,v in K.pairs(cache)],config=model.config)

class Engine(R.Engine):
    def __init__(self):
        self.budget=gpu_budget();self.tok,self.model=C.load_model(C.CFG['models']['receiver'],'cuda',dtype=torch.float16)
        self.prefix_ids=C.read(C.ROOT/'manifest.json')['prefix_ids']
        with torch.no_grad():self.header=C.decoder(self.model)(torch.tensor([self.prefix_ids],device='cuda'),use_cache=True).past_key_values
        self.snapshots=C.versions(self.model);self.mode='train';self.student_test_native_prefills=0;self.norm_counts=None;self.norm_hooks=[]
        self.prefix_keys=[C._cache_layer(self.header,l)[0][0].transpose(0,1).detach() for l in range(36)]
        self.prefix_values=[C._cache_layer(self.header,l)[1][0].transpose(0,1).detach() for l in range(36)]
        for l,block in enumerate(C.layers_of(self.model)):
            def count(module,args,index=l):
                if self.norm_counts is not None:self.norm_counts[index]+=1
            self.norm_hooks.append(block.self_attn.k_norm.register_forward_pre_hook(count))
    def unchanged(self):
        assert C.versions(self.model)==self.snapshots
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
    @torch.no_grad()
    def base(self,row):return saved_base(row)
    @torch.no_grad()
    def native_cache(self,row,*,baseline=False):
        # Native KV lives on host between teacher operations, not during student backward.
        cuda=R.Engine.native_cache(self,row,baseline=baseline)
        return to_cache(cuda,self.model,'cpu')
    def cache(self,row,base,corrections):
        p=len(self.prefix_ids);positions=torch.arange(p,p+len(row['content_ids']),device='cuda');tables=C._rotary_tables(self.model,positions);data=[]
        for l in range(36):
            k,v=(x[l].to('cuda').float() for x in base)
            if corrections is not None:k,v=corrections[l].key(k),corrections[l].value(v)
            cosine,sine=tables(l);k,v=C.finalize_layer(self.model,l,k,v,cosine,sine);hk,hv=C._cache_layer(self.header,l)
            data.append((torch.cat([hk.detach(),k[None]],2),torch.cat([hv.detach(),v[None]],2)))
        return C.DynamicCache(ddp_cache_data=data,config=self.model.config)
