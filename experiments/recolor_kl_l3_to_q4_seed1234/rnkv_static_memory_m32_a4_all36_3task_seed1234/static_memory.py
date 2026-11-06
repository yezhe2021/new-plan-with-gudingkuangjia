"""Input-independent native-projected KV, with differentiable RoPE relocation."""
from experiment_common import *
from heterofold.attention import rotary_cos_sin

def rotate(x):return torch.cat([-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],-1)

def rephase(key,source_cos,source_sin,target_cos,target_sin):
    # Native cache K already has k_norm and RoPE. Undo only the source rotation,
    # then rotate at the recipient positions; never apply k_norm a second time.
    sc,ss,tc,ts=[v.float()[None,None] for v in (source_cos,source_sin,target_cos,target_sin)]
    raw=(key.float()*sc-rotate(key.float())*ss)/(sc.square()+ss.square()).clamp_min(1e-12)
    return (raw*tc+rotate(raw)*ts).to(key.dtype)

class StaticMemory(M.RNKV):
    def __init__(self,engine,config=None):
        cfg=dict(CFG if config is None else config);cfg['replace_layers']=list(range(1,37))
        super().__init__(engine,cfg)
        self.routing.close();self.routing=AR.KeepBothRouting(engine,self.memory,cfg['replace_layers'])
        self._canonical_inference=None;self._canonical_versions=None

    def canonical(self):
        # This function deliberately accepts no row, translated body or length.
        versions=tuple(p._version for p in self.parameters())
        reuse=not self.training and not torch.is_grad_enabled()
        if reuse and versions==self._canonical_versions and self._canonical_inference is not None:
            return self._canonical_inference
        H=len(self.engine.prefix_ids)
        assert self.engine.header.get_seq_length()==H
        cache=super().construct(None,self.engine.header)
        values=tuple(x[:,:,H:H+self.M].clone() for pair in K.pairs(cache) for x in pair)
        if reuse:
            self._canonical_inference=tuple(x.detach() for x in values);self._canonical_versions=versions
            return self._canonical_inference
        else:
            self._canonical_inference=None;self._canonical_versions=None
        return values

    def relocate(self,values,length):
        H=len(self.engine.prefix_ids)
        if length==H:return values
        device=self.memory.prompt.device;model=self.engine.model
        source=torch.arange(H,H+self.M,device=device);target=torch.arange(length,length+self.M,device=device)
        out=[]
        for layer in range(model.config.num_hidden_layers):
            sc,ss=rotary_cos_sin(model,source,layer,model.dtype)
            tc,ts=rotary_cos_sin(model,target,layer,model.dtype)
            out.extend([rephase(values[2*layer],sc,ss,tc,ts),values[2*layer+1]])
        return tuple(out)

    def construct(self,row,translated):
        # Only this attachment step sees recipient coordinates. The fixed KV
        # content, including every V and pre-RoPE K, is shared across examples.
        values=self.canonical()
        return K.append(translated,self.relocate(values,translated.get_seq_length()),self.engine.model)
