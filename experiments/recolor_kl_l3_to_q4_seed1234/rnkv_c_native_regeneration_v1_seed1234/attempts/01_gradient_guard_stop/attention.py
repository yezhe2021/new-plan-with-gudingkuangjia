"""Explicit per-layer key-source routing and memory-only adapter hooks."""
from contextlib import contextmanager
from common import C,torch

class Routing:
    def __init__(self,engine,memory,replace):
        self.engine=engine;self.memory=memory;self.replace=set(replace)
        self.state=None;self.handles=[];self.calls={'construct':0,'answer':0}
        for i,block in enumerate(C.layers_of(engine.model),1):
            def mask(module,args,kwargs,layer=i):
                if self.state is None:return
                phase,header,L,count,q=self.state
                length=kwargs['past_key_values'].get_seq_length(layer-1)
                # This hook runs before layer's cache update: its own past remains
                # L or L+M while earlier layers may already have appended tokens.
                key_length=length+q.numel()
                allowed=self.allowed(phase,layer,header,L,count,q,key_length)
                assert allowed.any(-1).all()
                value=torch.zeros(q.numel(),key_length,device=q.device,dtype=self.engine.model.dtype).masked_fill(~allowed,float('-inf'))
                kwargs=dict(kwargs);kwargs['attention_mask']=value[None,None]
                self.calls[phase]+=1
                return args,kwargs
            self.handles.append(block.self_attn.register_forward_pre_hook(mask,with_kwargs=True))
            if str(i) in memory.adapters:
                def adapt(module,args,output,layer=i):
                    if self.state is None or self.state[0]!='construct':return output
                    assert output.ndim==3 and output.shape[1]==self.state[3]
                    return self.memory.adapt(layer,output)
                self.handles.append(block.register_forward_hook(adapt))

    def allowed(self,phase,layer,header,L,M,q,key_length):
        keys=torch.arange(key_length,device=q.device)[None,:]
        value=keys<=q[:,None]
        if phase=='answer':
            start,end=(header,L) if layer in self.replace else (L,L+M)
            value[:,start:end]=False
        else:assert phase=='construct'
        return value

    @contextmanager
    def context(self,phase,header,L,M,q):
        previous=self.state
        self.state=None if phase=='plain' else (phase,header,L,M,q)
        try:yield
        finally:self.state=previous

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
