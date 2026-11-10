"""Token-parallel full-depth retrieval; raw full-head KV output, no Memory."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn

@dataclass(frozen=True)
class DRTConfig:
    sender_layers:int=28
    receiver_layers:int=36
    source_width:int=1024
    target_width:int=1024
    state_width:int=256
    retrieval_heads:int=4

class Direct(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.direct_k=nn.Linear(cfg.source_width,cfg.target_width)
        self.direct_v=nn.Linear(cfg.source_width,cfg.target_width)

class DepthRetrievalTranslator(nn.Module):
    def __init__(self,cfg=DRTConfig(),seed=1234):
        super().__init__();self.cfg=cfg
        s,r,w=cfg.sender_layers,cfg.receiver_layers,cfg.state_width
        assert min(s,r,w,cfg.source_width,cfg.target_width,cfg.retrieval_heads)>0 and w%cfg.retrieval_heads==0
        taps=[int(math.floor(l*(s-1)/max(1,r-1)+.5)) for l in range(r)]
        self.register_buffer('sender_taps',torch.tensor(taps,dtype=torch.long));self.taps_tuple=tuple(taps)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers=nn.ModuleList(Direct(cfg) for _ in taps)
            self.encoders=nn.ModuleList(nn.Linear(2*cfg.source_width+2,w) for _ in range(s))
            self.source_embedding=nn.Parameter(torch.randn(s,w)*.02)
            self.target_embedding=nn.Parameter(torch.randn(r,w)*.02)
            self.query=nn.Linear(w,w);self.key=nn.Linear(w,w);self.value=nn.Linear(w,w);self.output=nn.Linear(w,w)
            self.norm=nn.LayerNorm(w);self.ffn=nn.Sequential(nn.Linear(w,4*w),nn.GELU(),nn.Linear(4*w,w))
            self.readouts=nn.ModuleList(nn.Linear(w,2*cfg.target_width) for _ in taps)
            for layer in self.readouts:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        prior=-2*(torch.linspace(0,1,r)[:,None]-torch.linspace(0,1,s)[None,:]).abs()
        self.depth_bias=nn.Parameter(prior[:,None,:].repeat(1,cfg.retrieval_heads,1))

    def forward(self,source_k,source_v,source_indices,*,return_states=False):
        c=self.cfg;device=next(self.parameters()).device;dtype=next(self.parameters()).dtype
        assert len(source_k)==len(source_v)==c.sender_layers
        index=torch.as_tensor(source_indices,dtype=torch.long,device='cpu');assert index.ndim==1 and index.numel()>0
        keys=[];values=[];features=[]
        # Each source layer copied once per token chunk, reused by every target layer.
        for s,(ks,vs) in enumerate(zip(source_k,source_v)):
            assert ks.ndim==2 and ks.shape==vs.shape and ks.shape[-1]==c.source_width
            k=ks.index_select(0,index.to(ks.device)).to(device=device,dtype=dtype)
            v=vs.index_select(0,index.to(vs.device)).to(device=device,dtype=dtype)
            kr=k.square().mean(-1,keepdim=True).add(1e-6).sqrt();vr=v.square().mean(-1,keepdim=True).add(1e-6).sqrt()
            h=torch.nn.functional.gelu(self.encoders[s](torch.cat([k/kr,v/vr,kr.log(),vr.log()],-1)))+self.source_embedding[s]
            keys.append(k);values.append(v);features.append(h)
        # [tokens, source-depth, width]; attention only across source depth.
        h=torch.stack(features,1);t=h.shape[0];heads=c.retrieval_heads;d=c.state_width//heads
        q=(self.query(h[:,self.taps_tuple])+self.target_embedding).reshape(t,c.receiver_layers,heads,d)
        k=self.key(h).reshape(t,c.sender_layers,heads,d);v=self.value(h).reshape(t,c.sender_layers,heads,d)
        scores=torch.einsum('tlhd,tshd->tlhs',q,k)/math.sqrt(d)+self.depth_bias
        weights=scores.softmax(-1)
        u=self.output(torch.einsum('tlhs,tshd->tlhd',weights,v).reshape(t,c.receiver_layers,c.state_width))
        z=u+self.ffn(self.norm(u));outk=[];outv=[]
        for l,tap in enumerate(self.taps_tuple):
            dk,dv=self.readouts[l](z[:,l]).split(c.target_width,-1)
            outk.append(self.layers[l].direct_k(keys[tap])+dk)
            outv.append(self.layers[l].direct_v(values[tap])+dv)
        outputs=(torch.stack(outk),torch.stack(outv))
        return (*outputs,z.permute(1,0,2)) if return_states else outputs

    def specification(self):
        return dict(config=asdict(self.cfg),parameters=sum(p.numel() for p in self.parameters()),sender_taps=list(self.taps_tuple),output_space='receiver pre-k_norm / pre-RoPE',token_interaction=False,depth_recurrence=False,input_dependent_depth_retrieval=True,readout_initialization='zero',memory=False)
