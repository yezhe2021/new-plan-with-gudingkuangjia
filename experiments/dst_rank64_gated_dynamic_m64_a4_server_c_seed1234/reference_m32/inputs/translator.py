"""Token-parallel, Receiver-depth recurrent K/V translation; no token attention."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn

@dataclass(frozen=True)
class DSTConfig:
    sender_layers:int=28
    receiver_layers:int=36
    source_width:int=1024
    target_width:int=1024
    state_width:int=256
    carry_state:bool=True

class DepthLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__();h=cfg.state_width;s=cfg.source_width;r=cfg.target_width
        self.state_norm=nn.LayerNorm(h)
        self.input=nn.Linear(2*s,h)
        self.previous=nn.Linear(h,h,bias=False)
        self.update=nn.Linear(h,h)
        # These dense, full-width readouts mix all KV heads; the original
        # features bypass the low-dimensional state, without a Recolor base.
        self.direct_k=nn.Linear(s,r);self.direct_v=nn.Linear(s,r)
        self.state_k=nn.Linear(h,r,bias=False);self.state_v=nn.Linear(h,r,bias=False)
        self.scale=1/math.sqrt(cfg.receiver_layers)
        nn.init.xavier_uniform_(self.update.weight,gain=.1);nn.init.zeros_(self.update.bias)
        nn.init.xavier_uniform_(self.state_k.weight,gain=.1);nn.init.xavier_uniform_(self.state_v.weight,gain=.1)

    def forward(self,k,v,z):
        # Normalize only the recurrent input; the direct readouts retain raw
        # amplitudes. No cross-token reduction or state is introduced.
        kn=k*torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-6)
        vn=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)
        delta=self.update(torch.nn.functional.gelu(self.input(torch.cat([kn,vn],-1))+self.previous(self.state_norm(z))))
        z=z+self.scale*delta
        return self.direct_k(k)+self.state_k(z),self.direct_v(v)+self.state_v(z),z

class DepthStateTranslator(nn.Module):
    def __init__(self,cfg=DSTConfig(),seed=1234):
        super().__init__();self.cfg=cfg
        if min(cfg.sender_layers,cfg.receiver_layers,cfg.source_width,cfg.target_width,cfg.state_width)<1:raise ValueError('Positive geometry required')
        taps=[int(math.floor(l*(cfg.sender_layers-1)/max(1,cfg.receiver_layers-1)+.5)) for l in range(cfg.receiver_layers)]
        self.register_buffer('sender_taps',torch.tensor(taps,dtype=torch.long))
        self.taps_tuple=tuple(taps)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.layers=nn.ModuleList(DepthLayer(cfg) for _ in taps)

    def forward(self,source_k,source_v,source_indices,*,return_states=False):
        if len(source_k)!=self.cfg.sender_layers or len(source_v)!=self.cfg.sender_layers:raise ValueError('Sender layer count mismatch')
        device=next(self.parameters()).device;dtype=next(self.parameters()).dtype
        index=torch.as_tensor(source_indices,dtype=torch.long,device='cpu')
        if index.ndim!=1 or index.numel()==0:raise ValueError('Nonempty 1D source indices required')
        z=torch.zeros(index.numel(),self.cfg.state_width,device=device,dtype=dtype);keys=[];values=[];states=[]
        for l,block in enumerate(self.layers):
            tap=self.taps_tuple[l];ks=source_k[tap];vs=source_v[tap]
            if ks.ndim!=2 or ks.shape!=vs.shape or ks.shape[-1]!=self.cfg.source_width:raise ValueError('Sender features must be [tokens,full KV width]')
            k=ks.index_select(0,index.to(ks.device)).to(device=device,dtype=dtype)
            v=vs.index_select(0,index.to(vs.device)).to(device=device,dtype=dtype)
            previous=z if self.cfg.carry_state else torch.zeros_like(z)
            k,v,z=block(k,v,previous);keys.append(k);values.append(v)
            if return_states:states.append(z)
        outputs=(torch.stack(keys),torch.stack(values))
        return (*outputs,torch.stack(states)) if return_states else outputs

    def specification(self):return dict(config=asdict(self.cfg),parameters=sum(p.numel() for p in self.parameters()),sender_taps=self.sender_taps.cpu().tolist(),output_space='receiver pre-k_norm / pre-RoPE',token_interaction=False)
