"""Per-Qwen-layer separate K/V dense2048→1024 projection, initialized as an equal mean."""
import torch
from torch import nn
class ConcatProjection(nn.Module):
    def __init__(self,layers=36,heads=8,dim=128):
        super().__init__();self.layers,self.heads,self.dim=layers,heads,dim
        width=heads*dim
        def layer():
            module=nn.Linear(2*width,width,bias=False)
            with torch.no_grad():
                eye=torch.eye(width,device=module.weight.device,dtype=module.weight.dtype)*0.5
                module.weight[:,:width].copy_(eye);module.weight[:,width:].copy_(eye)
            return module
        self.k=nn.ModuleList([layer() for _ in range(layers)])
        self.v=nn.ModuleList([layer() for _ in range(layers)])
    def _map(self,a,b,branches):
        return torch.stack([branches[i](torch.cat((a[i].flatten(-2),b[i].flatten(-2)),dim=-1)).reshape_as(a[i]) for i in range(self.layers)])
    def forward(self,lk,lv,gk,gv):
        if not(lk.shape==lv.shape==gk.shape==gv.shape) or lk.ndim!=4 or lk.shape[0]!=self.layers or lk.shape[-2:]!=(self.heads,self.dim):raise ValueError('Expected matching [layers,tokens,heads,dim] inputs')
        return self._map(lk,gk,self.k),self._map(lv,gv,self.v)
