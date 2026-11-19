import torch
from torch import nn
from layer_alignment import features


class Correction(nn.Module):
    def __init__(self,dim=1024,rank=16):
        super().__init__()
        self.B=nn.Parameter(torch.randn(dim,rank)*.02)
        self.G=nn.Parameter(torch.zeros(rank,dim))

    def forward(self,y): return y+(y@self.B)@self.G


def fold_correction(mapping,correction):
    # Avoid forming a large I+BG; algebraically identical to Eq9.
    a,b=mapping['A'].float(),mapping['b'].float()
    B,G=correction.B.detach().cpu(),correction.G.detach().cpu()
    return {'A':a+(a@B)@G,'b':b+(b@B)@G,'rms':mapping['rms']}


def apply_map(mapping,x,device=None):
    device=device or x.device
    return x.float().to(device)/mapping['rms'].to(device)@mapping['A'].to(device)+mapping['b'].to(device)


def translate(source,maps):
    result=[]
    for layer,mapping in enumerate(maps):
        result.append(apply_map(mapping,features(source,layer,len(maps)),source.device).reshape(-1,8,128))
    return torch.stack(result)
