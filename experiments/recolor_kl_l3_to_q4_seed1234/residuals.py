"""Identical zero-up factors; the sole linear/MLP structural change is GELU."""
import torch
from torch import nn
from heterofold.calibration import LayerCorrection


class MLPCorrection(LayerCorrection):
    def _correct(self,value,role):
        return value+torch.nn.functional.gelu(value@getattr(self,'left_'+role))@getattr(self,'right_'+role)


def make_corrections(kind,rank,seed=1234,device='cuda'):
    if kind=='none':return None
    torch.manual_seed(seed)
    cls=LayerCorrection if kind=='linear' else MLPCorrection
    result=nn.ModuleList([cls(1024,rank) for _ in range(36)]).to(device).float()
    assert sum(p.numel() for p in result.parameters())==36*4*1024*rank
    return result
