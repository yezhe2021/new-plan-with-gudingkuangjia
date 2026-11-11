from experiment_common import *
from residual import ResidualTranslator
BodyReader=Z.DE.BodyReader
raw_cache=Z.DE.raw_cache
class Engine(Z.DE.Engine):
    def __init__(self):
        super().__init__();path=D.ROOT/'runs/stage1/epoch4_step1536.pt';assert C.sha(path)==CFG['dst_stage1_sha256']
        self.frozen_dst=Z.DT.DepthStateTranslator(Z.DT.DSTConfig(**D.CFG['model']),D.CFG['seed']).cuda().float()
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==D.identity() and cp['stage']=='stage1' and cp['step']==1536
        self.frozen_dst.load_state_dict(cp['state'],strict=True);self.frozen_dst.eval().requires_grad_(False)
        self.base_versions=C.versions(self.frozen_dst)
        self.correction=ResidualTranslator(self.frozen_dst,rank=CFG['rank'],seed=CFG['seed']).cuda().float()
        assert sum(p.numel() for p in self.correction.parameters())==9437184
    def unchanged(self):
        super().unchanged();assert C.versions(self.frozen_dst)==self.base_versions
        assert all(not p.requires_grad and p.grad is None for p in self.frozen_dst.parameters())
