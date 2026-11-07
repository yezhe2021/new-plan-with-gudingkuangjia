"""Frozen DST body generation adapted to the established memory training API."""
from experiment_common import *

class Engine(Z.DE.Engine):
    def __init__(self):
        super().__init__()
        path=DST/'runs/stage2/epoch4_step1536.pt';assert C.sha(path)==CFG['dst_checkpoint_sha256']
        self.dst=Z.DT.DepthStateTranslator(Z.DT.DSTConfig(**D.CFG['model']),D.CFG['seed']).cuda().float()
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==D.identity() and cp['stage']=='stage2' and cp['step']==1536 and cp['epoch']==4
        self.dst.load_state_dict(cp['state'],strict=True);self.dst.eval().requires_grad_(False)
        self.dst_versions=C.versions(self.dst)
        assert all(torch.equal(v.detach().cpu(),cp['state'][n]) for n,v in self.dst.state_dict().items())

    @torch.no_grad()
    def base(self,row):return self.source(row)

    @torch.no_grad()
    def cache(self,row,base,correction):
        assert correction is self.dst
        return self.translated(row,self.dst,source=base,checkpointed=False)

    def unchanged(self):
        super().unchanged()
        assert C.versions(self.dst)==self.dst_versions
        assert all(not p.requires_grad and p.grad is None for p in self.dst.parameters())
