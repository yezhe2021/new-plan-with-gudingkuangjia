from experiment_common import *
BE=Z.scoped_module('rank64_b_engine',B.ROOT/'engine.py',B)
class Engine(BE.Engine):
    def __init__(self):
        super().__init__();path=B.ROOT/'runs/stage2/epoch4_step1536.pt'
        assert C.sha(path)==C.read(ROOT/'manifest.json')['b_checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==B.identity() and cp['step']==1536 and cp['stage']=='stage2'
        self.correction.load_state_dict(cp['state'],strict=True);self.correction.eval().requires_grad_(False)
        self.dst=self.correction;self.correction_versions=C.versions(self.correction)
    @torch.no_grad()
    def base(self,row):return self.source(row)
    @torch.no_grad()
    def cache(self,row,base,correction):
        assert correction is self.correction
        return self.translated(row,correction,source=base,checkpointed=False)
    def unchanged(self):
        super().unchanged();assert C.versions(self.correction)==self.correction_versions
        assert all(p.grad is None and not p.requires_grad for p in self.correction.parameters())
