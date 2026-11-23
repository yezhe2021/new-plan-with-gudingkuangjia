"""Use the original DST cache writer with a frozen raw Ridge predictor."""

from experiment_common import BASE, C, CFG, RIDGE, ROOT, Z, identity, load_module, torch
from ridge import RawRidge

ResidualTranslator = load_module("residual", BASE / "residual.py").ResidualTranslator
BodyReader = Z.DE.BodyReader
raw_cache = Z.DE.raw_cache


class Engine(Z.DE.Engine):
    def __init__(self):
        super().__init__()
        self.ridge_sha256 = C.sha(RIDGE)
        summary = C.read(ROOT / "stage_a/summary.json")
        assert summary["manifest_sha256"] == identity() and summary["checkpoint_sha256"] == self.ridge_sha256
        if (ROOT / "checks/gpu_completed.json").exists():
            assert C.read(ROOT / "checks/gpu_completed.json")["ridge_sha256"] == self.ridge_sha256
        payload = torch.load(RIDGE, map_location="cpu", weights_only=True, mmap=True)
        assert payload["manifest_sha256"] == identity()
        self.ridge = RawRidge(payload).cuda().eval().requires_grad_(False)
        self.ridge_versions = [b._version for b in self.ridge.buffers()]
        self.correction = ResidualTranslator(self.ridge, rank=CFG["rank"], seed=CFG["seed"]).cuda().float()
        assert sum(p.numel() for p in self.correction.parameters()) == 9_437_184

    def unchanged(self):
        super().unchanged()
        assert [b._version for b in self.ridge.buffers()] == self.ridge_versions
