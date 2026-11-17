"""Load the published method's locally calibrated Llama3.2-3B → Qwen3-4B map."""
from collections import Counter
from pathlib import Path

from heterofold.maps import load_map


def load_r16(core):
    path = Path(core.CFG["historical_heterofold"])
    digest = core.sha(path)
    assert digest == core.CFG["historical_heterofold_sha256"]
    mapping = load_map(path, device="cuda", dtype=core.torch.float32, mmap=True)
    calibration = mapping["functional_correction"]
    assert (mapping["sender"], mapping["receiver"]) == ("llama3", "qwen4")
    assert mapping["functional_variant"] == "output_aware_rank16"
    assert calibration["rank"] == 16 and calibration["inference_extra_modules"] == 0
    assert calibration["structure"] == "folded_kv_separated_low_rank_residual"
    assert calibration["epochs"] == 4
    best = min(calibration["history"], key=lambda epoch: epoch["validation"]["loss"])
    assert best["validation"]["loss"] == calibration["best_validation_loss"]
    assert mapping["depth_stencil"]["span"] == 1
    for role in "kv":
        assert all(len(taps) == 3 for taps in mapping["depth_stencil"][f"taps_{role}"])
    data = Path(core.CFG["calibration"])
    counts = Counter((r["split"], r["dataset"]) for r in core.read(data)["rows"])
    assert sorted(counts.values()) == [200, 200, 800, 800]
    assert calibration["calibration_train_examples"] == 1600
    assert calibration["calibration_validation_examples"] == 400
    return mapping, dict(path=str(path), sha256=digest, calibration_sha256=core.sha(data),
                         selected_epoch=best["epoch"], validation_loss=best["validation"]["loss"],
                         rank=16, epochs=4, train=1600, validation=400, map_dtype="float32")
