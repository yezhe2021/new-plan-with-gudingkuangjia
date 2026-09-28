import json
from pathlib import Path

import generation_common as C

assert C.CFG["train_samples"] == 1024 and C.CFG["val_samples"] == 128
assert C.CFG["test_samples"] == 128 and C.CFG["seed"] == 1234
assert C.CFG["stage_a_epochs"] == 2 and C.CFG["stage_b_epochs"] == 2
assert C.CFG["stage_a_learning_rate"] == 1e-3
assert C.CFG["stage_b_learning_rate"] == 1e-4
assert C.CFG["effective_batch"] == 8 and C.CFG["adapter_rank"] == 64
source, suffix, full = C.prompt_parts("2+2?")
assert (source, suffix, full) == ("Question: 2+2?\n", "Answer:", "Question: 2+2?\nAnswer:")
old = json.loads(Path(C.CFG["old_train_manifest"]).read_text())
assert old["count"] == 1024 and old["test_not_used"] is True
print("GSM8K retrained Stage-A + Hybrid Stage-B CPU tests passed")
