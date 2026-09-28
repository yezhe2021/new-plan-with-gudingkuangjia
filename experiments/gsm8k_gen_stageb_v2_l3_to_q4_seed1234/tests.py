import json
from pathlib import Path

from generation_common import CFG, prompt_parts

assert CFG["seed"] == 1234
assert CFG["train_samples"] == 1024 and CFG["val_samples"] == 128 and CFG["test_samples"] == 128
assert CFG["epochs"] == 2 and CFG["effective_batch"] == 8
assert CFG["adapter_rank"] == 64 and CFG["gradient_clip"] == 30
assert CFG["eos_weight"] == 0.1 and CFG["hybrid_ce_weight"] == 0.1
source, suffix, full = prompt_parts("2+2?")
assert source == "Question: 2+2?\n"
assert suffix == "Answer:" and full == "Question: 2+2?\nAnswer:"
old = json.loads(Path(CFG["old_train_manifest"]).read_text(encoding="utf-8"))
assert old["count"] == 1024 and old["test_not_used"] is True
print("Generation Stage-B v2 CPU tests passed")
