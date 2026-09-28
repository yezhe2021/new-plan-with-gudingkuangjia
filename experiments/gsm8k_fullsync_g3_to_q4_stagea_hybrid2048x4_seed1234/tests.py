import json
from pathlib import Path

import generation_common as C

assert C.CFG["train_samples"] == 2048 and C.CFG["val_samples"] == 128
assert C.CFG["test_samples"] == 128 and C.CFG["seed"] == 1234
assert C.CFG["stage_a_epochs"] == 2 and C.CFG["stage_b_epochs"] == 4
assert C.CFG["stage_a_train_samples"] == 1024
assert C.CFG["stage_a_learning_rate"] == 1e-3
assert C.CFG["stage_b_learning_rate"] == 1e-4
assert C.CFG["effective_batch"] == 8 and C.CFG["adapter_rank"] == 64
source, suffix, full = C.prompt_parts("2+2?")
assert (source, suffix, full) == ("Question: 2+2?\n", "Answer:", "Question: 2+2?\nAnswer:")
old = json.loads(Path(C.CFG["old_train_manifest"]).read_text())
assert old["count"] == 1024 and old["test_not_used"] is True
train, val, manifest = C.prepare_manifest()
assert len(train) == 2048 and len(val) == 128
assert set(manifest["train_indices"]).isdisjoint(manifest["val_indices"])
assert manifest["train_indices"][:1024] == old["selected_indices"]
assert all(a["question"] == b["question"] for a,b in zip(train[:1024],
    [C.read_jsonl(C.CFG["data_train"])[i] for i in manifest["train_indices"][:1024]]))
print("GSM8K Gemma→Qwen 2048x4 CPU tests passed")
