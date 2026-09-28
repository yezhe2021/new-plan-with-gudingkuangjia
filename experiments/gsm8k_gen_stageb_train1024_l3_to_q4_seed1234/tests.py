import json
from pathlib import Path

cfg = json.loads((Path(__file__).parent / "config.json").read_text())
assert cfg["train_samples"] == 1024
assert cfg["epochs"] == 2
assert cfg["effective_batch"] == 8
assert cfg["adapter_rank"] == 64
assert cfg["learning_rate"] == 0.0001
assert cfg["test_samples"] == 32
print("Generation-only Stage-B CPU configuration tests passed")
