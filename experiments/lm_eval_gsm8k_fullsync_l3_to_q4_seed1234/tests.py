import json
from pathlib import Path

here = Path(__file__).resolve().parent
cfg = json.loads((here / "config.json").read_text())
assert cfg["pilot_limit"] == 10
assert cfg["pilot_num_fewshot"] == 0
assert cfg["batch_size"] == 1
assert cfg["max_gen_toks"] == 256
for value in ("llama_model", "qwen_model", "reference_eval", "translator_module",
              "stage_a_checkpoint", "generation_stage_b_checkpoint"):
    assert cfg[value].startswith("/home/yezhe/")
print("lm-eval Full-Sync configuration tests passed")
