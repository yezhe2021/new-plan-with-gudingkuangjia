"""CPU-only alignment and reduced-shape reverse Writer tests."""

import sys
import torch

from alignment import source_rank
from experiment import BASE, fullsync_map, settings

sys.path.insert(0, str(BASE))
from common import tokenizer  # noqa: E402
from data import manifest_rows  # noqa: E402
from translator import NativeKVTranslator, ResidualKVAdapter  # noqa: E402


def main():
    assert [source_rank(2, 1, 0)] == [1]
    assert [source_rank(1, 2, i) for i in range(2)] == [0, 0]
    cfg = settings(True)
    row = manifest_rows(cfg, "test")[0]
    qwen = tokenizer(cfg["models"]["qwen"])
    llama = tokenizer(cfg["models"]["llama"])
    target, source, _ = fullsync_map(row, qwen, llama)
    assert len(target) == len(source) == len(row["encoded"]["llama"]["option_token_indices"])
    assert len(set(target)) == len(target)

    writer = NativeKVTranslator("full36_mlp", source_layers=3, target_layers=2,
                                heads=2, dim=4, hidden_dim=5)
    key = torch.randn(1, 3, 4, 2, 4)
    mapped_k, mapped_v = writer(key, key)
    assert mapped_k.shape == mapped_v.shape == (1, 2, 4, 2, 4)
    assert all(m.bias is None for m in writer.modules() if isinstance(m, torch.nn.Linear))
    adapter = ResidualKVAdapter(layers=2, heads=2, dim=4, rank=2)
    adapted_k, adapted_v, _, _ = adapter(mapped_k, mapped_v)
    assert torch.equal(adapted_k, mapped_k) and torch.equal(adapted_v, mapped_v)
    print("Qwen/Llama Full-Sync CPU tests passed", flush=True)


if __name__ == "__main__":
    main()
