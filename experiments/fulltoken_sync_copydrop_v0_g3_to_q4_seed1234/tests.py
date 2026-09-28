"""CPU-only alignment and reduced-shape Writer smoke tests."""

import torch

from alignment import source_rank
from experiment import BASE, fullsync_map, settings

import sys
sys.path.insert(0, str(BASE))
from common import tokenizer  # noqa: E402
from data import manifest_rows  # noqa: E402
from translator import NativeKVTranslator, ResidualKVAdapter  # noqa: E402


def main():
    assert [source_rank(2, 1, 0)] == [1]
    assert [source_rank(1, 2, i) for i in range(2)] == [0, 0]
    cfg = settings(True)
    row = manifest_rows(cfg, "test")[0]
    gemma = tokenizer(cfg["models"]["gemma"])
    qwen = tokenizer(cfg["models"]["qwen"])
    target, source, _ = fullsync_map(row, gemma, qwen)
    assert len(target) == len(source) == len(row["encoded"]["qwen"]["option_token_indices"])
    assert len(set(target)) == len(target)

    writer = NativeKVTranslator("full34_headmix256", source_layers=3, target_layers=2,
                                source_heads=2, target_heads=4, source_dim=4,
                                target_dim=2, hidden_dim=5, depth_output_dim=4,
                                head_mapping="full_head")
    key = torch.randn(1, 3, 4, 2, 4)
    mapped_k, mapped_v = writer(key, key)
    assert mapped_k.shape == mapped_v.shape == (1, 2, 4, 4, 2)
    assert all(m.bias is None for m in writer.modules() if isinstance(m, torch.nn.Linear))
    adapter = ResidualKVAdapter(layers=2, heads=4, dim=2, rank=2)
    adapted_k, adapted_v, _, _ = adapter(mapped_k, mapped_v)
    assert torch.equal(adapted_k, mapped_k) and torch.equal(adapted_v, mapped_v)
    print("Gemma/Qwen Full-Sync CPU tests passed", flush=True)


if __name__ == "__main__":
    main()
