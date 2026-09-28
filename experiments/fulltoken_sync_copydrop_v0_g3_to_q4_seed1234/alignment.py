"""Ordered Gemma-to-Qwen copy/drop on shared UTF-8 causal boundaries."""

import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYNC = HERE.parent / "sync_boundary_nativekv_full28_diagonal_translator_llama3_2_3b_to_qwen3_4b_seed1234"
sys.path.insert(0, str(SYNC))
from offsets import token_spans  # noqa: E402
from units import build_synchronized_units  # noqa: E402


def source_rank(source_count, target_count, target_rank):
    if source_count < 1 or target_count < 1 or not 0 <= target_rank < target_count:
        raise ValueError((source_count, target_count, target_rank))
    if source_count > target_count:
        return math.ceil((target_rank + 1) * source_count / target_count) - 1
    return target_rank * source_count // target_count


def fullsync_map(row, gemma_tok, qwen_tok):
    text = row["body"]
    gemma, qwen = row["encoded"]["gemma"], row["encoded"]["qwen"]
    source_spans = token_spans(gemma_tok, text, gemma["body"])
    target_spans = token_spans(qwen_tok, text, qwen["body"])
    # The upstream SyncUnit names its first stream "llama"; the algorithm is
    # tokenizer-independent and that first stream is Gemma here.
    units = build_synchronized_units(text, source_spans, target_spans)
    option_indices = list(qwen["option_token_indices"])
    option_set = set(option_indices)
    mapping, counts = {}, Counter()
    for unit in units:
        source, target = unit.llama_indices, unit.qwen_indices
        m, n = len(source), len(target)
        touched = [index for index in target if index in option_set]
        if not touched:
            continue
        kind = ("one_to_one" if m == n == 1 else
                "copy" if m == 1 else "drop" if n == 1 else "many_to_many")
        counts[kind + "_units"] += 1
        counts[kind + "_target_tokens"] += len(touched)
        for rank, target_index in enumerate(target):
            if target_index in option_set:
                if target_index in mapping:
                    raise RuntimeError(f"Duplicate Qwen token {target_index}: {row['id']}")
                mapping[target_index] = source[source_rank(m, n, rank)]
    if set(mapping) != option_set:
        missing = sorted(option_set - set(mapping))
        raise RuntimeError(f"Unmapped Qwen Options tokens in {row['id']}: {missing[:16]}")
    source_indices = [mapping[index] for index in option_indices]
    if any(not 0 <= index < len(gemma["body"]) for index in source_indices):
        raise RuntimeError(f"Out-of-range Gemma source index: {row['id']}")
    return option_indices, source_indices, dict(counts)
