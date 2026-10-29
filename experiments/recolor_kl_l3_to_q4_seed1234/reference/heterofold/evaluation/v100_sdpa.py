"""V100 evaluation-only SDPA dispatch, preserving attention math and context."""
import os


def configure():
    if os.environ.get('HETEROFOLD_V100_SDPA') != '1':
        return
    # Torch 2.5 memory-efficient attention does not implement enable_gqa=True.
    # Transformers already has the equivalent explicit repeat_kv code path.
    from transformers.integrations import sdpa_attention
    sdpa_attention.use_gqa_in_sdpa = explicit_gqa


def explicit_gqa(attention_mask, key):
    return False
