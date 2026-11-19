"""HeteroFold TA: latest shared character-END, not interval Copy/Drop."""
from bisect import bisect_right


def align_tokens(sender_offsets, receiver_offsets):
    sender = {b:i for i,(a,b) in enumerate(sender_offsets) if b>a}
    valid = [i for i,(a,b) in enumerate(sender_offsets) if b>a]
    if not valid: raise ValueError('No nonempty source text tokens')
    shared = sorted(set(sender)&{b for a,b in receiver_offsets if b>a})
    result = []
    for a,b in receiver_offsets:
        if b<=a: raise ValueError('Special/empty target offsets require separate handling')
        p = bisect_right(shared,b)-1
        result.append(sender[shared[p]] if p>=0 else valid[0])
    return result
