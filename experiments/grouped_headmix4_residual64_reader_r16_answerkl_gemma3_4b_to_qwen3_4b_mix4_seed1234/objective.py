"""Answer-boundary contract for the current MCQ datasets, independent of gold."""


def mcq_positions(row, encode_label):
    if row['task'] not in {'openbookqa', 'arc_challenge', 'mmlu_pro'}:
        raise ValueError('Answer-boundary training only supports configured MCQ tasks')
    candidates = row.get('candidates')
    if not candidates:
        raise ValueError('MCQ requires explicit option labels')
    encoded = [list(encode_label(label)) for label in candidates]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError('Multi-token labels require a separate full-label objective; refusing first-token truncation')
    if len({ids[0] for ids in encoded}) != len(encoded):
        raise ValueError('Option labels must have distinct token IDs')
    # No gold answer or generated label is fed into the input. Position 0 is
    # the next-token distribution at the end of the existing receiver suffix.
    return [0]
