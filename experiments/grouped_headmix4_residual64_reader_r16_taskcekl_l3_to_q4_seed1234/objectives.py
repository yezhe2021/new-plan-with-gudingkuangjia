"""Explicit candidate loss and reference-answer block alignment."""
import re
import torch
import torch.nn.functional as F


def clean_solution(text):
    # Remove GSM8K calculator annotations, retaining the written calculation/result.
    return re.sub(r'<<[^>]*>>', '', text).strip()


def kl(teacher, student):
    t = teacher.float().log_softmax(-1).detach()
    s = student.float().log_softmax(-1)
    return F.kl_div(s, t, log_target=True, reduction='none').sum(-1).mean()


def candidate_loss(teacher, student, label_ids, gold, weight):
    assert len(label_ids) == len(set(label_ids)) and 0 <= gold < len(label_ids)
    t, s = teacher[:, label_ids], student[:, label_ids]
    ce = F.cross_entropy(s.float(), torch.tensor([gold], device=s.device))
    auxiliary = kl(t, s)
    return ce + weight * auxiliary, ce, auxiliary, int(s.argmax(-1).item())


def answer_blocks(suffix, targets, size):
    """Inputs are preceding tokens; each target is scored exactly once."""
    assert suffix and targets and size > 0
    n = len(targets)
    for start in range(0, n, size):
        end = min(n, start + size)
        if start == 0:
            inputs = list(suffix) + targets[:end - 1]
            offset = len(suffix) - 1
        else:
            inputs = targets[start - 1:end - 1]
            offset = 0
        yield dict(start=start, end=end, inputs=inputs, offset=offset,
                   labels=targets[start:end], weight=(end - start) / n)
