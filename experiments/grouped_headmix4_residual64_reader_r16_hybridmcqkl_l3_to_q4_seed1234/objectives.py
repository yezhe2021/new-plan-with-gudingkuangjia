import re
import torch
import torch.nn.functional as F

def clean_controls(text):
    return re.sub(r"<\|[^>]*\|>", "", text)


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

