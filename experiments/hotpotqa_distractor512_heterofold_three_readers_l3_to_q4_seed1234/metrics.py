"""HotpotQA answer normalization, EM/F1 and paired question bootstrap."""
import re
import string
from collections import Counter


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_metrics(prediction, gold):
    p, g = normalize(prediction), normalize(gold)
    em = float(p == g)
    if (p in ['yes', 'no', 'noanswer'] or g in ['yes', 'no', 'noanswer']) and p != g:
        return dict(em=em, f1=0.)
    overlap = sum((Counter(p.split()) & Counter(g.split())).values())
    f1 = 2. * overlap / (len(p.split()) + len(g.split())) if overlap else 0.
    return dict(em=em, f1=f1)


def clean_prediction(text):
    return re.sub(r'<\|[^<>]*\|>', '', text).strip()


def paired_interval(values, seed=1234):
    import numpy as np
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return dict(n=len(x), mean=float(x.mean()), ci95=np.quantile(means, [.025, .975]).tolist())
