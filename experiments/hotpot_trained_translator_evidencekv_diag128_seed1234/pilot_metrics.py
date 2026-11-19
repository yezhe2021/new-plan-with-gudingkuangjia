"""HotpotQA answer EM/F1 and paired question bootstrap; no fuzzy answer extraction."""
import hashlib
import json
import re
import string
from collections import Counter
import numpy as np


def seed_for(*parts):
    return int(hashlib.sha256(json.dumps(parts,ensure_ascii=False).encode()).hexdigest()[:15],16)


def normalize(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def answer_metrics(prediction,gold):
    p,g=normalize(prediction),normalize(gold)
    em=float(p==g)
    if (p in ['yes','no','noanswer'] or g in ['yes','no','noanswer']) and p!=g:
        return dict(em=em,f1=0.)
    overlap=sum((Counter(p.split())&Counter(g.split())).values())
    f1=2.*overlap/(len(p.split())+len(g.split())) if overlap else 0.
    return dict(em=em,f1=f1)


def bootstrap(values,seed=1234):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(x))
