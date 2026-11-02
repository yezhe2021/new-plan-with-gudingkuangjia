"""Answer-only scoring matching original 2Wiki and MuSiQue-Ans conventions."""
import re
import string
from collections import Counter


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_one(prediction, gold, dataset):
    p, g = normalize(prediction), normalize(gold)
    em = float(p == g)
    if dataset == '2wiki' and ((p in ['yes','no','noanswer'] or g in ['yes','no','noanswer']) and p != g):
        return dict(em=em, f1=0.)
    pt, gt = p.split(), g.split()
    if dataset == 'musique' and (not pt or not gt):
        return dict(em=em, f1=float(pt == gt))
    overlap = sum((Counter(pt) & Counter(gt)).values())
    return dict(em=em, f1=2.*overlap/(len(pt)+len(gt)) if overlap else 0.)


def answer_metrics(prediction, answers, dataset):
    assert dataset in ['2wiki','musique'] and answers and all(isinstance(s,str) for s in answers)
    values = [score_one(prediction, gold, dataset) for gold in answers]
    return {key:max(v[key] for v in values) for key in ['em','f1']}
