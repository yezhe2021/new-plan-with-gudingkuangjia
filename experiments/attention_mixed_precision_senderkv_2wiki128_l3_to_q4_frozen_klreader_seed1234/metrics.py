"""Alignment and statistics without model/framework dependencies."""
import re
import math
import numpy as np
from scipy.stats import rankdata


def text_units(text):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+(?:\s+|$)", text)]
    assert spans and spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    return spans


def projection(offsets, evidence_start, units):
    matrix = np.zeros((len(offsets), len(units)), dtype=np.float64)
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        for j, (c, d) in enumerate(units):
            matrix[i, j] = max(0, min(b, evidence_start + d) - max(a, evidence_start + c)) / (b - a)
    assert np.all(matrix.sum(1) <= 1 + 1e-9)
    return matrix


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all() and x.min() >= -1e-12
    x = np.maximum(x, 0)
    mass = x.sum(-1, keepdims=True)
    return np.divide(x, mass, out=np.zeros_like(x), where=mass > 0)


def jsd(p, q):
    p, q = normalize(p), normalize(q)
    if p.sum() == 0 or q.sum() == 0:
        return None
    middle = (p + q) / 2
    a, b = p > 0, q > 0
    return float((np.sum(p[a] * np.log2(p[a] / middle[a])) + np.sum(q[b] * np.log2(q[b] / middle[b]))) / 2)


def top_membership(scores, k):
    threshold = np.partition(scores, -k)[-k]
    above, equal = scores > threshold, scores == threshold
    return above.astype(float) + equal * (k - above.sum()) / equal.sum()


def compare(p, q, fractions, seed=0):
    p, q = np.asarray(p), np.asarray(q)
    assert p.shape == q.shape and p.ndim == 1 and len(p) >= 2
    ranks_p, ranks_q = rankdata(p), rankdata(q)
    rho = None if np.std(ranks_p) == 0 or np.std(ranks_q) == 0 else float(np.corrcoef(ranks_p, ranks_q)[0, 1])
    result = {"spearman": rho, "jsd_bits": jsd(p, q), "units": len(p)}
    rng = np.random.default_rng(seed)
    for fraction in fractions:
        k = max(1, min(len(p), math.ceil(fraction * len(p))))
        # Independent seeded tie-breaking prevents a shared position tie-break
        # from creating artificial overlap on uniform/low-information heads.
        pa = np.lexsort((rng.random(len(p)), -p))[:k]
        qa = np.lexsort((rng.random(len(q)), -q))[:k]
        overlap = len(set(pa) & set(qa)) / k
        chance = k / len(p)
        result[str(fraction)] = dict(k=k, overlap=overlap, chance=chance,
            chance_adjusted=(overlap - chance) / (1 - chance) if k < len(p) else None,
            expected_overlap_under_ties=float(np.dot(top_membership(p, k), top_membership(q, k)) / k))
    return result


def to_token_scores(unit_scores, matrix):
    weights = matrix.sum(1)
    scores = np.divide(matrix @ unit_scores, weights, out=np.zeros(len(matrix)), where=weights > 0)
    return scores, np.flatnonzero(weights > 0)


def ranked_positions(scores, eligible, budget, reverse=False, seed=0):
    assert 0 < budget <= len(eligible)
    rng = np.random.default_rng(seed)
    ranking = np.lexsort((rng.random(len(eligible)), scores[eligible] if reverse else -scores[eligible]))
    return sorted(int(x) for x in eligible[ranking[:budget]])


def bootstrap(values, seed=0, repetitions=1000):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(values):
        return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(1)
    return dict(n=len(values), mean=float(values.mean()), ci95=np.quantile(sampled, [.025, .975]).tolist())


def answer_normalize(text):
    import string
    import unicodedata
    text = unicodedata.normalize("NFKC", text).lower().strip().split("\n")[0]
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())
