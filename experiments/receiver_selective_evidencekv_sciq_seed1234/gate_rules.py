"""Label-free inference decisions; calibration labels only choose global thresholds."""
import hashlib
import json
import numpy as np


def seed_for(*parts):
    return int(hashlib.sha256(json.dumps(parts,ensure_ascii=False).encode()).hexdigest()[:15],16)


def feature(accept,reject):
    return dict(question_gain=accept['question_logp']-reject['question_logp'],
                confidence_gain=reject['choice_entropy']-accept['choice_entropy'])


def decide(features,rule):
    kind=rule['kind']
    if kind=='always_accept':return True
    if kind=='always_reject':return False
    if kind in ['question_gain','confidence_gain']:return features[kind]>rule['threshold']
    raise ValueError(kind)


def fit(calibration):
    """Fit one threshold per feature; tie-break toward fewer accepted packets."""
    rules=dict(always_accept=dict(kind='always_accept'),always_reject=dict(kind='always_reject'))
    for key in ['question_gain','confidence_gain']:
        values=np.array([r['features'][key] for r in calibration])
        # Small preregistered grid; +/-1e30 include accept/reject all.
        thresholds=sorted(set([-1e30,0.,1e30]+np.quantile(values,[.1,.25,.5,.75,.9]).tolist()))
        candidates=[]
        for threshold in thresholds:
            rule=dict(kind=key,threshold=float(threshold))
            accepted=[decide(r['features'],rule) for r in calibration]
            correct=sum(r['accept' if a else 'reject']['correct'] for r,a in zip(calibration,accepted))
            candidates.append((correct,-sum(accepted),float(threshold),rule))
        best=max(candidates,key=lambda x:x[:3])
        rules[key+'_calibrated']=best[-1]
        rules[key+'_zero']=dict(kind=key,threshold=0.)
    # Select deployed rule on calibration only. Preserve both families as diagnostics.
    eligible=['always_reject','always_accept','question_gain_calibrated','confidence_gain_calibrated']
    scores={name:dict(correct=sum(r['accept' if decide(r['features'],rules[name]) else 'reject']['correct'] for r in calibration),
                     accept_count=sum(decide(r['features'],rules[name]) for r in calibration)) for name in eligible}
    selected=max(eligible,key=lambda name:(scores[name]['correct'],-scores[name]['accept_count'],-eligible.index(name)))
    return dict(rules=rules,selected_rule=selected,calibration_scores=scores,n=len(calibration))


def bootstrap(values,seed=1234):
    x=np.array(values,dtype=float);rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(x))
