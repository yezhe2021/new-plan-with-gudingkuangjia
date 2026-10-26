"""Global threshold selection on pilot only. Inference features never use gold."""
import numpy as np


def features(accept,reject):
    return dict(question_gain=accept['question_logp']-reject['question_logp'],
        first_token_entropy_gain=reject['first_token_entropy']-accept['first_token_entropy'])


def decide(x,rule):
    if rule['kind']=='always_accept':return True
    if rule['kind']=='always_reject':return False
    return x[rule['kind']]>rule['threshold']


def score(samples,rule):
    accepted=[decide(s['features'],rule) for s in samples]
    chosen=[s['accept' if a else 'reject'] for s,a in zip(samples,accepted)]
    return dict(f1=float(np.mean([v['f1'] for v in chosen])),em=float(np.mean([v['em'] for v in chosen])),accept_count=sum(accepted))


def fit(records,conditions):
    samples=[dict(accept=r['conditions'][c],reject=r['conditions']['reject_all'],
        features=features(r['conditions'][c],r['conditions']['reject_all'])) for r in records for c in conditions]
    rules={k:dict(kind=k) for k in ['always_reject','always_accept']}
    for key in ['question_gain','first_token_entropy_gain']:
        values=[s['features'][key] for s in samples]
        thresholds=sorted(set([-1e30,0.,1e30]+np.quantile(values,[.1,.25,.5,.75,.9]).tolist()))
        trials=[]
        for t in thresholds:
            rule=dict(kind=key,threshold=float(t));s=score(samples,rule)
            trials.append((s['f1'],s['em'],-s['accept_count'],t,rule))
        rules[key+'_calibrated']=max(trials,key=lambda t:t[:4])[-1]
        rules[key+'_zero']=dict(kind=key,threshold=0.)
    eligible=['always_reject','always_accept','question_gain_calibrated','first_token_entropy_gain_calibrated']
    scores={name:score(samples,rule) for name,rule in rules.items()}
    selected=max(eligible,key=lambda name:(scores[name]['f1'],scores[name]['em'],-scores[name]['accept_count'],-eligible.index(name)))
    return dict(rules=rules,selected_rule=selected,calibration_scores=scores,question_count=len(records),case_count=len(samples),
        conditions=conditions,primary_metric='equal-condition mean F1',label_free_features=True)
