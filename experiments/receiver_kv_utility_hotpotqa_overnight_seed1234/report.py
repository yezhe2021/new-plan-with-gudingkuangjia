"""Held-out policy evaluation; question-clustered intervals and utility diagnostics."""
import backend as B
from backend import ROOT,CFG,np,read,save,sha,ident,E
from estimators import records,cases,predict
from pilot_metrics import bootstrap
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from pathlib import Path
import joblib,json,time,fcntl,os,csv


def diagnostic(samples,score,target):
    truth=np.array([s['utility_'+target] for s in samples]);em=np.array([s['utility_em'] for s in samples]);active=em!=0
    result=dict(n=len(samples),unique_questions=len({s['id'] for s in samples}),helpful=int((em>0).sum()),harmful=int((em<0).sum()),
        informative_unique_questions=len({s['id'] for s,keep in zip(samples,active) if keep}),
        mse=float(np.mean((score-truth)**2)),spearman=float(spearmanr(score,truth).statistic) if np.std(score)>0 and np.std(truth)>0 else None,
        helpful_vs_harmful_auroc=float(roc_auc_score(em[active]>0,score[active])) if len(np.unique(em[active]))==2 else None)
    bounds=np.unique(np.quantile(score,np.linspace(0,1,6)));bins=[]
    if len(bounds)>1:
        for i in range(len(bounds)-1):
            ix=(score>=bounds[i])&(score<=bounds[i+1] if i==len(bounds)-2 else score<bounds[i+1])
            if ix.any():bins.append(dict(n=int(ix.sum()),predicted=float(score[ix].mean()),observed=float(truth[ix].mean())))
    else:bins=[dict(n=len(samples),predicted=float(score.mean()),observed=float(truth.mean()))]
    result['utility_calibration_bins']=bins;return result


def policy_metrics(samples,score,threshold):
    take=score>threshold;chosen=[s['accept' if a else 'reject'] for s,a in zip(samples,take)]
    y=np.array([s['utility_em'] for s in samples]);helpful=y>0;harmful=y<0
    return dict(n_cases=len(samples),accept_rate=float(take.mean()),em=float(np.mean([s['em'] for s in chosen])),
        f1=float(np.mean([s['f1'] for s in chosen])),helpful_recall=float(take[helpful].mean()) if helpful.any() else None,
        harmful_rejection_rate=float((~take[harmful]).mean()) if harmful.any() else None,
        helpful_cases=int(helpful.sum()),harmful_cases=int(harmful.sum()))


def main():
    B.verify();assert read(ROOT/'test_completed.json')['manifest_sha256']==ident()
    lock=read(ROOT/'selector.json');assert lock['manifest_sha256']==ident() and lock['estimator_sha256']==sha(ROOT/'estimators.joblib')
    models=joblib.load(ROOT/'estimators.joblib');rr=records('test');conditions=CFG['conditions']+CFG['stress_conditions']
    assert all(r['selector_sha256']==sha(ROOT/'selector.json') for r in rr)
    allcases=cases(rr,conditions);scores={name:predict(e,allcases) for name,e in models.items()}
    decisions={name:v>models[name]['threshold'] for name,v in scores.items()}
    selected={name:{metric:np.array([s['accept' if a else 'reject'][metric] for s,a in zip(allcases,take)]) for metric in ['em','f1']} for name,take in decisions.items()}
    summary=dict(experiment=CFG['name'],status='completed',timestamp=time.strftime('%F %T'),manifest_sha256=ident(),
        counts=CFG['counts'],selector_sha256=sha(ROOT/'selector.json'),primary_model=lock['primary_model'],
        conditions={},mixtures={},diagnostics={},baselines={},runtime={})
    for name in ['compact','native','reject']:
        summary['baselines'][name]={m:float(np.mean([r['baselines'][name][m] for r in rr])) for m in ['em','f1','answer_token_nll']}
    for c in conditions:
        ix=np.array([s['condition']==c for s in allcases]);group=[s for s in allcases if s['condition']==c]
        summary['conditions'][c]={name:policy_metrics(group,values[ix],models[name]['threshold']) for name,values in scores.items()}
        summary['diagnostics'][c]={name:diagnostic(group,values[ix],'em' if name.endswith('_em') or name=='calibrated_confidence' else 'f1') for name,values in scores.items() if name not in ['always_accept','always_reject']}
    ids=[r['id'] for r in rr]
    for groupname,cc in [('in_distribution',CFG['conditions']),('unseen_corruptions',CFG['stress_conditions'])]:
        ix=np.array([s['condition'] in cc for s in allcases]);group=[s for s in allcases if s['condition'] in cc]
        matrix_indices=np.array([[i for i,s in enumerate(allcases) if s['id']==rid and s['condition'] in cc] for rid in ids])
        assert matrix_indices.shape==(len(rr),len(cc))
        output={}
        for name,values in scores.items():
            m=policy_metrics(group,values[ix],models[name]['threshold'])
            for metric in ['em','f1']:
                per_question=selected[name][metric][matrix_indices].mean(1)
                m[metric+'_interval']=bootstrap(per_question)
                m[metric+'_gains']={baseline:bootstrap((selected[name][metric]-selected[baseline][metric])[matrix_indices].mean(1))
                    for baseline in ['always_accept','always_reject','question_gain_zero','calibrated_confidence','utility_output_f1','utility_cross_f1']}
                a=np.array([s['accept'][metric] for s in allcases]);b=np.array([s['reject'][metric] for s in allcases])
                m[metric+'_rate_matched_random_expected']=float((m['accept_rate']*a[ix]+(1-m['accept_rate'])*b[ix]).mean())
            output[name]=m
        summary['mixtures'][groupname]=dict(conditions=cc,n_questions=len(rr),n_cases=len(group),policies=output,
            oracle={metric:float(np.mean([max(s['accept'][metric],s['reject'][metric]) for s in group])) for metric in ['em','f1']},
            both_wrong_cases=sum(s['accept']['em']==s['reject']['em']==0 for s in group))
    summary['runtime']=dict(test_wall_seconds=read(ROOT/'test_completed.json')['elapsed_seconds'],
        development_wall_seconds=read(ROOT/'development_completed.json')['elapsed_seconds'],
        mean_feature_seconds=float(np.mean([s['feature_seconds'] for s in allcases])),
        mean_case_seconds=float(np.mean([s['case_seconds'] for s in allcases])),
        perturbations_per_case=len(CFG['perturbations']),candidate_sequences_per_probe=2,
        both_accept_and_reject_generations_required=True,persistent_kv_tensors=False)
    summary['caveats']=['Controlled two-gold-paragraph HotpotQA bridge subset; not official distractor benchmark.',
        'All hypotheses, model grids and thresholds frozen before test inference; primary model is utility_stability8_f1.',
        'Equal clean/wrong/partial50 mixture is a controlled distribution; stress corruption types excluded from fitting and calibration.',
        'Question/document disjointness includes repeated corruption versions and donor sources.',
        'Gaussian retains target aggregate moments; synthetic corruption does not establish reliability on natural retrieval errors.',
        'Candidate pair scores are sequence likelihood features, not calibrated probabilities of correctness.',
        'No inference compute reduction claimed. Stability2/4/8 use nested subsets of the same probes.',
        'Reported intervals resample original questions, preserving all their corruption variants. Exploratory secondary comparisons are not multiplicity-corrected.']
    save(ROOT/'summary.json',summary)
    with (ROOT/'evaluation/decisions.jsonl').open('w') as f:
        for i,s in enumerate(allcases):
            f.write(json.dumps(dict(id=s['id'],condition=s['condition'],selector_sha256=sha(ROOT/'selector.json'),
                policies={name:dict(utility=float(values[i]),accept=bool(decisions[name][i])) for name,values in scores.items()}))+'\n')
    lines=['# Overnight HotpotQA KV utility experiment','',f'Train {CFG["counts"]["train"]}, calibration {CFG["counts"]["calibration"]}, independent test {len(rr)}.',
        '',f'Primary model: `{lock["primary_model"]}`. F1 is the primary metric; EM is secondary.','',
        '|Policy|EM|F1|Accept rate|Helpful recall|Harmful rejection|','|---|---:|---:|---:|---:|---:|']
    for name in ['always_accept','always_reject','question_gain_zero','question_gain_calibrated','calibrated_confidence','utility_output_f1','utility_cross_f1','utility_stability2_f1','utility_stability4_f1','utility_stability8_f1']:
        m=summary['mixtures']['in_distribution']['policies'][name]
        lines.append(f'|{name}|{m["em"]:.4f}|{m["f1"]:.4f}|{m["accept_rate"]:.4f}|{m["helpful_recall"]}|{m["harmful_rejection_rate"]}|')
    lines+=['','Full paired intervals, per-condition results, unseen corruption stress tests and utility AUROC/calibration: summary.json.','',*summary['caveats']]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    shared=dict(experiment=CFG['name'],counts=CFG['counts'],primary_model=lock['primary_model'],
        in_distribution=summary['mixtures']['in_distribution'],result=str(ROOT/'summary.json'))
    with Path(CFG['shared_results']).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(shared,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=ident(),no_persistent_kv=True))
    B.status(status='completed',phase='all');E.log('OVERNIGHT EXPERIMENT COMPLETE')


if __name__=='__main__':main()
