"""Per-question associations; bootstrap whole questions, never individual heads."""
import json
import numpy as np


def ranks(a):
    a=np.asarray(a,dtype=float);order=np.argsort(a,axis=-1,kind='stable')
    values=np.take_along_axis(a,order,axis=-1);n=a.shape[-1];p=np.arange(n)
    starts=np.concatenate([np.ones(a.shape[:-1]+(1,),bool),np.diff(values,axis=-1)!=0],axis=-1)
    ends=np.concatenate([np.diff(values,axis=-1)!=0,np.ones(a.shape[:-1]+(1,),bool)],axis=-1)
    lo=np.maximum.accumulate(np.where(starts,p,0),axis=-1)
    hi=np.minimum.accumulate(np.where(ends,p,n-1)[...,::-1],axis=-1)[...,::-1]
    ranked=np.empty_like(a);np.put_along_axis(ranked,order,(lo+hi)/2+1,axis=-1)
    return ranked


def residual_rank(a,groups):
    out=ranks(a)
    for g in sorted(set(groups)):
        mask=groups==g;out[...,mask]-=out[...,mask].mean(axis=-1,keepdims=True)
    return out


def rho(a,b):
    den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),den,out=np.full_like(den,np.nan,dtype=float),where=den>1e-20)


def finite(v):return float(v) if np.isfinite(v) else None
def interval(v):
    v=np.asarray(v);good=v[np.isfinite(v)]
    return [float(x) for x in np.quantile(good,[.025,.975])] if len(good)>=.95*len(v) else None


def extract(r):
    d=r['diagnostics'];fixed=d['fixed_native_q'];trans=d['fixed_translated_q'];actual=d['actual_q']['TT']['attention']['mean']
    f={k:v for k,v in r['distances'].items() if k!='per_layer'}
    f.update(route_kl=fixed['cases']['TT']['route_kl'],wo_nmse=fixed['cases']['TT']['wo_nmse'],
        translated_query_route_kl=trans['cases']['TT']['route_kl'],translated_query_wo_nmse=trans['cases']['TT']['wo_nmse'],
        actual_route_kl=actual['route_kl'],actual_wo_nmse=actual['actual_wo_nmse'],query_relative_l2=actual['query_native_relative_l2'],
        compatibility_synergy=fixed['output_damage_synergy'],output_interaction_nmse=fixed['output_interaction_nmse'])
    targets={}
    for branch in ['actual_q','clamped_native_q']:
        for metric,name in [('target_token_nll','nll'),('next_token_kl','kl')]:
            targets[branch+'_'+name+'_gain']=d[branch]['TT']['functional'][metric]-d[branch]['NN']['functional'][metric]
        targets[branch+'_interaction_nll']=d['functional_interactions'][branch]['target_token_nll']
    targets['clamped_k_only_nll_gain']=d['clamped_native_q']['TT']['functional']['target_token_nll']-d['clamped_native_q']['NT']['functional']['target_token_nll']
    targets['clamped_v_only_nll_gain']=d['clamped_native_q']['TT']['functional']['target_token_nll']-d['clamped_native_q']['TN']['functional']['target_token_nll']
    key='correct' if r['task']=='arc' else 'f1'
    targets['oracle_task_gain']=r['oracle_free']['NN'][key]-r['oracle_free']['TT'][key]
    if r['task']=='arc':
        targets['historical_choice_margin_gain']=r['historical_choice_scores']['NN']['gold_margin']-r['historical_choice_scores']['TT']['gold_margin']
        for branch in ['actual_q','clamped_native_q']:
            targets[branch+'_correct_option_margin_gain']=d[branch]['NN']['functional']['correct_option_margin']-d[branch]['TT']['functional']['correct_option_margin']
    return f,targets


def associations(records,cfg):
    records=sorted(records,key=lambda r:(r['id'],r['group']))
    ids=sorted({r['id'] for r in records});assert len(ids)==64 and len(records)==192
    assert all([r['id'] for r in records[i:i+3]]==[records[i]['id']]*3 for i in range(0,192,3))
    group_names=sorted(cfg['groups']);groups=np.array([group_names.index(r['group']) for r in records])
    extracted=[extract(r) for r in records]
    features={k:np.array([v[0][k] for v in extracted]) for k in extracted[0][0]}
    targets={k:np.array([v[1][k] for v in extracted]) for k in extracted[0][1]}
    rng=np.random.default_rng(cfg['seed']);draw=rng.integers(0,64,size=(cfg['bootstrap_samples'],64))
    indices=(draw[:,:,None]*3+np.arange(3)[None,None,:]).reshape(cfg['bootstrap_samples'],192)
    fr={k:residual_rank(v,groups) for k,v in features.items()};tr={k:residual_rank(v,groups) for k,v in targets.items()}
    fb={k:residual_rank(v[indices],groups) for k,v in features.items()};tb={k:residual_rank(v[indices],groups) for k,v in targets.items()}
    table={};boot={}
    for feature in features:
        table[feature]={};boot[feature]={}
        for target in targets:
            value=rho(fr[feature],tr[target]);samples=rho(fb[feature],tb[target]);boot[feature][target]=samples
            bygroup={}
            for g in group_names:
                ix=groups==group_names.index(g);x=ranks(features[feature][ix]);y=ranks(targets[target][ix]);x-=x.mean();y-=y.mean()
                bygroup[g]=finite(rho(x,y))
            table[feature][target]=dict(partial_spearman=finite(value),question_cluster_ci95=interval(samples),
                by_group_spearman=bygroup,n_questions=64,n_group_records=192)
    target='clamped_native_q_nll_gain';contrasts={}
    for feature in ['route_kl','wo_nmse','compatibility_synergy','query_relative_l2']:
        a=table[feature][target]['partial_spearman'];b=table['kv_relative_l2'][target]['partial_spearman']
        contrasts[feature+'_minus_kv_l2']=dict(delta_rho=None if a is None or b is None else a-b,
            paired_question_cluster_ci95=interval(boot[feature][target]-boot['kv_relative_l2'][target]))
    return dict(associations=table,metric_comparisons=contrasts,
        mean_features={k:float(v.mean()) for k,v in features.items()},mean_targets={k:float(v.mean()) for k,v in targets.items()},
        method='Rank variables, remove group fixed effects, correlate residuals; resample64 question clusters with all3 groups together. No head/token treated as an independent sample.')


def report(C,root,cfg,manifest):
    all_records=[]
    for task in cfg['test_counts']:
        done=C.read(root/f'evaluation/{task}/completed.json');assert done['manifest_sha256']==manifest()
        for p in sorted((root/f'evaluation/{task}/per_group').rglob('*.json')):
            v=C.read(p);assert v['manifest_sha256']==manifest();all_records.append(v)
    assert len(all_records)==128*3*2*3
    values={}
    for task in cfg['test_counts']:
        values[task]={}
        for method in cfg['methods']:
            values[task][method]={}
            for context in ['gold_answer','student_prefix']:
                selected=[r for r in all_records if r['task']==task and r['method']==method and r['context']==context]
                result=associations(selected,cfg);values[task][method][context]=result
                C.save(root/f'analysis/{task}/{method}/{context}.json',dict(manifest_sha256=manifest(),**result))
    supported={}
    for task in cfg['test_counts']:
        supported[task]={'route':[],'compatibility':[]}
        for method in cfg['methods']:
            gold=values[task][method]['gold_answer'];student=values[task][method]['student_prefix']
            route=gold['metric_comparisons']['route_kl_minus_kv_l2']
            confirm=student['metric_comparisons']['route_kl_minus_kv_l2']['delta_rho']
            positive=gold['associations']['route_kl']['clamped_native_q_nll_gain']['partial_spearman']
            if route['paired_question_cluster_ci95'] and route['paired_question_cluster_ci95'][0]>0 and positive is not None and positive>0 and confirm is not None and confirm>0:
                supported[task]['route'].append(method)
            output=gold['metric_comparisons']['wo_nmse_minus_kv_l2']
            inter=gold['associations']['compatibility_synergy']['clamped_native_q_interaction_nll']
            student_inter=student['associations']['compatibility_synergy']['clamped_native_q_interaction_nll']['partial_spearman']
            if output['paired_question_cluster_ci95'] and output['paired_question_cluster_ci95'][0]>0 and inter['question_cluster_ci95'] and inter['question_cluster_ci95'][0]>0 and student_inter is not None and student_inter>0:
                supported[task]['compatibility'].append(method)
    a=all(len(v['route'])>=2 for v in supported.values());b=all(len(v['compatibility'])>=2 for v in supported.values())
    label='A_and_B_exploratory_candidates' if a and b else 'A_route_exploratory_candidate' if a else 'B_compatibility_exploratory_candidate' if b else 'C_no_stable_signal_in_this_cohort'
    summary=dict(manifest_sha256=manifest(),status='completed',phase=1,exploratory_signal=label,
        candidate_methods=supported,statistics=values,config=cfg,
        interpretation='No independent128-question confirmation has been run. C means insufficient stable evidence in this selected64-question cohort, not proof that the mechanisms do not exist.')
    C.save(root/'summary.json',summary)
    C.write_jsonl(root/'per_sample_analysis.jsonl',[dict(id=r['id'],task=r['task'],method=r['method'],context=r['context'],group=r['group'],
        predictors=extract(r)[0],outcomes=extract(r)[1]) for r in all_records])
    lines=['# KV functional sensitivity: Phase1, A/C/D, ARC64+Hotpot64','',f'Exploratory signal: **{label}**.','',
        '| Task | Method | Context | K/V L2 ρ | Route KL ρ | Output NMSE ρ | Route−L2 Δρ95%CI |',
        '|---|---|---|---:|---:|---:|---|']
    def fmt(v):return 'undefined' if v is None else f'{v:.3f}'
    for task in values:
        for method in values[task]:
            for context,v in values[task][method].items():
                target='clamped_native_q_nll_gain'
                rr=[v['associations'][f][target]['partial_spearman'] for f in ['kv_relative_l2','route_kl','wo_nmse']]
                ci=v['metric_comparisons']['route_kl_minus_kv_l2']['paired_question_cluster_ci95']
                lines.append('|'+task+'|'+method+'|'+context+'|'+'|'.join(fmt(x) for x in rr)+'|'+str(ci)+'|')
    lines+=['','Primary table outcome: native-KV repair improvement in target-token NLL under group-native-Q clamp. Gold context uses all gold answer tokens; student context uses the same fixed selected positions in the unmodified student rollout.','',
        'Full per-question diagnostics, actual-Q controls, K-only/V-only effects, output interactions, route/head statistics and Oracle free-answer gains are in per_group JSON and analysis/ summaries.','',
        *['- '+s for s in cfg['caveats']]]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    # Tied ranks and clustered partial correlations are scientific checks.
    assert np.array_equal(ranks(np.array([3.,1.,1.,4.])),np.array([3.,1.5,1.5,4.]))
    x=np.array([[1.,2.,2.],[3.,3.,1.]])
    assert np.array_equal(ranks(x),np.array([[1.,2.5,2.5],[2.5,2.5,1.]]))
    g=np.array([0,1,0,1]);r=residual_rank(np.array([1.,2.,3.,4.]),g)
    assert abs(float(rho(r,r))-1)<1e-12
    assert finite(rho(np.zeros(4),np.ones(4))) is None
    print(json.dumps(dict(status='completed',tied_ranks=True,group_residualization=True,constant_undefined=True)))
