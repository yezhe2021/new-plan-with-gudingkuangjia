"""Question-paired diagnostic report. Oracle reads gold; deployable policies do not."""
import json,time
import numpy as np
from backend import ROOT,CFG,read,save,sha,ident,rows
from pilot_metrics import bootstrap


def comparison(accept,reject):
    # Arrays have one row per original question and one column per condition.
    result={}
    for metric in ['em','f1']:
        a=np.asarray(accept[metric]);r=np.asarray(reject[metric]);oracle=np.maximum(a,r)
        best='accept' if a.mean()>=r.mean() else 'reject';fixed=a if best=='accept' else r
        result[metric]=dict(accept=float(a.mean()),reject=float(r.mean()),oracle=float(oracle.mean()),
            accept_minus_reject=bootstrap((a-r).mean(1)),best_fixed=best,
            oracle_minus_best_fixed=bootstrap((oracle-fixed).mean(1)),
            helpful_cases=int((a>r).sum()),harmful_cases=int((a<r).sum()),
            helpful_questions=int((a>r).any(1).sum()),harmful_questions=int((a<r).any(1).sum()))
    a=np.asarray(accept['em']);r=np.asarray(reject['em'])
    result['both_wrong_cases']=int(((a==0)&(r==0)).sum())
    result['questions']=len(a);result['cases']=int(a.size)
    return result


def report():
    records=rows();current=[];old=[];m=read(ROOT/'manifest.json')
    for row in records:
        r=read(ROOT/'evaluation'/f"{row['id']}.json")
        assert r['complete'] and r['manifest_sha256']==ident();current.append(r)
        p=ROOT/'reference_results'/f"{row['id']}.json"
        assert sha(p)==m['prior_results_sha256'][p.name];old.append(read(p))
    summary=dict(n=len(records),manifest_sha256=ident(),scope=CFG['scope'],checkpoints=m['checkpoints'],
        primary_diagnostic='Clean KV utility and oracle headroom; not a fitted selector result',versions={},paired_checkpoint_delta={},
        timing=read(ROOT/'evaluation_completed.json'))
    for label,data in [('mixed_task_reference',old),('hotpot_trained',current)]:
        version=dict(baselines={},conditions={})
        for b in ['compact','reject','native']:
            version['baselines'][b]={metric:float(np.mean([r['baselines'][b][metric] for r in data])) for metric in ['em','f1']}
        for conds,name in [( [c],c) for c in CFG['conditions']]+[(CFG['conditions'],'equal_mixture')]:
            aa={metric:[[r['conditions'][c]['accept'][metric] for c in conds] for r in data] for metric in ['em','f1']}
            rr={metric:[[r['baselines']['reject'][metric] for c in conds] for r in data] for metric in ['em','f1']}
            version['conditions'][name]=comparison(aa,rr)
            # Prespecified historical question-gain rule, with no fitted threshold.
            gate={metric:[[r['conditions'][c]['accept'][metric] if
                r['conditions'][c]['accept']['question_logp']>r['baselines']['reject']['question_logp']
                else r['baselines']['reject'][metric] for c in conds] for r in data] for metric in ['em','f1']}
            version['conditions'][name]['question_gain_zero']={metric:dict(value=float(np.mean(gate[metric])),
                minus_accept=bootstrap((np.array(gate[metric])-np.array(aa[metric])).mean(1)),
                minus_reject=bootstrap((np.array(gate[metric])-np.array(rr[metric])).mean(1))) for metric in ['em','f1']}
        summary['versions'][label]=version
    for c in CFG['conditions']:
        summary['paired_checkpoint_delta'][c]={metric:bootstrap([
            a['conditions'][c]['accept'][metric]-b['conditions'][c]['accept'][metric] for a,b in zip(current,old)]) for metric in ['em','f1']}
    summary['baseline_prediction_mismatches']={b:sum(a['baselines'][b]['generated_ids']!=o['baselines'][b]['generated_ids'] for a,o in zip(current,old)) for b in ['compact','reject','native']}
    save(ROOT/'summary.json',summary)
    lines=['# HotpotQA-trained Translator: 128-question evidence KV diagnostic','',
        'Same 128 previously evaluated official-dev questions, same prompt and donor mapping. No overlap with the HotpotQA checkpoint train/validation questions. No Reader or parameter training. This is an exploratory checkpoint comparison, not a new independent selector test.','',
        '| Version / arm | EM | F1 |','|---|---:|---:|']
    for label,v in summary['versions'].items():
        for b,x in v['baselines'].items():lines.append(f"| {label} / {b} | {100*x['em']:.2f}% | {100*x['f1']:.2f}% |")
        for c in CFG['conditions']:
            x=v['conditions'][c];lines.append(f"| {label} / {c} | {100*x['em']['accept']:.2f}% | {100*x['f1']['accept']:.2f}% |")
    lines+=['','## Potential for selection','',
        '| New checkpoint condition | EM helpful / harmful cases | EM oracle | F1 oracle | F1 oracle minus best fixed |',
        '|---|---:|---:|---:|---:|']
    for c,x in summary['versions']['hotpot_trained']['conditions'].items():
        interval=x['f1']['oracle_minus_best_fixed']
        lines.append(f"| {c} | {x['em']['helpful_cases']} / {x['em']['harmful_cases']} | {100*x['em']['oracle']:.2f}% | {100*x['f1']['oracle']:.2f}% | {100*interval['mean']:.2f} pp [{100*interval['ci95'][0]:.2f}, {100*interval['ci95'][1]:.2f}] |")
    lines+=['','## Paired checkpoint changes','']
    for c,d in summary['paired_checkpoint_delta'].items():
        x=d['f1'];lines.append(f"- {c} F1: {100*x['mean']:+.2f} pp; paired 95% CI [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}].")
    lines+=['','The oracle uses gold only after generation, and is an upper bound, not an operational selector. All mixture bootstrap intervals resample the 128 original questions with their conditions together (5000 replicates). The equal mixture has 384 cases, not 384 independent questions.','',
        'Question-gain > 0 is included as a fixed historical diagnostic in summary.json; no threshold is fitted. Stability features and learned selectors are deliberately not run until task utility is established.','',
        'The new checkpoint was trained on full distractor prompts. This evidence-only interface remains a distribution shift even though the dataset name matches. Previously reported distractor512 accuracy is not directly comparable.','',
        'Baseline generation mismatches versus historical run: '+json.dumps(summary['baseline_prediction_mismatches']),
        f"\nEvaluation elapsed seconds: {summary['timing']['elapsed_seconds']:.1f}."]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=ident(),finished_at=time.strftime('%F %T')))
    save(ROOT/'status.json',dict(status='completed',completed_questions=len(records),total_questions=len(records)))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':report()
