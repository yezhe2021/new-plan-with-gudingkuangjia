"""Lock pilot-calibrated selector before touching reserved evaluation answers."""
import backend as B
from backend import ROOT,CFG,E,np,read,save,sha,ident,rows,temp,status
from pilot_metrics import bootstrap
from selector_rules import fit,features,decide
from pathlib import Path
import json
import time
import os
import fcntl


def prepare():
    E.verify();parent=Path(CFG['parent_experiment']);pm=read(parent/'manifest.json')
    assert read(parent/'completed.json')['manifest_sha256']==sha(parent/'manifest.json')
    assert read(parent/'summary.json')['pilot_gate']['passed']
    reserved=parent/'prepared/reserved_test.jsonl'
    assert sha(reserved)==pm['reserved_test_sha256']
    records=[json.loads(x) for x in reserved.read_text().splitlines()]
    assert len(records)==CFG['test_count']
    for r in records:r['split']='test'
    assert not (ROOT/'manifest.json').exists()
    (ROOT/'prepared').mkdir(exist_ok=True)
    out=ROOT/'prepared/questions.jsonl';out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (ROOT/'prepared/reserved_test.jsonl').write_bytes(reserved.read_bytes())
    calibration=[read(p) for p in sorted((parent/'evaluation/calibration').glob('hotpot_*.json'))]
    assert len(calibration)==CFG['pilot_count'] and all(r['complete'] for r in calibration)
    assert all(r['manifest_sha256']==sha(parent/'manifest.json') for r in calibration)
    assert not {r['id'] for r in calibration}&{r['id'] for r in records}
    anchors={tuple(r['frames']['qwen']['body_ids'][:r['frames']['qwen']['native_prefix_length']]) for r in records}
    assert anchors=={tuple(pm['anchor_ids'])}
    selector=fit(calibration,CFG['corruption_conditions'])
    manifest=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),prepared_sha256=sha(out),
        reserved_test_sha256=sha(ROOT/'prepared/reserved_test.jsonl'),reference_manifest_sha256=E.identity(),
        code={p.name:sha(p) for p in ROOT.glob('*.py')},checkpoints=pm['checkpoints'],anchor_ids=pm['anchor_ids'],
        parent_manifest_sha256=sha(parent/'manifest.json'),calibration_results_sha256={p.name:sha(p) for p in sorted((parent/'evaluation/calibration').glob('hotpot_*.json'))},
        no_reader_installed=True,no_test_outcome_used_in_selection=True,test_count=len(records))
    save(ROOT/'manifest.json',manifest)
    selector.update(manifest_sha256=ident(),locked_at=time.strftime('%F %T'),calibration_only=True)
    save(ROOT/'selector.json',selector)
    save(ROOT/'prepared/completed.json',dict(manifest_sha256=ident(),selector_sha256=sha(ROOT/'selector.json')))
    status(status='prepared',phase='prepare');E.log('LOCKED SELECTOR '+json.dumps(selector,ensure_ascii=False))


def summarize():
    B.verify();selector=read(ROOT/'selector.json');selected=selector['selected_rule'];rules=selector['rules']
    records=[read(B.result_path(r)) for r in rows()];n=len(records);conditions=CFG['corruption_conditions']
    assert all(r['complete'] and r['manifest_sha256']==ident() and r['selector_sha256']==sha(ROOT/'selector.json') for r in records)
    summary=dict(experiment=CFG['name'],status='completed',timestamp=time.strftime('%F %T'),n=n,
        manifest_sha256=ident(),selector=selector,baselines={},conditions={},balanced_mixture={})
    for c in CFG['conditions']:
        group=[r['conditions'][c] for r in records]
        summary['baselines'][c]=dict(n=n,correct=sum(r['em'] for r in group),
            **{k:float(np.mean([r[k] for r in group])) for k in ['em','f1','answer_token_nll','eos','generated_tokens']})
    for c in conditions:
        data=[]
        for r in records:
            a,b=r['conditions'][c],r['conditions']['reject_all'];x=features(a,b)
            assert r['selection'][c]['features']==x
            accepted=decide(x,rules[selected]);assert accepted==r['selection'][c]['decisions'][selected]
            data.append((a,b,a if accepted else b,accepted))
        summary['conditions'][c]=dict(n=n,selected_rule=selected,accept_count=sum(d[3] for d in data),
            em=float(np.mean([d[2]['em'] for d in data])),f1=float(np.mean([d[2]['f1'] for d in data])),
            gain_vs_always_accept={m:bootstrap([d[2][m]-d[0][m] for d in data]) for m in ['em','f1']},
            gain_vs_always_reject={m:bootstrap([d[2][m]-d[1][m] for d in data]) for m in ['em','f1']},
            clean_or_corrupt_useful_em=sum(d[0]['em']>d[1]['em'] for d in data),
            clean_or_corrupt_harmful_em=sum(d[0]['em']<d[1]['em'] for d in data),
            oracle_em=float(np.mean([max(d[0]['em'],d[1]['em']) for d in data])),
            prediction_changes_vs_reject=sum(d[2]['prediction']!=d[1]['prediction'] for d in data))
    for name,rule in rules.items():
        accept_rate=float(np.mean([decide(features(r['conditions'][c],r['conditions']['reject_all']),rule) for r in records for c in conditions]))
        metrics={}
        for metric in ['em','f1']:
            scores=[];gain_a=[];gain_b=[];random_expected=[]
            for r in records:
                b=r['conditions']['reject_all'];aa=[r['conditions'][c] for c in conditions]
                values=[a if decide(features(a,b),rule) else b for a in aa]
                scores.append(np.mean([v[metric] for v in values]))
                gain_a.append(np.mean([v[metric]-a[metric] for v,a in zip(values,aa)]))
                gain_b.append(np.mean([v[metric]-b[metric] for v in values]))
                random_expected.append(np.mean([accept_rate*a[metric]+(1-accept_rate)*b[metric] for a in aa]))
            metrics[metric]=dict(score=bootstrap(scores),gain_vs_accept=bootstrap(gain_a),gain_vs_reject=bootstrap(gain_b),
                matched_accept_rate_random_expected=float(np.mean(random_expected)))
        summary['balanced_mixture'][name]=dict(accept_rate=accept_rate,**metrics)
    summary['caveats']=['128 heldout questions, fixed before calibration32 inference; not official full HotpotQA benchmark.',
        'Short-answer generation uses entire normalized output EM/F1. No gold-dependent answer extraction.',
        'Selected rule fixed on pilot F1. Other rules are diagnostics, not alternative test-selected winners.',
        'Balanced synthetic clean/wrong/partial corruption is a controlled mixture, not an estimate of deployment prevalence.',
        'Both accept and reject branches are computed: no compute savings claimed.',
        'First-token entropy and question likelihood are proxies, not calibrated truth probabilities.',
        'No Reader, training, or quantization. Bootstrap resamples questions with their corruption variants together.']
    save(ROOT/'summary.json',summary)
    lines=['# HotpotQA heldout128 selective receiver','',f'Locked selector: `{selected}`','',
        '|Condition|Accept-all EM|Reject-all EM|Selected EM|Selected F1|Acceptance|','|---|---:|---:|---:|---:|---:|']
    for c,x in summary['conditions'].items():
        lines.append(f'|{c}|{summary["baselines"][c]["em"]:.4f}|{summary["baselines"]["reject_all"]["em"]:.4f}|{x["em"]:.4f}|{x["f1"]:.4f}|{x["accept_count"]/n:.4f}|')
    lines+=['','Mixture results and paired intervals: see summary.json.','',*summary['caveats']]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with Path(CFG['shared_results']).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    for r in rows():temp(r,'mapped').unlink()
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=ident(),temporary_tensors_cleaned=True))
    status(status='completed',phase='all');E.log('HOTPOT SELECTOR EVAL COMPLETE')
