"""Matched question results and stepwise contrasts, without test-time tuning."""
import time
import numpy as np
from collections import Counter
from common import ROOT,CFG,read,save,identity,rows
from pilot_metrics import bootstrap


def report():
    selected=rows();records=[read(ROOT/'evaluation'/(r['id']+'.json')) for r in selected]
    assert len(records)==128 and all(r['complete'] and r['manifest_sha256']==identity() for r in records)
    reproduction=read(ROOT/'reproduction.json');assert reproduction['passed']
    conditions=CFG['conditions'];summary=dict(n=len(records),manifest_sha256=identity(),conditions={},contrasts={},
        reproduction=reproduction,timing=read(ROOT/'evaluation_completed.json'),
        question_types=dict(Counter(r['question_type'] for r in selected)),scope=CFG['scope'])
    for name in conditions:
        summary['conditions'][name]={}
        for arm in ['native','translated']:
            data=[r['conditions'][name][arm] for r in records]
            summary['conditions'][name][arm]=dict(correct=sum(int(x['em']) for x in data),
                em=float(np.mean([x['em'] for x in data])),f1=float(np.mean([x['f1'] for x in data])),
                mean_generated_tokens=float(np.mean([x['generated_tokens'] for x in data])),cap_count=sum(x['reached_cap'] for x in data))
    for previous,current in zip(conditions,conditions[1:]):
        key=current+' minus '+previous;contrast={}
        for arm in ['native','translated']:
            contrast[arm]={metric:bootstrap([r['conditions'][current][arm][metric]-r['conditions'][previous][arm][metric]
                for r in records]) for metric in ['em','f1']}
            contrast[arm]['prediction_changed']=sum(r['conditions'][current][arm]['prediction']!=r['conditions'][previous][arm]['prediction'] for r in records)
        contrast['translated_minus_native_change']={metric:bootstrap([
            (r['conditions'][current]['translated'][metric]-r['conditions'][previous]['translated'][metric])-
            (r['conditions'][current]['native'][metric]-r['conditions'][previous]['native'][metric]) for r in records]) for metric in ['em','f1']}
        summary['contrasts'][key]=contrast
    summary['historical_full512']=read(ROOT.parent/'grouped_headmix4_residual64_hotpotqa_only_l3_to_q4_seed1234/summary.json')['metrics']
    summary['diagnostic_cleaning_changes']={arm:sum(r['conditions']['E_diagnostic_decoder'][arm]['prediction']!=
        r['conditions']['E_diagnostic_decoder'][arm]['prior_diagnostic_cleaning'] for r in records) for arm in ['native','translated']}
    summary['total_change_E_minus_A']={arm:{metric:bootstrap([r['conditions']['E_diagnostic_decoder'][arm][metric]-
        r['conditions']['A_original'][arm][metric] for r in records]) for metric in ['em','f1']} for arm in ['native','translated']}
    save('summary.json',summary)
    lines=['# Same128 HotpotQA interface ablation','',
        'Same HotpotQA-trained A+B, no Reader. Uniform 128-question subset of the original512; no length/type/outcome exclusions. Original native and translated predictions reproduced exactly for all128 before proceeding.','',
        '| Condition | Native correct | Native EM / F1 | Translated correct | Translated EM / F1 |',
        '|---|---:|---:|---:|---:|']
    for name,value in summary['conditions'].items():
        n=value['native'];t=value['translated']
        lines.append(f"| {name} | {n['correct']}/128 | {100*n['em']:.2f}% / {100*n['f1']:.2f}% | {t['correct']}/128 | {100*t['em']:.2f}% / {100*t['f1']:.2f}% |")
    lines+=['','A: original full distractor prompt and complete prompt KV. B: remove distractors, preserve original supporting passage numbers. C: use the previous selector evidence formatting/instructions, still transfer the complete question-containing prompt. D: same Qwen input token sequence as C, but transfer evidence only and process the question natively. E: same evidence cache as D, use the prior diagnostic greedy loop; common answer cleaning throughout.','',
        '## Paired stepwise changes','',
        '| Change | Native F1 delta (95% CI), pp | Translated F1 delta (95% CI), pp |','|---|---:|---:|']
    for key,value in summary['contrasts'].items():
        cells=[]
        for arm in ['native','translated']:
            x=value[arm]['f1'];cells.append(f"{100*x['mean']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]")
        lines.append('| '+key+' | '+' | '.join(cells)+' |')
    lines+=['','Bootstrap resamples the same128 original questions, 5000 replicates. These are exploratory ordered contrasts; effects are conditional on earlier changes, not order-independent attributions. No selector is fitted or selected.','',
        'C versus D keeps the complete Qwen input tokens identical; it changes which model computes the question states and the transfer/prefill boundary. Native C/D/E results measure sensitivity to the prefill/decoder split. E retains the original model loading/backend, so this check alone does not isolate every runtime difference from the earlier diagnostic.','',
        'The question subset differs from the earlier short bridge-only128. Compare same-question A-E for interface effects; do not treat cross-subset accuracy gaps as pure interface effects. Support-only variants use gold supporting titles and are controlled diagnostics, not the official distractor task.','',
        f"Total evaluation time: {summary['timing']['elapsed_seconds']/60:.2f} minutes."]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save('completed.json',dict(status='completed',manifest_sha256=identity(),finished_at=time.strftime('%F %T')))
    save('status.json',dict(status='completed',completed_questions=len(records),total_questions=len(records)))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':report()
