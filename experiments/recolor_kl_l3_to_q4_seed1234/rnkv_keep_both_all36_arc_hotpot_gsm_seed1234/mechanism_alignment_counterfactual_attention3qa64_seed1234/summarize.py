"""Question-level paired summaries; preserve negative cases and synthetic gates."""
from common_diag import *

def mean_dict(values):
    if not values:return None
    return {k:sum(v[k] for v in values)/len(values) for k in values[0] if isinstance(values[0][k],(int,float))}

def stats(records):
    if not records:return dict(n=0)
    functional={a:mean_dict([r['experiment1_functional'][a] for r in records]) for a in records[0]['experiment1_functional']}
    generation={a:{k:sum(r['experiment2_generation'][a][k] for r in records)/len(records) for k in ['em','f1']} for a in records[0]['experiment2_generation']}
    effects={}
    for name,left,right in [('keep_vs_same_position_masked','translated_shifted_masked','keep'),('correct_vs_wrong','wrong_memory','keep'),('correct_vs_header_only','header_only_memory','keep'),('position_shift_only','translated_shifted_masked','C')]:
        effects[name]=C.paired([r['experiment1_functional'][left]['answer_nll']-r['experiment1_functional'][right]['answer_nll'] for r in records])
    gen_diffs={a:{k:C.paired([r['experiment2_generation']['keep'][k]-r['experiment2_generation'][a][k] for r in records]) for k in ['em','f1']} for a in ['C','translated_shifted_masked','wrong_memory','header_only_memory']}
    groups={g:C.paired([r['experiment3_group_effects'][g]['gold_nll_gain_from_group_access'] for r in records]) for g in CFG['layer_groups']}
    local={source:{str(l):mean_dict([next(v for v in r['experiment3_fixedq'] if v['query_source']==source and v['layer']==l) for r in records]) for l in range(1,37)} for source in ['native_shifted_q','keep_q']}
    return dict(n=len(records),functional=functional,generation=generation,paired_nll_effects=effects,paired_generation_differences=gen_diffs,group_access_gold_nll_gain=groups,fixedq=local)

def main():
    verify();assert C.read(ROOT/'evaluation/completed.json')['status']=='completed'
    data=[C.read(ROOT/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json') for r in rows()]
    assert all(r['manifest_sha256']==identity() for r in data)
    tasks={t:stats([r for r in data if r['task']==t]) for t in CFG['test_counts']}
    strata={t:{s:stats([r for r in data if r['task']==t and r['stratum']==s]) for s in ['rescued','harmed','both_correct','both_wrong']} for t in CFG['test_counts']}
    cf=[C.read(p) for p in sorted((ROOT/'counterfactual/per_pair').glob('*.json'))];assert len(cf)==16
    verified=[r for r in cf if r['native_both_correct']]
    def cf_stats(rr):
        return dict(n=len(rr),conditions={a:{k:sum(r['conditions'][a][k] for r in rr)/len(rr) for k in ['em','f1','matches_body_answer','matches_memory_answer']} for a in cf[0]['conditions']}) if rr else dict(n=0)
    mark('summary.json',status='completed',tasks=tasks,strata=strata,counterfactual=dict(all_pairs=cf_stats(cf),native_verified_pairs=cf_stats(verified)),
        training=False,checkpoint_sha256=CFG['checkpoint_sha256'])
    lines=['# Llama to Qwen frozen Memory mechanism analysis','','All three diagnostics use the same trained3-task all-layer checkpoint.192 existing QA questions; no training.','',
        '## Free generation: identical64-token protocols','','| Task | Metric | Native | C | Keep | Same-position masked | Wrong donor | Header-only Memory |','|---|---|---:|---:|---:|---:|---:|---:|']
    for task,s in tasks.items():
        for k in ['em','f1']:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{s["generation"][a][k]*100:.2f}%' for a in ['native','C','keep','translated_shifted_masked','wrong_memory','header_only_memory'])+' |')
    lines+=['','## Gold-answer prediction: per-question means, EOS separate','','| Task | Condition | Gold NLL | KL to natural Native | KL to shifted Native | Native top1 match |','|---|---|---:|---:|---:|---:|']
    for task,s in tasks.items():
        for a,v in s['functional'].items():lines.append(f'| {task} | {a} | {v["answer_nll"]:.4f} | {v["kl_to_native"]:.4f} | {v["kl_to_native_shifted"]:.4f} | {v["top1_match_native"]*100:.2f}% |')
    lines+=['',f'Synthetic fact pairs: {len(verified)}/{len(cf)} have both Native endpoints correct. All pairs remain reported; no resampling by correctness.','',
        'Positive paired NLL gain for keep_vs_same_position_masked means Memory improves answer probability beyond position shift. Positive group_access gain means blocking that group harms answer NLL, conditional on all other groups; gains are not additive.',
        'Fixed-Q cosines compare local operators with a common background and suffix states. Keep-Q native-body target is a hybrid oracle. Actual trajectory similarity is observational. Fixed-Q background token ratios cannot change when only Memory keys are added.',
        '',*['- '+s for s in CFG['caveats']]]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Static scientific figures: aggregate data only, no raw attention tensors.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,(task,s) in zip(axes,tasks.items()):
        ls=list(range(1,37));v=s['fixedq']['native_shifted_q']
        for key,label in [('wo_cos_C_to_native','C'),('wo_cos_keep_to_native','Keep')]:ax.plot(ls,[v[str(l)][key] for l in ls],label=label)
        ax.axvspan(19,30,color='grey',alpha=.12);ax.set(title=task,xlabel='Receiver layer',ylabel='Output cosine to native operator');ax.legend()
    fig.suptitle('Fixed Native-shifted Q and common nonmemory KV; diagnostic operator comparison')
    (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/fixed_query_output_cosine.png',dpi=160);plt.close(fig)
    verify();mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

if __name__=='__main__':main()
