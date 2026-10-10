import collections
import importlib.util
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from runtime import CFG,ROOT,save,seed,load_models,tokenizers,base,adapter,freeze,make_cache,prefill,decode,answer_with_memory,digest_state
from kv_backend import PaperCache,mapped,memory
from prompt_protocol import query_only


def official():
    spec=importlib.util.spec_from_file_location('qasper_official',Path(CFG['data'])/'qasper_evaluator.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    data=json.loads((Path(CFG['data'])/'qasper-dev-v0.3.json').read_text())
    data.update(json.loads((Path(CFG['data'])/'qasper-train-v0.3.json').read_text()))
    return m,m.get_answers_and_evidence(data,True)


def summarize(records):
    evaluator,refs=official(); output={}
    full={r['question_id']:r for r in records if r['condition']=='native_full'}
    for name in sorted({r['condition'] for r in records}):
        rr=[r for r in records if r['condition']==name]; ids=[r['question_id'] for r in rr]
        gold={qid:refs[qid] for qid in ids}
        predictions={r['question_id']:{'answer':r['text'],'evidence':[]} for r in rr}
        report=evaluator.evaluate(gold,predictions); report.pop('Evidence F1',None)
        f1={r['question_id']:max(evaluator.token_f1_score(r['text'],g['answer']) for g in refs[r['question_id']]) for r in rr}
        nf1={qid:max(evaluator.token_f1_score(full[qid]['text'],g['answer']) for g in refs[qid]) for qid in ids}
        answerable=[qid for qid in ids if any(g['type']!='none' for g in refs[qid])]
        correct={qid for qid in ids if f1[qid]>=CFG['retention_f1_threshold']}
        native_correct={qid for qid in ids if nf1[qid]>=CFG['retention_f1_threshold']}
        bypaper=collections.defaultdict(list)
        for r in rr: bypaper[r['paper_id']].append(f1[r['question_id']])
        strata={}
        for typ in sorted({refs[qid][0]['type'] for qid in ids}):
            typed=[r for r in rr if refs[r['question_id']][0]['type']==typ]
            strata[typ]={'count':len(typed),'Answer F1':sum(f1[r['question_id']] for r in typed)/len(typed),
                'EOS_rate':sum(r['stop']=='eos' for r in typed)/len(typed),
                'limit_hit_rate':sum(r['stop']=='max_tokens' for r in typed)/len(typed),
                'unanswerable_rate':sum(r['text'].strip().lower()=='unanswerable' for r in typed)/len(typed)}
        report.update(first_annotation_type_strata=strata,
            Answerable_Only_F1=sum(f1[q] for q in answerable)/len(answerable) if answerable else None,
            answerable_count=len(answerable), EOS_rate=sum(r['stop']=='eos' for r in rr)/len(rr),
            predicted_unanswerable_rate=sum(r['text'].strip().lower()=='unanswerable' for r in rr)/len(rr),
            limit_hit_rate=sum(r['stop']=='max_tokens' for r in rr)/len(rr),
            think_marker_rate=sum('</think>' in r['text'] or '<think>' in r['text'] for r in rr)/len(rr),
            correctness_threshold=CFG['retention_f1_threshold'],native_correct=len(native_correct),
            both_correct=len(correct&native_correct),native_only_correct=len(native_correct-correct),
            translated_only_correct=len(correct-native_correct),both_below_threshold=len(set(ids)-(correct|native_correct)),
            retention=len(correct&native_correct)/len(native_correct) if native_correct else None,
            native_minus_translated_F1=sum(nf1[q]-f1[q] for q in ids)/len(ids),
            paper_average_F1={p:sum(v)/len(v) for p,v in bypaper.items()},paper_worst_query_F1={p:min(v) for p,v in bypaper.items()},
            first_token_KL_mean=sum(r.get('first_token_kl',0) for r in rr)/len(rr))
        output[name]=report
    return output


@torch.no_grad()
def run_eval(cache,receiver,rt,papers,conditions,folder,heldout=False):
    records=[]; timings=[]
    for row in papers:
        item=cache.get(row); proto=item['protocol']
        states={}; build_times={}; mapped_models={}
        start=time.perf_counter(); _,nc=prefill(receiver,proto['prefix_ids'])
        states['native_cache']=freeze(nc); del nc; build_times['native_cache']=time.perf_counter()-start
        for name,(mapper,residual) in conditions.items():
            start=time.perf_counter()
            if id(mapper) not in mapped_models: mapped_models[id(mapper)]=mapped(mapper,item)
            k,v=mapped_models[id(mapper)]
            states[name]=memory(receiver,item,k,v,residual)
            build_times[name]=time.perf_counter()-start
        before={name:digest_state(state) for name,state in states.items()}
        del mapped_models,k,v
        qids={q['question_id'] for q in row['heldout_questions']} if heldout else {q['question_id'] for q in row['questions']}
        per_paper=collections.Counter()
        for q in proto['queries']:
            if q['question_id'] not in qids: continue
            start=time.perf_counter(); logits,nc=prefill(receiver,q['full_ids'])
            teacher=logits.cpu(); pred=decode(receiver,rt,logits,nc); del nc
            rr={'native_full':(pred,time.perf_counter()-start,0.)}
            for name,state in states.items():
                start=time.perf_counter(); pred,first=answer_with_memory(receiver,rt,state,q['suffix_ids'])
                kl=F.kl_div(first.log_softmax(-1),teacher.softmax(-1),reduction='sum')
                rr[name]=(pred,time.perf_counter()-start,float(kl))
            start=time.perf_counter(); logits,nc=prefill(receiver,query_only(rt,q['question']))
            pred=decode(receiver,rt,logits,nc); del nc
            rr['query_only']=(pred,time.perf_counter()-start,0.)
            for name,(pred,secs,kl) in rr.items():
                records.append({'paper_id':row['paper_id'],'question_id':q['question_id'],'question':q['question'],
                    'condition':name,**pred,'seconds':secs,'first_token_kl':kl,'memory_sha256':before.get(name),
                    'native_evidence_slots':0 if name in conditions else item['tokens'] if name.startswith('native') else 0})
                per_paper[name]+=secs
            save(folder/'per_sample.json',records)
        assert before=={name:digest_state(state) for name,state in states.items()}, 'A Query mutated reusable memory'
        timings.append({'paper_id':row['paper_id'],'reuse_count':len(qids),'immutable':True,'memory_hashes':before,
            'cached_reader_seconds':dict(per_paper),'mapping_build_seconds':build_times,
            'source_capture_count_in_session':cache.capture_counts[row['paper_id']],
            'timing_scope':'Warm-model mapping/build/read; source+teacher capture costs excluded, not end-to-end speedup'})
        save(folder/'timing.json',timings)
        print(f'EVALUATE {folder.name} paper={len(timings)}/{len(papers)}',flush=True)
    metrics=summarize(records); save(folder/'summary.json',metrics)
    for name in metrics: save(folder/(name+'_predictions.json'),{r['question_id']:{'answer':r['text'],'evidence':[]} for r in records if r['condition']==name})
    return metrics


def main():
    seed(); sender,receiver=load_models(); st,rt=tokenizers(); cache=PaperCache(sender,receiver,st,rt)
    prepared=json.loads((ROOT/'prepared.json').read_text())
    a=base(ROOT/'runs/stage_a_fresh/best.pt').eval().requires_grad_(False)
    b=adapter(ROOT/'runs/stage_b/best.pt').eval().requires_grad_(False)
    oa=base(Path(CFG['legacy'])/'runs/gsm8k/stage_a/best.pt',old=True).eval().requires_grad_(False)
    ob=adapter(Path(CFG['legacy'])/'runs/gsm8k/stage_b/best.pt',old=True).eval().requires_grad_(False)
    conditions={'old_gsm8k_stage_a':(oa,None),'old_gsm8k_stage_b':(oa,ob),
                'new_stage_a':(a,None),'new_stage_b_attention':(a,b)}
    result={'unseen_dev':run_eval(cache,receiver,rt,prepared['dev'],conditions,ROOT/'evaluation/dev'),
            'seen_evidence_heldout_query':run_eval(cache,receiver,rt,[p for p in prepared['train'] if p['heldout_questions']],conditions,ROOT/'evaluation/seen_heldout',True)}
    save(ROOT/'evaluation/summary.json',result)
    lines=['# QASPER Evidence-KV — expanded training data','',
        'No Test split tuning/evaluation. Protocol: independent evidence/question chat messages, full-render tokenization.',
        'HeteroFold-style objective only; not the original HeteroFold architecture or mapping domain.','',
        '| Condition | Dev Answer F1 | Answerable-only F1 | EOS | Unanswerable prediction rate |','|---|---:|---:|---:|---:|']
    for name,m in result['unseen_dev'].items():
        ans=m['Answerable_Only_F1']; lines.append(f"| {name} | {100*m['Answer F1']:.2f} | {100*ans:.2f} | {100*m['EOS_rate']:.2f}% | {100*m['predicted_unanswerable_rate']:.2f}% |" if ans is not None else f'| {name} | {100*m["Answer F1"]:.2f} | N/A | | |')
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('QASPER STRONG BASELINE PILOT COMPLETED',flush=True)


if __name__=='__main__': main()
