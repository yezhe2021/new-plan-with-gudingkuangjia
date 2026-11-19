"""Frozen mapper evaluation. Hub formatting/parser are loaded unchanged from pilot."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import torch
from common import ROOT,CFG,load_model,tokenizers,seed,save
from capture_kv import capture,protocol
from token_alignment import align_tokens
from inference import mapped_memory,freeze,prefill,generate,digest
from train import load_maps


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def conditions(folder):
    return {'ta_three_layer_ridge':load_maps(folder/'maps/ridge.pt'),
            'recolor':load_maps(folder/'maps/recolor.pt'),
            'k_calibration':load_maps(folder/'calibration_k/folded.pt'),
            'kv_calibration':load_maps(folder/'calibration_kv/folded.pt')}


def unload(model):
    del model
    gc.collect();torch.cuda.empty_cache()


def summary(records,evaluator,refs):
    metrics={}
    native={r['question_id']:r for r in records if r['condition']=='native_full'}
    for name in sorted({r['condition'] for r in records}):
        rows=[r for r in records if r['condition']==name];ids={r['question_id'] for r in rows}
        gold={qid:refs[qid] for qid in ids}
        report=evaluator.evaluate(gold,{r['question_id']:{'answer':r['text'],'evidence':[]} for r in rows})
        report.pop('Evidence F1',None)
        def score(r): return max(evaluator.token_f1_score(r['text'],g['answer']) for g in refs[r['question_id']])
        good={r['question_id'] for r in rows if score(r)>=.5};ngood={i for i in ids if score(native[i])>=.5}
        report.update(count=len(rows),both_correct=len(good&ngood),native_only_correct=len(ngood-good),
            translated_only_correct=len(good-ngood),both_below_threshold=len(ids-(good|ngood)),threshold=.5,
            EOS_rate=sum(r['stop']=='eos' for r in rows)/len(rows))
        metrics[name]=report
    return metrics


@torch.no_grad()
def hub(folder):
    pilot=Path(CFG['hub_pilot']); p=module('unchanged_hub_protocol',pilot/'prompt_protocol.py')
    pcfg=json.loads((pilot/'config.json').read_text()); data=Path(pcfg['data'])
    ev=module('official_qasper',data/'qasper_evaluator.py')
    refs=ev.get_answers_and_evidence(json.loads((data/'qasper-dev-v0.3.json').read_text()),True)
    papers=json.loads((pilot/'prepared.json').read_text())['dev'];st,rt=tokenizers()
    maps=conditions(folder);records=[];audits=[]
    for row in papers:
        # Reuse exact old render, NOT its copy/drop alignment algorithm.
        s=p.render(st,row['paper']);targets=[p.render(rt,row['paper'],q['question']) for q in row['questions']]
        t=targets[0];prefix=t['ids'][:t['stop']]
        assert all(z['ids'][:z['stop']]==prefix for z in targets)
        ta=align_tokens(s['spans'],t['spans'])
        sender=load_model('sender');src=capture(sender,s['ids'][:s['stop']]);del sender;gc.collect();torch.cuda.empty_cache()
        src={r:src[r][:,s['head']:][:,ta] for r in ('k','v')}
        receiver=load_model('receiver')
        # Independent prefix-only capture: translated branches cannot read native evidence.
        head=capture(receiver,prefix[:t['head']],storage_dtype=next(receiver.parameters()).dtype)
        states={name:mapped_memory(receiver,src,head,m) for name,m in maps.items()}
        _,cache=prefill(receiver,prefix);states['native_cache']=freeze(cache);del cache
        hashes={name:digest(state) for name,state in states.items()}
        for q,z in zip(row['questions'],targets):
            results={'native_full':generate(receiver,rt,z['ids'],max_tokens=pcfg['max_new_tokens'])[0]}
            for name,state in states.items():
                results[name]=generate(receiver,rt,z['ids'][z['stop']:],state,pcfg['max_new_tokens'])[0]
            assert results['native_full']['ids']==results['native_cache']['ids'], 'Native full/cache generation mismatch'
            for name,result in results.items():
                records.append(dict(paper_id=row['paper_id'],question_id=q['question_id'],question=q['question'],
                    condition=name,**result,native_evidence_slots=0 if name in maps else len(ta)))
            save(folder/'evaluation/hub/per_sample.json',records)
        assert hashes=={name:digest(state) for name,state in states.items()},'Reusable memory mutated'
        audits.append({'paper_id':row['paper_id'],'query_count':len(targets),'source_capture_count':1,
                       'memory_hashes':hashes,'immutable':True,'native_evidence_fallback':False})
        save(folder/'evaluation/hub/cache_audit.json',audits)
        del receiver,states,src,head;gc.collect();torch.cuda.empty_cache()
        print('HUB EVALUATE',len(audits),len(papers),flush=True)
    save(folder/'evaluation/hub/summary.json',summary(records,ev,refs))


@torch.no_grad()
def full_context(folder):
    if not CFG['evaluation_manifest']: raise ValueError('Set evaluation_manifest; official eval samples are not calibration inputs')
    rows=[json.loads(x) for x in Path(CFG['evaluation_manifest']).read_text().splitlines() if x.strip()]
    st,rt=tokenizers();maps=conditions(folder);records=[]
    scorer=module('qasper_reference',Path(CFG['hub_pilot'])/'qasper_evaluator_reference.py')
    for i,row in enumerate(rows):
        proto=protocol(st,rt,row);s,t=proto['source'],proto['target']
        sender=load_model('sender');src=capture(sender,s['ids']);del sender;gc.collect();torch.cuda.empty_cache()
        src={r:src[r][:,s['head']:][:,proto['mapping']] for r in ('k','v')}
        receiver=load_model('receiver');head=capture(receiver,t['head_ids'],storage_dtype=next(receiver.parameters()).dtype)
        states={name:mapped_memory(receiver,src,head,m) for name,m in maps.items()}
        inputs={'native_full':(t['ids']+t['tail_ids'],None),**{name:(t['tail_ids'],state) for name,state in states.items()}}
        for name,(ids,state) in inputs.items():
            torch.manual_seed(CFG['seed']+i)
            result,_=generate(receiver,rt,ids,state,CFG['max_new_tokens'],CFG['full_context_decode'])
            records.append({'id':row['id'],'task':row['task'],'condition':name,**result,
                'F1':max(scorer.token_f1_score(result['text'],a) for a in row['references'])})
        save(folder/'evaluation/full_context/per_sample.json',records)
        del receiver,states,src,head;gc.collect();torch.cuda.empty_cache()
        print('FULL CONTEXT EVALUATE',i+1,len(rows),flush=True)
    metrics={}
    for task in sorted({r['task'] for r in records}):
        metrics[task]={}
        for name in inputs:
            rr=[r for r in records if r['task']==task and r['condition']==name]
            metrics[task][name]={'F1':sum(r['F1'] for r in rr)/len(rr),'count':len(rr)}
    save(folder/'evaluation/full_context/summary.json',metrics)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['smoke','formal'],default='formal')
    a.add_argument('--interface',choices=['hub','full_context'],required=True);args=a.parse_args();seed()
    (hub if args.interface=='hub' else full_context)(ROOT/'runs'/args.mode)
