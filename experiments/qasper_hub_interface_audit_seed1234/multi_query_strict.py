"""Strict evidence-only memory, 16 papers x 3 queries, frozen Writers.

V2 deliberately tokenizes wrapper/body/query separately. This gives a fixed
paper grid and forbids boundary tokens with native evidence information.
Full Native uses the identical concatenated token sequence, not a different
retokenization. Reader receives only fresh cloned cache and question suffix IDs.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import time
import torch
from transformers import AutoTokenizer
import experiment as e
from native_chat_fix import SYSTEM
from kv_backend import capture


def encode(tok, text):
    return list(tok.encode(text, add_special_tokens=False))


def template_parts(tok):
    sentinel = '__QASPER_MEMORY_BOUNDARY_1234__'
    rendered = tok.apply_chat_template([{'role':'system','content':SYSTEM},
        {'role':'user','content':'Paper:\n'+sentinel}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    assert rendered.count(sentinel) == 1
    return rendered.split(sentinel)


def sync_indices(st, rt, text):
    s = st(text, add_special_tokens=False, return_offsets_mapping=True)
    t = rt(text, add_special_tokens=False, return_offsets_mapping=True)
    sb, tb = s['offset_mapping'], t['offset_mapping']
    bounds = sorted({0} | ({b for a,b in sb if a<b} & {b for a,b in tb if a<b}))
    result = {}
    for left,right in zip(bounds,bounds[1:]):
        si = [i for i,(a,b) in enumerate(sb) if a<b and left<b<=right]
        ti = [i for i,(a,b) in enumerate(tb) if a<b and left<b<=right]
        if not si or not ti: continue
        for rank,i in enumerate(ti):
            j = math.ceil((rank+1)*len(si)/len(ti))-1 if len(si)>len(ti) else rank*len(si)//len(ti)
            result[i] = si[j]
    if set(result) != set(range(len(t['input_ids']))):
        raise RuntimeError('Full-Sync must cover EVERY evidence slot; no native fallback')
    return s['input_ids'],t['input_ids'],[result[i] for i in range(len(result))]


def state_hash(state):
    digest = hashlib.sha256()
    for k,v in state:
        for tensor in (k,v):
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def clock_start():
    torch.cuda.synchronize()
    return time.perf_counter()


def elapsed(start):
    torch.cuda.synchronize()
    return time.perf_counter()-start


def read_memory(receiver, tokenizer, state, query_ids):
    # This API has NO evidence text, evidence IDs, or native evidence tensors.
    logits, cache = e.prefill(receiver, query_ids, e.restore(receiver,state))
    first = logits.cpu()
    return e.generate(receiver,logits,cache,tokenizer),first


def paper_text(p):
    return 'Title: '+e.evidence(p).split('Title: ',1)[1]


def select_papers(data, rt):
    eligible = []
    for pid,p in data.items():
        qa = {q['question_id']:q for q in p['qas']}
        body = paper_text(p)
        if len(qa)>=3 and 512<=len(encode(rt,body))<=3072:
            eligible.append((pid,p,list(qa.values()),body))
    rng = random.Random(1234)
    eligible.sort(key=lambda x:x[0]); rng.shuffle(eligible)
    if len(eligible)<16: raise RuntimeError(f'Need 16 eligible papers, found {len(eligible)}')
    selected = []
    for pid,p,qa,body in eligible[:16]:
        qa.sort(key=lambda q:q['question_id']); rng.shuffle(qa)
        selected.append({'paper_id':pid,'paper':body,'questions':qa[:3]})
    return selected


def anomalies(pred):
    words = pred['text'].lower().split()
    grams = collections.Counter(tuple(words[i:i+4]) for i in range(max(0,len(words)-3)))
    return {'unanswerable':pred['text'].strip().lower()=='unanswerable',
        'limit_hit':pred['stop']=='max_tokens', 'think_marker':'</think>' in pred['text'] or '<think>' in pred['text'],
        'repeated_4gram':bool(grams and max(grams.values())>=4)}


def run(smoke=False):
    e.seed_all(1234)
    out = e.ROOT/('results_multiquery_strict_smoke_v2' if smoke else 'results_multiquery_strict_v2'); out.mkdir(exist_ok=True)
    rt = AutoTokenizer.from_pretrained(e.CFG['models']['qwen'],local_files_only=True)
    st = AutoTokenizer.from_pretrained(e.CFG['models']['llama'],local_files_only=True)
    rh,tail = template_parts(rt); sh,_ = template_parts(st)
    rhids,shids = encode(rt,rh),encode(st,sh)
    data = json.loads((e.DATA/'qasper-dev-v0.3.json').read_text())
    spec = importlib.util.spec_from_file_location('qasper_eval',e.DATA/'qasper_evaluator.py')
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    refs = evaluator.get_answers_and_evidence(data,True)
    papers = select_papers(data,rt)
    if smoke: papers = papers[:1]
    total_queries = len(papers)*3
    e.save_json(out/'manifest.json',{'seed':1234,'split':'dev','papers':papers})
    e.save_json(out/'config.json',{'protocol':'strict-segmented-chat-evidence-memory-v2',
        'papers':len(papers),'queries_per_paper':3,'smoke':smoke,'training':False,'max_new_tokens':128,
        'no_truncation':True,'system':SYSTEM,'receiver_head':rh,'receiver_tail':tail,
        'sender_head':sh,'enable_thinking':False,'dtype':'float16','attention':'eager',
        'tokenization':'head + separately encoded paper + separately encoded question/tail; same grid for every condition',
        'reader_inputs':'question/tail IDs and clone of immutable memory ONLY',
        'native_allowed':'head wrapper computed before evidence only; ALL body slots translated',
        'privacy_note':'interface isolation only, not formal privacy; length grid is visible',
        'sampling':'paper-based, not answer-type balanced; first annotation type used for strata',
        'checkpoints':{g:{x:str(e.LEGACY/f'runs/{g}/stage_{x}/best.pt') for x in ('a','b')} for g in ('mcq','gsm8k')}})
    cfg = {'models':e.CFG['models'],'attention_implementation':e.CFG['attention_implementation']}
    receiver,sender = e.load_model(cfg,'qwen'),e.load_model(cfg,'llama')
    models = {g:(e.load_base(e.LEGACY/f'runs/{g}/stage_a/best.pt'),
                 e.load_adapter(e.LEGACY/f'runs/{g}/stage_b/best.pt')) for g in ('mcq','gsm8k')}
    start = clock_start(); wk,wv = capture(receiver,rhids); wrapper_seconds = elapsed(start)
    records,audits,timings = [],[],[]
    for number,p in enumerate(papers):
        pid,body = p['paper_id'],p['paper']
        sid,tid,mapping = sync_indices(st,rt,body)
        prefix_ids = rhids+tid
        start = clock_start(); sk,sv = capture(sender,shids+sid); sender_seconds = elapsed(start)
        item = {'source_k':sk[:,[len(shids)+i for i in mapping]],
                'source_v':sv[:,[len(shids)+i for i in mapping]],'tokens':len(tid)}
        # Native evidence computation below is confined to reference/audit branch.
        start = clock_start(); _,nc = e.prefill(receiver,prefix_ids)
        states = {'native_cache':e.freeze_cache(nc)}; native_build = elapsed(start); del nc
        start = clock_start(); nk,nv = capture(receiver,prefix_ids)
        states['native_rebuild'] = e.freeze_cache(e.make_cache(receiver,nk.cuda(),nv.cuda(),torch.arange(len(prefix_ids),device='cuda')))
        rebuild_build = elapsed(start); del nk,nv
        builds = {'native_cache':native_build,'native_rebuild':rebuild_build}
        for group,(base,adapter) in models.items():
            start = clock_start(); bk,bv = e.map_base(base,item)
            fk,fv = torch.cat((wk.cuda(),bk),1),torch.cat((wv.cuda(),bv),1)
            states[group+'_stage_a'] = e.freeze_cache(e.make_cache(receiver,fk,fv,torch.arange(len(prefix_ids),device='cuda')))
            a_build = elapsed(start)
            start = clock_start()
            with torch.amp.autocast('cuda',dtype=torch.float16): k,v,_,_ = adapter(bk[None],bv[None])
            fk,fv = torch.cat((wk.cuda(),k[0]),1),torch.cat((wv.cuda(),v[0]),1)
            states[group+'_stage_b'] = e.freeze_cache(e.make_cache(receiver,fk,fv,torch.arange(len(prefix_ids),device='cuda')))
            builds[group+'_stage_a'] = sender_seconds+a_build+wrapper_seconds/len(papers)
            builds[group+'_stage_b'] = builds[group+'_stage_a']+elapsed(start)
            del bk,bv,fk,fv,k,v
        del item,sk,sv
        before = {name:state_hash(state) for name,state in states.items()}
        query_seconds = collections.Counter()
        for q in p['questions']:
            qid = q['question_id']; suffix = '\n\nQuestion: '+q['question']+tail
            qids = encode(rt,suffix)
            start = clock_start(); logits,cache = e.prefill(receiver,prefix_ids+qids)
            reference = logits.cpu(); full = e.generate(receiver,logits,cache,rt)
            full_seconds = elapsed(start); del cache
            results = {'full':(full,full_seconds)}
            parity = {}
            for name,state in states.items():
                start = clock_start(); pred,first = read_memory(receiver,rt,state,qids)
                seconds = elapsed(start); results[name] = (pred,seconds); query_seconds[name]+=seconds
                if name.startswith('native'):
                    parity[name] = {'generation_identical':pred['ids']==full['ids'],
                        'argmax_identical':int(first.argmax())==int(reference.argmax()),
                        'max_logit_error':float((first-reference).abs().max())}
            query_seconds['full']+=full_seconds
            audits.append({'paper_id':pid,'question_id':qid,'checks':parity})
            for name,(pred,seconds) in results.items():
                records.append({'paper_id':pid,'question_id':qid,'question':q['question'],
                    'type':refs[qid][0]['type'],'condition':name,**pred,**anomalies(pred),
                    'seconds':seconds,'memory_id':before.get(name),'prefix_tokens':len(prefix_ids),
                    'query_input_ids':qids,'native_evidence_slots':0 if name.startswith(('mcq','gsm8k')) else len(tid)})
            e.save_json(out/'per_sample.json',records); e.save_json(out/'parity.json',audits)
            print(f'MULTI QUERY paper={number+1}/{len(papers)} query={len(audits)}/{total_queries} native_parity={parity}',flush=True)
        after = {name:state_hash(state) for name,state in states.items()}
        if before != after: raise RuntimeError('Reusable memory was mutated by a Query')
        timings.append({'paper_id':pid,'sender_prefills':1,'stage_a_maps_per_group':1,
            'stage_b_adapter_calls_per_group':1,'queries_per_memory':3,'memory_immutable':True,
            'memory_hashes':before,'sender_seconds':sender_seconds,'build_seconds':builds,
            'query_seconds':dict(query_seconds),
            'three_query_total_seconds':{name:builds.get(name,0)+secs for name,secs in query_seconds.items()}})
        e.save_json(out/'reuse_and_timing.json',timings)
        del states; torch.cuda.empty_cache()
    gold = {r['question_id']:refs[r['question_id']] for r in records if r['condition']=='full'}
    metrics = {}
    for name in sorted({r['condition'] for r in records}):
        rows = [r for r in records if r['condition']==name]
        predictions = {r['question_id']:{'answer':r['text'].strip(),'evidence':[]} for r in rows}
        report = evaluator.evaluate(gold,predictions); report.pop('Evidence F1',None)
        strata = {}
        for typ in sorted({r['type'] for r in rows}):
            rr = [r for r in rows if r['type']==typ]
            sub = evaluator.evaluate({r['question_id']:gold[r['question_id']] for r in rr},
                {r['question_id']:predictions[r['question_id']] for r in rr})
            strata[typ] = {'count':len(rr),'Answer F1':sub['Answer F1'],
                **{x+'_rate':sum(r[x] for r in rr)/len(rr) for x in ('unanswerable','limit_hit','think_marker','repeated_4gram')},
                'EOS_rate':sum(r['stop']=='eos' for r in rr)/len(rr)}
        report.update(first_annotation_type_metrics=strata,EOS_rate=sum(r['stop']=='eos' for r in rows)/len(rows),
            unanswerable_rate=sum(r['unanswerable'] for r in rows)/len(rows))
        metrics[name] = report; e.save_json(out/(name+'_predictions.json'),predictions)
    baseline = evaluator.evaluate(gold,{qid:{'answer':'Unanswerable','evidence':[]} for qid in gold})
    baseline.pop('Evidence F1',None)
    checks = {name:{'generation_agreement':sum(a['checks'][name]['generation_identical'] for a in audits)/total_queries,
                   'argmax_agreement':sum(a['checks'][name]['argmax_identical'] for a in audits)/total_queries,
                   'max_logit_error':max(a['checks'][name]['max_logit_error'] for a in audits)}
              for name in ('native_cache','native_rebuild')}
    e.save_json(out/'summary.json',{'status':'completed','count':total_queries,'papers':len(papers),'metrics':metrics,
        'always_unanswerable_baseline':baseline,'native_parity':checks,
        'all_memory_immutable':True,'wrapper_build_seconds':wrapper_seconds,
        'reuse_total_seconds':{name:sum(t['three_query_total_seconds'][name] for t in timings) for name in metrics},
        'note':'Strict V2 protocol differs from V1 boundary tokenization; compare within V2, not as a single-variable improvement.'})
    print('MULTIQUERY STRICT AUDIT COMPLETED',flush=True)
    if smoke and any(c['argmax_agreement'] != 1 for c in checks.values()):
        raise RuntimeError('Smoke native first-token parity failed; do not start full audit blindly')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    with torch.inference_mode(): run(parser.parse_args().smoke)
