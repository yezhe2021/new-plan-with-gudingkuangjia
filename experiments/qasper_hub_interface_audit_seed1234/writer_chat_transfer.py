"""Frozen old Writers on the repaired QASPER chat protocol. No training."""
import importlib.util
import json
import math
import time
import torch
from transformers import AutoTokenizer
import experiment as e
from native_chat_fix import SYSTEM
from kv_backend import capture


def body_mapping(source_tok, target_tok, source_prompt, target_prompt, paper, split):
    s = source_tok(source_prompt, add_special_tokens=False, return_offsets_mapping=True)
    t = target_tok(target_prompt, add_special_tokens=False, return_offsets_mapping=True)
    sa, ta = source_prompt.index(paper), target_prompt.index(paper)
    # Control tokens and tokens crossing the paper boundary remain receiver-native.
    sb = [(i, a-sa, b-sa) for i, (a, b) in enumerate(s['offset_mapping'])
          if sa <= a < b <= sa+len(paper)]
    tb = [(i, a-ta, b-ta) for i, (a, b) in enumerate(t['offset_mapping'])
          if i < split and ta <= a < b <= ta+len(paper)]
    boundaries = sorted({0} | ({b for _, _, b in sb} & {b for _, _, b in tb}))
    mapping = {}
    for left, right in zip(boundaries, boundaries[1:]):
        si = [i for i, _, b in sb if left < b <= right]
        ti = [i for i, _, b in tb if left < b <= right]
        if not si or not ti:
            continue
        for rank, index in enumerate(ti):
            rank_s = (math.ceil((rank+1)*len(si)/len(ti))-1 if len(si)>len(ti)
                      else rank*len(si)//len(ti))
            mapping[index] = si[rank_s]
    wanted = {i for i, _, _ in tb}
    if set(mapping) != wanted or not wanted:
        raise RuntimeError(f'Incomplete paper-body Full-Sync mapping: {len(wanted-set(mapping))}')
    target = sorted(mapping)
    return s['input_ids'], target, [mapping[i] for i in target]


def run():
    e.seed_all(1234)
    out = e.ROOT/'results_chat_transfer_v1'
    out.mkdir(exist_ok=True)
    rows = json.loads((e.ROOT/'results/manifest.json').read_text())['rows']
    native = json.loads((e.ROOT/'results_native_chat_v1/per_sample.json').read_text())
    native_by_id = {r['question_id']: r for r in native}
    cfg = {'models': e.CFG['models'], 'attention_implementation': e.CFG['attention_implementation']}
    receiver = e.load_model(cfg, 'qwen')
    sender = e.load_model(cfg, 'llama')
    tok = AutoTokenizer.from_pretrained(e.CFG['models']['qwen'], local_files_only=True)
    st = AutoTokenizer.from_pretrained(e.CFG['models']['llama'], local_files_only=True)
    eos = receiver.generation_config.eos_token_id
    eos_set = {eos} if isinstance(eos, int) else set(eos)
    e.save_json(out/'config.json', {'seed':1234, 'split':'dev', 'count':len(rows),
        'source_manifest':'results/manifest.json', 'native_reference':'results_native_chat_v1',
        'source_chat_template':st.chat_template, 'receiver_chat_template':tok.chat_template,
        'system':SYSTEM, 'enable_thinking':False, 'max_new_tokens':128, 'do_sample':False,
        'training':False, 'source_question_visible':False,
        'translation_scope':'paper-body only, Full-Sync copy/drop; native receiver wrapper and Question',
        'note':'Off-distribution frozen-checkpoint diagnostic, not a trained QASPER benchmark.',
        'checkpoints':{g:{x:str(e.LEGACY/f"runs/{g}/stage_{x}/best.pt") for x in ('a','b')}
                       for g in ('mcq','gsm8k')}})
    records = []
    for group in ('mcq','gsm8k'):
        base = e.load_base(e.LEGACY/f'runs/{group}/stage_a/best.pt')
        adapter = e.load_adapter(e.LEGACY/f'runs/{group}/stage_b/best.pt')
        for index, row in enumerate(rows):
            nr = native_by_id[row['question_id']]
            paper = 'Title: '+row['evidence'].split('Title: ',1)[1]
            sp = st.apply_chat_template([{'role':'system','content':SYSTEM},
                {'role':'user','content':'Paper:\n'+paper}], tokenize=False, add_generation_prompt=False)
            sid, target, selected = body_mapping(st,tok,sp,nr['prompt'],paper,nr['cache_split'])
            sk, sv = capture(sender,sid)
            nk, nv = capture(receiver,nr['input_ids'][:nr['cache_split']])
            item = {'source_k':sk[:,selected], 'source_v':sv[:,selected], 'tokens':len(selected)}
            bk, bv = e.map_base(base,item)
            ids = torch.tensor([nr['input_ids']],device='cuda',dtype=torch.long)
            args = dict(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=128,
                        do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id or tok.eos_token_id)
            pos = torch.arange(nr['cache_split'],device='cuda')
            # Audit the exact capture/re-RoPE path used by the Writers before substitution.
            if group == 'mcq' and index < 2:
                rebuilt = receiver.generate(**args,past_key_values=e.make_cache(receiver,nk.cuda(),nv.cuda(),pos))[0,len(nr['input_ids']):].tolist()
                e.save_json(out/f'rebuild_oracle_{index}.json',{'question_id':row['question_id'],
                    'generation_identical':rebuilt==nr['full_ids'],'ids':rebuilt,'reference_ids':nr['full_ids']})
                if rebuilt != nr['full_ids']:
                    raise RuntimeError('Native pre-RoPE rebuild failed exact generation parity; audit before translation')
            for stage in ('a','b'):
                k,v = bk,bv
                if stage == 'b':
                    with torch.amp.autocast('cuda',dtype=torch.float16):
                        k,v,_,_ = adapter(bk[None],bv[None])
                    k,v = k[0],v[0]
                fk,fv = nk.cuda().clone(),nv.cuda().clone()
                fk[:,target],fv[:,target] = k.to(fk.dtype),v.to(fv.dtype)
                start = time.perf_counter()
                generated = receiver.generate(**args,past_key_values=e.make_cache(receiver,fk,fv,pos))[0,len(nr['input_ids']):].tolist()
                records.append({'condition':group+'_stage_'+stage,'question_id':row['question_id'],
                    'paper_id':row['paper_id'],'type':row['type'],'text':tok.decode(generated,skip_special_tokens=True).strip(),
                    'generated_ids':generated,'eos':bool(generated and generated[-1] in eos_set),
                    'seconds':time.perf_counter()-start,'translated_tokens':len(target),
                    'native_prefix_tokens':nr['cache_split']-len(target), 'target_positions':target,
                    'source_positions':selected,'source_prompt':sp})
                e.save_json(out/'per_sample.json',records)
            del sk,sv,nk,nv,bk,bv,k,v,fk,fv
            print(f'CHAT TRANSFER {group} {index+1}/{len(rows)}',flush=True)
        del base,adapter
        torch.cuda.empty_cache()
    spec = importlib.util.spec_from_file_location('qasper_eval',e.DATA/'qasper_evaluator.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    refs = evaluator.get_answers_and_evidence(json.loads((e.DATA/'qasper-dev-v0.3.json').read_text()),True)
    gold = {r['question_id']:refs[r['question_id']] for r in rows}
    metrics = {}
    for name in sorted({r['condition'] for r in records}):
        rr = [r for r in records if r['condition']==name]
        predictions = {r['question_id']:{'answer':r['text'],'evidence':[]} for r in rr}
        report = evaluator.evaluate(gold,predictions)
        report.pop('Evidence F1',None)
        report.update(EOS_rate=sum(r['eos'] for r in rr)/len(rr),
                      mean_generated_tokens=sum(len(r['generated_ids']) for r in rr)/len(rr))
        metrics[name] = report
        e.save_json(out/(name+'_predictions.json'),predictions)
    summary = {'status':'completed','count':len(rows),'metrics':metrics,
        'native_chat_metrics':json.loads((e.ROOT/'results_native_chat_v1/summary.json').read_text())['metrics'],
        'note':'Same samples and raw official answer F1; frozen old Writers; no QASPER training.'}
    e.save_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    print('CHAT TRANSFER COMPLETED',flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        run()
