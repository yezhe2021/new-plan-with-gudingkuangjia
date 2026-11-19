"""Answer-only gold-evidence pilot; raw outputs, paired gates, and frozen fusion."""
import argparse
from collections import Counter
from pathlib import Path
import json
import random
import re
import string
import torch
from common import ROOT, CFG, read, rows, save, append, log, verify, signature, release, digest
from models import Pool, generate, capture, activate, offload
from protocol import build, INSTRUCTION
from algorithms import Engine, our_model, our_map, residual, hf_map
from train import load_a, folder


def normalize(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def score(pred, gold):
    p,g=normalize(pred),normalize(gold);em=float(p==g)
    if (p in ('yes','no','noanswer') or g in ('yes','no','noanswer')) and not em:return em,0.
    a,b=p.split(),g.split();n=sum((Counter(a)&Counter(b)).values())
    return em,(2*n/(len(a)+len(b)) if n else 0.)


def interval(differences):
    import numpy as np
    x=np.asarray(differences,dtype=float);rng=np.random.default_rng(CFG['seed'])
    means=x[rng.integers(len(x),size=(CFG['validation_gate']['bootstrap_replicates'],len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],'n':len(x)}


def plain_question(tok, question):
    value=tok.apply_chat_template([{'role':'user','content':INSTRUCTION+'\n\nQuestion: '+question+'\nAnswer:'}],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,date_string=CFG['native_template_date'])
    return list(value['input_ids'] if hasattr(value,'keys') else value)


def load_models(methods, family):
    result={}
    if 'our' in methods:
        base=load_a(family)
        item=read(folder(family,'our_b')/'completed.json')['checkpoint']
        assert digest(Path(item['path']))==item['sha256'],'Stage B checkpoint identity changed'
        a=residual().eval().requires_grad_(False)
        payload=torch.load(item['path'],map_location='cpu',weights_only=True);a.load_state_dict(payload['state']);del payload
        result['our']=(base,a)
    if 'heterofold' in methods:
        item=read(folder(family,'hf_calib')/'completed.json')['checkpoint']
        assert digest(Path(item['path']))==item['sha256'],'HeteroFold checkpoint identity changed'
        result['heterofold']=torch.load(item['path'],map_location='cpu',weights_only=True)
    return result


def mapped_group(engine, group, family, methods, learned, units):
    pairs=engine.pairs(group,family,units);engine.pool.cpu();memories={}
    if 'our' in methods:
        base,adapter=learned['our'];activate(base);activate(adapter)
        with torch.no_grad():
            for pair in pairs:
                k,v=our_map(base,pair);k,v,_,_=adapter(k[None].cuda(),v[None].cuda())
                memories[('our',family,pair['row']['id'],pair['unit'])]=(k[0].cpu(),v[0].cpu())
        offload(base);offload(adapter)
    if 'heterofold' in methods:
        receiver=engine.pool.gpu('qwen')
        with torch.no_grad():
            for pair in pairs:
                memories[('heterofold',family,pair['row']['id'],pair['unit'])]=hf_map(learned['heterofold'],pair,receiver)
        del receiver;engine.pool.cpu()
    return memories,{(p['row']['id'],p['unit']):p for p in pairs}


def run(split, family=None, native_only=False, fusion=False):
    verify()
    pool=Pool();engine=Engine(pool)
    output=ROOT/'evaluation'/split/('native' if native_only else family or 'all')
    output.mkdir(parents=True,exist_ok=True)
    dest=output/'generations.jsonl'
    previous=rows(dest) if dest.exists() else []
    done={(r['id'],r['arm']) for r in previous}
    data=rows(ROOT/f'prepared/{split}.jsonl')
    # Nearby lengths make wrong-document controls less sensitive to relocation;
    # no resampling/truncation is applied. Length differences are recorded.
    data.sort(key=lambda r:len(r['documents'][0])+len(r['documents'][1]))
    methods=[] if native_only else ['our','heterofold']
    families=[] if native_only else [family] if family else ['llama','gemma']
    learned={f:load_models(methods,f) for f in families}
    size=CFG['capture_batch_questions']
    for offset in range(0,len(data),size):
        group=data[offset:offset+size];memories={};pairs={}
        units=('0','1','joint') if fusion else ('joint',)
        for f in families:
            mm,pp=mapped_group(engine,group,f,methods,learned[f],units);memories.update(mm);pairs.update(pp)
        receiver=pool.gpu('qwen')
        for index,row in enumerate(group):
            proto=build(pool.tokenizers,row);t=proto['target']
            if (row['id'],'native_joint') not in done or native_only:
                native=capture(receiver,t['ids'][:t['end']])
            elif (row['id'],'joint') in pairs:
                native=pairs[(row['id'],'joint')]['target']
            else:
                native=capture(receiver,t['ids'][:t['end']])
            prefix=native['k'][:,:t['begin']],native['v'][:,:t['begin']]
            arms=[('question_only',None),('native_joint',None)]
            for f in families:
                for method in methods:
                    arms.append((f'{method}_{f}_joint',memories[(method,f,row['id'],'joint')]))
                    if split=='validation':
                        donor=min((r for r in group if r['id']!=row['id']),key=lambda r:abs(len(r['documents'][0])+len(r['documents'][1])-len(row['documents'][0])-len(row['documents'][1])))
                        arms.append((f'{method}_{f}_wrong',memories[(method,f,donor['id'],'joint')]))
            if fusion:
                # Native independent blocks are prefetched separately with the
                # common prefix, then post-normalization keys are re-RoPE'd at
                # final global positions. No equality to joint is assumed.
                independent=[]
                for unit in ('0','1'):
                    pt=build(pool.tokenizers,row,unit)['target']
                    record=capture(receiver,pt['ids'][:pt['end']])
                    independent.append((record['k'][:,pt['begin']:pt['end']],record['v'][:,pt['begin']:pt['end']]))
                arms.append(('native_independent',tuple(torch.cat([p[j] for p in independent],1) for j in (0,1))))
                for method in methods:
                    for assignment in CFG['fusion_assignments']:
                        left,right=assignment.split('_')
                        a=memories[(method,left,row['id'],'0')];b=memories[(method,right,row['id'],'1')]
                        arms.append((f'{method}_{assignment}',(torch.cat((a[0],b[0]),1),torch.cat((a[1],b[1]),1))))
            for arm,body in arms:
                if (row['id'],arm) in done:continue
                if arm=='question_only':
                    ids,stop=generate(receiver,plain_question(pool.tokenizers['qwen'],row['question']))
                    memory_tokens=0
                elif arm=='native_joint':
                    ids,stop=generate(receiver,t['ids']);memory_tokens=len(t['body_ids'])
                else:
                    memory=(torch.cat((prefix[0],body[0]),1),torch.cat((prefix[1],body[1]),1))
                    # Every condition starts before the question suffix; no
                    # correctly-conditioned question KVs survive an intervention.
                    ids,stop=generate(receiver,t['suffix_ids'],memory);memory_tokens=body[0].shape[1]
                text=pool.tokenizers['qwen'].decode(ids,skip_special_tokens=True).strip()
                em,f1=score(text,row['answer'])
                item={'id':row['id'],'arm':arm,'question':row['question'],'gold':row['answer'],'prediction':text,
                      'ids':ids,'em':em,'f1':f1,'stop':stop,'generated_tokens':len(ids),'memory_tokens':memory_tokens,
                      'correct_evidence_tokens':len(t['body_ids']),'question_suffix_recomputed':True,'gold_evidence_mechanism_pilot':True}
                append(dest,item);previous.append(item);done.add((row['id'],arm))
                save(output/'status.json',{'phase':'evaluating','completed_generations':len(previous),'last_id':row['id'],'last_arm':arm})
                log(f'EVAL {split} {len(previous)} {arm} EM={em} F1={f1:.3f} stop={stop}')
            del native
        del receiver;pool.cpu();del memories,pairs
    summary={}
    for arm in sorted({r['arm'] for r in previous}):
        rr=[r for r in previous if r['arm']==arm]
        assert len(rr)==len(data)
        summary[arm]={'n':len(rr),'em':sum(r['em'] for r in rr)/len(rr),'f1':sum(r['f1'] for r in rr)/len(rr),
                      'eos_rate':sum(r['stop']=='eos' for r in rr)/len(rr),
                      'mean_generated_tokens':sum(r['generated_tokens'] for r in rr)/len(rr)}
    save(output/'summary.json',{'completed':True,'split':split,'pilot_not_official_full_benchmark':True,'arms':summary})
    by_arm={arm:{r['id']:r for r in previous if r['arm']==arm} for arm in summary}
    comparisons=[]
    if 'native_independent' in by_arm:comparisons.append(('native_joint','native_independent'))
    for arm in sorted(summary):
        if arm not in ('question_only','native_joint') and not arm.endswith('_wrong'):
            comparisons.extend([(arm,'question_only'),(arm,'native_joint')])
            if 'native_independent' in by_arm and arm!='native_independent':comparisons.append((arm,'native_independent'))
    if fusion:
        for method in methods:
            for mixed in ('llama_gemma','gemma_llama'):
                for same in ('llama_llama','gemma_gemma'):
                    comparisons.append((f'{method}_{mixed}',f'{method}_{same}'))
        for assignment in CFG['fusion_assignments']:
            comparisons.append((f'our_{assignment}',f'heterofold_{assignment}'))
    paired={}
    for a,b in dict.fromkeys(comparisons):
        if a not in by_arm or b not in by_arm:continue
        ids=sorted(by_arm[a]);assert ids==sorted(by_arm[b])
        paired[a+' minus '+b]={metric:interval([by_arm[a][i][metric]-by_arm[b][i][metric] for i in ids]) for metric in ('em','f1')}
    save(output/'paired_comparisons.json',{'descriptive_only_not_checkpoint_selection':True,'contrasts':paired})
    if split=='validation':
        qonly={r['id']:r for r in previous if r['arm']=='question_only'}
        conditions=['native_joint'] if native_only else [f'{m}_{family}_joint' for m in methods]
        gates={}
        for arm in conditions:
            correct={r['id']:r for r in previous if r['arm']==arm}
            gain=interval([correct[i]['f1']-qonly[i]['f1'] for i in correct])
            passed=gain['mean']>=CFG['validation_gate']['min_f1_gain_over_question_only'] and gain['ci95'][0]>0
            wrong_drop=None
            if not native_only:
                wrong={r['id']:r for r in previous if r['arm']==arm.replace('_joint','_wrong')}
                wrong_drop=interval([correct[i]['f1']-wrong[i]['f1'] for i in correct])
                passed=passed and wrong_drop['mean']>0
            gates[arm]={'passed':passed,'f1_gain_over_question_only':gain,'correct_minus_wrong_f1':wrong_drop}
        save(output/'gate.json',{'passed':all(r['passed'] for r in gates.values()),'conditions':gates,
             'selected_on_validation_only':True,'wrong_control_length_matched_within_batch_not_exact_length':True})
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['validation','pilot']);ap.add_argument('--family',choices=['llama','gemma']);ap.add_argument('--native',action='store_true');ap.add_argument('--fusion',action='store_true')
    args=ap.parse_args();run(args.split,args.family,args.native,args.fusion)
