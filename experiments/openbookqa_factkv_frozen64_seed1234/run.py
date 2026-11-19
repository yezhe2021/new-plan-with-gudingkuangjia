"""OpenBookQA Additional fact1-only frozen MCQ Writer: 64 official test questions."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mcq_evidenceonly_frozen64_seed1234'))
from run import kb, LEGACY, digest, body_map, scores, prediction


def data():
    archive=ROOT/'data/OpenBookQA-V1-Sep2018.zip'
    with zipfile.ZipFile(archive) as z:
        filename=next(n for n in z.namelist() if n.endswith('Additional/test_complete.jsonl'))
        extra=[json.loads(line) for line in z.read(filename).decode().splitlines() if line.strip()]
    byid={str(r['id']):r for r in extra}
    main=[json.loads(line) for line in (Path(kb.CFG['datasets']['openbookqa_root'])/'test.jsonl').read_text().splitlines() if line.strip()]
    rows=[]
    for row in main[:64]:
        add=byid[str(row['id'])]
        assert row['question']==add['question'] and row['answerKey']==add['answerKey'], 'Main/Additional mismatch'
        assert isinstance(add['fact1'],str) and add['fact1'].strip(), 'Missing fact1; no substitution'
        rows.append(add)
    assert len(rows)==64
    return rows,{'archive':str(archive),'sha256':digest(archive),'member':filename,'alignment':'exact ID/question/choices/answerKey equality'}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    kb.seed_all(1234)
    assert torch.cuda.is_available()
    rows,source=data(); rows=rows[:2] if a.smoke else rows
    out=ROOT/('smoke' if a.smoke else 'results');out.mkdir(parents=True,exist_ok=True)
    ca=LEGACY/'runs/mcq/stage_a/best.pt';cb=LEGACY/'runs/mcq/stage_b/best.pt'
    cfg={'models':kb.CFG['models'],'attention_implementation':kb.CFG['attention_implementation']}
    rt=AutoTokenizer.from_pretrained(cfg['models']['qwen'],local_files_only=True)
    st=AutoTokenizer.from_pretrained(cfg['models']['llama'],local_files_only=True)
    receiver=kb.load_model(cfg,'qwen');sender=kb.load_model(cfg,'llama')
    base=kb.load_base(ca);adapter=kb.load_adapter(cb)
    kb.save_json(out/'config.json',{'training':False,'count':len(rows),'seed':1234,'source':source,
        'selection':'first64 Main official test IDs joined to official Additional fact1',
        'checkpoints':{str(ca):digest(ca),str(cb):digest(cb)},
        'sender':'newline + fact1 only; BOS added; never question/choices/answerKey',
        'receiver':'native newline token0 + optional translated fact KV + native Question/Options/Answer',
        'alignment':'Full-Sync copy/drop, all fact tokens excluding fixed native newline anchor',
        'baseline':'identical native newline anchor + identical Question/Options/Answer; no fact text or fact KV',
        'scoring':'answer-text summed likelihood and character-length-normalized likelihood; 0shot, no chat template',
        'scope':'gold supporting-fact diagnostic, not retrieval or standard closed-book benchmark',
        'position_protocol':'contiguous actual positions; question shifted by inserted memory length'})
    records=[]
    for i,row in enumerate(rows):
        q=row['question']['stem']; choices=[x['text'] for x in row['question']['choices']]
        labels=[x['label'] for x in row['question']['choices']];gold=labels.index(row['answerKey'])
        readout='\n\nQuestion: '+q+'\n\nOptions:\n'+'\n'.join(f'{l}. {c}' for l,c in zip(labels,choices))+'\n\nAnswer:'
        # Constant newline anchor keeps token0 native without leaking any fact word.
        body='\n'+row['fact1'];bids=rt.encode(body,add_special_tokens=False)
        anchor=rt.encode('\n',add_special_tokens=False)
        assert len(anchor)==1 and bids[0]==anchor[0], 'Fact word entered native anchor'
        enc=rt(body+readout,add_special_tokens=False,return_offsets_mapping=True)
        n=sum(b<=len(body) for _,b in enc['offset_mapping'])
        assert enc['input_ids'][:n]==bids[:n] and n>1
        for left,right in enc['offset_mapping'][n:]:
            assert left>=len(body) or not body[left:min(right,len(body))].strip(), 'Fact content leaked into native readout'
        ctx=enc['input_ids'];nk,nv=kb.capture(receiver,bids[:n])
        sid,selected=body_map(st,rt,body,body);selected=selected[:n-1]
        assert len(selected)==n-1
        sk,sv=kb.capture(sender,sid)
        item={'source_k':sk[:,selected],'source_v':sv[:,selected],'tokens':len(selected)}
        bk,bv=kb.map_base(base,item)
        with torch.amp.autocast('cuda',dtype=torch.float16): k,v,_,_=adapter(bk[None],bv[None])
        fk=torch.cat((nk[:,:1].cuda(),k[0]),1);fv=torch.cat((nv[:,:1].cuda(),v[0]),1)
        without=rt.encode('\n'+readout,add_special_tokens=False)
        conditions={'no_fact_kv':prediction(scores(receiver,rt,without,choices),choices),
                    'translated_fact_kv':prediction(scores(receiver,rt,ctx,choices,(fk,fv)),choices)}
        parity=None
        if a.smoke:
            full=scores(receiver,rt,ctx,choices)
            reused=scores(receiver,rt,ctx,choices,(nk.cuda(),nv.cuda()))
            parity={'full_scores':full,'native_cache_scores':reused,'max_score_difference':max(abs(x-y) for x,y in zip(full,reused))}
            assert prediction(full,choices)['pred']==prediction(reused,choices)['pred'], 'Native cache prediction mismatch'
        records.append({'index':i,'id':row['id'],'question':q,'choices':choices,'gold':gold,
            'fact1':row['fact1'],'sender_prompt':body,'receiver_readout':readout,
            'fact_memory_tokens':n-1,'native_anchor_tokens':1,'conditions':conditions,'smoke_parity':parity})
        kb.save_json(out/'per_sample.json',records)
        print(f'DONE {i+1}/{len(rows)} id={row["id"]}',flush=True)
        del nk,nv,sk,sv,item,bk,bv,k,v,fk,fv
    metrics={c:{'n':len(records),'correct':sum(r['conditions'][c]['pred']==r['gold'] for r in records),
                  'correct_norm':sum(r['conditions'][c]['pred_norm']==r['gold'] for r in records)} for c in conditions}
    paired={}
    for key in ['pred','pred_norm']:
        counts={'both_correct':0,'no_fact_only':0,'fact_only':0,'both_wrong':0}
        for r in records:
            x=r['conditions']['no_fact_kv'][key]==r['gold'];y=r['conditions']['translated_fact_kv'][key]==r['gold']
            counts['both_correct' if x and y else 'no_fact_only' if x else 'fact_only' if y else 'both_wrong']+=1
        paired[key]=counts
    summary={'status':'completed','metrics':metrics,'paired':paired}
    kb.save_json(out/'summary.json',summary);print(json.dumps(summary),flush=True);print('COMPLETED',flush=True)


if __name__=='__main__':main()
