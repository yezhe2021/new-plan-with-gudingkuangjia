"""Freeze all examples, donor assignments and synthetic fact pairs before diagnostics."""
import math
import random
from common_diag import *

def main():
    if (ROOT/'manifest.json').exists():verify();return
    N.verify();assert N.identity()==CFG['parent_manifest_sha256']
    assert C.sha(PARENT/'runs/epoch4_step1536.pt')==CFG['checkpoint_sha256']
    data=[r for r in N.rows('test') if r['task'] in CFG['test_counts']];assert len(data)==192
    donors={};assignments=[]
    for task in CFG['test_counts']:
        rr=sorted([r for r in data if r['task']==task],key=lambda r:(len(r['content_ids']),r['id']))
        assert len(rr)==64
        for start in range(0,64,8):
            group=rr[start:start+8];choices=list(range(1,8));random.Random(CFG['seed']+start).shuffle(choices)
            shift=min(choices,key=lambda s:sum(C.norm(r['gold'])==C.norm(group[(i+s)%8]['gold']) for i,r in enumerate(group))*1000+sum(abs(math.log(len(group[(i+s)%8]['content_ids'])/len(r['content_ids']))) for i,r in enumerate(group)))
            for i,row in enumerate(group):
                donor=group[(i+shift)%8];assert row['id']!=donor['id'];donors[row['id']]=donor['id']
                assignments.append(dict(id=row['id'],donor=donor['id'],task=task,body_tokens=len(row['content_ids']),donor_body_tokens=len(donor['content_ids']),same_gold=C.norm(row['gold'])==C.norm(donor['gold'])))
    strata={}
    for r in data:
        old=prior(r);a=old['C']['em'];b=old[N.ARM]['em']
        strata[r['id']]='rescued' if a==0 and b==1 else 'harmed' if a==1 and b==0 else 'both_correct' if a==1 else 'both_wrong'
    C.write_jsonl(ROOT/'prepared/test.jsonl',data);C.save(ROOT/'prepared/donors.json',donors);C.save(ROOT/'prepared/strata.json',strata)
    tok=C.AutoTokenizer.from_pretrained(C.CFG['models']['receiver'],local_files_only=True)
    sender=C.AutoTokenizer.from_pretrained(C.CFG['models']['sender'],local_files_only=True);affix=C.read(C.ROOT/'manifest.json')
    projects=['Amber','Birch','Cedar','Dahlia','Elm','Flint','Garnet','Hazel','Iris','Jasper','Kestrel','Laurel','Maple','Nettle','Opal','Pine']
    cities=['Paris','Rome','Oslo','Lima','Tokyo','Seoul','Cairo','Delhi','Riga','Bern','Doha','Perth','Sofia','Accra','Dakar','Kyoto','Lisbon','Berlin','Madrid','Vienna','Dublin','Athens','Boston','Austin']
    pairs=[];synthetic=[]
    for i,project in enumerate(projects):
        org='Oriole '+project+' Foundation';question='In which city is the archive of the organization that acquired Project '+project+' located?'
        def build(city,tag):
            prompt=('Answer the question using the passages below. Output only the short answer, without an explanation.\n\n'
                +'Passage 1:\nProject '+project+'\nProject '+project+' was acquired by the '+org+'.\n\n'
                +'Passage 2:\n'+org+'\nThe '+org+' operates its archive in '+city+'.\n\n'
                +'Passage 3:\nUnrelated organization\nThe Meadow Council operates an unrelated archive in Wellington.\n\nQuestion: '+question+'\nGive only the answer.')
            row=dict(id=f'cf_{i:02d}_{tag}',task='hotpotqa',question=question,prompt=prompt,gold=city,answers=[city],kind='generation',continuation_suffix=affix['assistant'])
            row['content_ids']=tok.encode(prompt,add_special_tokens=False);row['suffix_ids']=tok.encode(affix['assistant'],add_special_tokens=False)
            row['sender_tokens']=len(sender.encode(prompt,add_special_tokens=False));return row
        candidates=[build(city,'a') for city in cities]
        valid=[(a,b) for a in range(len(cities)) for b in range(a+1,len(cities)) if len(candidates[a]['content_ids'])==len(candidates[b]['content_ids']) and candidates[a]['sender_tokens']==candidates[b]['sender_tokens']]
        assert valid;ai,bi=valid[i%len(valid)];a=build(cities[ai],'a');b=build(cities[bi],'b')
        assert a['question']==b['question'] and a['prompt'].replace(a['gold'],b['gold'])==b['prompt']
        synthetic.extend([a,b]);pairs.append(dict(id=f'cf_{i:02d}',a=a['id'],b=b['id'],same_receiver_and_sender_token_length=True))
    C.write_jsonl(ROOT/'prepared/counterfactual.jsonl',synthetic);C.save(ROOT/'prepared/counterfactual_pairs.json',pairs)
    refs=[PARENT/'manifest.json',PARENT/'runs/epoch4_step1536.pt',PARENT/'prepared/test.jsonl',Path(CFG['diagnostics'])]
    refs+=list(PARENT.glob('*.py'))+[PARENT/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json' for r in data]
    refs+=list((ROOT/'prepared').glob('*'))
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},
        counts=CFG['test_counts'],donor_assignments=assignments,strata_counts={s:sum(v==s for v in strata.values()) for s in set(strata.values())},training=False,synthetic_pairs=16))
    mark('prepared/completed.json',status='completed',n=192,counterfactual_pairs=16);C.log('MECHANISM PREPARED',192,16)

if __name__=='__main__':main()
