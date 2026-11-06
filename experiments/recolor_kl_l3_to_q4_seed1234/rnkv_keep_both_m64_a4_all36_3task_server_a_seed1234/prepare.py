"""Copy byte-identical cohorts and pin one-dimensional Memory ablation."""
import shutil
from experiment_common import *
from runtime_new import batches

def probe_rows():
    data=rows('train');chosen={}
    for task in CFG['train_counts']:
        rr=[r for r in data if r['task']==task];chosen[rr[0]['id']]=rr[0]
        for key in [lambda r:len(r['content_ids']),lambda r:len(r['content_ids'])+len(r['gold_tokens'])]:
            row=max(rr,key=key);chosen[row['id']]=row
    for split in ['validation','test']:
        for task in CFG[split+'_counts']:
            rr=[r for r in rows(split) if r['task']==task];row=max(rr,key=lambda r:len(r['content_ids'])+len(r.get('gold_tokens',[])));chosen[row['id']]=row
    for batch in batches(1)[:4]:
        for row,kind in batch:chosen[row['id']]=row
    for task in CFG['test_counts']:
        row=next(r for r in rows('test') if r['task']==task);chosen[row['id']]=row
    return list(chosen.values())

def main():
    if (ROOT/'manifest.json').exists():verify();return
    N.verify();assert N.identity()==CFG['reference_manifest_sha256']
    for split in ['train','validation','test']:
        p=ROOT/f'prepared/{split}.jsonl';p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REFERENCE/f'prepared/{split}.jsonl',p)
        rr=rows(split);assert {t:sum(r['task']==t for r in rr) for t in CFG[split+'_counts']}==CFG[split+'_counts']
        for row in rr:
            assert len(C.read(C.ROOT/'manifest.json')['prefix_ids'])+len(row['content_ids'])+64+len(row['suffix_ids'])+max(64,len(row.get('gold_tokens',[])))<CFG['max_context_tokens']
    probes=probe_rows();C.write_jsonl(ROOT/'prepared/probe.jsonl',probes)
    data=rows('train')+rows('validation')+rows('test');unique={str(cache_file(r)):r for r in data}
    expected=sum(36*len(r['content_ids'])*1024*4+8192 for r in unique.values())
    assert shutil.disk_usage(ROOT).free>expected+CFG['disk_reserve_gib']*2**30
    refs=[REFERENCE/'manifest.json',REFERENCE/'config.json',C.ROOT/'runs/C/epoch4_step800.pt',Path(C.CFG['recolor'])]
    refs+=list(REFERENCE.glob('*.py'))+[ROOT/f'prepared/{s}.jsonl' for s in ['train','validation','test','probe']]
    refs+=[REFERENCE/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json' for r in rows('test')]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},
        counts={s:len(rows(s)) for s in ['train','validation','test']},memory_tokens=64,trainable_parameters=491520,initialization_shared_with_m32=True,
        expected_base_cache_gib=expected/2**30,unique_cache_entries=len(unique),probe_n=len(probes),
        batch_schedules={str(e):C.digest([[(r['id'],k) for r,k in b] for b in batches(e)]) for e in range(1,5)}))
    mark('prepared/completed.json',status='completed',expected_base_cache_gib=expected/2**30,probe_n=len(probes));assert not torch.cuda.is_initialized()
    C.log('M64 PREPARED',expected/2**30,'GiB exact base cache',len(probes),'probe rows')
if __name__=='__main__':main()
