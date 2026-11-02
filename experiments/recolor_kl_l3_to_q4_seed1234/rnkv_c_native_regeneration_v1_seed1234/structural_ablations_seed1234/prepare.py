"""Reuse exact original train/validation and all seven fixed test cohorts."""
from ablation_common import *

def main():
    if (ROOT/'manifest.json').exists():verify();return
    R.verify();assert R.identity()==CFG['parent_manifest_sha256']
    cross=PARENT/CFG['cross_experiment'];assert C.sha(cross/'manifest.json')==CFG['cross_manifest_sha256']
    assert C.read(cross/'completed.json')['status']=='completed'
    assert C.sha(PARENT/'runs/epoch4_step1024.pt')==CFG['parent_checkpoint_sha256']
    for k in ['seed','epochs','effective_batch','updates','learning_rate','weight_decay','gradient_clip','checkpoint_every','lambda_kl','prefix_max_new_tokens','supervised_positions_max','prefix_quotas_per_batch']:
        assert CFG[k]==R.CFG[k],k
    data={s:R.rows(s) for s in ['train','validation','test']}
    data['test']+=C.read_jsonl(cross/'prepared/test.jsonl')
    assert len(data['train'])==2048 and len(data['validation'])==128 and len(data['test'])==448
    for r in data['test']:
        if r['task']=='arc':r['kind']='label'
        elif r['task']=='hotpotqa':r.update(kind='generation',answers=[r['gold']])
        baseline(r)
    assert all(sum(r['task']==t for r in data['test'])==n for t,n in CFG['test_counts'].items())
    for s,rr in data.items():C.write_jsonl(ROOT/'prepared'/(s+'.jsonl'),rr)
    for s in ['train','validation']:assert C.sha(ROOT/'prepared'/(s+'.jsonl'))==C.sha(PARENT/'prepared'/(s+'.jsonl'))
    refs=[PARENT/n for n in ['manifest.json','config.json','prepared/train.jsonl','prepared/validation.jsonl','prepared/test.jsonl','runs/epoch4_step1024.pt']]
    refs+=list(PARENT.glob('*.py'))+[cross/'manifest.json',cross/'prepared/test.jsonl']+list(cross.glob('*.py'))
    refs+=[ROOT/'prepared'/(s+'.jsonl') for s in data]
    refs+=[(PARENT if r['task'] in ['arc','hotpotqa'] else cross)/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json' for r in data['test']]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},
        sources={str(f):C.sha(f) for f in refs},counts={s:len(rr) for s,rr in data.items()},
        training_data_identical=True,validation_data_identical=True,test_cohorts_identical=True,
        batch_schedules={str(e):C.digest([[(r['id'],k) for r,k in b] for b in M.batches(e)]) for e in range(1,5)},
        maxima={s:max(len(r['content_ids']) for r in rr) for s,rr in data.items()}))
    mark('prepared/completed.json',status='completed',counts={s:len(rr) for s,rr in data.items()},original_splits_byte_identical=True)
    C.log('STRUCTURAL ABLATIONS PREPARED',{s:len(rr) for s,rr in data.items()})

if __name__=='__main__':main()
