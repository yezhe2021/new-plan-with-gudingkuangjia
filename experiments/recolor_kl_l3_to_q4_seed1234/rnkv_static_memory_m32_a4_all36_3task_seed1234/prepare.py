"""Byte-identical existing splits, budgets and native rollouts; no resampling."""
import shutil
from experiment_common import *
from runtime_new import batches

def main():
    if (ROOT/'manifest.json').exists():verify();return
    N.verify();assert N.identity()==CFG['reference_manifest_sha256'] and N.done('completed.json')
    cp=PARENT/'runs/epoch4_step1536.pt';assert C.sha(cp)==CFG['reference_checkpoint_sha256']
    for key in ['seed','memory_tokens','hidden_size','adapter_rank','adapter_after_layers','replace_layers','train_counts','validation_counts','test_counts','epochs','effective_batch','updates','learning_rate','weight_decay','gradient_clip','lambda_kl','prefix_max_new_tokens','supervised_positions_max','prefix_quotas_per_batch','updates_per_epoch','evaluation_generation_caps','c_checkpoint_sha256']:
        assert CFG[key]==N.CFG[key],key
    refs=[PARENT/'manifest.json',PARENT/'config.json',cp]+list(PARENT.glob('*.py'))
    for split in ['train','validation','test']:
        source=PARENT/f'prepared/{split}.jsonl';target=ROOT/f'prepared/{split}.jsonl'
        target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);assert C.sha(source)==C.sha(target)
        refs.extend([source,target]);assert {t:sum(r['task']==t for r in rows(split)) for t in CFG[split+'_counts']}==CFG[split+'_counts']
    refs.extend(baseline_path(r) for r in rows('test'))
    for row in rows('test'):baseline(row)
    prefixes={str(p):C.sha(p) for p in (PARENT/'native_prefix').glob('*.json')}
    schedules={str(e):C.digest([[(r['id'],kind) for r,kind in b] for b in batches(e)]) for e in range(1,5)}
    assert schedules==C.read(PARENT/'manifest.json')['batch_schedules']
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},
        counts={s:len(rows(s)) for s in ['train','validation','test']},trainable_parameters=409600,batch_schedules=schedules,native_prefix_sources=prefixes,construction_input='fixed canonical public header only',reference_checkpoint_sha256=C.sha(cp)))
    mark('prepared/completed.json',status='completed',splits_byte_identical=True,training_schedules_identical=True,native_prefix_files=len(prefixes))
    C.log('STATIC MEMORY PREPARED',identity())

if __name__=='__main__':main()
