"""Reuse old Static training budget and rows with the DST's fixed Full-Sync grid."""
import shutil
from experiment_common import *
from runtime_new import batches

def main():
    if (ROOT/'manifest.json').exists():verify();return
    Z.verify();assert Z.done('completed.json') and Z.identity()==CFG['transfer_manifest_sha256']
    assert D.identity()==CFG['dst_manifest_sha256'] and S.identity()==CFG['static_manifest_sha256']
    assert C.sha(DST/'runs/stage2/epoch4_step1536.pt')==CFG['dst_checkpoint_sha256']
    for key in ['seed','memory_tokens','adapter_rank','adapter_after_layers','replace_layers','train_counts','validation_counts','test_counts','epochs','updates','effective_batch','updates_per_epoch','learning_rate','weight_decay','gradient_clip','lambda_kl','prefix_quotas_per_batch','prefix_max_new_tokens','supervised_positions_max','evaluation_generation_caps']:
        assert CFG[key]==S.CFG[key],key
    refs=[TRANSFER/'manifest.json',DST/'manifest.json',SOURCE/'manifest.json',DST/'runs/stage2/epoch4_step1536.pt']
    for split in ['train','validation','test']:
        source=DST/f'prepared/{split}.jsonl';target=ROOT/f'prepared/{split}.jsonl';target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
        old={r['id']:r for r in S.rows(split)};data=rows(split)
        assert len(old)==len(data)==sum(CFG[split+'_counts'].values())
        for row in data:assert all(row[k]==v for k,v in old[row['id']].items()),row['id']
        refs.extend([source,target,SOURCE/f'prepared/{split}.jsonl'])
    for row in rows('test'):baseline(row);refs.append(baseline_path(row))
    refs+=list(TRANSFER.glob('*.py'))+list(SOURCE.glob('*.py'))+list(DST.glob('*.py'))
    refs+=[C.ROOT/'metrics.py',ORIGINAL/'cross_dataset5x64_seed1234/qa_metrics.py',ORIGINAL/'cross_dataset5x64_seed1234/experiment.py',GSM/'numeric.py']
    schedules={str(e):C.digest([[(r['id'],kind) for r,kind in b] for b in batches(e)]) for e in range(1,5)}
    assert schedules==C.read(SOURCE/'manifest.json')['batch_schedules']
    prefixes={str(f):C.sha(f) for f in (PARENT/'native_prefix').glob('*.json')}
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},batch_schedules=schedules,native_prefix_sources=prefixes,trainable_parameters=409600,fresh_initialization=True,construction_input='fixed public header only',counts={s:len(rows(s)) for s in ['train','validation','test']}))
    mark('prepared/completed.json',status='completed',original_static_budget_and_rows_identical=True,frozen_dst_checkpoint_sha256=CFG['dst_checkpoint_sha256'])
    C.log('DST-SPECIFIC STATIC MEMORY PREPARED',identity())

if __name__=='__main__':main()
