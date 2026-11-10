"""Byte-identical Static-control data and schedule; only construction changes."""
import shutil
from experiment_common import *
from runtime_new import batches

def main():
    if (ROOT/'manifest.json').exists():verify();return
    F.verify();assert F.identity()==CFG['static_reference_manifest_sha256'] and F.done('completed.json')
    assert C.sha(STATIC_ROOT/'runs/epoch4_step1536.pt')==CFG['static_reference_checkpoint_sha256']
    assert C.sha(DST/'runs/stage2/epoch4_step1536.pt')==CFG['dst_checkpoint_sha256']
    for key in ['seed','memory_tokens','hidden_size','adapter_rank','adapter_after_layers','replace_layers','train_counts','validation_counts','test_counts','epochs','updates','effective_batch','updates_per_epoch','learning_rate','weight_decay','gradient_clip','lambda_kl','prefix_quotas_per_batch','prefix_max_new_tokens','supervised_positions_max','evaluation_generation_caps','dst_checkpoint_sha256']:
        assert CFG[key]==F.CFG[key],key
    refs=[STATIC_ROOT/'manifest.json',STATIC_ROOT/'runs/epoch4_step1536.pt',DST/'runs/stage2/epoch4_step1536.pt']
    for split in ['train','validation','test']:
        source=STATIC_ROOT/f'prepared/{split}.jsonl';target=ROOT/f'prepared/{split}.jsonl';target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
        assert C.sha(source)==C.sha(target);refs.extend([source,target])
    for row in rows('test'):baseline(row);refs.append(baseline_path(row))
    refs+=list(STATIC_ROOT.glob('*.py'))+list(DST.glob('*.py'))
    refs+=[C.ROOT/'metrics.py',ORIGINAL/'cross_dataset5x64_seed1234/qa_metrics.py',ORIGINAL/'cross_dataset5x64_seed1234/experiment.py',GSM/'numeric.py']
    schedules={str(e):C.digest([[(r['id'],kind) for r,kind in b] for b in batches(e)]) for e in range(1,5)}
    assert schedules==C.read(STATIC_ROOT/'manifest.json')['batch_schedules']
    prefixes={str(f):C.sha(f) for f in (PARENT/'native_prefix').glob('*.json')}
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},batch_schedules=schedules,native_prefix_sources=prefixes,trainable_parameters=409600,fresh_initialization=True,construction_input='fixed public header + current full frozen DST body',counts={s:len(rows(s)) for s in ['train','validation','test']}))
    mark('prepared/completed.json',status='completed',static_control_rows_byte_identical=True,static_control_budget_and_schedules_identical=True,frozen_dst_checkpoint_sha256=CFG['dst_checkpoint_sha256'])
    C.log('DST DYNAMIC MEMORY PREPARED',identity())

if __name__=='__main__':main()
