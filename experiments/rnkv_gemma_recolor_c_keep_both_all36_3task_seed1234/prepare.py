"""Pin the completed Gemma19-30 reference and copy byte-identical cohorts."""
import shutil
from experiment_common import *
from runtime_new import batches

def main():
    if (ROOT/'manifest.json').exists():verify();return
    PG.verify();assert PG.identity()==CFG['reference_manifest_sha256'] and PG.done('completed.json')
    assert PG.map_sha()==CFG['recolor_sha256'] and C.sha(PARENT/'runs/C/epoch4_step800.pt')==CFG['c_checkpoint_sha256']
    assert C.sha(PARENT/'runs/epoch4_step1536.pt')==CFG['partial_checkpoint_sha256']
    same=['seed','memory_tokens','hidden_size','adapter_rank','adapter_after_layers','train_counts','validation_counts','test_counts','epochs','effective_batch','updates','learning_rate','weight_decay','gradient_clip','lambda_kl','prefix_max_new_tokens','supervised_positions_max','prefix_quotas_per_batch','updates_per_epoch','evaluation_generation_caps','sender','receiver','source_dtype']
    assert all(CFG[k]==PG.CFG[k] for k in same);assert PG.CFG['replace_layers']==list(range(19,31)) and CFG['replace_layers']==list(range(1,37))
    refs=[PARENT/n for n in ['manifest.json','config.json','completed.json','maps/recolor.pt','maps/completed.json','runs/C/epoch4_step800.pt','runs/C/completed.json','runs/epoch4_step1536.pt']]
    refs+=list(PARENT.glob('*.py'))
    maxima={}
    for split in ['train','validation','test']:
        path=ROOT/f'prepared/{split}.jsonl';path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/f'prepared/{split}.jsonl',path)
        rr=rows(split);assert {t:sum(r['task']==t for r in rr) for t in CFG[split+'_counts']}==CFG[split+'_counts']
        refs.append(path);maxima[split]={t:dict(body_tokens=max(len(r['content_ids']) for r in rr if r['task']==t),gold_tokens=max(len(r.get('gold_tokens',[])) for r in rr if r['task']==t)) for t in CFG[split+'_counts']}
    for row in rows('test'):baseline(row);refs.append(reference_path(row))
    schedules={str(e):C.digest([[(r['id'],k) for r,k in b] for b in batches(e)]) for e in range(1,5)}
    assert schedules==C.read(PARENT/'manifest.json')['batch_schedules']
    assert shutil.disk_usage(ROOT).free>CFG['disk_reserve_gib']*2**30
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},
        counts={s:len(rows(s)) for s in ['train','validation','test']},maxima=maxima,batch_schedules=schedules,reference_settings_identical=same,
        only_training_setting_change='Answer-time Memory access19-30 ->1-36',reference_data_byte_identical=True,recolor_frozen_sha256=CFG['recolor_sha256'],c_frozen_sha256=CFG['c_checkpoint_sha256'],trainable_parameters=409600))
    assert not torch.cuda.is_initialized();mark('prepared/completed.json',status='completed',counts={s:len(rows(s)) for s in ['train','validation','test']},single_factor='Memory answer access scope',translator_retrained=False)
    C.log('GEMMA ALL36 PREPARED same data/schedule/translator as19-30; fresh Memory training')
if __name__=='__main__':main()
