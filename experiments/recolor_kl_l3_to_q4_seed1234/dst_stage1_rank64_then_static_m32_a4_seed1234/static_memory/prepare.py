from experiment_common import *
from runtime_new import batches
import shutil
def main():
    if (ROOT/'manifest.json').exists():verify();return
    B.verify();assert B.done('completed.json') and B.done('runs/stage2/completed.json')
    path=B.ROOT/'runs/stage2/epoch4_step1536.pt';digest=C.sha(path)
    assert digest==C.read(B.ROOT/'runs/stage2/completed.json')['checkpoint_sha256']
    for key in ['seed','memory_tokens','adapter_rank','adapter_after_layers','replace_layers','epochs','updates','effective_batch','learning_rate','weight_decay','gradient_clip','lambda_kl','prefix_quotas_per_batch','prefix_max_new_tokens','supervised_positions_max','evaluation_generation_caps']:
        assert CFG[key]==F.CFG[key],key
    refs=[B.ROOT/'manifest.json',path,F.ROOT/'manifest.json']
    for split in ['train','validation','test']:
        src=B.ROOT/f'prepared/{split}.jsonl';dest=ROOT/f'prepared/{split}.jsonl';dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest);refs.extend([src,dest])
        assert rows(split)==F.rows(split)
    for row in rows('test'):baseline(row);refs.append(B.ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    schedules={str(e):C.digest([[(r['id'],k) for r,k in batch] for batch in batches(e)]) for e in range(1,5)}
    assert schedules==C.read(F.ROOT/'manifest.json')['batch_schedules']
    C.save(ROOT/'manifest.json',dict(config_sha256=C.sha(ROOT/'config.json'),b_manifest_sha256=B.identity(),b_checkpoint_sha256=digest,code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},native_prefix_sources={str(f):C.sha(f) for f in (PARENT/'native_prefix').glob('*.json')},batch_schedules=schedules,trainable_parameters=409600))
    mark('prepared/completed.json',status='completed');C.log('D PREPARED',identity())
if __name__=='__main__':main()
