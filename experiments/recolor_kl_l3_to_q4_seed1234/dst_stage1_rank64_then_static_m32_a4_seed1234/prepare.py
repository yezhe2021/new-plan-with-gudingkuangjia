from experiment_common import *
import shutil
def main():
    if (ROOT/'manifest.json').exists():verify();return
    F.verify();D.verify();assert F.identity()==CFG['static_reference_manifest_sha256'] and D.identity()==CFG['dst_manifest_sha256']
    cp=D.ROOT/'runs/stage1/epoch4_step1536.pt';assert C.sha(cp)==CFG['dst_stage1_sha256']
    refs=[cp,D.ROOT/'manifest.json',F.ROOT/'manifest.json']
    for split in ['train','validation','test']:
        src=D.ROOT/f'prepared/{split}.jsonl';dest=ROOT/f'prepared/{split}.jsonl';dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest)
        assert len(rows(split))==sum(CFG[split+'_counts'].values());refs.extend([src,dest])
    for row in rows('test'):
        baseline(row)
        refs.extend([D.ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json',F.ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'])
    C.save(ROOT/'manifest.json',dict(config_sha256=C.sha(ROOT/'config.json'),config=CFG,code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},trainable_parameters=9437184,frozen_stage1_sha256=CFG['dst_stage1_sha256']))
    mark('prepared/completed.json',status='completed');C.log('B PREPARED',identity())
if __name__=='__main__':main()
