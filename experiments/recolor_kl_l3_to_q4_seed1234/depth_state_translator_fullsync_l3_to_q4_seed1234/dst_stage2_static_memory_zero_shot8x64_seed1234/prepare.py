"""Pin the final existing checkpoints and the same512 already-fixed questions."""
import shutil
from experiment_common import *

def main():
    if (ROOT/'manifest.json').exists():verify();return
    S.verify();D.verify();assert S.done('completed.json') and D.done('completed.json')
    assert S.identity()==CFG['static_manifest_sha256'] and D.identity()==CFG['dst_manifest_sha256']
    dp=DST/'runs/stage2/epoch4_step1536.pt';mp=SOURCE/'runs/epoch4_step1536.pt'
    assert C.sha(dp)==CFG['dst_checkpoint_sha256'] and C.sha(mp)==CFG['static_checkpoint_sha256']
    assert C.sha(EXPORT/'manifest.json')==CFG['export_manifest_sha256'] and C.sha(EXPORT/'export/canonical.pt')==CFG['canonical_export_sha256']
    em=C.read(EXPORT/'export/completed.json');assert em['source_checkpoint_sha256']==CFG['static_checkpoint_sha256']
    assert D.N.identity()==S.N.identity() and D.C.CFG['models']['receiver']==S.C.CFG['models']['receiver']
    for key in ['test_counts','evaluation_generation_caps']:assert D.CFG[key]==S.CFG[key]==CFG[key]
    original={r['id']:r for r in S.rows('test')};dr=D.rows('test')
    assert len(dr)==512 and set(original)=={r['id'] for r in dr}
    for row in dr:
        assert all(row[k]==v for k,v in original[row['id']].items()),row['id']
        assert len(row['sync_source_indices'])==len(row['content_ids'])
        baseline(row)
    target=ROOT/'prepared/test.jsonl';target.parent.mkdir(exist_ok=True);shutil.copyfile(DST/'prepared/test.jsonl',target)
    refs=[DST/'manifest.json',SOURCE/'manifest.json',dp,mp,DST/'prepared/test.jsonl',SOURCE/'prepared/test.jsonl',target,EXPORT/'manifest.json',EXPORT/'export/completed.json',EXPORT/'export/canonical.pt']
    refs+=list(DST.glob('*.py'))+list(SOURCE.glob('*.py'))
    refs+=[C.ROOT/'metrics.py',S.ORIGINAL/'cross_dataset5x64_seed1234/experiment.py',S.ORIGINAL/'cross_dataset5x64_seed1234/qa_metrics.py',S.GSM/'numeric.py']
    refs+=[dst_record(r) for r in dr]+[static_record(r) for r in dr]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},n=512,optimizer_updates=0,trainable_parameters=0,base_checkpoint='Stage2 epoch4/step1536',memory_checkpoint='original Static epoch4/step1536',input_texts_and_scoring_rows_identical=True))
    mark('prepared/completed.json',status='completed',n=512,optimizer_updates=0,checkpoint_selection='predeclared final checkpoints')
    C.log('DST PLUS STATIC PREPARED',identity())

if __name__=='__main__':main()
