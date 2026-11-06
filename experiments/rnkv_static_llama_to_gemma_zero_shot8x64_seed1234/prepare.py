"""Pin existing checkpoints and byte-identical evaluation rows; no training."""
import shutil
from experiment_common import *

def main():
    if (ROOT/'manifest.json').exists():verify();return
    S.verify();G.verify();assert S.done('completed.json') and G.done('completed.json')
    assert S.identity()==CFG['static_manifest_sha256'] and G.identity()==CFG['gemma_manifest_sha256']
    static_checkpoint()
    assert C.sha(GEMMA/'runs/epoch4_step1536.pt')==CFG['gemma_memory_checkpoint_sha256']
    assert G.map_sha()==CFG['gemma_recolor_sha256'] and G.CFG['c_checkpoint_sha256']==CFG['gemma_c_checkpoint_sha256']
    assert Path(S.C.CFG['models']['receiver']).resolve()==Path(G.PG.CFG['receiver']).resolve()
    assert C.read(S.C.ROOT/'manifest.json')['prefix_ids']==C.read(G.PG.ROOT/'manifest.json')['prefix_ids']
    for key in ['memory_tokens','adapter_rank','adapter_after_layers','replace_layers','test_counts','evaluation_generation_caps']:
        assert S.CFG[key]==G.CFG[key],key
    a=SOURCE/'prepared/test.jsonl';b=GEMMA/'prepared/test.jsonl';assert C.sha(a)==C.sha(b),'Identical test row bytes required'
    out=ROOT/'prepared/test.jsonl';out.parent.mkdir(exist_ok=True);shutil.copyfile(a,out)
    assert len(rows())==512 and {t:sum(r['task']==t for r in rows()) for t in CFG['test_counts']}==CFG['test_counts']
    for row in rows():baseline(row)
    refs=[SOURCE/'manifest.json',SOURCE/'config.json',SOURCE/'runs/epoch4_step1536.pt',GEMMA/'manifest.json',GEMMA/'config.json',GEMMA/'runs/epoch4_step1536.pt',a,b,out]
    refs+=list(SOURCE.glob('*.py'))+list(GEMMA.glob('*.py'))+[baseline_path(r) for r in rows()]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},test_count=512,optimizer_updates=0,trainable_parameters=0,transferred_parameter_count=409600,test_rows_byte_identical=True))
    mark('prepared/completed.json',status='completed',n=512,optimizer_updates=0,receiver_and_header_identical=True)
    C.log('STATIC SENDER TRANSFER PREPARED',identity())

if __name__=='__main__':main()
