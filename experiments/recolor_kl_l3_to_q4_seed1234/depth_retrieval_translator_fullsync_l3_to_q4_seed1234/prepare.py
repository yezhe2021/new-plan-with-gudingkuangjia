"""Reuse byte-identical DST alignment/splits, not its weights or training state."""
from experiment_common import *
from translator import DepthRetrievalTranslator,DRTConfig
import shutil

def main():
    if (ROOT/'manifest.json').exists():verify();return
    N.verify();assert N.identity()==CFG['data_reference_manifest_sha256']
    parent=Path(CFG['alignment_reference']);assert C.sha(parent/'manifest.json')==CFG['alignment_reference_manifest_sha256']
    pm=C.read(parent/'manifest.json');refs=[parent/'manifest.json',REFERENCE/'manifest.json',REFERENCE/'config.json']
    for split in ['train','validation','test']:
        src=parent/f'prepared/{split}.jsonl';assert C.sha(src)==pm['sources'][str(src)]
        dest=ROOT/f'prepared/{split}.jsonl';dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest)
        assert len(rows(split))==sum(CFG[split+'_counts'].values());refs.extend([src,dest,REFERENCE/f'prepared/{split}.jsonl'])
    for row in rows('test'):
        baseline(row);refs.append(REFERENCE/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
        f=parent/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json';refs.append(f)
        v=C.read(f);assert v['manifest_sha256']==CFG['alignment_reference_manifest_sha256'] and v['gold']==row['gold']
    spec=DepthRetrievalTranslator(DRTConfig(**CFG['model']),CFG['seed']).specification()
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={str(p.relative_to(ROOT)):C.sha(p) for p in list(ROOT.glob('*.py'))+list((ROOT/'reference').glob('*.py'))},sources={str(p):C.sha(p) for p in refs},model=spec,alignment=pm['alignment']))
    mark('prepared/completed.json',status='completed',model=spec,alignment=pm['alignment']);C.log('DRT PREPARED',spec)

if __name__=='__main__':main()
