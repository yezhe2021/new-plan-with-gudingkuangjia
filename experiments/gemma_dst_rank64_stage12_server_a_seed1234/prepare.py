"""Preserve existing task splits; audit Full-Sync over complete RAW bodies."""
from experiment_common import *
from alignment import fullsync,spans,synchronized_units,build_synchronized_units
from translator import DepthStateTranslator,DSTConfig

def main():
    if (ROOT/'manifest.json').exists():verify();return
    N.verify();assert N.identity()==CFG['data_reference_manifest_sha256']
    sender=C.AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True)
    receiver=C.AutoTokenizer.from_pretrained(CFG['receiver'],local_files_only=True)
    report={};refs=[REFERENCE/'manifest.json',REFERENCE/'config.json',ROOT/'gemma_baseline_manifest.json',ROOT/'model_identity.json']
    assert C.sha(ROOT/'gemma_baseline_manifest.json')==CFG['baseline_manifest_sha256']
    assets=C.read(ROOT/'model_identity.json');assert assets['status']=='completed'
    for name,digest in assets['sha256'].items():assert C.sha(Path(CFG['sender'])/name)==digest,name
    for split in ['train','validation','test']:
        data=[];audits=[]
        original=N.rows(split)
        for i,row in enumerate(original):
            a=fullsync(row['prompt'],sender,receiver,row['content_ids'])
            if i<3:
                se=sender(row['prompt'],add_special_tokens=False,return_offsets_mapping=True);re=receiver(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
                ss,rs=spans(row['prompt'],se),spans(row['prompt'],re)
                assert synchronized_units(row['prompt'],ss,rs)==build_synchronized_units(row['prompt'],ss,rs),'Changed legacy sync units'
            data.append(dict(row,sender_ids=list(a.source_ids),sync_source_indices=list(a.source_indices),sync_audit=a.audit));audits.append(a.audit)
            if i%256==0:C.log('DST ALIGN',split,i,len(original))
        C.write_jsonl(ROOT/f'prepared/{split}.jsonl',data)
        assert len(data)==sum(CFG[split+'_counts'].values())
        report[split]=dict(n=len(data),receiver_tokens=sum(a['receiver_tokens'] for a in audits),intra_unit_lookahead_tokens=sum(a['intra_unit_lookahead_tokens'] for a in audits),max_intra_unit_lookahead_bytes=max(a['max_intra_unit_lookahead_bytes'] for a in audits),max_unit_bytes=max(a['max_unit_bytes'] for a in audits))
        refs.extend([REFERENCE/f'prepared/{split}.jsonl',ROOT/f'prepared/{split}.jsonl'])
    for row in rows('test'):
        baseline(row);refs.append(ROOT/f'baseline/{row["task"]}/per_sample/{row["id"]}.json')
    spec=DepthStateTranslator(DSTConfig(**CFG['model']),CFG['seed']).specification()
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={str(p.relative_to(ROOT)):C.sha(p) for p in list(ROOT.glob('*.py'))+list((ROOT/'reference').glob('*.py'))},sources={str(p):C.sha(p) for p in refs},model=spec,alignment=report))
    mark('prepared/completed.json',status='completed',alignment=report,model=spec)
    C.log('DST PREPARED',spec['parameters'],report)

if __name__=='__main__':main()
