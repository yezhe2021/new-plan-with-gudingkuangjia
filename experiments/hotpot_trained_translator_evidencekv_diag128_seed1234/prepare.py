"""Freeze a prediction-independent, paired 128-question checkpoint diagnostic."""
import json
import time
from pathlib import Path
import backend as B
from backend import ROOT, CFG, E, read, save, sha


def prepare():
    if (ROOT/'manifest.json').exists():
        B.verify()
        return
    E.verify()
    parent=Path(CFG['parent']);cp_parent=Path(CFG['checkpoint_parent'])
    pm=read(parent/'manifest.json');pcm=sha(cp_parent/'manifest.json')
    assert read(parent/'completed.json')['manifest_sha256']==sha(parent/'manifest.json')
    assert read(cp_parent/'completed.json')['manifest_sha256']==pcm
    source=parent/'prepared/questions.jsonl'
    assert sha(source)==pm['prepared_sha256']
    records=[json.loads(s) for s in source.read_text().splitlines()]
    selected=[r for r in records if r['split']=='test'][:CFG['n']]
    assert len(selected)==CFG['n']
    ids={r['id'] for r in selected}
    assert len(ids)==CFG['n']
    assert all(r['donor_id'] in ids and r['donor_id']!=r['id'] for r in selected)
    excluded=[];exclusion_hashes={}
    for split in ['train','validation']:
        p=cp_parent/'prepared'/f'{split}.jsonl'
        exclusion_hashes[str(p)]=sha(p)
        excluded.extend(json.loads(s) for s in p.read_text().splitlines())
    excluded_ids={r['source_id'] for r in excluded}
    excluded_questions={B.norm(r['question']) for r in excluded}
    assert not {r['source_id'] for r in selected}&excluded_ids
    assert not {B.norm(r['question']) for r in selected}&excluded_questions
    assert len({t for r in selected for t in r['titles']})==2*len(selected)
    out=ROOT/'prepared/questions.jsonl';out.parent.mkdir(exist_ok=True)
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    checkpoints={}
    for stage,rel in [('stage_a','runs/stage_a/best.pt'),('stage_b','runs/stage_b/last.pt')]:
        p=cp_parent/rel;digest=sha(p);done=read(cp_parent/'runs'/stage/'completed.json')
        assert done['selected_sha256']==digest and done['manifest_sha256']==pcm
        checkpoints[stage]=dict(path=str(p),sha256=digest)
    # Save the historical paired reference, keeping original result identities.
    prior_dir=ROOT/'reference_results';prior_dir.mkdir(exist_ok=True)
    hashes={}
    for row in selected:
        p=parent/'evaluation/test'/(row['id']+'.json');old=read(p)
        assert old['complete'] and old['manifest_sha256']==sha(parent/'manifest.json')
        assert old['donor_id']==row['donor_id'] and old['answer']==row['answer']
        save(prior_dir/p.name,dict(id=row['id'],manifest_sha256=old['manifest_sha256'],baselines=old['baselines'],
             conditions={c:dict(accept=old['conditions'][c]['accept']) for c in CFG['conditions']}))
        hashes[p.name]=sha(prior_dir/p.name)
    anchors={tuple(r['frames']['qwen']['body_ids'][:r['frames']['qwen']['native_prefix_length']]) for r in selected}
    assert len(anchors)==1
    save(ROOT/'manifest.json',dict(config_sha256=sha(ROOT/'config.json'),prepared_sha256=sha(out),
        reference_manifest_sha256=E.identity(),parent_manifest_sha256=sha(parent/'manifest.json'),
        checkpoint_parent_manifest_sha256=pcm,checkpoints=checkpoints,anchor_ids=list(next(iter(anchors))),
        exclusion_file_sha256=exclusion_hashes,train_validation_question_overlap=0,
        selection_uses_predictions=False,prior_results_sha256=hashes,
        code={p.name:sha(p) for p in ROOT.glob('*.py')},frozen_at=time.strftime('%F %T'),
        scope=CFG['scope'],no_reader=True,n=len(selected)))
    print('PREPARED',len(selected),'disjoint from',len(excluded),'checkpoint train/validation rows',flush=True)


if __name__=='__main__':prepare()
