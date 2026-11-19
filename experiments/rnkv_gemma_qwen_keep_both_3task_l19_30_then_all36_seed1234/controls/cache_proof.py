"""Validate cache miss and hit on real frozen Gemma KV before resuming training."""
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;TOP=HERE.parent
sys.path.insert(0,str(TOP))
import gemma_common as H
from latent_cache import install
m=json.loads((HERE/'cache_manifest.json').read_text());assert H.identity()==m['parent_manifest_sha256']
for n,d in m['code'].items():assert H.C.sha(HERE/n)==d
H.verify();install(H,m['contract'],'cache_proof')
engine=H.Engine();checks=[]
for task in H.CFG['train_counts']:
    row=max([r for r in H.rows('train') if r['task']==task],key=lambda r:len(r['gemma_body_ids'])+len(r['gold_tokens']))
    a=engine.base(row);b=engine.base(row)
    assert all(H.torch.equal(x,y) for x,y in zip(a,b))
    engine.unchanged();checks.append(dict(task=task,id=row['id'],miss_and_hit_kv_bitwise_identical=True));del a,b
H.C.save(HERE/'cache_proof.json',dict(status='completed',parent_manifest_sha256=H.identity(),checks=checks))
print('LOSSLESS LATENT CACHE PROOF PASSED',checks,flush=True)
