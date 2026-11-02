import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
import core as C
import train as T
torch=C.torch

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    f=ROOT/name
    if not f.exists():return False
    x=C.read(f);assert x['manifest_sha256']==identity() and x['status']=='completed'
    return True
def verify():
    C.verify();m=C.read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(ROOT/n)==d,n
    for p,d in m['sources'].items():assert C.sha(p)==d,p

class Engine(C.Engine):
    @torch.no_grad()
    def native_cache(self,row,*,baseline=False):
        cache=super().native_cache(row,baseline=baseline)
        for l in range(36):
            for t,h in zip(C._cache_layer(cache,l),C._cache_layer(self.header,l)):
                t[:,:,:len(self.prefix_ids),:].copy_(h)
        return cache
