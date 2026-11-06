"""M64/A4 same-data ablation on server A; shared pinned baseline implementations."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text());REFERENCE=Path(CFG['reference'])
spec=importlib.util.spec_from_file_location('canonical_m32_common',REFERENCE/'experiment_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
R=N.R;M=N.M;K=N.K;O=N.O;AR=N.AR;C=N.C;T=N.T;torch=N.torch
ORIGINAL=N.ORIGINAL;GSM=N.GSM
ARM='m64_a4_all36'

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    p=ROOT/name
    if not p.exists():return False
    v=C.read(p);assert v['manifest_sha256']==identity() and v['status']=='completed',name
    return True

def verify():
    N.verify();assert N.identity()==CFG['reference_manifest_sha256'];v=C.read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n

def baseline(row):
    path=REFERENCE/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json';v=C.read(path)
    assert v['manifest_sha256']==CFG['reference_manifest_sha256'] and v['id']==row['id'] and v['question']==row['question'] and v['gold']==row['gold']
    return v['conditions']

def cache_file(row):return ROOT/'base_cache'/(C.digest(dict(prompt=row['prompt'],content_ids=row['content_ids']))+'.pt')
def saved_base(row):
    v=torch.load(cache_file(row),map_location='cpu',weights_only=True)
    assert v['manifest_sha256']==identity() and v['prompt_sha256']==C.digest(dict(prompt=row['prompt'],content_ids=row['content_ids']))
    assert v['recolor_sha256']==C.CFG['recolor_sha256']
    k,vv=v['kv'];assert k.dtype==vv.dtype==torch.float16 and k.shape==vv.shape==(36,len(row['content_ids']),1024)
    return k,vv

def gpu_budget():
    import subprocess
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
    total=torch.cuda.get_device_properties(0).total_memory
    budget=(free-CFG['gpu_reserve_mib'])*2**20;assert budget>0
    torch.cuda.set_per_process_memory_fraction(min(.94,budget/total),0)
    return dict(initial_free_mib=free,allocator_budget_mib=budget/2**20,external_reserve_mib=CFG['gpu_reserve_mib'])
