import importlib.util
import json
import os
import sys
import time
from pathlib import Path
TOP=Path(__file__).resolve().parent
CFG=json.loads((TOP/'config.json').read_text());ARM=os.environ.get('RNKV_GEMMA_ARM')
assert ARM is None or ARM in CFG['arms']
ROOT=TOP/'arms'/ARM if ARM else TOP
REFERENCE=Path(CFG['data_reference']);GP=Path(CFG['gemma_parent'])
sys.path.insert(0,str(REFERENCE))
import experiment_common as N
C=N.C;R=N.R;M=N.M;K=N.K;T=N.T;O=N.O;torch=N.torch

def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
oldpath=list(sys.path)
sys.path.insert(0,str(Path(CFG['legacy_core']).parent))
L=module_at('gemma_fullsync_protocol_readonly',CFG['legacy_core'])
sys.path[:]=oldpath
G=module_at('gemma_frozen_grouped',GP/'grouped_translator.py')
S=module_at('gemma_frozen_residual',GP/'residual_translator.py')

def identity():return C.sha(TOP/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),arm=ARM,**v))
def status(**v):
    value=dict(time=time.strftime('%F %T'),arm=ARM,**v);C.save(TOP/'status.json',value)
    if ARM:C.save(ROOT/'status.json',value)
def rows(split):return C.read_jsonl(TOP/f'prepared/{split}.jsonl')
def paused():return (TOP/'controls/pause_requested.json').exists()
def done(name):
    f=ROOT/name
    if not f.exists():return False
    v=C.read(f);assert v['manifest_sha256']==identity() and v['arm']==ARM and v['status']=='completed';return True
def verify():
    N.verify();m=C.read(TOP/'manifest.json');assert C.sha(TOP/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(TOP/n)==d,n
    for n,d in m['sources'].items():assert C.sha(n)==d,n
def verify_gemma():
    assert C.sha(GP/'manifest.json')==CFG['gemma_manifest_sha256'];m=C.read(GP/'manifest.json')
    assert C.sha(GP/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(GP/n)==d,n
    for n,d in m['dependencies'].items():assert C.sha(n)==d,n
    for s,d in m['prepared_sha256'].items():assert C.digest(C.read_jsonl(GP/f'prepared/{s}.jsonl'))==d
    for role,name in [('a','best'),('b','last')]:assert C.sha(GP/f'runs/stage_{role}/{name}.pt')==CFG[f'stage_{role}_sha256']

def load_translator():
    a=torch.load(GP/'runs/stage_a/best.pt',map_location='cpu',weights_only=True);assert a['manifest_sha256']==CFG['gemma_manifest_sha256']
    base=G.GroupedKVTranslator(**C.read(GP/'config.json')['architecture']);base.load_state_dict(a['state'],strict=True);del a
    translator=S.FrozenTranslatorWithResidual(base,64)
    b=torch.load(GP/'runs/stage_b/last.pt',map_location='cpu',weights_only=True)
    assert b['manifest_sha256']==CFG['gemma_manifest_sha256'] and b['frozen_stage_a_sha256']==CFG['stage_a_sha256'] and b['state_scope']=='cross_head_residual_only'
    translator.residual.load_state_dict(b['state'],strict=True);del b
    return translator.cuda().eval().requires_grad_(False)

class Engine(R.Engine):
    def __init__(self):
        from transformers import Gemma3ForConditionalGeneration
        self.tok,self.model=C.load_model(CFG['receiver'],'cuda',dtype=torch.float16)
        self.sender_tok=C.AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True)
        self.sender=Gemma3ForConditionalGeneration.from_pretrained(CFG['sender'],dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
        self.translator=load_translator();self.prefix_ids=C.read(C.ROOT/'manifest.json')['prefix_ids']
        with torch.no_grad():self.header=C.decoder(self.model)(torch.tensor([self.prefix_ids],device='cuda'),use_cache=True).past_key_values
        self.mode='train';self.student_test_native_prefills=0
        self.snapshots=[C.versions(m) for m in [self.model,self.sender,self.translator]]

    @torch.no_grad()
    def base(self,row):
        ids=row['gemma_body_ids'];k,v,_=L.capture(self.sender,ids,len(ids),[0])
        assert torch.isfinite(k).all() and torch.isfinite(v).all(),('Gemma FP32 nonfinite',row['id'])
        ix=row['gemma_source_indices'];k=k[:,ix].half().contiguous();v=v[:,ix].half().contiguous()
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        assert k.shape==v.shape==(34,len(row['content_ids'])-1,4,256)
        pairs=[]
        for start in range(0,k.shape[1],64):
            with torch.amp.autocast('cuda',dtype=torch.float16):
                out=self.translator(k[:,start:start+64][None].cuda(),v[:,start:start+64][None].cuda())
            pairs.append(tuple(x[0] for x in out))
        key,value=tuple(torch.cat([p[j] for p in pairs],1) for j in range(2))
        assert key.shape==value.shape==(36,len(row['content_ids'])-1,8,128)
        assert torch.isfinite(key).all() and torch.isfinite(value).all()
        return key,value

    @torch.no_grad()
    def cache(self,row,base,correction=None):
        assert correction is None or correction is self.translator
        ids=self.prefix_ids+row['content_ids'][:1];h=len(self.prefix_ids);anchor=len(ids)
        # Only the public header and the first prompt token are prefetched.
        cache=C.decoder(self.model)(torch.tensor([ids],device='cuda'),use_cache=True).past_key_values
        positions=torch.arange(anchor,h+len(row['content_ids']),device='cuda');data=[]
        for l in range(36):
            ak,av=C._cache_layer(cache,l);hk,hv=C._cache_layer(self.header,l)
            ak=torch.cat([hk,ak[:,:,h:]],2);av=torch.cat([hv,av[:,:,h:]],2)
            # Stage-A+B output already lives AFTER Qwen k_norm. RoPE only.
            key=L.apply_rope(self.model,base[0][l],positions,l).transpose(0,1)[None]
            value=base[1][l].transpose(0,1)[None]
            data.append((torch.cat([ak,key],2),torch.cat([av,value],2)))
        return C.DynamicCache(ddp_cache_data=data,config=self.model.config)

    def unchanged(self):
        assert [C.versions(m) for m in [self.model,self.sender,self.translator]]==self.snapshots
        assert all(not p.requires_grad and p.grad is None for m in [self.model,self.sender,self.translator] for p in m.parameters())
