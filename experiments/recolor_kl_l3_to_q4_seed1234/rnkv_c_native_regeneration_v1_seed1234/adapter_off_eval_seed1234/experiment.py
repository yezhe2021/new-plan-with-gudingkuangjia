"""Post-training bypass of four memory adapters; no parent code modifications."""
from contextlib import contextmanager
from types import MethodType
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
import common as R
import model as M
import cache as K
C=R.C;torch=R.torch

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def paused():return (ROOT/'controls/pause_requested.json').exists()
def parent_result(row):
    value=C.read(PARENT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    assert value['manifest_sha256']==R.identity() and value['id']==row['id'] and value['gold']==row['gold']
    return value

def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    R.verify();assert R.identity()==CFG['parent_manifest_sha256']
    assert C.read(PARENT/'completed.json')['status']=='completed'
    assert C.sha(PARENT/'runs/epoch4_step1024.pt')==CFG['checkpoint_sha256']
    rr=R.rows('test');assert len(rr)==128
    assert all(sum(r['task']==t for r in rr)==n for t,n in CFG['test_counts'].items())
    C.write_jsonl(ROOT/'prepared/test.jsonl',rr)
    refs=[PARENT/n for n in ['manifest.json','completed.json','config.json','prepared/test.jsonl','runs/epoch4_step1024.pt']]
    refs += list(PARENT.glob('*.py'))+[ROOT/'prepared/test.jsonl']
    refs += [PARENT/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json' for r in rr]
    for row in rr:parent_result(row)
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},training=False,
        test_ids={t:[r['id'] for r in rr if r['task']==t] for t in CFG['test_counts']},new_predictions=128))
    mark('prepared/completed.json',status='completed',n=128,training=False)

def verify():
    R.verify();m=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in m['sources'].items():assert C.sha(n)==d,n

class Bypass:
    def __init__(self,module):self.module=module;self.active=False;self.counts={}
    @contextmanager
    def disabled(self):
        assert not self.active;self.active=True;self.counts={l:0 for l in CFG['disabled_adapters_after_layers']}
        memory=self.module.memory;original=memory.adapt
        def identity_adapter(memory_,layer,hidden):
            assert memory_ is memory and layer in self.counts
            self.counts[layer]+=1
            return hidden
        memory.adapt=MethodType(identity_adapter,memory)
        try:yield
        finally:memory.adapt=original;self.active=False
    def check(self):assert self.counts=={l:1 for l in CFG['disabled_adapters_after_layers']},self.counts

def assert_frozen(engine,correction,module,snapshots,prompt):
    engine.unchanged();assert C.versions(correction)==snapshots[0] and C.versions(module)==snapshots[1]
    assert torch.equal(module.memory.prompt,prompt)
    assert all(not p.requires_grad and p.grad is None for m in [correction,module] for p in m.parameters())

@torch.no_grad()
def smoke(engine,correction,module,bypass,snapshots,prompt):
    selected=[next(r for r in rows() if r['task']==t) for t in CFG['test_counts']]
    selected += [max([r for r in rows() if r['task']=='hotpotqa'],key=lambda r:len(r['content_ids']))]
    checks=[]
    for row in {r['id']:r for r in selected}.values():
        base=engine.base(row);translated=engine.cache(row,base,correction)
        enabled=module.construct(row,translated);on=module.score(row,enabled,rn=True)
        old=parent_result(row)['conditions']['rnkv']
        assert on['prediction']==old['prediction'],('Enabled-model prediction parity',row['id'])
        if row['task']=='hotpotqa':
            assert on['generation']==old['generation'] and on['generated_tokens']==old['generated_tokens']
        score_error=max([abs(a-b) for a,b in zip(on.get('scores',[]),old.get('scores',[]))] or [0.])
        assert score_error<.001,('Enabled score parity',score_error)
        with bypass.disabled():
            off=module.construct(row,translated);result=module.score(row,off,rn=True)
        bypass.check();L=translated.get_seq_length();assert off.get_seq_length()==enabled.get_seq_length()==L+32
        for i,(onpair,offpair,originalpair) in enumerate(zip(K.pairs(enabled),K.pairs(off),K.pairs(translated))):
            for a,b,t in zip(onpair,offpair,originalpair):
                assert torch.equal(a[:,:,:L,:],t) and torch.equal(b[:,:,:L,:],t)
                if i<6:assert torch.equal(a[:,:,L:,:],b[:,:,L:,:]),'A post-block6 adapter changed earlier/native layer6 KV'
        changed=sum(not torch.equal(a[:,:,L:,:],b[:,:,L:,:]) for l in range(18,30) for a,b in zip(K.pairs(enabled)[l],K.pairs(off)[l]))
        assert changed>0,'Adapter bypass did not alter any selected memory state'
        assert_frozen(engine,correction,module,snapshots,prompt)
        checks.append(dict(id=row['id'],task=row['task'],body_tokens=len(row['content_ids']),enabled_score_error=score_error,
            bypass_counts=dict(bypass.counts),changed_selected_kv_tensors=changed,early_memory_exact=True,prompt_and_weights_unchanged=True))
        del base,translated,enabled,off
        C.log('ADAPTER-OFF SMOKE PASSED',row['id'])
    mark('smoke/completed.json',status='completed',checks=checks,student_native_body_prefills=engine.student_test_native_prefills)

@torch.no_grad()
def evaluate(engine,correction,module,bypass,snapshots,prompt,task):
    records=[];data=[r for r in rows() if r['task']==task]
    for i,row in enumerate(data,1):
        if paused():status(status='paused',task=task,sample=i);sys.exit(75)
        path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
        if path.exists():
            value=C.read(path);assert value['manifest_sha256']==identity();records.append(value);continue
        base=engine.base(row);translated=engine.cache(row,base,correction)
        with bypass.disabled():
            memory=module.construct(row,translated);off=module.score(row,memory,rn=True)
        bypass.check()
        values=dict(parent_result(row)['conditions'],adapter_off=off)
        mark(str(path.relative_to(ROOT)),id=row['id'],task=task,gold=row['gold'],conditions=values,bypass_counts=dict(bypass.counts),parent_manifest_sha256=R.identity())
        records.append(C.read(path));del base,translated,memory
        assert_frozen(engine,correction,module,snapshots,prompt);assert engine.student_test_native_prefills==0
        status(status='running',phase='evaluation',task=task,sample=i,total=64)
        if i==1 or i%8==0:C.log('ADAPTER-OFF EVAL',task,i,64)
    keys=['correct'] if task=='arc' else ['em','f1'];names=['C','rnkv','adapter_off','native']
    metrics={w:{k:sum(r['conditions'][w][k] for r in records)/64 for k in keys} for w in names}
    comparisons={b:{k:C.paired([r['conditions']['adapter_off'][k]-r['conditions'][b][k] for r in records]) for k in keys} for b in ['rnkv','C']}
    rescue_harm={b:{tag:[r['id'] for r in records if (r['conditions']['adapter_off'][keys[0]]-r['conditions'][b][keys[0]])*sign>0]
                    for tag,sign in [('rescued_ids',1),('harmed_ids',-1)]} for b in ['rnkv','C']}
    mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,adapter_off_minus=comparisons,rescued_harmed=rescue_harm)
    mark(f'evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))

def summarize():
    tasks={t:C.read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    assert all(x['status']=='completed' and x['manifest_sha256']==identity() for x in tasks.values())
    mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# RN-KV adapter-off inference ablation','','| Method | ARC64 accuracy | Hotpot64 EM | Hotpot64 F1 |','|---|---:|---:|---:|']
    for w in ['C','rnkv','adapter_off','native']:
        vv=[tasks['arc']['metrics'][w]['correct']]+[tasks['hotpotqa']['metrics'][w][k] for k in ['em','f1']]
        lines.append('| '+w+' | '+' | '.join(f'{v*100:.2f}%' for v in vv)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    verify();mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

def run():
    verify();engine,correction,module=M.load('final');engine.mode='test';module.eval().requires_grad_(False)
    snapshots=(C.versions(correction),C.versions(module));prompt=module.memory.prompt.detach().clone();bypass=Bypass(module)
    smoke(engine,correction,module,bypass,snapshots,prompt)
    for task in CFG['test_counts']:evaluate(engine,correction,module,bypass,snapshots,prompt,task)
    summarize()

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError(sys.argv[1])
