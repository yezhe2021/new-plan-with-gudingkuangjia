"""Inference-only all-layer memory access using the exact trained Keep-both."""
from contextlib import contextmanager
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());STRUCT=Path(CFG['structural_experiment'])
sys.path.insert(0,str(STRUCT))
import ablation_common as A
import ablation_runtime as AR
import ablation_evaluate as E
C=A.C;R=A.R;K=A.K;torch=A.torch
ARM=CFG['arm'];BASELINES=['C','rnkv','keep_both','native'];ARMS=BASELINES[:3]+[ARM,'native']

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def pause():
    if (ROOT/'controls/pause_requested.json').exists():status(status='paused');raise SystemExit(75)

def baseline_path(row):return STRUCT/f'arms/keep_both/evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    value=C.read(baseline_path(row))
    assert value['manifest_sha256']==CFG['structural_manifest_sha256'] and value['id']==row['id'] and value['gold']==row['gold']
    assert value['question']==row['question'] and set(value['conditions'])==set(BASELINES)
    return value['conditions']

def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    A.verify();assert A.identity()==CFG['structural_manifest_sha256']
    cp=STRUCT/'arms/keep_both/runs/epoch4_step1024.pt';assert C.sha(cp)==CFG['keep_both_checkpoint_sha256']
    assert C.read(STRUCT/'arms/keep_both/evaluation/completed.json')['status']=='completed'
    source=STRUCT/'prepared/test.jsonl';data=C.read_jsonl(source)
    assert len(data)==448 and all(sum(r['task']==t for r in data)==n for t,n in CFG['test_counts'].items())
    for r in data:baseline(r)
    dest=ROOT/'prepared/test.jsonl';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    refs=[STRUCT/'manifest.json',cp,source,dest]+[baseline_path(r) for r in data]+list(STRUCT.glob('*.py'))
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},
        training=False,test_cohorts_byte_identical=True,new_prediction_configurations=448))
    mark('prepared/completed.json',status='completed',n=448,test_counts=CFG['test_counts']);C.log('ALL-LAYER KEEP-BOTH PREPARED',448)

def verify():
    A.verify();value=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==value['config_sha256']
    for n,d in value['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in value['sources'].items():assert C.sha(n)==d,n

@contextmanager
def all_layers(module):
    old=module.routing.replace
    assert old==set(range(19,31)) and module.routing.state is None
    module.routing.replace=set(CFG['answer_memory_visible_layers'])
    try:yield
    finally:
        assert module.routing.state is None
        module.routing.replace=old

def unchanged(engine,correction,module,snapshots):
    engine.unchanged();assert engine.student_test_native_prefills==0
    assert [C.versions(correction),C.versions(module)]==snapshots
    assert all(not p.requires_grad and p.grad is None for m in [correction,module] for p in m.parameters())
    assert module.routing.replace==set(range(19,31)) and module.routing.state is None

def compare(a,b):
    assert a['prediction']==b['prediction'],('Existing baseline prediction changed',a,b)
    for key in ['generation','generated_tokens','correct','em','f1','reached_cap']:
        if key in b:assert a[key]==b[key],(key,a[key],b[key])
    error=max([abs(x-y) for x,y in zip(a.get('scores',[]),b.get('scores',[]))] or [0.])
    assert error<.001,error
    return error

@torch.no_grad()
def smoke(engine,correction,module,snapshots):
    checks=[]
    for task in CFG['test_counts']:
        pause();row=next(r for r in rows() if r['task']==task)
        base=engine.base(row);translated=engine.cache(row,base,correction);memory=module.construct(row,translated)
        old=E.score(module,row,memory);error=compare(old,baseline(row)['keep_both'])
        L=translated.get_seq_length();h=len(engine.prefix_ids);q=torch.tensor([L+32,L+33],device='cuda')
        assert memory.get_seq_length()==L+32
        with all_layers(module):
            # Changing answer visibility cannot alter the memory construction.
            rebuilt=module.construct(row,translated)
            assert all(torch.equal(x,y) for x,y in zip(K.flat(memory),K.flat(rebuilt)));del rebuilt
            for layer in range(1,37):
                allowed=module.routing.allowed('answer',layer,h,L,32,q,L+34)
                assert allowed[:,:L+32].all() and allowed[0,L+32] and not allowed[0,L+33] and allowed[1,L+33]
            value=E.score(module,row,memory)
        unchanged(engine,correction,module,snapshots)
        mark(f'smoke/per_sample/{row["id"]}.json',status='completed',id=row['id'],task=task,conditions={**baseline(row),ARM:value},
            baseline_score_max_error=error,construction_cache_bitwise_identical=True,all36_masks_causal_body_and_memory_visible=True)
        checks.append(dict(task=task,id=row['id'],baseline_score_error=error));del base,translated,memory
        C.log('ALL-LAYER KEEP-BOTH SMOKE PASSED',task)
    mark('smoke/completed.json',status='completed',checks=checks,all_parameters_frozen=True,student_native_body_prefills=0,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)

@torch.no_grad()
def evaluate(engine,correction,module,snapshots,task):
    data=[r for r in rows() if r['task']==task];records=[]
    for i,row in enumerate(data,1):
        pause();path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
        if path.exists():
            value=C.read(path);assert value['manifest_sha256']==identity();records.append(value);continue
        status(status='running',phase='evaluation',task=task,sample=i,total=64)
        reuse=ROOT/f'smoke/per_sample/{row["id"]}.json'
        if reuse.exists():
            v=C.read(reuse);assert v['manifest_sha256']==identity();conditions=v['conditions']
        else:
            base=engine.base(row);translated=engine.cache(row,base,correction);memory=module.construct(row,translated)
            with all_layers(module):result=E.score(module,row,memory)
            conditions={**baseline(row),ARM:result};del base,translated,memory
        mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=conditions,
            reused_smoke=str(reuse) if reuse.exists() else None)
        records.append(C.read(path));unchanged(engine,correction,module,snapshots)
        if i==1 or i%8==0:C.log('ALL-LAYER KEEP-BOTH EVAL',task,i,64)
    keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct']
    metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in ARMS}
    differences={b:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][b][k] for r in records]) for k in keys} for b in ['keep_both','C','rnkv']}
    changes={b:{name:[r['id'] for r in records if (r['conditions'][ARM][keys[0]]-r['conditions'][b][keys[0]])*sign>0]
        for name,sign in [('rescued_ids',1),('harmed_ids',-1)]} for b in ['keep_both','C']}
    mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,all_layers_minus=differences,rescued_harmed=changes,
        cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in ARMS})
    mark(f'evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))

def run():
    verify();status(status='running',phase='loading_frozen_keep_both');engine,correction,module=AR.load('keep_both',final=True)
    engine.mode='test';module.eval().requires_grad_(False);snapshots=[C.versions(correction),C.versions(module)]
    smoke(engine,correction,module,snapshots)
    for task in CFG['test_counts']:evaluate(engine,correction,module,snapshots,task)
    unchanged(engine,correction,module,snapshots);verify()
    tasks={t:C.read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    mark('summary.json',status='completed',tasks=tasks,training=False,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    lines=['# All-layer Keep-both inference mask intervention','','Same existing checkpoint; no training; seven identical64-question cohorts.','',
        '| Task | Metric | C | RN-KV | Keep-both19-30 | Keep-both1-36 | Native |','|---|---|---:|---:|---:|---:|---:|']
    for task,s in tasks.items():
        for k in s['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{s["metrics"][a][k]*100:.2f}%' for a in ARMS)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed');C.log('ALL-LAYER KEEP-BOTH COMPLETE')

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError(sys.argv[1])
