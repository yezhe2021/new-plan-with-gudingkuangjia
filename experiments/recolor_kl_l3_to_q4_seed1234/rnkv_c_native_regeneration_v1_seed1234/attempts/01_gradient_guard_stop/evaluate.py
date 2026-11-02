"""Same prompts, scoring and golds; C and Native retain original positions."""
from common import *
from model import load

@torch.no_grad()
def evaluate(task):
    verify();assert done('runs/completed.json')
    if done(f'evaluation/{task}/completed.json'):return
    engine,correction,module=load('final');module.eval();engine.mode='test';records=[]
    data=[r for r in rows('test') if r['task']==task]
    for i,row in enumerate(data,1):
        if paused():sys.exit(75)
        path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
        if path.exists():
            item=C.read(path);assert item['manifest_sha256']==identity();records.append(item);continue
        base=engine.base(row);translated=engine.cache(row,base,correction);regenerated=module.construct(row,translated)
        values={w:module.score(row,cache,rn=(w=='rnkv')) for w,cache in [('C',translated),('rnkv',regenerated)]}
        native=engine.native_cache(row,baseline=True);values['native']=module.score(row,native,rn=False)
        mark(str(path.relative_to(ROOT)),id=row['id'],task=task,gold=row['gold'],conditions=values)
        records.append(C.read(path));del base,translated,regenerated,native
        engine.unchanged();assert engine.student_test_native_prefills==0
        status(status='running',phase='evaluation',task=task,sample=i,total=64)
        if i%8==0:C.log('RNKV EVAL',task,i,64)
    keys=['correct'] if task=='arc' else ['em','f1']
    metrics={w:{k:sum(r['conditions'][w][k] for r in records)/64 for k in keys} for w in ['C','rnkv','native']}
    comparisons={k:C.paired([r['conditions']['rnkv'][k]-r['conditions']['C'][k] for r in records]) for k in keys}
    correctness=keys[0]
    rescue=[r['id'] for r in records if r['conditions']['rnkv'][correctness]>r['conditions']['C'][correctness]]
    harm=[r['id'] for r in records if r['conditions']['rnkv'][correctness]<r['conditions']['C'][correctness]]
    recovery={k:((metrics['rnkv'][k]-metrics['C'][k])/(metrics['native'][k]-metrics['C'][k]) if metrics['native'][k]>metrics['C'][k] else None) for k in keys}
    mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,rnkv_minus_C=comparisons,rescued_ids=rescue,harmed_ids=harm,recovery_ratio=recovery)
    mark(f'evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))

def summarize():
    verify();tasks={t:C.read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    assert all(done(f'evaluation/{t}/completed.json') for t in tasks)
    mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# RN-KV v1: fixed epoch4/step1024','','| Method | ARC64 accuracy | Hotpot64 EM | Hotpot64 F1 |','|---|---:|---:|---:|']
    for w in ['C','rnkv','native']:
        vv=[tasks['arc']['metrics'][w]['correct']]+[tasks['hotpotqa']['metrics'][w][k] for k in ['em','f1']]
        lines.append('| '+w+' | '+' | '.join(f'{v*100:.2f}%' for v in vv)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

if __name__=='__main__':
    if sys.argv[1]=='summarize':summarize()
    else:evaluate(sys.argv[1])
