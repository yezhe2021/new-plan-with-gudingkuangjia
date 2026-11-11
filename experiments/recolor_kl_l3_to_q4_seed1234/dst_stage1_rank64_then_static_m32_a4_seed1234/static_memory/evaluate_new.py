"""Fixed final DST-specific Static Memory; same eight64-question cohorts."""
from experiment_common import *
from runtime_new import load
from score import score

@torch.no_grad()
def main():
    verify();assert done('runs/completed.json') and done('parity/completed.json')
    if done('completed.json'):return
    engine,correction,module=load(final=True);module.eval().requires_grad_(False);engine.mode='test'
    before=[C.versions(correction),C.versions(module)];names=['DST_stage1',BASE,'DST_stage2',OLD,ARM,'C_static','native'];tasks={}
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():status(status='paused',phase='evaluation',task=task,sample=i);raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity();records.append(v);continue
            base=engine.base(row);body=engine.cache(row,base,correction);cache=module.construct(row,body)
            result=score(module,row,cache,rn=True)
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions={**baseline(row),ARM:result},b_checkpoint_sha256=C.read(ROOT/'manifest.json')['b_checkpoint_sha256'],memory_checkpoint_sha256=C.read(ROOT/'runs/completed.json')['checkpoint_sha256'])
            records.append(C.read(path));del base,body,cache
            engine.unchanged();assert before==[C.versions(correction),C.versions(module)] and engine.student_test_native_prefills==0
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
            if i==1 or i%8==0:C.log('DST-SPECIFIC STATIC EVAL',task,i,64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        diffs={a:{k:C.paired([r['conditions'][ARM][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in [BASE,OLD,'C_static','native']}
        changes={a:{k:{label:[r['id'] for r in records if (r['conditions'][ARM][k]-r['conditions'][a][k])*sign>0] for label,sign in [('rescued_ids',1),('harmed_ids',-1)]} for k in keys} for a in [BASE,OLD]}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,retrained_minus=diffs,rescued_harmed=changes,cap_counts={a:sum(r['conditions'][a].get('reached_cap',0) for r in records) for a in names})
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json')
    verify();engine.unchanged();assert before==[C.versions(correction),C.versions(module)]
    mark('summary.json',status='completed',tasks=tasks,config=CFG)
    lines=['# D: frozen DST Stage1 + Rank64 + fresh Static M32+A4 all36','','| Task | Metric | A Stage1 | B Rank64 | E Full Stage2 | F Full+Static | D Rank64+Static | C+Static | Native |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,t in tasks.items():
        for k in t['metrics'][ARM]:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{t["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed');C.log('DST-SPECIFIC STATIC ALL COMPLETED')

if __name__=='__main__':main()
