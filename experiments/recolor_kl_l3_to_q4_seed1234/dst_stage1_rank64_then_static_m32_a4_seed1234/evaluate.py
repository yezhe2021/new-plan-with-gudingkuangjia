from experiment_common import *
from runtime import load_stage
from engine import Engine,BodyReader
from score import score
@torch.no_grad()
def main():
    verify();assert done('runs/stage2/completed.json')
    if done('completed.json'):return
    engine=Engine();module=engine.correction;load_stage(module,'stage2');module.eval().requires_grad_(False)
    reader=BodyReader(engine);engine.mode='test';versions=C.versions(module)
    names=['DST_stage1','B_rank64','DST_stage2','F_static','C','C_static','native'];tasks={}
    for task,n in CFG['test_counts'].items():
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                r=C.read(path);assert r['manifest_sha256']==identity();records.append(r);continue
            old=baseline(row);source=engine.source(row);cache=engine.translated(row,module,source,checkpointed=False)
            result=score(reader,row,cache)
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions={**old,'B_rank64':result},checkpoint_sha256=C.read(ROOT/'runs/stage2/completed.json')['checkpoint_sha256'])
            records.append(C.read(path));del source,cache;engine.unchanged()
            assert C.versions(module)==versions and engine.student_test_native_prefills==0
            status(status='running',phase='B_evaluation',task=task,sample=i,total=n)
            if i==1 or i%8==0:C.log('B EVAL',task,i,n)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        assert len(records)==n
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/n for k in keys} for a in names}
        differences={a:{k:C.paired([r['conditions']['B_rank64'][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in names if a!='B_rank64'}
        mark(f'evaluation/{task}/summary.json',status='completed',n=n,metrics=metrics,B_minus=differences);tasks[task]=metrics
    verify();mark('summary.json',status='completed',tasks=tasks)
    lines=['# B: DST Stage1 + Rank64 functional calibration','','| Task | Metric | A Stage1 | B Rank64 | E Full Stage2 | F Full+Static | C | C+Static | Native |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,metrics in tasks.items():
        for key in metrics['B_rank64']:lines.append('| '+task+' | '+key+' | '+' | '.join(f'{metrics[a][key]*100:.2f}%' for a in names)+' |')
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='B_completed_D_next')
if __name__=='__main__':main()
