"""Independent DST Stage1/Stage2 comparison; Static Memory remains disabled."""
from experiment_common import *
from runtime import make,load_stage
from engine import Engine,BodyReader
from score import score

@torch.no_grad()
def main():
    verify();assert done('runs/stage1/completed.json') and done('runs/stage2/completed.json')
    engine=Engine();reader=BodyReader(engine);models={}
    for stage in ['stage1','stage2']:
        module=make(stage,engine);load_stage(module,stage);models[stage]=module.eval().requires_grad_(False)
    before={s:C.versions(m) for s,m in models.items()};engine.mode='test';tasks={};parity=[]
    names=['C','DST_stage1','DST_rank64','native']
    for task in CFG['test_counts']:
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            if paused():raise SystemExit(75)
            path=ROOT/f'evaluation/{task}/per_sample/{row["id"]}.json'
            if path.exists():
                v=C.read(path);assert v['manifest_sha256']==identity();records.append(v);continue
            old=baseline(row)
            if i==1:
                native=engine.native_cache(row,baseline=True);prediction=score(reader,row,native)
                for key in ['prediction','correct','legacy_correct','em','f1','generation']:
                    if key in old['native']:assert prediction[key]==old['native'][key],(task,key,'Native scoring parity')
                parity.append(dict(task=task,id=row['id'],native_prediction_parity=True));del native
            source=engine.source(row);values={k:old[k] for k in ['C','native']}
            for stage,module in models.items():
                cache=engine.translated(row,module,source,checkpointed=False);values['DST_stage1' if stage=='stage1' else 'DST_rank64']=score(reader,row,cache);del cache
            mark(str(path.relative_to(ROOT)),id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=values,sync_audit=row['sync_audit'])
            records.append(C.read(path));del source;engine.unchanged()
            assert engine.student_test_native_prefills==0 and all(before[s]==C.versions(m) for s,m in models.items())
            status(status='running',phase='evaluation',task=task,sample=i,total=64)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/64 for k in keys} for a in names}
        differences={a:{k:C.paired([r['conditions']['DST_rank64'][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in ['C','DST_stage1','native']}
        mark(f'evaluation/{task}/summary.json',status='completed',n=64,metrics=metrics,stage2_minus=differences)
        tasks[task]=C.read(ROOT/f'evaluation/{task}/summary.json')
    verify();mark('summary.json',status='completed',tasks=tasks,config=CFG,native_parity=parity)
    lines=['# Gemma to Qwen: Full-Sync DST plus frozen-base Rank64','','| Task | Metric | Gemma Recolor+C | DST Stage1 | DST+Rank64 | Native |','|---|---|---:|---:|---:|---:|']
    for task,t in tasks.items():
        for k in t['metrics']['DST_rank64']:lines.append('| '+task+' | '+k+' | '+' | '.join(f'{t["metrics"][a][k]*100:.2f}%' for a in names)+' |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')

if __name__=='__main__':main()
