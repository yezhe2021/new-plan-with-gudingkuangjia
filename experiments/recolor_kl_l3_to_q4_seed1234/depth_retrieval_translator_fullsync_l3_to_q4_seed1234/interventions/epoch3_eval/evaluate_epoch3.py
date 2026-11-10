"""Diagnostic epoch3 evaluation, same eight cohorts/scorers; no checkpoint selection."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiment_common import *
from runtime import make
from engine import Engine,BodyReader
from score import score

@torch.no_grad()
def main():
    verify();out=HERE/'evaluation';path=HERE/'checkpoint.pt'
    digest=C.sha(path);cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==identity() and cp['stage']=='stage2' and cp['epoch']==3 and cp['step']==1152
    code_sha=C.sha(Path(__file__))
    def write(path,**v):C.save(path,dict(manifest_sha256=identity(),epoch=3,step=1152,checkpoint_sha256=digest,evaluator_sha256=code_sha,**v))
    engine=Engine();reader=BodyReader(engine);module=make();module.load_state_dict(cp['state'],strict=True);del cp
    module.eval().requires_grad_(False);versions=C.versions(module);engine.mode='test';tasks={}
    names=['C','DST_stage2','DRT_epoch3','native']
    for task,n in CFG['test_counts'].items():
        records=[]
        for i,row in enumerate([r for r in rows('test') if r['task']==task],1):
            file=out/task/'per_sample'/(row['id']+'.json')
            if file.exists():
                record=C.read(file);assert record['checkpoint_sha256']==digest and record['evaluator_sha256']==code_sha and record['manifest_sha256']==identity()
                records.append(record);continue
            old=baseline(row)
            if i==1:
                native=engine.native_cache(row,baseline=True);prediction=score(reader,row,native)
                for key in ['prediction','correct','legacy_correct','em','f1','generation']:
                    if key in old['native']:assert prediction[key]==old['native'][key],(task,key)
                del native
            prior=C.read(Path(CFG['alignment_reference'])/f'evaluation/{task}/per_sample/{row["id"]}.json')
            assert prior['manifest_sha256']==CFG['alignment_reference_manifest_sha256'] and prior['gold']==row['gold']
            source=engine.source(row);cache=engine.translated(row,module,source,checkpointed=False)
            result=score(reader,row,cache);del source,cache
            conditions=dict(C=old['C'],DST_stage2=prior['conditions']['DST_stage2'],DRT_epoch3=result,native=old['native'])
            write(file,id=row['id'],task=task,question=row['question'],gold=row['gold'],conditions=conditions)
            records.append(C.read(file));engine.unchanged();assert versions==C.versions(module) and engine.student_test_native_prefills==0
            C.save(HERE/'evaluation_status.json',dict(status='running',task=task,sample=i,total=n,time=time.strftime('%F %T')))
            if i==1 or i%8==0:C.log('DRT EPOCH3 EVAL',task,i,n)
        keys=['em','f1'] if task in ['hotpotqa','2wiki','musique'] else ['correct','legacy_correct'] if task=='gsm8k' else ['correct']
        assert len(records)==n
        metrics={a:{k:sum(r['conditions'][a][k] for r in records)/n for k in keys} for a in names}
        differences={a:{k:C.paired([r['conditions']['DRT_epoch3'][k]-r['conditions'][a][k] for r in records]) for k in keys} for a in ['C','DST_stage2','native']}
        write(out/task/'summary.json',status='completed',n=n,metrics=metrics,epoch3_minus=differences);tasks[task]=metrics
    verify();write(HERE/'completed.json',status='completed',tasks=tasks,finished_at=time.strftime('%F %T'),selection='diagnostic only; always resume epoch4')
    lines=['# DRT Stage2 Epoch3 diagnostic','','| Task | Metric | C | DST final | DRT epoch3 | Native |','|---|---|---:|---:|---:|---:|']
    for task,metrics in tasks.items():
        for key in metrics['DRT_epoch3']:lines.append('| '+task+' | '+key+' | '+' | '.join(f'{metrics[a][key]*100:.2f}%' for a in names)+' |')
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n');C.log('DRT EPOCH3 EVALUATION COMPLETED')

if __name__=='__main__':main()
