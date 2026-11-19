"""All six historical test subsets, Native / Gemma A+B / Gemma A+B+Reader."""
from common import *
import stage_reader as R

@torch.no_grad()
def evaluate(task):
    folder=ROOT/'evaluation'/task
    if done(folder/'completed.json'):return
    assert done(ROOT/'runs/reader/completed.json')
    tok,model,base,reader=R.load_all()
    ck=torch.load(ROOT/'runs/reader/epoch4_step1024.pt',map_location='cpu',weights_only=True)
    assert ck['manifest_sha256']==identity() and ck['step']==1024
    assert ck['frozen_stage_a_sha256']==R.CFG['base_stage_a_sha256'] and ck['frozen_stage_b_sha256']==R.CFG['frozen_stage_b_sha256']
    reader.load_state_dict(ck['state']);del ck
    runtime=C.runtime(model,tok);rr=[r for r in rows('test') if r['task']==task]
    assert len(rr)==U.CFG['formal_counts'][task]
    for i,row in enumerate(rr,1):
        path=folder/'per_sample'/(U.fname(row)+'.json')
        if path.exists():assert json.loads(path.read_text())['manifest_sha256']==identity();continue
        previous=OLD_U_ROOT/'evaluation'/task/'per_sample'/(U.fname(row)+'.json')
        rec=json.loads(previous.read_text());assert rec['manifest_sha256']==OLD_U_ID
        for k in ['id','task','prompt','receiver_suffix','candidates','gold']:assert rec[k]==row[k]
        conditions=dict(qwen_native=rec['conditions']['qwen_native'])
        native,k,v=R.fixed_state(model,base,reader,row)
        for arm,enabled in [('grouped_residual',False),('grouped_residual_reader',True)]:
            with reader.mode(enabled):
                if task=='gsm8k':
                    text,count=C.R.generate(runtime,torch.tensor([row['suffix_ids']],device='cuda'),U.CFG['gsm8k_formal_max_new_tokens'],past_key_values=U.translated_cache(model,row,native,k,v),past_length=len(row['frames']['qwen']['body_ids']),temperature=0.0)
                    pred=C.S.extract_numeric_answer(text)
                    value=dict(prediction=pred,correct=int(pred==C.S.normalize_number(row['gold'])),generation=text,generated_tokens=count,reached_cap=int(count>=U.CFG['gsm8k_formal_max_new_tokens']))
                else:value=U.candidate_scores(model,tok,row,native,k,v)
            conditions[arm]=value
        mark(path,id=row['id'],task=task,prompt=row['prompt'],gold=row['gold'],conditions=conditions,native_source=str(previous))
        status(status='running',phase='evaluation',task=task,sample=i,total=len(rr))
        if i%8==0 or task=='gsm8k':C.log(f'Evaluation {task} {i}/{len(rr)}')
    records=[json.loads((folder/'per_sample'/(U.fname(r)+'.json')).read_text()) for r in rr]
    C.write_jsonl(folder/'per_sample.jsonl',records);mark(folder/'completed.json',status='completed',n=len(records))

def summarize():
    records=[];tasks={}
    for task,n in U.CFG['formal_counts'].items():
        assert done(ROOT/'evaluation'/task/'completed.json')
        rr=C.read_jsonl(ROOT/'evaluation'/task/'per_sample.jsonl');assert len(rr)==n
        metrics={a:dict(n=n,correct=sum(r['conditions'][a]['correct'] for r in rr),accuracy=sum(r['conditions'][a]['correct'] for r in rr)/n) for a in ['qwen_native','grouped_residual','grouped_residual_reader']}
        if task=='gsm8k':
            for a,m in metrics.items():m.update(mean_generated_tokens=sum(r['conditions'][a]['generated_tokens'] for r in rr)/n,cap_count=sum(r['conditions'][a]['reached_cap'] for r in rr))
        tasks[task]=dict(metrics=metrics);records.extend(rr)
    C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records)
    mark(ROOT/'summary.json',status='completed',tasks=tasks,config=CFG,caveats=CFG['reader']['caveats'])
    mark(ROOT/'completed.json',status='completed');status(status='completed',phase='all')

if __name__=='__main__':
    verify()
    if sys.argv[1]=='summarize':summarize()
    else:evaluate(sys.argv[1])
