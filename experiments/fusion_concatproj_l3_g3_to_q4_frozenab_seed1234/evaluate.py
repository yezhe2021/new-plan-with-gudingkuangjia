"""Fixed-final-checkpoint complete evaluation against the frozen equal-average baseline."""
from runtime import *

@torch.inference_mode()
def run():
    em=verify()
    if done('completed.json'):return
    assert done('runs/projection/completed.json')
    p=ROOT/'runs/projection/epoch4_step1024.pt';r=torch.load(p,map_location='cpu',weights_only=True)
    assert r['manifest_sha256']==identity() and r['step']==1024 and r['state_scope']=='concat_projection_only' and r['frozen_frontends']==E.identity()
    module=fresh();module.load_state_dict(r['state'],strict=True);module.eval().requires_grad_(False);del r
    lfront=E.load_front('llama',em);gfront=E.load_front('gemma',em);tok,model=C.model('qwen');rt=C.runtime(model,tok)
    baselines={r['id']:r for r in C.read_jsonl(EP/'evaluation/per_sample.jsonl')}
    for task in ['openbookqa','arc_challenge','hellaswag','mmlu_pro','sciq','gsm8k']:
        if done('evaluation/'+task+'/completed.json'):continue
        items=[x for x in rows('test') if x['gemma_row']['task']==task]
        assert len(items)==CFG['formal_counts'][task]
        for i,item in enumerate(items,1):
            row=item['gemma_row'];path=ROOT/'evaluation'/task/'per_sample'/(U.fname(row)+'.json')
            if path.exists():assert json.loads(path.read_text())['manifest_sha256']==identity();continue
            status(status='running',phase='evaluation',task=task,sample=i,total=len(items))
            ls,gs=E.source_pair(item);lk,lv=E.map_source(lfront,ls);gk,gv=E.map_source(gfront,gs)
            k,v=project(module,[lk,lv,gk,gv]);native=U.native_body(model,row)
            if task=='gsm8k':
                text,count=C.R.generate(rt,torch.tensor([row['suffix_ids']],device='cuda'),4096,past_key_values=U.translated_cache(model,row,native,k,v),past_length=len(row['frames']['qwen']['body_ids']),temperature=0.0)
                pred=C.S.extract_numeric_answer(text);result=dict(prediction=pred,correct=int(pred==C.S.normalize_number(row['gold'])),generation=text,generated_tokens=count,reached_cap=int(count>=4096))
            else:result=U.candidate_scores(model,tok,row,native,k,v)
            baseline=baselines[row['id']]
            for key in ['id','task','prompt','gold']:assert baseline[key]==row[key]
            conditions=dict(baseline['conditions'],concat_projection=result)
            C.save(path,dict(id=row['id'],task=task,prompt=row['prompt'],gold=row['gold'],conditions=conditions,manifest_sha256=identity(),projection_checkpoint_sha256=C.sha(p) if i==1 else json.loads((ROOT/'runs/projection/completed.json').read_text())['checkpoint_sha256']))
            if i%8==0 or task=='gsm8k':C.log(f'Concat projection evaluation {task} {i}/{len(items)}')
        records=[json.loads((ROOT/'evaluation'/task/'per_sample'/(U.fname(x['gemma_row'])+'.json')).read_text()) for x in items]
        C.write_jsonl(ROOT/'evaluation'/task/'per_sample.jsonl',records);mark('evaluation/'+task+'/completed.json',status='completed',n=len(records))
    summary()

def summary():
    tasks={};records=[]
    for task,n in CFG['formal_counts'].items():
        assert done('evaluation/'+task+'/completed.json');rr=C.read_jsonl(ROOT/'evaluation'/task/'per_sample.jsonl');assert len(rr)==n
        metrics={a:dict(n=n,correct=sum(r['conditions'][a]['correct'] for r in rr),accuracy=sum(r['conditions'][a]['correct'] for r in rr)/n) for a in ['qwen_native','gemma_native','llama_ab','gemma_ab','equal50_fusion','concat_projection']}
        if task=='gsm8k':
            for a,m in metrics.items():m.update(mean_generated_tokens=sum(r['conditions'][a]['generated_tokens'] for r in rr)/n,cap_count=sum(r['conditions'][a]['reached_cap'] for r in rr))
        paired={}
        for a in ['llama_ab','gemma_ab','equal50_fusion']:
            pairs=[(r['conditions']['concat_projection']['correct'],r['conditions'][a]['correct']) for r in rr]
            paired[a]=dict(both_correct=sum(x and y for x,y in pairs),projection_only=sum(x and not y for x,y in pairs),other_only=sum(not x and y for x,y in pairs),both_wrong=sum(not x and not y for x,y in pairs))
        tasks[task]=dict(metrics=metrics,paired=paired);records.extend(rr)
    C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records)
    mark('summary.json',status='completed',tasks=tasks,config=CFG,caveats=CFG['caveats'])
    mark('completed.json',status='completed',n=544);status(status='completed',phase='all')
if __name__=='__main__':run()
