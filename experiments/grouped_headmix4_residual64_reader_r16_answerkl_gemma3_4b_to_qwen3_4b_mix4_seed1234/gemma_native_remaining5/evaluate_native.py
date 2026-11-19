"""Complete Gemma native baselines on the five remaining fixed test subsets."""
import os,sys,json,time,fcntl,subprocess,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from common import C,U,torch,CFG as PCFG,identity as parent_identity,verify as verify_parent
from capture_source import load_gemma
TASKS=dict(openbookqa=128,arc_challenge=128,hellaswag=64,mmlu_pro=128,sciq=64)
CONFIG=dict(model=PCFG['sender'],compute_dtype='float32',attention='sdpa',seed=1234,counts=TASKS,prompt='Exact existing question/option/context text, Gemma native chat template, same assistant answer cue.',scoring='Standalone candidate tokenization, summed conditional token log-likelihood; HellaSwag divides by candidate token count. No answer generation for multiple choice.',gsm_reuse='Existing gemma_native_gsm8k32 result; no GSM rerun',continuous_monitoring=False,automatic_upload=False)
GSM=PARENT/'gemma_native_gsm8k32'
def save(name,obj):C.save(ROOT/name,obj)
def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**kw):save(name,dict(manifest_sha256=identity(),**kw))
def status(**kw):save('status.json',dict(time=time.strftime('%F %T'),**kw))
def prepare():
    verify_parent()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(CONFIG['model'],local_files_only=True)
    records=[]
    for row in U.rows('test'):
        if row['task'] not in TASKS:continue
        messages=[dict(role='user',content=row['prompt'])]
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        direct=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if hasattr(direct,'keys'):direct=direct['input_ids']
        assert list(direct)==tok.encode(rendered,add_special_tokens=False)
        context=rendered+row['cue'];ids=tok.encode(context,add_special_tokens=False)
        candidates=[C.R.candidate_ids(tok,c) for c in row['candidates']]
        assert all(candidates) and len(candidates)==len(row['candidates'])
        records.append(dict(id=row['id'],task=row['task'],prompt=row['prompt'],cue=row['cue'],candidates=row['candidates'],gold=row['gold'],normalize=row['normalize'],input_ids=ids,candidate_ids=candidates,parent_row_sha256=C.digest(row)))
    assert dict(C.Counter(r['task'] for r in records))==TASKS
    assert len(records)==512 and len({r['id'] for r in records})==512
    gsm_summary=json.loads((GSM/'summary.json').read_text())
    assert gsm_summary['n']==32 and gsm_summary['status']=='completed' and gsm_summary['parent_manifest_sha256']==parent_identity()
    gsm_manifest=C.sha(GSM/'manifest.json');assert json.loads((GSM/'completed.json').read_text())['manifest_sha256']==gsm_manifest
    m=dict(config=CONFIG,parent_manifest_sha256=parent_identity(),prepared_sha256=C.digest(records),code_sha256=C.sha(__file__),gsm_files={str(GSM/f):C.sha(GSM/f) for f in ['manifest.json','summary.json','completed.json','per_sample.jsonl']},model_metadata={str(Path(CONFIG['model'])/f):C.sha(Path(CONFIG['model'])/f) for f in ['config.json','tokenizer_config.json']})
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==m
    else:save('manifest.json',m)
    C.write_jsonl(ROOT/'prepared.jsonl',records);mark('prepared_completed.json',status='completed',n=512,counts=TASKS)

def verify():
    verify_parent();m=json.loads((ROOT/'manifest.json').read_text())
    assert m['code_sha256']==C.sha(__file__) and m['config']==CONFIG and m['parent_manifest_sha256']==parent_identity()
    rr=C.read_jsonl(ROOT/'prepared.jsonl');assert len(rr)==512 and C.digest(rr)==m['prepared_sha256']
    for field in ['gsm_files','model_metadata']:
        for p,d in m[field].items():assert C.sha(p)==d
    return rr

@torch.inference_mode()
def score_candidates(model,row):
    context=row['input_ids'];candidates=row['candidate_ids']
    if all(len(x)==1 for x in candidates):
        tensor=torch.tensor([context],device='cuda')
        logits=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1).logits
        assert torch.isfinite(logits).all()
        scores=logits[0,-1].float().log_softmax(-1)[torch.tensor([x[0] for x in candidates],device='cuda')].cpu().tolist()
    else:
        scores=[]
        for target in candidates:
            # All native context + candidate history, causal scoring; no truncation or cache substitution.
            tensor=torch.tensor([context+target[:-1]],device='cuda')
            output=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=len(target))
            logits=output.logits;assert logits.shape[1]==len(target) and torch.isfinite(logits).all()
            labels=torch.tensor([target],device='cuda')
            scores.append(C.R._candidate_log_likelihood(logits,labels,row['normalize']))
    assert all(math.isfinite(s) for s in scores)
    return scores

@torch.inference_mode()
def smoke(model,rr):
    checks=[]
    for task in ['openbookqa','hellaswag']:
        row=next(r for r in rr if r['task']==task)
        actual=score_candidates(model,row);expected=[]
        for target in row['candidate_ids']:
            # Independent full-sequence, full-logit reference for candidate index offsets.
            ids=row['input_ids']+target
            tensor=torch.tensor([ids],device='cuda')
            output=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False)
            start=len(row['input_ids'])-1
            logits=output.logits[:,start:start+len(target)]
            expected.append(C.R._candidate_log_likelihood(logits,torch.tensor([target],device='cuda'),row['normalize']))
        torch.testing.assert_close(torch.tensor(actual),torch.tensor(expected),rtol=1e-4,atol=1e-4)
        assert max(range(len(actual)),key=actual.__getitem__)==max(range(len(expected)),key=expected.__getitem__)
        checks.append(dict(task=task,id=row['id'],max_score_abs_error=max(abs(a-b) for a,b in zip(actual,expected)),prediction_identical=True))
    mark('smoke.json',status='completed',checks=checks,native_full_sequence_parity=True)

def summarize(rr):
    rows=[json.loads((ROOT/'per_sample'/r['task']/(U.fname(r)+'.json')).read_text()) for r in rr]
    gm=C.read_jsonl(GSM/'per_sample.jsonl');assert len(gm)==32
    native={x['id']:x for x in rows+gm};assert len(native)==544
    current=C.read_jsonl(PARENT/'evaluation/per_sample.jsonl');assert len(current)==544
    combined=[];tasks={}
    for old in current:
        g=native[old['id']];assert old['prompt']==g['prompt'] and old['gold']==g['gold']
        c=dict(old['conditions']);c['gemma_native']={k:g[k] for k in ['prediction','correct']}
        if old['task']=='gsm8k':
            c['gemma_native'].update({k:g[k] for k in ['generation','generated_tokens','reached_cap']})
        else:c['gemma_native']['scores']=g['scores']
        combined.append(dict(id=old['id'],task=old['task'],prompt=old['prompt'],gold=old['gold'],conditions=c,manifest_sha256=identity()))
    counts=dict(TASKS,gsm8k=32)
    for task,n in counts.items():
        ss=[r for r in combined if r['task']==task];assert len(ss)==n
        metrics={arm:dict(n=n,correct=sum(x['conditions'][arm]['correct'] for x in ss),accuracy=sum(x['conditions'][arm]['correct'] for x in ss)/n) for arm in ['qwen_native','gemma_native','grouped_residual','grouped_residual_reader']}
        tasks[task]=dict(metrics=metrics)
    C.write_jsonl(ROOT/'per_sample.jsonl',rows)
    C.write_jsonl(ROOT/'comparison_per_sample.jsonl',combined)
    mark('summary.json',status='completed',tasks=tasks,counts=counts,total=544,parent_manifest_sha256=parent_identity(),caveats=['MMLU-Pro custom split, not untouched official full test.','Gemma FP32 and Qwen FP16 use model-native chat templates and shared task scoring.','Sender and Stage-A data mixture both differ from the old Llama run.'])
    mark('completed.json',status='completed',newly_evaluated=512,reused_gsm8k=32,total=544);status(status='completed',phase='all')

@torch.inference_mode()
def run():
    with (ROOT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rr=verify();C.seed();assert torch.cuda.mem_get_info()[0]/2**30>=24
        status(status='running',phase='loading_model',pid=os.getpid());tok,model=load_gemma(torch.float32)
        smoke(model,rr)
        for task in TASKS:
            subset=[r for r in rr if r['task']==task]
            for i,row in enumerate(subset,1):
                path=ROOT/'per_sample'/task/(U.fname(row)+'.json')
                if path.exists():assert json.loads(path.read_text())['manifest_sha256']==identity();continue
                status(status='running',phase='candidate_scoring',task=task,sample=i,total=len(subset))
                start=time.monotonic();scores=score_candidates(model,row)
                value=C.record(scores,row['gold'])
                C.save(path,dict(id=row['id'],task=task,prompt=row['prompt'],gold=row['gold'],candidates=row['candidates'],normalize=row['normalize'],elapsed_seconds=time.monotonic()-start,manifest_sha256=identity(),**value))
                if i%8==0 or i==len(subset):print(time.strftime('%F %T'),f'Gemma Native {task} {i}/{len(subset)}',flush=True)
            records=[json.loads((ROOT/'per_sample'/task/(U.fname(r)+'.json')).read_text()) for r in subset]
            mark('evaluation/'+task+'/completed.json',status='completed',n=len(records),correct=sum(r['correct'] for r in records))
        summarize(rr)

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():raise RuntimeError('Already launched: inspect existing status before resuming')
        prepare();(ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/evaluation.log').open('a') as out:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        info=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True);save('launch.json',info);print(json.dumps(info))
if __name__=='__main__':
    try:
        if sys.argv[1]=='--launch':launch()
        else:run()
    except Exception as e:
        save('failure.json',dict(type=type(e).__name__,message=str(e)));status(status='failed',error=str(e));raise
