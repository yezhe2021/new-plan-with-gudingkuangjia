"""Frozen Llama A+B and Gemma A+B; fixed equal raw-KV mean; no Reader or training."""
import sys,os,json,time,hashlib,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
GP=Path(CFG['gemma_parent']);LP=Path(CFG['llama_residual_parent']);LB=Path(CFG['llama_baseline_parent'])
sys.path.insert(0,str(GP))
from common import C,U,torch,identity as gemma_identity,verify as verify_gemma,OLD_U_ROOT,OLD_U_ID,rows as gemma_rows,loaded_source as gemma_source
from grouped_translator import GroupedKVTranslator
from residual_translator import FrozenTranslatorWithResidual
sys.path.insert(0,str(ROOT))
from fusion import fuse

def identity():return C.sha(ROOT/'manifest.json')
def save(name,value):C.save(ROOT/name,value)
def mark(name,**kw):save(name,dict(manifest_sha256=identity(),**kw))
def status(**kw):save('status.json',dict(time=time.strftime('%F %T'),**kw))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def selected_paths():
    lc=json.loads((LP/'config.json').read_text())
    return {'llama_a':Path(lc['base_parent'])/'runs/stage_a/best.pt','llama_b':LP/'runs/stage_b/last.pt','gemma_a':GP/'runs/stage_a/best.pt','gemma_b':GP/'runs/stage_b/last.pt'}

def prepare():
    verify_gemma();lc=json.loads((LP/'config.json').read_text())
    assert json.loads((LP/'completed.json').read_text())['status']=='completed'
    original={r['id']:r for r in C.read_jsonl(OLD_U_ROOT/'prepared/test.jsonl')}
    assert len(original)==544
    prepared=[];baselines={};dependencies={}
    for gr in gemma_rows('test'):
        lr=original[gr['id']]
        for key in ['id','task','prompt','receiver_suffix','candidates','gold','normalize','cue','suffix_ids','target_indices','native_context_ids']:
            assert gr[key]==lr[key],(gr['id'],key)
        assert gr['frames']['qwen']==lr['frames']['qwen']
        source=Path(lr['source_ref']) if lr.get('source_ref') else OLD_U_ROOT/'source'/(U.fname(lr)+'.pt')
        assert source.exists() and (GP/'source'/(U.fname(gr)+'.pt')).exists()
        record=dict(gemma_row=gr,llama_source_path=str(source),llama_source_identity=lr.get('source_identity',OLD_U_ID),llama_source_length=len(lr['source_indices']))
        assert len(gr['source_indices'])==record['llama_source_length']
        prepared.append(record)
        lpath=LB/'evaluation'/gr['task']/'per_sample'/(U.fname(gr)+'.json')
        gpath=GP/'evaluation'/gr['task']/'per_sample'/(U.fname(gr)+'.json')
        lrec=json.loads(lpath.read_text());grec=json.loads(gpath.read_text())
        assert lrec['reader_manifest_sha256']==C.sha(LB/'manifest.json') and grec['manifest_sha256']==gemma_identity()
        for rec in [lrec,grec]:
            for key in ['id','task','prompt','gold']:assert rec[key]==gr[key]
        assert lrec['conditions']['qwen_native']==grec['conditions']['qwen_native']
        baselines[gr['id']]=dict(qwen_native=grec['conditions']['qwen_native'],llama_ab=lrec['conditions']['grouped_residual'],gemma_ab=grec['conditions']['grouped_residual'])
        for p in [lpath,gpath]:dependencies[str(p)]=C.sha(p)
    assert dict(C.Counter(r['gemma_row']['task'] for r in prepared))==CFG['tasks']
    native=GP/'gemma_native_remaining5/comparison_per_sample.jsonl'
    natives={r['id']:r for r in C.read_jsonl(native)}
    for rid,baseline in baselines.items():baseline['gemma_native']=natives[rid]['conditions']['gemma_native']
    dependencies[str(native)]=C.sha(native)
    for folder in [GP,LP,LB]:
        for p in list(folder.glob('*.py'))+[folder/'config.json',folder/'manifest.json',folder/'completed.json']:
            dependencies[str(p)]=C.sha(p)
    for p in [OLD_U_ROOT/'prepared/test.jsonl',GP/'prepared/test.jsonl']:dependencies[str(p)]=C.sha(p)
    weights={k:dict(path=str(p),sha256=C.sha(p)) for k,p in selected_paths().items()}
    assert weights['llama_a']['sha256']==lc['base_stage_a_sha256']
    for role,file in [('gemma_a','stage_a_selected.json'),('gemma_b','stage_b_selected.json')]:assert weights[role]['sha256']==json.loads((GP/file).read_text())['sha256']
    manifest=dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},dependencies=dependencies,weights=weights,prepared_sha256=C.digest(prepared),baseline_sha256=C.digest(baselines),gemma_manifest_sha256=gemma_identity(),llama_a_manifest_sha256=lc['base_parent_manifest_sha256'],llama_b_manifest_sha256=C.sha(LP/'manifest.json'),llama_architecture=lc['architecture'],gemma_architecture=json.loads((GP/'config.json').read_text())['architecture'],readers_attached=False,all_target_frames_identical=True)
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==manifest
    else:save('manifest.json',manifest)
    C.write_jsonl(ROOT/'prepared/test.jsonl',prepared);save('prepared/baselines.json',baselines)
    mark('prepared/completed.json',status='completed',n=544,counts=CFG['tasks']);status(status='prepared',phase='ready')

def verify():
    verify_gemma();m=json.loads((ROOT/'manifest.json').read_text())
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for p,d in m['code'].items():assert C.sha(ROOT/p)==d
    for p,d in m['dependencies'].items():assert C.sha(p)==d
    for w in m['weights'].values():assert C.sha(w['path'])==w['sha256']
    assert C.digest(rows())==m['prepared_sha256']
    assert C.digest(json.loads((ROOT/'prepared/baselines.json').read_text()))==m['baseline_sha256']
    return m

def load_front(role,m):
    pa,pb=m['weights'][role+'_a'],m['weights'][role+'_b']
    a=torch.load(pa['path'],map_location='cpu',weights_only=True)
    expected=m['llama_a_manifest_sha256'] if role=='llama' else m['gemma_manifest_sha256']
    assert a['manifest_sha256']==expected and a['stage']=='a'
    base=GroupedKVTranslator(**m[role+'_architecture']);base.load_state_dict(a['state'],strict=True);del a
    with torch.random.fork_rng(devices=[]):front=FrozenTranslatorWithResidual(base,64)
    b=torch.load(pb['path'],map_location='cpu',weights_only=True)
    expected=m['llama_b_manifest_sha256'] if role=='llama' else m['gemma_manifest_sha256']
    assert b['manifest_sha256']==expected and b['state_scope']=='cross_head_residual_only'
    assert b['frozen_stage_a_sha256']==pa['sha256'] and b['step']==1024 and b['epoch_completed']
    front.residual.load_state_dict(b['state'],strict=True);del b
    return front.cuda().eval().requires_grad_(False)

def source_pair(item):
    gr=item['gemma_row'];g=gemma_source(gr)
    l=torch.load(item['llama_source_path'],map_location='cpu',weights_only=True)
    assert l['manifest_sha256']==item['llama_source_identity']
    if 'id' in l:assert l['id']==gr['id']
    assert l['key'].shape==l['value'].shape==(28,item['llama_source_length'],8,128)
    assert l['key'].dtype==l['value'].dtype==torch.float16
    return l,g

@torch.inference_mode()
def map_source(front,source):
    result=[]
    for start in range(0,source['key'].shape[1],CFG['chunk_tokens']):
        k=source['key'][:,start:start+64][None].cuda();v=source['value'][:,start:start+64][None].cuda()
        with torch.amp.autocast('cuda',dtype=torch.float16):result.append(front(k,v))
    return tuple(torch.cat([part[i][0] for part in result],dim=1) for i in range(2))

@torch.inference_mode()
def smoke(model,tok,lfront,gfront):
    baselines=json.loads((ROOT/'prepared/baselines.json').read_text());checks=[]
    for task in ['openbookqa','hellaswag']:
        item=next(r for r in rows() if r['gemma_row']['task']==task);row=item['gemma_row']
        ls,gs=source_pair(item);lk,lv=map_source(lfront,ls);gk,gv=map_source(gfront,gs)
        assert lk.shape==lv.shape==gk.shape==gv.shape==(36,len(row['source_indices']),8,128)
        native=U.native_body(model,row)
        for branch,k,v in [('llama_ab',lk,lv),('gemma_ab',gk,gv)]:
            actual=U.candidate_scores(model,tok,row,native,k,v);expected=baselines[row['id']][branch]
            err=max(abs(a-b) for a,b in zip(actual['scores'],expected['scores']))
            assert actual['prediction']==expected['prediction'] and err<0.02,(branch,err)
            checks.append(dict(id=row['id'],branch=branch,baseline_prediction_identical=True,max_score_abs_error=err))
        for weight,k,v in [(1.0,lk,lv),(0.0,gk,gv)]:
            fk,fv=fuse(lk,lv,gk,gv,weight);assert torch.equal(k,fk) and torch.equal(v,fv)
        fk,fv=fuse(lk,lv,gk,gv)
        assert fk.shape==lk.shape and torch.isfinite(fk).all() and torch.isfinite(fv).all()
        # Raw native reconstruction confirms single application of target RoPE and correct anchor.
        ids=row['frames']['qwen']['body_ids'];anchor=row['frames']['qwen']['native_prefix_length']
        k,v,_=C.capture(model,ids,len(ids),[0])
        a=U.suffix_logits(model,row,U.clone_cache(native,model),[],[0])
        b=U.suffix_logits(model,row,U.translated_cache(model,row,native,k[:,anchor:].cuda(),v[:,anchor:].cuda()),[],[0])
        kl=float(U.full_kl(a,b));assert kl<0.001 and int(a.argmax())==int(b.argmax())
        checks.append(dict(id=row['id'],native_cache_parity_kl=kl,zero_one_endpoints_bitwise=True,cache_token_count_unchanged=True))
    assert all(not p.requires_grad and p.grad is None for m in [model,lfront,gfront] for p in m.parameters())
    assert all('LoRA' not in type(m).__name__ for m in model.modules())
    mark('smoke/completed.json',status='completed',checks=checks,reader_absent=True,all_parameters_frozen=True)

@torch.inference_mode()
def evaluate(model,tok,lfront,gfront,task):
    folder=ROOT/'evaluation'/task
    if (folder/'completed.json').exists():assert json.loads((folder/'completed.json').read_text())['manifest_sha256']==identity();return
    items=[r for r in rows() if r['gemma_row']['task']==task];assert len(items)==CFG['tasks'][task]
    baseline=json.loads((ROOT/'prepared/baselines.json').read_text());runtime=C.runtime(model,tok)
    for i,item in enumerate(items,1):
        row=item['gemma_row'];path=folder/'per_sample'/(U.fname(row)+'.json')
        if path.exists():assert json.loads(path.read_text())['manifest_sha256']==identity();continue
        start=time.monotonic();status(status='running',phase='evaluation',task=task,sample=i,total=len(items))
        ls,gs=source_pair(item);lk,lv=map_source(lfront,ls);gk,gv=map_source(gfront,gs)
        k,v=fuse(lk,lv,gk,gv,CFG['weight_llama'])
        assert k.shape==v.shape==(36,len(row['source_indices']),8,128) and torch.isfinite(k).all() and torch.isfinite(v).all()
        native=U.native_body(model,row)
        if task=='gsm8k':
            text,count=C.R.generate(runtime,torch.tensor([row['suffix_ids']],device='cuda'),CFG['max_new_tokens'],past_key_values=U.translated_cache(model,row,native,k,v),past_length=len(row['frames']['qwen']['body_ids']),temperature=0.0)
            pred=C.S.extract_numeric_answer(text)
            result=dict(prediction=pred,correct=int(pred==C.S.normalize_number(row['gold'])),generation=text,generated_tokens=count,reached_cap=int(count>=CFG['max_new_tokens']))
        else:result=U.candidate_scores(model,tok,row,native,k,v)
        conditions=dict(baseline[row['id']],equal50_fusion=result)
        C.save(path,dict(id=row['id'],task=task,prompt=row['prompt'],gold=row['gold'],conditions=conditions,manifest_sha256=identity(),elapsed_seconds=time.monotonic()-start))
        if i%8==0 or task=='gsm8k':C.log(f'Equal50 fusion {task} {i}/{len(items)} correct={result["correct"]}')
        del ls,gs,lk,lv,gk,gv,k,v,native
    records=[json.loads((folder/'per_sample'/(U.fname(r['gemma_row'])+'.json')).read_text()) for r in items]
    C.write_jsonl(folder/'per_sample.jsonl',records);mark('evaluation/'+task+'/completed.json',status='completed',n=len(records),correct=sum(r['conditions']['equal50_fusion']['correct'] for r in records))

def summarize():
    records=[];tasks={}
    for task,n in CFG['tasks'].items():
        assert (ROOT/'evaluation'/task/'completed.json').exists()
        rr=C.read_jsonl(ROOT/'evaluation'/task/'per_sample.jsonl');assert len(rr)==n
        metrics={a:dict(n=n,correct=sum(r['conditions'][a]['correct'] for r in rr),accuracy=sum(r['conditions'][a]['correct'] for r in rr)/n) for a in ['qwen_native','gemma_native','llama_ab','gemma_ab','equal50_fusion']}
        if task=='gsm8k':
            for a in metrics:
                vv=[r['conditions'][a] for r in rr]
                metrics[a].update(mean_generated_tokens=sum(v['generated_tokens'] for v in vv)/n,cap_count=sum(v['reached_cap'] for v in vv))
        paired={}
        for a in ['llama_ab','gemma_ab']:
            pairs=[(r['conditions']['equal50_fusion']['correct'],r['conditions'][a]['correct']) for r in rr]
            paired[a]=dict(both_correct=sum(x and y for x,y in pairs),fusion_only=sum(x and not y for x,y in pairs),single_only=sum(not x and y for x,y in pairs),both_wrong=sum(not x and not y for x,y in pairs))
        tasks[task]=dict(metrics=metrics,paired=paired);records.extend(rr)
    C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records)
    mark('summary.json',status='completed',tasks=tasks,config=CFG,n=544,caveats=CFG['caveats'])
    mark('completed.json',status='completed',n=544);status(status='completed',phase='all')

@torch.inference_mode()
def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=verify();C.seed();assert torch.cuda.mem_get_info()[0]/2**30>=CFG['queue_min_free_gpu_gib']
        status(status='running',phase='loading_models',pid=os.getpid())
        lfront=load_front('llama',m);gfront=load_front('gemma',m);tok,model=C.model('qwen')
        smoke(model,tok,lfront,gfront)
        # Complete short candidate-scoring tasks first; GSM uses the unchanged4096-token cap.
        for task in ['openbookqa','arc_challenge','hellaswag','mmlu_pro','sciq','gsm8k']:evaluate(model,tok,lfront,gfront,task)
        summarize()

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():raise RuntimeError('Already launched: inspect prior process before resuming')
        prepare();(ROOT/'logs').mkdir(exist_ok=True)
        subprocess.run([sys.executable,'-m','unittest','-v','test_fusion'],cwd=ROOT,check=True,stdout=(ROOT/'logs/unit_tests.log').open('w'),stderr=subprocess.STDOUT)
        with (ROOT/'logs/pipeline.log').open('a') as out:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=process.pid,started_at=time.strftime('%F %T'),detached=True,training=False,reader=False)
        save('launch.json',record);print(json.dumps(record))

if __name__=='__main__':
    try:
        if sys.argv[1]=='--launch':launch()
        else:run()
    except Exception as e:
        save('failure.json',dict(type=type(e).__name__,message=str(e)));status(status='failed',error=str(e));raise
