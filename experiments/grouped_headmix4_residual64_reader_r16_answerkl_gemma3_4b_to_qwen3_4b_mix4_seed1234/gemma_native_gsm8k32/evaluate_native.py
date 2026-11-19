"""Native Gemma evaluation on the exact32 GSM questions of the completed transfer run."""
import os,sys,json,time,hashlib,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from common import C,U,torch,CFG as PARENT_CFG,identity as parent_identity,verify as verify_parent
from capture_source import load_gemma
CONFIG=dict(name='gemma3_4b_it_native_gsm8k32_seed1234',model=PARENT_CFG['sender'],compute_dtype='float32',attention='sdpa',seed=1234,max_new_tokens=4096,do_sample=False,n=32,prompt='Exact original question prompt in the native Gemma user chat template; no answer or demonstrations appended.',scoring='Same pinned extract_numeric_answer / normalize_number as the transfer experiment',stopping='Native Gemma generation_config EOS IDs, including end_of_turn; generated_tokens includes terminal EOS if emitted.',continuous_monitoring=False,automatic_upload=False)
def save(name,obj):C.save(ROOT/name,obj)
def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**kw):save(name,dict(manifest_sha256=identity(),**kw))
def status(**kw):save('status.json',dict(time=time.strftime('%F %T'),**kw))
def prepare():
    verify_parent()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(CONFIG['model'],local_files_only=True)
    rr=[r for r in U.rows('test') if r['task']=='gsm8k'];assert len(rr)==32
    prepared=[]
    for r in rr:
        rendered=tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=False,add_generation_prompt=True)
        direct=tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True)
        if hasattr(direct,'keys'):direct=direct['input_ids']
        ids=tok.encode(rendered,add_special_tokens=False);assert list(direct)==ids
        assert r['cue']=='' and rendered.count(r['prompt'])==1
        prepared.append(dict(id=r['id'],task='gsm8k',doc_sha256=r['doc_sha256'],prompt=r['prompt'],gold=r['gold'],input_ids=ids,rendered_prompt=rendered,parent_row_sha256=C.digest(r)))
    m=dict(config=CONFIG,parent_manifest_sha256=parent_identity(),parent_config_sha256=C.sha(PARENT/'config.json'),prepared_sha256=C.digest(prepared),code_sha256=C.sha(__file__),model_metadata={str(Path(CONFIG['model'])/f):C.sha(Path(CONFIG['model'])/f) for f in ['config.json','tokenizer_config.json']})
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==m
    else:save('manifest.json',m)
    C.write_jsonl(ROOT/'prepared.jsonl',prepared);mark('prepared_completed.json',status='completed',n=32)
    return prepared

def verify():
    verify_parent();m=json.loads((ROOT/'manifest.json').read_text())
    assert m['parent_manifest_sha256']==parent_identity() and m['config']==CONFIG
    assert m['code_sha256']==C.sha(__file__)
    rr=C.read_jsonl(ROOT/'prepared.jsonl');assert len(rr)==32 and C.digest(rr)==m['prepared_sha256']
    for p,d in m['model_metadata'].items():assert C.sha(p)==d
    return rr

def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():raise RuntimeError('Already launched: inspect prior process before resuming')
        prepare();(ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/evaluation.log').open('a') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        info=dict(pid=p.pid,started_at=time.strftime('%F %T'),detached=True)
        save('launch.json',info);print(json.dumps(info))

@torch.inference_mode()
def run():
    with (ROOT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rr=verify();C.seed()
        free=torch.cuda.mem_get_info()[0]/2**30;assert free>=24,free
        status(status='running',phase='loading_model',pid=os.getpid())
        tok,model=load_gemma(torch.float32)
        native_eos=getattr(model.generation_config,'eos_token_id',None)
        stops=set(native_eos if isinstance(native_eos,list) else [native_eos]);stops.add(tok.eos_token_id);stops.discard(None)
        assert 106 in stops,('Gemma end_of_turn missing',stops)
        save('runtime.json',dict(dtype=str(next(model.parameters()).dtype),eos_token_ids=sorted(stops),gpu=torch.cuda.get_device_name(),versions=dict(torch=torch.__version__),parent_manifest_sha256=parent_identity()))
        for i,r in enumerate(rr,1):
            path=ROOT/'per_sample'/(U.fname(r)+'.json')
            if path.exists():
                assert json.loads(path.read_text())['manifest_sha256']==identity();continue
            start=time.monotonic();status(status='running',phase='generation',sample=i,total=32,id=r['id'])
            tensor=torch.tensor([r['input_ids']],device='cuda')
            # Check real full-model native logits before the first decode.
            if i==1:
                out=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1)
                assert torch.isfinite(out.logits).all(),'Nonfinite native Gemma logits'
                mark('smoke.json',status='completed',finite_native_logits=True,native_eos_ids=sorted(stops),first_prompt_tokens=len(r['input_ids']))
                del out
            output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),max_new_tokens=4096,do_sample=False,use_cache=True,eos_token_id=sorted(stops),pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,logits_to_keep=1)
            generated=output[0,tensor.shape[1]:].tolist();assert generated
            text=tok.decode(generated,skip_special_tokens=False)
            pred=C.S.extract_numeric_answer(text);count=len(generated)
            eos=generated[-1] in stops
            result=dict(id=r['id'],doc_sha256=r['doc_sha256'],prompt=r['prompt'],gold=r['gold'],prediction=pred,correct=int(pred==C.S.normalize_number(r['gold'])),generation=text,generated_tokens=count,stopped_eos=eos,reached_cap=int(count>=4096 and not eos),elapsed_seconds=time.monotonic()-start,manifest_sha256=identity())
            C.save(path,result)
            saved=[json.loads(p.read_text()) for p in (ROOT/'per_sample').glob('*.json')]
            mark('progress.json',status='running',completed=len(saved),expected=32,correct_so_far=sum(r['correct'] for r in saved),mean_seconds_per_item=sum(r['elapsed_seconds'] for r in saved)/len(saved))
            print(time.strftime('%F %T'),f'Gemma Native {i}/32 correct={result["correct"]} tokens={count} seconds={result["elapsed_seconds"]:.1f}',flush=True)
        saved=[json.loads((ROOT/'per_sample'/(U.fname(r)+'.json')).read_text()) for r in rr]
        C.write_jsonl(ROOT/'per_sample.jsonl',saved)
        mark('summary.json',status='completed',n=32,correct=sum(r['correct'] for r in saved),accuracy=sum(r['correct'] for r in saved)/32,mean_generated_tokens=sum(r['generated_tokens'] for r in saved)/32,cap_count=sum(r['reached_cap'] for r in saved),config=CONFIG,parent_manifest_sha256=parent_identity())
        mark('completed.json',status='completed');status(status='completed',phase='all')
if __name__=='__main__':
    try:
        if sys.argv[1]=='--launch':launch()
        else:run()
    except Exception as e:
        save('failure.json',dict(type=type(e).__name__,message=str(e)));status(status='failed',error=str(e));raise
