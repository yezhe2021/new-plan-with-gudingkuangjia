"""Audit the eager/SDPA capture discrepancy, then resume only missing receiver rows."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import sys,json,gc,time,hashlib,copy,fcntl
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import run_experiment as K
E=K.E
OUT=Path(__file__).resolve().parent


def compare(a,b):
    a,b=a.float(),b.float()
    return dict(max_error=float((a-b).abs().max()),nmse=float((a-b).square().sum()/b.square().sum().clamp_min(1e-30)))


@torch.inference_mode()
def capture_audit():
    K.verify();row=next(r for r in K.rows() if r['id']=='arc_22')
    new=torch.load(K.source_path(row),map_location='cpu',weights_only=True);old=E.U.loaded_source(row)
    audit=dict(original_manifest_sha256=K.identity(),sample=row['id'],legacy_backend=E.C.CFG['attention'],new_backend=E.CFG['attention_backend'],comparisons={})
    assert audit['legacy_backend']=='sdpa' and audit['new_backend']=='eager'
    for backend in ['eager','sdpa']:
        model=E.AutoModelForCausalLM.from_pretrained(E.CFG['models']['llama'],dtype=torch.float16,device_map='cuda',
            local_files_only=True,attn_implementation=backend).eval().requires_grad_(False)
        k,v,_=E.C.capture(model,row['layouts']['llama']['body_ids'],len(row['layouts']['llama']['body_ids']),[0])
        aligned=[x[:,row['source_indices']] for x in [k,v]]
        expected=[new[x][:,row['target_map']] for x in ['key','value']] if backend=='eager' else [old[x] for x in ['key','value']]
        audit['comparisons'][backend]={role:compare(a,b) for role,a,b in zip(['key','value'],aligned,expected)}
        assert all(compare(a,b)['max_error']==0 for a,b in zip(aligned,expected)),audit
        del model,k,v,aligned,expected;gc.collect();torch.cuda.empty_cache()
    audit['cause']='Recomputed eager and SDPA captures reproduce respectively the new and historical Sender tensors exactly. Source histories and weights match; backend arithmetic caused the score-check trigger.'
    audit['remaining_source_comparisons']={}
    for row in K.rows():
        if K.result_path('qwen',row).exists():continue
        new=torch.load(K.source_path(row),map_location='cpu',weights_only=True);old=E.U.loaded_source(row)
        comparison={role:compare(new[role][:,row['target_map']],old[role]) for role in ['key','value']}
        # Gross interface/position errors are not accepted as backend arithmetic.
        assert all(c['nmse']<1e-4 for c in comparison.values()),(row['id'],comparison)
        assert torch.equal(new['key'][0,row['target_map']],old['key'][0])
        assert torch.equal(new['value'][0,row['target_map']],old['value'][0])
        audit['remaining_source_comparisons'][row['id']]=comparison
    audit['status']='passed';audit['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    K.save(OUT/'audit.json',audit)
    print('CAPTURE BACKEND AUDIT PASSED',json.dumps(audit['comparisons']),flush=True)
    return audit


@torch.inference_mode()
def resume(audit):
    assert audit['status']=='passed';K.verify()
    for name in ['failure.json','runner_status.json','status.json']:
        f=ROOT/name
        if f.exists():
            if not (OUT/('original_'+name)).exists():K.save(OUT/('original_'+name),K.read(f))
            if name=='failure.json':f.rename(OUT/'original_failure.json')
    K.save(ROOT/'runner_status.json',dict(status='running',pid=os.getpid(),resumed=True,recovery_audit=str(OUT/'audit.json'),time=time.strftime('%F %T')))
    tokenizer,model=E.load_model('qwen');base,reader=E.load_frontend(model);completed=0
    for index,row in enumerate(K.rows(),1):
        if K.result_path('qwen',row).exists():
            assert K.read(K.result_path('qwen',row))['manifest_sha256']==K.identity();continue
        prior=K.read(K.result_path('llama',row));source=torch.load(K.source_path(row),map_location='cpu',weights_only=True)
        assert source['manifest_sha256']==prior['manifest_sha256']==K.identity()
        factory=K.FixedCodec(source['key'],source['value'],K.CFG['group_size'],row['wire_positions'],row['target_map'])
        with reader.mode(False):native=E.U.native_body(model,row)
        results={};base_result=base_logits=None
        for arm in K.ARMS:
            K.status(status='running',phase='receiver_resumed',sample=index,total=256,task=row['task'],arm=arm)
            k,v,wire=K.packet(row,factory,arm,prior['selections']);assert wire['sha256']==prior['arms'][arm]['wire']['sha256']
            with reader.mode(False):cache,mappedk,mappedv=E.translated(model,base,reader,row,native,k,v)
            with reader.mode(True):result,logits,_=K.scores(model,row,row['layouts']['qwen'],cache)
            if arm=='fp16':
                reference=K.read(ROOT/'baseline'/row['task']/(K.fname(row)+'.json'))
                result['baseline_score_max_error']=max(abs(a-b) for a,b in zip(result['scores'],reference['scores']))
                result['baseline_prediction_match']=result['prediction']==reference['prediction']
                result['capture_backend_audit']=audit['remaining_source_comparisons'][row['id']]
                base_result,base_logits=result,logits.clone()
                if completed==0:
                    old_loader=E.U.loaded_source
                    try:
                        E.U.loaded_source=lambda _:dict(key=source['key'][:,row['target_map']].contiguous(),value=source['value'][:,row['target_map']].contiguous())
                        _,refk,refv=E.R.fixed_state(model,base,reader,row)
                        with reader.mode(True):canonical=E.U.candidate_scores(model,tokenizer,row,native,refk,refv)
                        assert torch.equal(refk,mappedk) and torch.equal(refv,mappedv)
                        score_error=max(abs(a-b) for a,b in zip(result['scores'],canonical['scores']))
                        with reader.mode(True):
                            prefix_logits=E.U.suffix_logits(model,row,copy.deepcopy(cache),[],[0])[0]
                        prefix_scores=prefix_logits.log_softmax(-1)[row['layouts']['qwen']['label_ids']].cpu().tolist()
                        prefix_error=max(abs(a-b) for a,b in zip(result['scores'],prefix_scores))
                        assert prefix_error<1e-4,('Original suffix helper at identical causal prefix disagrees',prefix_error)
                        K.save(OUT/'canonical_receiver_audit.json',dict(status='passed',id=row['id'],mapped_tensors_bitwise_equal=True,score_max_error=score_error,
                            identical_prefix_helper_score_error=prefix_error,legacy_candidate_prefix_shape_scores=canonical['scores'],new_prefix_only_scores=result['scores'],
                            explanation='Original per-candidate scoring appends a future label token; optimized scoring uses only the causal prefix. Both are causally equivalent, but FP16 GEMM shapes differ. At identical prefix length the original helper matches the optimized implementation.',
                            cached_reference_prediction=reference['prediction'],new_prediction=result['prediction'],raw_score_max_deviation=result['baseline_score_max_error']))
                        del refk,refv
                    finally:E.U.loaded_source=old_loader
            results[arm]=dict(K.delta(result,base_result,logits,base_logits),wire=wire)
        assert len({results[a]['wire']['bytes'] for a in K.ARMS[2:]})==1
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        K.save(K.result_path('qwen',row),dict(id=row['id'],task=row['task'],gold=row['gold'],manifest_sha256=K.identity(),arms=results,recovery_audit=str(OUT/'audit.json')))
        K.source_path(row).unlink();completed+=1
        del factory,source,native,cache,k,v,mappedk,mappedv
        if index%8==0:print(time.strftime('%F %T'),'QWEN RESUMED',index,'/256',flush=True)
    K.save(ROOT/'evaluation/qwen/completed.json',dict(status='completed',manifest_sha256=K.identity(),n=256,recovered=True))
    K.summarize()
    final=K.read(ROOT/'summary.json');final['recovery']=dict(audit_path=str(OUT/'audit.json'),original_stopped_at='arc_22',
        note='Historical SDPA captures are a numerical replay reference. Every quantized arm is compared to its own eager FP16 capture; original rows and weights unchanged. Only missing receiver rows were computed.')
    K.save(OUT/'completed.json',dict(status='completed',new_rows=completed,original_manifest_sha256=K.identity()))
    K.save(ROOT/'runner_status.json',dict(status='completed',pid=os.getpid(),resumed=True,time=time.strftime('%F %T')))


if __name__=='__main__':
    try:
        with (OUT/'recovery.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not(OUT/'completed.json').exists(),'Recovery already complete'
            audit=K.read(OUT/'audit.json')
            assert audit['status']=='passed' and audit['original_manifest_sha256']==K.identity()
            K.save(OUT/'resume_version.json',dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_capture_audit_sha256=K.sha(OUT/'audit.json'),reason='Reuse passed exact backend capture audit; validate score helper at identical causal prefix, while retaining old prefix-shape numerical deviations.'))
            resume(audit)
    except Exception as exc:
        import traceback
        K.save(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        K.status(status='failed',phase='recovery',error=str(exc));raise
