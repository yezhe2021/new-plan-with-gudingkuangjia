"""Resumable extraction. Test collection starts only after estimator lock."""
import backend as B
from backend import ROOT,CFG,E,read,save,sha,ident,rows,status,torch
from engine import Engine,perturb_positions
from pathlib import Path
import sys,time,traceback,copy,json


def preflight(engine):
    checks=[];lookup={r['id']:r for r in rows('train')}
    for row in rows('train')[:4]:
        own=engine.mapped(row);cache=engine.cache(row,own);blocked=engine.reject_columns(row)
        out,_=engine.prompt(row,cache,blocked);masked=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        noisy=engine.cache(row,engine.variant(row,'gaussian',own,own))
        out,_=engine.prompt(row,noisy,blocked);other=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        err=float((masked-other).abs().max());assert err==0,err
        native=engine.native(row['frames']['qwen']['body_ids']);out,_=engine.prompt(row,native)
        cached=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        out=engine.forward(row['frames']['qwen']['body_ids']+row['continuation_ids'],use_cache=False)
        full=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float();diff=float((cached-full).abs().max())
        assert diff<.3,diff
        assert int(cached.argmax())==int(full.argmax()),'Native cache/full first token mismatch'
        for spec in CFG['perturbations']:perturb_positions(row,spec)
        # Duplicate candidates must receive identical scores, including under masking.
        candidates=[row['continuation_ids'][-3:]]*2
        pair=engine.pair_scores(row,cache,candidates,blocked);assert pair[0]==pair[1]
        assert all(not p.requires_grad for model in [*engine.models.values(),engine.translator] for p in model.parameters())
        checks.append(dict(id=row['id'],masked_content_logit_error=err,native_cache_full_max_logit_error=diff,first_token_equal=True))
    save(ROOT/'preflight/completed.json',dict(manifest_sha256=ident(),checks=checks))


def process(engine,row,lookup,folder,condition_names,selector_sha=None,audit=False):
    path=folder/(row['id']+'.json');began=time.time()
    result=read(path) if path.exists() else dict(id=row['id'],split=row['split'],question=row['question'],answer=row['answer'],
        donor_id=row['donor_id'],manifest_sha256=ident(),selector_sha256=selector_sha,baselines={},conditions={},complete=False)
    assert result['manifest_sha256']==ident() and result['selector_sha256']==selector_sha
    if result['complete']:return result
    own=engine.mapped(row);donor=engine.mapped(lookup[row['donor_id']]);reject_cache=engine.cache(row,own)
    for name in ['reject','compact','native']:
        if name in result['baselines']:continue
        if name=='compact':cache=None;blocked=[];compact=True
        elif name=='native':cache=engine.native(row['frames']['qwen']['body_ids']);blocked=[];compact=False
        else:cache=reject_cache;blocked=engine.reject_columns(row);compact=False
        generated=engine.generate(row,cache,blocked,compact)
        result['baselines'][name]=engine.annotate(row,generated,cache,blocked,compact);save(path,result)
    reject=result['baselines']['reject']
    for name in condition_names:
        if name in result['conditions']:continue
        status(status='running',phase='collect_'+row['split'],question_id=row['id'],condition=name)
        case_start=time.time();kv=engine.variant(row,name,own,donor);cache=engine.cache(row,kv)
        accept=engine.generate(row,cache)
        response=engine.extract(row,cache,reject_cache,accept,reject,audit=audit)
        # Labels are appended after inference features have been computed.
        response['accept']=engine.annotate(row,accept,cache)
        response['reject']=reject
        response['utility_em']=response['accept']['em']-reject['em']
        response['utility_f1']=response['accept']['f1']-reject['f1']
        response['case_seconds']=time.time()-case_start
        result['conditions'][name]=response;save(path,result)
    result['complete']=True;result['row_seconds']=time.time()-began;save(path,result)
    return result


def run(mode):
    B.verify();engine=Engine();selected=rows('train')[:4] if mode=='smoke' else [r for split in (['train','calibration'] if mode=='development' else ['test']) for r in rows(split)]
    if mode=='smoke':preflight(engine)
    else:assert read(ROOT/'preflight/completed.json')['manifest_sha256']==ident()
    locked=None
    if mode=='test':
        lock=read(ROOT/'selector.json');assert lock['manifest_sha256']==ident()
        assert sha(ROOT/'estimators.joblib')==lock['estimator_sha256'];locked=sha(ROOT/'selector.json')
    lookup={r['id']:r for r in rows()};start=time.time();recent=[]
    audit_ids={r['id'] for r in rows('calibration')[:CFG['perturb_generation_audit_count']]}
    for i,row in enumerate(selected,1):
        folder=ROOT/'smoke' if mode=='smoke' else ROOT/'evaluation'/row['split']
        conditions=CFG['conditions']+CFG['stress_conditions'] if mode=='test' else CFG['conditions']
        audit=mode=='smoke' or (row['split']=='calibration' and row['id'] in audit_ids)
        result=process(engine,row,lookup,folder,conditions,locked,audit)
        recent.append(result['row_seconds'])
        status(status='running',phase=mode,completed_questions=i,total_questions=len(selected),question_id=row['id'],
            elapsed_seconds=time.time()-start,mean_recent_question_seconds=sum(recent[-32:])/len(recent[-32:]))
        if i%8==0 or mode=='smoke':E.log(f'{mode.upper()} {i}/{len(selected)} row_sec={result["row_seconds"]:.1f} elapsed_min={(time.time()-start)/60:.1f}')
    save(ROOT/(mode+'_completed.json'),dict(manifest_sha256=ident(),n=len(selected),elapsed_seconds=time.time()-start))


if __name__=='__main__':
    try:run(sys.argv[1])
    except Exception as exc:
        save(ROOT/'failure.json',dict(action='collect_'+sys.argv[1],message=str(exc),traceback=traceback.format_exc(),time=time.strftime('%F %T')))
        status(status='failed',phase='collect_'+sys.argv[1],error=str(exc));raise
