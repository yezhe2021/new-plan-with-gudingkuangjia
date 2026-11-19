"""Frozen checkpoint diagnostic; no selector fitting or stability probes."""
import sys,time,traceback
import backend as B
from backend import ROOT,CFG,E,read,save,ident,rows,status,torch
from engine import Engine


def preflight(engine,selected):
    checks=[]
    for row in selected[:2]:
        own=engine.mapped(row);cache=engine.cache(row,own);blocked=engine.reject_columns(row)
        out,_=engine.prompt(row,cache,blocked);masked=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        other_cache=engine.cache(row,engine.variant(row,'gaussian',own,own))
        out,_=engine.prompt(row,other_cache,blocked);other=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        error=float((masked-other).abs().max());assert error==0,error
        native=engine.native(row['frames']['qwen']['body_ids']);out,_=engine.prompt(row,native)
        cached=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        out=engine.forward(row['frames']['qwen']['body_ids']+row['continuation_ids'],use_cache=False)
        full=engine.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        diff=float((cached-full).abs().max());assert diff<.3,diff
        assert int(cached.argmax())==int(full.argmax())
        checks.append(dict(id=row['id'],masked_content_logit_error=error,native_cache_full_max_error=diff))
    assert all(not p.requires_grad for model in [*engine.models.values(),engine.translator] for p in model.parameters())
    save(ROOT/'preflight/completed.json',dict(manifest_sha256=ident(),checks=checks,models_frozen=True))


def run():
    B.verify();start=time.time();engine=Engine();selected=rows();lookup={r['id']:r for r in selected}
    preflight(engine,selected);recent=[]
    for i,row in enumerate(selected,1):
        path=ROOT/'evaluation'/f"{row['id']}.json";began=time.time()
        result=read(path) if path.exists() else dict(id=row['id'],question=row['question'],answer=row['answer'],
            donor_id=row['donor_id'],manifest_sha256=ident(),baselines={},conditions={},complete=False)
        assert result['manifest_sha256']==ident()
        if not result['complete']:
            own=engine.mapped(row);donor=engine.mapped(lookup[row['donor_id']]);own_cache=engine.cache(row,own)
            for name in ['reject','compact','native']:
                if name in result['baselines']:continue
                if name=='compact':cache=None;blocked=[];compact=True
                elif name=='native':cache=engine.native(row['frames']['qwen']['body_ids']);blocked=[];compact=False
                else:cache=own_cache;blocked=engine.reject_columns(row);compact=False
                generated=engine.generate(row,cache,blocked,compact)
                result['baselines'][name]=engine.annotate(row,generated,cache,blocked,compact);save(path,result)
            for name in CFG['conditions']:
                if name in result['conditions']:continue
                status(status='running',phase='diagnostic',completed_questions=i-1,total_questions=len(selected),condition=name,question_id=row['id'])
                cache=engine.cache(row,engine.variant(row,name,own,donor));generated=engine.generate(row,cache)
                accepted=engine.annotate(row,generated,cache);rejected=result['baselines']['reject']
                result['conditions'][name]=dict(accept=accepted,utility_em=accepted['em']-rejected['em'],
                    utility_f1=accepted['f1']-rejected['f1']);save(path,result)
            result.update(complete=True,row_seconds=time.time()-began);save(path,result)
        recent.append(result['row_seconds'])
        elapsed=time.time()-start;mean=sum(recent[-16:])/len(recent[-16:])
        status(status='running',phase='diagnostic',completed_questions=i,total_questions=len(selected),
               elapsed_seconds=elapsed,mean_recent_question_seconds=mean,eta_seconds=(len(selected)-i)*mean)
        if i%8==0 or i==1:print(time.strftime('%F %T'),f'PROGRESS {i}/{len(selected)} mean_sec={mean:.2f} eta_min={(len(selected)-i)*mean/60:.1f}',flush=True)
    save(ROOT/'evaluation_completed.json',dict(manifest_sha256=ident(),n=len(selected),elapsed_seconds=time.time()-start))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        save(ROOT/'failure.json',dict(message=str(exc),traceback=traceback.format_exc(),time=time.strftime('%F %T')))
        status(status='failed',error=str(exc));raise
