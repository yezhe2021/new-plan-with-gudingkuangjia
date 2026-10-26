"""Frozen existing Readers × native/translated KV on validation only."""
import json
import time
import experiment as E


@E.torch.no_grad()
def run():
    folder=E.ROOT/'reader_cache_diagnostic'
    if (folder/'completed.json').exists():
        assert json.loads((folder/'completed.json').read_text())['manifest_sha256']==E.identity()
        return
    tok,model,base,reader=E.R.load_all()
    frozen=E.R.base_digest(base)
    states={}
    for name,path,manifest in [
        ('old_reader',E.REFERENCE/'runs/reader/epoch4_step1024.pt',E.CFG['reference_reader_manifest_sha256']),
        ('task_reader',E.SUPERVISED/'runs/reader/epoch4_step1024.pt',E.CFG['supervised_manifest_sha256'])]:
        record=E.torch.load(path,map_location='cpu',weights_only=True)
        assert record['manifest_sha256']==manifest and record['step']==1024
        assert record['frozen_stage_a_sha256']==E.CFG['base_stage_a_sha256']
        assert record['frozen_stage_b_sha256']==E.CFG['frozen_stage_b_sha256']
        states[name]=record['state']
    rows=E.gsm_validation_rows(E.CFG['diagnostic_validation_n'])
    arms=[('native_no_reader',None,'native'),('native_old_reader','old_reader','native'),
          ('translated_old_reader','old_reader','translated'),('native_task_reader','task_reader','native'),
          ('translated_task_reader','task_reader','translated')]
    records=[]
    for i,row in enumerate(rows,1):
        native,k,v=E.R.fixed_state(model,base,reader,row)
        conditions={}
        for arm,state,cache_type in arms:
            path=folder/'per_arm'/arm/(E.U.fname(row)+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                assert record['manifest_sha256']==E.identity() and record['id']==row['id']
            else:
                if state:reader.load_state_dict(states[state])
                cache=E.U.clone_cache(native,model) if cache_type=='native' else E.U.translated_cache(model,row,native,k,v)
                tick=time.monotonic()
                with reader.mode(state is not None):
                    value=E.generate_value(model,tok,row,cache)
                E.mark(path,id=row['id'],gold=row['gold'],arm=arm,seconds=time.monotonic()-tick,**value)
                record=json.loads(path.read_text())
                del cache
            conditions[arm]=record
            E.status(status='running',phase='reader_cache_diagnostic',sample=i,total=len(rows),arm=arm)
            E.C.log(f'CACHE DIAGNOSTIC {i}/{len(rows)} {arm} correct={record["correct"]} tokens={record["generated_tokens"]}')
        records.append(dict(id=row['id'],gold=row['gold'],prompt=row['prompt'],conditions=conditions))
        E.C.write_jsonl(folder/'per_sample.jsonl',records)
        del native,k,v
    assert E.R.base_digest(base)==frozen
    metrics={arm:dict(n=len(rows),correct=sum(r['conditions'][arm]['correct'] for r in records),
                      accuracy=sum(r['conditions'][arm]['correct'] for r in records)/len(rows),
                      cap_count=sum(r['conditions'][arm]['reached_cap'] for r in records),
                      mean_generated_tokens=sum(r['conditions'][arm]['generated_tokens'] for r in records)/len(rows))
             for arm,_,_ in arms}
    paired={}
    for a,b in [('native_task_reader','native_old_reader'),('translated_task_reader','translated_old_reader'),
                ('native_old_reader','native_no_reader'),('native_task_reader','native_no_reader')]:
        vv=[(r['conditions'][a]['correct'],r['conditions'][b]['correct']) for r in records]
        paired[a+'_vs_'+b]=dict(both_correct=sum(x and y for x,y in vv),first_only=sum(x and not y for x,y in vv),
                              second_only=sum(not x and y for x,y in vv),both_wrong=sum(not x and not y for x,y in vv))
    E.mark(folder/'summary.json',status='completed',metrics=metrics,paired=paired,training=False,split='validation',
           caveat='Native body is prefetched with Reader disabled; Reader enabled only on suffix/continuation. This is an interface diagnostic, not a full native LoRA benchmark. Small fixed validation subset does not establish a unique causal mechanism.')
    E.mark(folder/'completed.json',status='completed')
    E.C.log('CACHE DIAGNOSTIC COMPLETED '+json.dumps(metrics))


if __name__=='__main__':
    try:
        E.verify()
        run()
    except Exception as exc:
        E.save(E.ROOT/'failure.json',dict(phase='reader_cache_diagnostic',type=type(exc).__name__,message=str(exc)))
        E.status(status='failed',phase='reader_cache_diagnostic',error=str(exc))
        raise
