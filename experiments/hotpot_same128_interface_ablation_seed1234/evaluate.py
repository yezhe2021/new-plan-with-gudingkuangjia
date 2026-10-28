"""First reproduce the original interface, then run ordered matched ablations."""
import copy,time,traceback
from common import ROOT,CFG,P,H,U,C,torch,read,save,sha,identity,rows,status,verify


@torch.inference_mode()
def generate_original(model,tok,row,cache,evidence=False):
    suffix=row['continuation_ids'] if evidence else row['suffix_ids']
    text,count=C.R.generate(C.runtime(model,tok),torch.tensor([suffix],device='cuda'),CFG['max_new_tokens'],
        past_key_values=cache,past_length=len(row['frames']['qwen']['body_ids']),temperature=0.,seed=CFG['seed'])
    prediction=H.clean_prediction(text)
    return dict(prediction=prediction,generation=text,generated_tokens=count,reached_cap=int(count>=CFG['max_new_tokens']))


@torch.inference_mode()
def generate_diagnostic(model,tok,row,cache):
    # Same greedy token loop as the prior evidence diagnostic; score with common cleaning.
    base=len(row['frames']['qwen']['body_ids']);ids=row['continuation_ids']
    def forward(ids,memory,base):
        return model.model(torch.tensor([ids],device='cuda'),past_key_values=memory,
            attention_mask=torch.ones(1,base+len(ids),device='cuda',dtype=torch.long),
            position_ids=torch.arange(base,base+len(ids),device='cuda')[None],use_cache=True)
    out=forward(ids,cache,base);total=base+len(ids);memory=out.past_key_values
    logits=model.lm_head(out.last_hidden_state[0,-1]).float()
    eos=model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos])
    stops.update([tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')]);stops.discard(None)
    tokens=[]
    for i in range(CFG['max_new_tokens']):
        token=int(logits.argmax());tokens.append(token)
        if token in stops:break
        if i+1<CFG['max_new_tokens']:
            out=forward([token],memory,total);total+=1;memory=out.past_key_values
            logits=model.lm_head(out.last_hidden_state[0,-1]).float()
    text=tok.decode(tokens,skip_special_tokens=False)
    return dict(prediction=H.clean_prediction(text),generation=text,generated_tokens=len(tokens),
        reached_cap=int(len(tokens)>=CFG['max_new_tokens']),prior_diagnostic_cleaning=tok.decode(tokens,skip_special_tokens=True).strip())


def score(value,answer):return dict(value,**H.answer_metrics(value['prediction'],answer))


@torch.inference_mode()
def evaluate_arm(model,tok,sender,translator,row,answer,evidence=False,decoder_bridge=False):
    native=U.native_body(model,row)
    native_result=score(generate_original(model,tok,row,U.clone_cache(native,model),evidence),answer)
    source=P.capture_pair(sender,model,row);k,v=P.map_source(translator,source)
    cache=U.translated_cache(model,row,native,k,v)
    mapped_result=score(generate_original(model,tok,row,cache,evidence),answer)
    results=dict(native=native_result,translated=mapped_result)
    bridge=None
    if decoder_bridge:
        cache=U.translated_cache(model,row,native,k,v)
        bridge=dict(native=score(generate_diagnostic(model,tok,row,U.clone_cache(native,model)),answer),
                    translated=score(generate_diagnostic(model,tok,row,cache),answer))
    return results,bridge


def result_path(row):return ROOT/'evaluation'/(row['id']+'.json')


@torch.inference_mode()
def preflight(model,tok,selected):
    checks=[]
    for item in selected[:2]:
        c=item['variants']['C_selector_prompt'];d=item['variants']['D_evidence_only']
        logits=[]
        for r,suffix in [(c,c['suffix_ids']),(d,d['continuation_ids'])]:
            cache=U.native_body(model,r);base=len(r['frames']['qwen']['body_ids'])
            out=model.model(torch.tensor([suffix],device='cuda'),past_key_values=cache,
                attention_mask=torch.ones(1,base+len(suffix),device='cuda',dtype=torch.long),
                position_ids=torch.arange(base,base+len(suffix),device='cuda')[None],use_cache=False)
            logits.append(model.lm_head(out.last_hidden_state[0,-1]).float())
        diff=float((logits[0]-logits[1]).abs().max());assert diff<.3,diff
        assert int(logits[0].argmax())==int(logits[1].argmax())
        checks.append(dict(id=item['id'],native_C_D_logit_max_error=diff,first_token_equal=True))
    save('preflight/completed.json',dict(manifest_sha256=identity(),checks=checks))


@torch.inference_mode()
def run():
    verify();began=time.time();selected=rows()
    tok,model,sender=P.models();translator=P.fresh_b().eval()
    m=read(ROOT/'manifest.json');state=torch.load(m['checkpoints']['stage_b']['path'],map_location='cpu',weights_only=True)
    assert state['manifest_sha256']==P.identity() and state['frozen_stage_a_sha256']==m['checkpoints']['stage_a']['sha256']
    translator.residual.load_state_dict(state['state'],strict=True);del state
    translator.requires_grad_(False)
    assert all(not p.requires_grad for module in [model,sender,translator] for p in module.parameters())
    # Reproduce all original128 before any interface ablation.
    mismatch=[];recent=[]
    for i,row in enumerate(selected,1):
        path=result_path(row);start=time.time()
        result=read(path) if path.exists() else dict(id=row['id'],source_id=row['source_id'],question=row['question'],
            answer=row['answer'],manifest_sha256=identity(),conditions={},complete=False)
        assert result['manifest_sha256']==identity()
        if 'A_original' not in result['conditions']:
            result['conditions']['A_original']=evaluate_arm(model,tok,sender,translator,row['variants']['A_original'],row['answer'])[0]
            result['original_seconds']=time.time()-start;save(path,result)
        for arm,oldarm in [('native','qwen_native'),('translated','hotpot_stage_ab')]:
            a=result['conditions']['A_original'][arm];b=row['historical'][oldarm]
            if a['prediction']!=b['prediction'] or a['em']!=b['em'] or abs(a['f1']-b['f1'])>1e-12:
                mismatch.append(dict(id=row['id'],arm=arm,current=a,historical=b))
        recent.append(result['original_seconds']);mean=sum(recent[-16:])/len(recent[-16:])
        status(status='running',phase='reproduction',completed_questions=i,total_questions=len(selected),
            elapsed_seconds=time.time()-began,mean_recent_question_seconds=mean,phase_eta_seconds=(len(selected)-i)*mean)
        if i==1 or i%8==0:print(time.strftime('%F %T'),f'REPRODUCE {i}/{len(selected)} mean_sec={mean:.1f} mismatches={len(mismatch)}',flush=True)
    save('reproduction.json',dict(manifest_sha256=identity(),n=len(selected),mismatches=mismatch,passed=not mismatch))
    assert not mismatch,'Original-interface predictions differ from history; inspect reproduction.json before continuing'
    preflight(model,tok,selected);recent=[];stage_start=time.time()
    for i,row in enumerate(selected,1):
        path=result_path(row);result=read(path);start=time.time()
        for name in ['B_support_only','C_selector_prompt','D_evidence_only']:
            if name in result['conditions'] and (name!='D_evidence_only' or 'E_diagnostic_decoder' in result['conditions']):continue
            status(status='running',phase='ablations',completed_questions=i-1,total_questions=len(selected),condition=name)
            outputs,bridge=evaluate_arm(model,tok,sender,translator,row['variants'][name],row['answer'],
                evidence=name=='D_evidence_only',decoder_bridge=name=='D_evidence_only')
            result['conditions'][name]=outputs
            if bridge is not None:result['conditions']['E_diagnostic_decoder']=bridge
            save(path,result)
        result.update(complete=True,ablation_seconds=time.time()-start);save(path,result)
        recent.append(result['ablation_seconds']);mean=sum(recent[-16:])/len(recent[-16:])
        status(status='running',phase='ablations',completed_questions=i,total_questions=len(selected),
            elapsed_seconds=time.time()-began,mean_recent_question_seconds=mean,phase_eta_seconds=(len(selected)-i)*mean)
        if i==1 or i%8==0:print(time.strftime('%F %T'),f'ABLATIONS {i}/{len(selected)} mean_sec={mean:.1f} eta_min={(len(selected)-i)*mean/60:.1f}',flush=True)
    save('evaluation_completed.json',dict(manifest_sha256=identity(),n=len(selected),elapsed_seconds=time.time()-began,
        ablation_seconds=time.time()-stage_start))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        save('failure.json',dict(message=str(exc),traceback=traceback.format_exc(),time=time.strftime('%F %T')))
        status(status='failed',error=str(exc));raise
