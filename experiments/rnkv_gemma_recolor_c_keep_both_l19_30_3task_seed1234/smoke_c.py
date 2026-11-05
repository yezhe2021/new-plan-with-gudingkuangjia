"""Real Gemma/Qwen: RAW official transfer, exactly one k_norm, CPU-streamed objective parity and gradients."""
import dataclasses
from experiment_common import *
from engine_new import Engine,attention_objective
from calibrate_c import update

def main():
    verify()
    if done('smoke_C/completed.json'):return
    engine=Engine();checks=[]
    for dataset in ['Open-R1 prompt-only','HotpotQA distractor train full-context']:
        row=min([r for r in cal_rows('train') if r['dataset']==dataset],key=lambda r:len(r['content_ids']))
        base=engine.base(row)
        raw=C.Engine.base(engine,row);hit=engine.base(row)
        assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(base,raw,hit));del raw,hit
        module=fresh_c()
        with torch.no_grad():
            zero=engine.cache(row,base,None);engine.norm_counts=[0]*36;initial=engine.cache(row,base,module)
            assert engine.norm_counts==[1]*36;engine.norm_counts=None
            assert all(torch.equal(a,b) for l in range(36) for a,b in zip(C._cache_layer(zero,l),C._cache_layer(initial,l)))
            offsets,k,v=C.capture_sender(engine.sender_tok,engine.sender,engine.mapping,row['prompt'],device='cpu')
            official,length=C.build_receiver_cache(engine.tok,engine.model,engine.mapping,offsets,k,v,row['prompt'],engine.prefix_ids)
            assert length==len(engine.prefix_ids)+len(row['content_ids'])
            assert all(torch.equal(a,b) for l in range(36) for a,b in zip(C._cache_layer(zero,l),C._cache_layer(official,l)))
            del official,initial,offsets,k,v
            native=engine.native_cache(row);cached=engine.logits(row,C.clone_cache(native,engine.model))
            out=C.decoder(engine.model)(torch.tensor([engine.prefix_ids+row['content_ids']+row['suffix_ids']],device='cuda'),use_cache=False)
            direct=engine.model.lm_head(out.last_hidden_state[0,-1:]).float();parity=float(C.vocab_kl(direct,cached))
            assert abs(parity)<.001 and int(direct.argmax())==int(cached.argmax());del direct,cached,out,native,zero
        ex=engine.attention_example(row,base)
        with torch.no_grad():
            gpu=dataclasses.replace(ex,**{name:torch.stack(getattr(ex,name)).cuda() for name in ['native_query','native_attention','native_output','native_value']})
            native=C.attention_objective(engine,module,gpu);own=attention_objective(engine,module,ex)
            original=C.HC.example_objective(engine.model,module,gpu,engine.prefix_keys,engine.prefix_values,lambda_k_output=1.,backward=False)
            errors={k:abs(own[k]-original[k]) for k in ['routing_kl','k_post_wo_mse','post_wo_nmse']}
            assert all(abs(own[k]-native[k])<1e-6 for k in own)
            assert all(errors[k]<.002+.02*abs(original[k]) for k in errors),(own,original)
            del gpu
        opt=torch.optim.AdamW(module.parameters(),lr=3e-4,weight_decay=1e-4);scaler=T.scaler_for()
        metrics,outcome,retries=update(engine,module,opt,scaler,[row],1,0,scope='smoke_C');engine.unchanged()
        checks.append(dict(id=row['id'],official_raw_cache_exact=True,cache_hit_exact=True,zero_residual_exact=True,k_norm_once=True,native_cached_full_kl=parity,
            streamed_teacher_objective_exact=True,official_objective_absolute_error=errors,metrics=metrics,gradient=outcome,amp_retries=retries))
        del base,ex,module,opt,scaler
    # Largest calibration example checks full teacher probes and backward memory, without truncation.
    row=max(cal_rows('train'),key=lambda r:len(r['content_ids'])*len(r['probe_positions']))
    module=fresh_c();opt=torch.optim.AdamW(module.parameters(),lr=3e-4,weight_decay=1e-4);scaler=T.scaler_for()
    metrics,outcome,retries=update(engine,module,opt,scaler,[row],1,0,scope='smoke_C_long')
    mark('smoke_C/completed.json',status='completed',checks=checks,longest_attention_id=row['id'],longest_probes=len(row['probe_positions']),longest_metrics=metrics,
        longest_gradient=outcome,amp_retries=retries,disposable_weights=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
if __name__=='__main__':main()
