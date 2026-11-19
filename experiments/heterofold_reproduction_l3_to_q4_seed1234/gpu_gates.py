"""Must pass before calibration; no threshold relaxation or native-evidence fallback."""
import gc
import torch
from common import CFG,ROOT,load_model,tokenizers,seed,save
from capture_kv import capture,protocol
from build_cache import build_cache
from inference import freeze,prefill,generate,digest
from calibration import objectives
from fold_mapper import Correction,apply_map,fold_correction
from build_cache import normalized_key,rope


def main():
    seed();st,rt=tokenizers()
    text='Context: Water freezes at zero degrees Celsius.\nQuestion: At what temperature does water freeze?'
    q='At what temperature does water freeze?'
    row={'text':text,'question':q,'question_start':text.index(q),'question_end':len(text)}
    t=protocol(st,rt,row)['target'];model=load_model('receiver');dtype=next(model.parameters()).dtype
    raw=capture(model,t['ids'],t['probe_positions'],storage_dtype=dtype)
    # Verify that the raw-domain recipe reconstructs native cache, including head.
    cache=build_cache(model,raw['k'].cuda(),raw['v'].cuda(),torch.arange(len(t['ids']),device='cuda'))
    state=freeze(cache);del cache;before=digest(state)
    native,nlog=generate(model,rt,t['ids']+t['tail_ids'],max_tokens=16)
    restored,rlog=generate(model,rt,t['tail_ids'],state,max_tokens=16)
    err=(nlog-rlog).abs();tol=1e-4 if dtype==torch.float32 else .02
    audit={'native_cache':{'generation_identical':native['ids']==restored['ids'],
        'first_argmax_identical':int(nlog.argmax())==int(rlog.argmax()),
        'first_logit_mae':float(err.mean()),'first_logit_max_error':float(err.max()),'mae_tolerance':tol},
        'immutable':before==digest(state),'key_domain':raw['key_domain'],'key_norm_count':1,'RoPE_count':1}
    save(ROOT/'gates/gpu_audit.json',audit)
    assert native['ids']==restored['ids'] and int(nlog.argmax())==int(rlog.argmax()) and err.mean()<=tol,audit
    assert before==digest(state)
    n=t['head'];layer=18;k=raw['k'][layer,n:].flatten(1).cuda().float();v=raw['v'][layer,n:].flatten(1).cuda().float()
    kc,vc=Correction().cuda(),Correction().cuda()
    lk,lv,metrics=objectives(model,layer,raw,t,k,v,kc,vc)
    # Native inputs should replay teacher outputs; zero is an implementation gate, not a performance score.
    assert abs(float(lk))<1e-4 and float(lv)<1e-4,metrics
    lk.backward();assert kc.G.grad is not None and vc.G.grad is None
    kc.zero_grad(set_to_none=True)
    lk,lv,_=objectives(model,layer,raw,t,k*.9,v*.9,kc,vc)
    lv.backward();assert kc.G.grad is None and vc.G.grad is not None
    assert all(p.grad is None for p in model.parameters())
    audit['gradient_isolation']=True;audit['native_attention_replay']=metrics
    # Verify folded affine parity after receiver K norm/RoPE, not just algebra on CPU.
    x=torch.randn(16,3072,device='cuda')
    mapping={'A':torch.randn(3072,1024)*.01,'b':torch.randn(1024)*.01,'rms':torch.tensor(1.)}
    correction=Correction().cuda();correction.G.data.normal_(std=.001)
    unf=correction(apply_map(mapping,x)).reshape(16,8,128)
    folded=apply_map(fold_correction(mapping,correction),x).reshape(16,8,128)
    positions=torch.arange(16,device='cuda')
    a=rope(model,normalized_key(model,layer,unf),positions)
    b=rope(model,normalized_key(model,layer,folded),positions)
    mae=float((a-b).abs().mean());threshold=1e-4 if dtype==torch.float32 else .01
    assert mae<=threshold,{'folded_post_norm_rope_mae':mae,'threshold':threshold}
    audit['folded_post_norm_rope_mae']=mae
    save(ROOT/'gates/gpu_completed.json',audit)
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
