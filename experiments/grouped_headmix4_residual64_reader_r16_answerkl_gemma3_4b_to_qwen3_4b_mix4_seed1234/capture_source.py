"""Gemma native forward; capture post-k_norm/pre-RoPE K and unrotated V."""
from common import *
from transformers import Gemma3ForConditionalGeneration,AutoTokenizer
import shutil,gc

def load_gemma(dtype):
    tok=AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True)
    model=Gemma3ForConditionalGeneration.from_pretrained(CFG['sender'],dtype=dtype,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    config=C.text_config(model)
    assert (config.num_hidden_layers,config.num_key_value_heads,config.head_dim)==(34,4,256)
    return tok,model

@torch.no_grad()
def audit_half():
    tok,model=load_gemma(torch.float16);checks=[]
    rr=C.read_jsonl(OLD_U_ROOT/'prepared/train.jsonl')
    for task in U.CFG['train_counts']:
        row=max((r for r in rr if r['task']==task),key=lambda r:len(r['prompt']))
        ids=tok.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        k,v,logits=C.capture(model,ids,len(ids),[0])
        checks.append(dict(task=task,id=row['id'],tokens=len(ids),finite_k=bool(torch.isfinite(k).all()),finite_v=bool(torch.isfinite(v).all()),finite_logits=bool(torch.isfinite(logits).all())))
        print(checks[-1],flush=True)
    passed=all(r[k] for r in checks for k in ['finite_k','finite_v','finite_logits'])
    C.save(ROOT/'gemma_fp16_audit.json',dict(passed=passed,checks=checks))
    print('FP16_AUDIT',passed,flush=True)

@torch.no_grad()
def capture(mode):
    verify()
    splits=['test'] if mode=='test' else ['train','validation']
    rr=[r for s in splits for r in rows(s,mode=='smoke')]
    todo=[r for r in rr if not source_path(r).exists()]
    if not todo:
        for r in rr:loaded_source(r)
        return
    _,model=load_gemma(getattr(torch,CFG['source_compute_dtype']))
    if mode=='smoke':
        # Validate all 34 layers, including both local/global attention, against native physical KV.
        row=min(todo,key=lambda r:len(r['frames']['gemma']['body_ids']))
        ids=row['frames']['gemma']['body_ids'];assert len(ids)<1024
        k,v,_=C.capture(model,ids,len(ids),[0])
        tensor=torch.tensor([ids],device='cuda')
        native=C.backbone(model)(input_ids=tensor,use_cache=True).past_key_values
        checks=[]
        for i in range(34):
            nk,nv=U._cache_layer(native,i)
            rotated=C.apply_rope(model,k[i].cuda(),torch.arange(len(ids),device='cuda'),i).transpose(0,1)[None]
            rawv=v[i].cuda().transpose(0,1)[None]
            torch.testing.assert_close(rotated,nk,rtol=1e-5,atol=1e-5)
            torch.testing.assert_close(rawv,nv,rtol=1e-5,atol=1e-5)
            checks.append(dict(layer=i,type=C.text_config(model).layer_types[i],key_max_error=float((rotated-nk).abs().max()),value_max_error=float((rawv-nv).abs().max())))
        mark(ROOT/'smoke/sender_interface.json',status='completed',checks=checks,dtype=CFG['source_compute_dtype'])
        del native,k,v,rotated,rawv,nk,nv
    for i,r in enumerate(todo,1):
        assert shutil.disk_usage(ROOT).free/2**30>CFG['disk_reserve_gib']
        ids=r['frames']['gemma']['body_ids'];k,v,logits=C.capture(model,ids,len(ids),[0])
        assert torch.isfinite(k).all() and torch.isfinite(v).all() and torch.isfinite(logits).all(),('Nonfinite Gemma',r['id'])
        ix=r['source_indices'];k=k[:,ix].to(torch.float16).contiguous();v=v[:,ix].to(torch.float16).contiguous()
        assert torch.isfinite(k).all() and torch.isfinite(v).all(),('FP16 storage overflow',r['id'])
        C.save_tensor(source_path(r),dict(key=k,value=v,id=r['id'],row_sha256=C.digest(r),manifest_sha256=identity(),source_compute_dtype=CFG['source_compute_dtype']))
        status(status='running',phase='gemma_capture_'+mode,sample=i,total=len(todo),id=r['id'])
        if i%16==0 or mode=='smoke' or i==len(todo):C.log(f'Gemma capture {mode} {i}/{len(todo)} {r["id"]}')
    mark(ROOT/f'source_{mode}_completed.json',status='completed',n=len(rr))

if __name__=='__main__':
    if sys.argv[1]=='audit-half':audit_half()
    else:capture(sys.argv[1])
