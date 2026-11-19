"""Cache frozen dual-translator outputs to task-specific RAM disk, without changing parent artifacts."""
from runtime import *
from capture_source import load_gemma

@torch.inference_mode()
def capture(mode):
    em=verify()
    rr=[]
    for split in ['train','validation']:
        selected=rows(split)
        if mode=='smoke':
            seen=set();chosen=[]
            for x in selected:
                task=x['gemma_row']['task']
                if task not in seen:chosen.append(x);seen.add(task)
            selected=chosen
        rr+=selected
    todo=[]
    for x in rr:
        p=features_path(x)
        if not p.exists():todo.append(x)
        else:
            value=torch.load(p,map_location='cpu',weights_only=True)
            assert value['manifest_sha256']==identity() and value['row_sha256']==feature_digest(x)
    if not todo:
        mark('features_'+mode+'_completed.json',status='completed',n=len(rr));return
    assert torch.cuda.mem_get_info()[0]/2**30>=22
    lfront=E.load_front('llama',em);gfront=E.load_front('gemma',em)
    _,gemma=load_gemma(torch.float32)
    for i,item in enumerate(todo,1):
        row=item['gemma_row'];assert shutil.disk_usage(CACHE).free/2**30>CFG['feature_cache_reserve_gib']
        ls=torch.load(item['llama_source_path'],map_location='cpu',weights_only=True)
        assert ls['manifest_sha256']==item['llama_source_identity']
        if 'id' in ls:assert ls['id']==row['id']
        assert ls['key'].shape==ls['value'].shape==(28,item['llama_source_length'],8,128)
        ids=row['frames']['gemma']['body_ids'];k,v,logits=C.capture(gemma,ids,len(ids),[0])
        assert torch.isfinite(k).all() and torch.isfinite(v).all() and torch.isfinite(logits).all()
        ix=row['source_indices'];gs=dict(key=k[:,ix].half().contiguous(),value=v[:,ix].half().contiguous())
        assert torch.isfinite(gs['key']).all() and torch.isfinite(gs['value']).all()
        lk,lv=E.map_source(lfront,ls);gk,gv=E.map_source(gfront,gs)
        assert lk.shape==lv.shape==gk.shape==gv.shape==(36,len(ix),8,128)
        value=dict(lk=lk.cpu(),lv=lv.cpu(),gk=gk.cpu(),gv=gv.cpu(),id=row['id'],manifest_sha256=identity(),row_sha256=feature_digest(item),feature_space='Frozen Stage-A+residual Stage-B raw Qwen pre-RoPE',dtype='float16')
        assert all(torch.isfinite(value[x]).all() for x in ['lk','lv','gk','gv'])
        C.save_tensor(features_path(item),value)
        status(status='running',phase='frozen_feature_capture',sample=i,total=len(todo),id=row['id'],cache=str(CACHE))
        if i%16==0 or mode=='smoke' or i==len(todo):C.log(f'Frozen feature capture {mode} {i}/{len(todo)} {row["id"]}')
        del ls,gs,k,v,logits,lk,lv,gk,gv,value
    mark('features_'+mode+'_completed.json',status='completed',n=len(rr))

if __name__=='__main__':capture(sys.argv[1])
