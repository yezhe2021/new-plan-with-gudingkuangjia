"""Lossless FP16 frozen-group features: nine times smaller than full36-layer KV."""
import json
import os
import shutil
from pathlib import Path

def install(H,contract,phase):
    torch=H.torch;C=H.C;top=H.TOP;scope=C.digest(contract)
    cache_root=top/'source_latent_cache'/scope[:16];cache_root.mkdir(parents=True,exist_ok=True)
    original=H.Engine.base;stats=dict(hits=0,misses=0,created_bytes=0,roundtrip_checks=0)

    def report():
        C.save(top/'controls/cache_status.json',dict(parent_manifest_sha256=H.identity(),cache_scope=scope,arm=H.ARM,phase=phase,**stats))

    @torch.no_grad()
    def decode(engine,value):
        base=engine.translator.base;result=[];n=value['key'].shape[1]
        for start in range(0,n,64):
            parts=[]
            with torch.amp.autocast('cuda',dtype=torch.float16):
                for role,networks in [('key',base.k_outputs),('value',base.v_outputs)]:
                    layers=[]
                    for group in range(4):
                        hidden=value[role][group,start:start+64].cuda()[None]
                        for layer in range(base.target_bounds[group],base.target_bounds[group+1]):
                            layers.append(networks[layer](hidden).reshape(1,hidden.shape[1],8,128))
                    parts.append(torch.stack(layers,1))
                mapped=engine.translator.residual(*parts)
            result.append(tuple(x[0] for x in mapped))
        return tuple(torch.cat([v[j] for v in result],1) for j in range(2))

    @torch.no_grad()
    def cached(engine,row):
        assert all(not p.requires_grad for p in engine.translator.parameters()) and not engine.translator.training
        row_sha=C.digest(row);path=cache_root/(row_sha+'.pt')
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=True)
            assert value['scope']==scope and value['row_sha256']==row_sha and value['roundtrip_exact']
            assert all(value[k].dtype==torch.float16 and tuple(value[k].shape)==(4,len(row['content_ids'])-1,1024) for k in ['key','value'])
            result=decode(engine,value);stats['hits']+=1
            assert all(torch.isfinite(x).all() for x in result)
            if stats['hits']%8==1:report()
            return result
        collected={role:[[] for _ in range(4)] for role in ['key','value']};handles=[]
        for role,encoders in [('key',engine.translator.base.k_encoders),('value',engine.translator.base.v_encoders)]:
            for group,encoder in enumerate(encoders):
                def hook(module,args,out,role=role,group=group):
                    assert out.ndim==3 and out.shape[0]==1 and out.shape[-1]==1024 and out.dtype==torch.float16
                    collected[role][group].append(out[0].detach().cpu())
                handles.append(encoder.register_forward_hook(hook))
        try:fresh=original(engine,row)
        finally:
            for handle in handles:handle.remove()
        value={role:torch.stack([torch.cat(v,0) for v in groups]) for role,groups in collected.items()}
        value.update(scope=scope,row_sha256=row_sha,id=row['id'])
        restored=decode(engine,value)
        # Save only when reconstructing every K/V element is bitwise identical
        # to the untouched original function on this actual example.
        assert all(torch.equal(a,b) for a,b in zip(fresh,restored)),('Latent cache changes frozen KV',row['id'])
        value['roundtrip_exact']=True;stats['roundtrip_checks']+=1
        size=sum(value[k].numel()*value[k].element_size() for k in ['key','value'])
        assert shutil.disk_usage(top).free>contract['reserve_gib']*2**30+size,'Latent cache disk reserve reached'
        temp=path.with_suffix('.pt.tmp.'+str(os.getpid()));torch.save(value,temp);temp.replace(path)
        stats['misses']+=1;stats['created_bytes']+=path.stat().st_size
        if stats['misses']%8==1:report()
        return fresh

    H.Engine.base=cached
    return scope
