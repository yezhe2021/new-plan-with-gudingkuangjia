"""Gemma FP32 RAW pre-normalization Recolor; canonical Qwen cache assembly unchanged."""
import shutil
from experiment_common import *
from transformers import Gemma3ForConditionalGeneration
from heterofold.attention import apply_rope,apply_native_query_scale,rotary_cos_sin
from heterofold.capture import capture_native_qkv,capture_kv
from heterofold.alignment import receiver_to_sender_indices

def models():
    tok,model=C.load_model(CFG['receiver'],'cuda',dtype=torch.float16)
    st=C.AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True)
    sender=Gemma3ForConditionalGeneration.from_pretrained(CFG['sender'],dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    assert len(C.layers_of(sender))==34 and len(C.layers_of(model))==36
    return (st,sender),(tok,model)

class Engine(R.Engine):
    def __init__(self):
        (self.sender_tok,self.sender),(self.tok,self.model)=models()
        self.recolor_sha256=map_sha();self.base_map=C.load_map(ROOT/'maps/recolor.pt')
        assert self.base_map['source_space']=='kv_prerope_joint' and self.base_map['target_space']=='kv_pre_norm'
        assert self.base_map['Ls']==34 and self.base_map['Lr']==36 and self.base_map['kv_width']==1024
        self.builder=C.HC.ExampleBuilder((self.sender_tok,self.sender),(self.tok,self.model),self.base_map,'cuda');self.mapping=self.builder.mapping
        self.prefix_ids=C.read(ROOT/'manifest.json')['prefix_ids'];assert self.prefix_ids==self.builder.prefix_ids.tolist()
        with torch.no_grad():self.header=C.decoder(self.model)(torch.tensor([self.prefix_ids],device='cuda'),use_cache=True).past_key_values
        self.prefix_keys=[C._cache_layer(self.header,l)[0][0].transpose(0,1).detach() for l in range(36)]
        self.prefix_values=[C._cache_layer(self.header,l)[1][0].transpose(0,1).detach() for l in range(36)]
        self.snapshots=[C.versions(m) for m in [self.model,self.sender]]
        self.map_tensors=[x for role in 'kv' for name in ['A_'+role,'b_'+role] for x in self.mapping[name]];self.map_versions=[x._version for x in self.map_tensors]
        self.mode='train';self.student_test_native_prefills=0;self.norm_counts=None;self.norm_hooks=[]
        for l,block in enumerate(C.layers_of(self.model)):
            def count(module,args,index=l):
                if self.norm_counts is not None:self.norm_counts[index]+=1
            self.norm_hooks.append(block.self_attn.k_norm.register_forward_pre_hook(count))
        self.cache_root=ROOT/'base_cache'/self.recolor_sha256[:16];self.cache_root.mkdir(parents=True,exist_ok=True)
        self.cache_bytes=sum(f.stat().st_size for f in self.cache_root.glob('*.pt'));self.cache_hits=0;self.cache_misses=0

    @torch.no_grad()
    def compute_base(self,row):
        encoding=self.tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        assert encoding['input_ids']==row['content_ids']
        offsets,keys,values=capture_kv(self.sender_tok,self.sender,row['prompt'],self.builder.source_layers,device='cpu',dtype=self.sender.dtype)
        source=torch.tensor(receiver_to_sender_indices(offsets,encoding['offset_mapping']))
        klist=[];vlist=[]
        for l in range(36):
            out=[]
            for role,features in [('k',keys),('v',values)]:
                x=torch.cat([features[int(tap)].float()[source] for tap in self.mapping['depth_stencil']['taps_'+role][l]],-1).cuda()
                value=x/float(self.mapping['rms_'+role][l])@self.mapping['A_'+role][l]+self.mapping['b_'+role][l]
                out.append(value.to(self.model.dtype));del x,value
            klist.append(out[0]);vlist.append(out[1])
        return torch.stack(klist),torch.stack(vlist)

    @torch.no_grad()
    def base(self,row):
        key=C.digest(dict(prompt=row['prompt'],ids=row['content_ids']));path=self.cache_root/(key+'.pt')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['manifest_sha256']==identity() and saved['recolor_sha256']==self.recolor_sha256 and saved['key']==key
            k,v=(x.cuda() for x in saved['kv']);self.cache_hits+=1
        else:
            k,v=self.compute_base(row);self.cache_misses+=1;size=(k.numel()+v.numel())*2+8192
            if self.cache_bytes+size<=CFG['base_cache_budget_gib']*2**30 and shutil.disk_usage(ROOT).free-size>CFG['disk_reserve_gib']*2**30:
                saved=dict(manifest_sha256=identity(),recolor_sha256=self.recolor_sha256,key=key,kv=(k.cpu(),v.cpu()))
                C.atomic_save(saved,path);self.cache_bytes+=path.stat().st_size
                proof=torch.load(path,map_location='cuda',weights_only=True)
                assert torch.equal(proof['kv'][0],k) and torch.equal(proof['kv'][1],v),'Cache serialization changed outputs'
                del proof,saved
        assert k.shape==v.shape==(36,len(row['content_ids']),1024) and k.dtype==v.dtype==torch.float16
        assert torch.isfinite(k).all() and torch.isfinite(v).all();return k,v

    @torch.no_grad()
    def attention_example(self,row,base):
        # Same native teacher formulas and all literal-question probes as C;
        # only storage differs, one layer's dense attention is on GPU at a time.
        p=len(self.prefix_ids);probes=torch.tensor(row['probe_positions'],device='cuda')+p
        full=torch.tensor([self.prefix_ids+row['content_ids']],device='cuda');length=full.shape[1]
        captures=capture_native_qkv(self.model,full,probes);positions=torch.arange(length,device='cuda')
        qout=[];aout=[];oout=[];vout=[]
        for l,block in enumerate(C.layers_of(self.model)):
            captured=captures[l];attn=block.self_attn
            cosine,sine=rotary_cos_sin(self.model,positions,l,self.model.dtype)
            query=apply_rope(captured['q'],cosine.index_select(0,probes),sine.index_select(0,probes));query=apply_native_query_scale(attn,query,probes)
            key=apply_rope(captured['k'],cosine,sine);mask=C.causal_mask(attn,probes,length)
            prob=C.attention_probabilities(attn,query,key,mask,4)
            context=torch.einsum('hpt,thd->phd',prob,captured['v'].repeat_interleave(4,dim=1)).reshape(probes.numel(),4096)
            output=C.output_projection(attn)(context)
            qout.append(query.to(self.model.dtype).cpu());aout.append(prob.to(self.model.dtype).cpu())
            oout.append(output.to(self.model.dtype).cpu());vout.append(captured['v'].to(self.model.dtype).cpu())
            captures[l]=None;del captured,query,key,prob,context,output
        return C.HC.CalibrationExample(p,len(row['content_ids']),probes,base[0],base[1],qout,aout,oout,vout)

def attention_objective(engine,corrections,example,*,scaler=None,divisor=1):
    # Reuse exact canonical objective by lazily moving one teacher layer per access.
    class GPUViews:
        def __init__(self,values):self.values=values
        def __getitem__(self,l):return self.values[l].to('cuda')
    import dataclasses
    ex=dataclasses.replace(example,**{k:GPUViews(getattr(example,k)) for k in ['native_query','native_attention','native_output','native_value']})
    return C.attention_objective(engine,corrections,ex,scaler=scaler,divisor=divisor)
