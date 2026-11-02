"""Small real Qwen3 on CPU checks the actual hooks/caches, without GPU use."""
from types import SimpleNamespace
from common import *
from model import RNKV,losses
import cache as K
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

def main():
    torch.set_num_threads(1);torch.manual_seed(1234)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    config._attn_implementation='sdpa';llm=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    class Tokenizer:
        def convert_tokens_to_ids(self,x):return 1
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3],tok=Tokenizer())
    def forward(ids,cache,length,use_cache=True):
        return llm.model(torch.tensor([ids]),past_key_values=cache,position_ids=torch.arange(length,length+len(ids))[None],
                         attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),use_cache=use_cache)
    engine.forward=forward
    cfg=dict(seed=1234,memory_tokens=4,adapter_rank=2,adapter_after_layers=[6,12,18,24],replace_layers=list(range(19,31)))
    module=RNKV(engine,cfg);row=dict(id='cpu_probe',content_ids=[4,5,6,7,8],suffix_ids=[9,10],gold_tokens=[11,12]);L=7
    with torch.no_grad():base=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
    before=[x.clone() for x in K.flat(base)];positions=[]
    def track(m,a,kw):
        if module.routing.state is not None:positions.append((module.routing.state[0],kw['position_ids'].clone()))
    handle=llm.model.register_forward_pre_hook(track,with_kwargs=True)
    captures={}
    def keyhook(m,a,out):
        if module.routing.state is not None and module.routing.state[0]=='construct':captures['key']=out.detach().clone()
    def valuehook(m,a,out):
        if module.routing.state is not None and module.routing.state[0]=='construct':captures['value']=out.detach().clone()
    hk=llm.model.layers[18].self_attn.k_norm.register_forward_hook(keyhook)
    hv=llm.model.layers[18].self_attn.v_proj.register_forward_hook(valuehook)
    with torch.no_grad():memory=module.construct(row,base)
    hk.remove();hv.remove()
    cos,sin=llm.model.rotary_emb(torch.zeros(1,4,32),torch.arange(L,L+4)[None])
    key=captures['key'].transpose(1,2)
    _,expected=apply_rotary_pos_emb(torch.zeros(1,4,4,8),key,cos,sin)
    assert torch.equal(K.pairs(memory)[18][0][:,:,L:],expected),'Memory key must be actual native post-norm/post-RoPE projection'
    expected_v=captures['value'].view(1,4,2,8).transpose(1,2)
    assert torch.equal(K.pairs(memory)[18][1][:,:,L:],expected_v)
    assert all(torch.equal(x,y) for x,y in zip(before,K.flat(base))),'Construction mutated translated cache'
    # Answer-source exclusion is a functional test, rather than only examining a mask array.
    with torch.no_grad():
        original=module.logits(row,memory,rn=True)
        changed=K.clone(memory,llm)
        for l,pair in enumerate(K.pairs(changed),1):
            start,end=(2,L) if l in cfg['replace_layers'] else (L,L+4)
            for x in pair:x[:,:,start:end,:]=torch.randn_like(x[:,:,start:end,:])*50
        blocked=module.logits(row,changed,rn=True)
        assert torch.equal(original,blocked),'A prohibited cache source changed answer logits'
        changed=K.clone(base,llm)
        for k,v in K.pairs(changed):v[:,:,2:,:]+=torch.randn_like(v[:,:,2:,:])*2
        other=module.construct(row,changed)
        assert not torch.equal(K.pairs(memory)[18][0],K.pairs(other)[18][0]),'Construction cannot read translated body'
    for phase,q in positions:
        assert int(q[0,0])==(L if phase=='construct' else L+4)
    # A post-block6 adapter affects memory K/V of layer7+, not layer6.
    with torch.no_grad():
        module.memory.adapters['6'][-1].weight.fill_(.05)
        other=module.construct(row,base)
        assert torch.equal(K.pairs(memory)[5][0],K.pairs(other)[5][0])
        assert not torch.equal(K.pairs(memory)[6][0],K.pairs(other)[6][0])
    vectors=[];out=[]
    for enabled in [False,True]:
        module.activation_checkpointing=enabled;module.zero_grad(set_to_none=True)
        cache=module.construct(row,base);logits=module.token_logits(row,cache,[11,12],rn=True)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([11,12]));loss.backward()
        pp=list(module.parameters());assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pp)
        assert float(module.memory.prompt.grad.norm())>0
        assert any(float(p.grad.norm())>0 for p in module.memory.adapters.parameters())
        vectors.append(torch.cat([p.grad.flatten() for p in pp]).clone());out.append(logits.detach())
    assert torch.allclose(out[0],out[1],atol=1e-6,rtol=1e-6)
    error=float((vectors[0]-vectors[1]).norm()/vectors[0].norm())
    assert error<1e-5,error
    # Exercise the two-branch task+KL graph sharing one differentiable memory
    # construction, not only a task-only surrogate loss.
    module.zero_grad(set_to_none=True)
    combined,detail=losses(engine,module,row,base,base,'gold',1)
    combined.backward()
    assert torch.isfinite(combined) and module.memory.prompt.grad.norm()>0
    assert abs(detail['loss']-(detail['task_loss']+.5*detail['full_kl']))<1e-5
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(p.grad is None for p in llm.parameters())
    for layer in [1,18,19,30,31,36]:
        q=torch.tensor([L+4,L+5]);a=module.routing.allowed('answer',layer,2,L,4,q,L+6)
        assert a[:,:2].all() and a[0,L+4] and not a[0,L+5] and a[1,L+5]
        assert bool(a[:,2:L].any())==(layer not in cfg['replace_layers'])
        assert bool(a[:,L:L+4].any())==(layer in cfg['replace_layers'])
    handle.remove();module.routing.close()
    assert not torch.cuda.is_initialized(),'CPU checks must not initialize CUDA while dependency is running'
    C.save(ROOT/'cpu_checks.json',dict(status='completed',device='cpu',native_projection_exact=True,post_block_adapter_timing=True,
        construction_reads_body=True,answer_inaccessible_sources_invariant=True,positions_correct=True,
        checkpoint_gradient_relative_error=error,task_plus_fullKL_shared_memory_gradient=True,frozen_llm_no_grad=True))
    print('RNKV CPU CHECKS PASSED',error,flush=True)

if __name__=='__main__':main()
