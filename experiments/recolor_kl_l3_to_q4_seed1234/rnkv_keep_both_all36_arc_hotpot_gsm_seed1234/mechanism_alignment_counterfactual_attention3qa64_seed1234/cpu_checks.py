"""Actual tiny Qwen tests Memory masking/null construction and RoPE roundtrip."""
from types import SimpleNamespace
from common_diag import *
from interventions import *
from transformers import Qwen3Config,Qwen3ForCausalLM

def main():
    verify();torch.manual_seed(1234);torch.set_num_threads(1)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    config._attn_implementation='sdpa';llm=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3]);engine.forward=lambda ids,cache,length,use_cache=True:llm.model(torch.tensor([ids]),past_key_values=cache,attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),position_ids=torch.arange(length,length+len(ids))[None],use_cache=use_cache)
    cfg=dict(seed=1234,memory_tokens=32,adapter_rank=2,adapter_after_layers=[6,12,18,24]);module=RT.AllLayerMemory(engine,cfg).eval().requires_grad_(False)
    row=dict(id='cpu',content_ids=[4,5,6,7,8],suffix_ids=[9,10]);tokens=[11,12,1]
    with torch.no_grad():
        base=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
        keep=module.construct(row,base)
        with access(module,header_only=True):null=module.construct(row,base)
        assert any(not torch.equal(a[:,:,7:],b[:,:,7:]) for p,q in zip(K.pairs(keep),K.pairs(null)) for a,b in zip(p,q))
        for p,q in zip(K.pairs(keep),K.pairs(base)):
            for a,b in zip(p,q):assert torch.equal(a[:,:,:7],b)
        changed=K.clone(keep,llm,detach=True)
        for pair in K.pairs(changed):
            for tensor in pair:tensor[:,:,7:39]=torch.randn_like(tensor[:,:,7:39])
        assert torch.equal(logits(module,row,keep,tokens,True,'masked'),logits(module,row,changed,tokens,True,'masked'))
        mem=memory_pairs(keep,7);moved=donor_memory(engine,mem,7,19);back=donor_memory(engine,moved,19,7)
        assert all(torch.equal(a[1],b[1]) and D.relative_l2(a[0],b[0])<.003 for a,b in zip(mem,back))
        same=donor_memory(engine,mem,7,7);assert all(torch.equal(a,b) for p,q in zip(mem,same) for a,b in zip(p,q))
        assert torch.isfinite(logits(module,row,padded(engine,base),tokens,True,'masked')).all()
    assert all(p.grad is None for p in llm.parameters()) and not torch.cuda.is_initialized()
    mark('cpu_checks.json',status='completed',actual_tiny_qwen=True,masked_memory_invariant=True,null_reads_no_body=True,body_unchanged=True,rope_roundtrip=True,self_rephase_exact=True)
    C.log('MECHANISM CPU CHECKS PASSED')

if __name__=='__main__':main()
