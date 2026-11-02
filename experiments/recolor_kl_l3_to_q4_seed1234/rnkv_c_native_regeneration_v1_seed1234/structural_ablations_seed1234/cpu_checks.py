"""Tiny actual Qwen3 validates both structural changes, without CUDA."""
from types import SimpleNamespace
from ablation_common import *
from ablation_runtime import AblationModel,losses
from transformers import Qwen3Config,Qwen3ForCausalLM

def main():
    torch.set_num_threads(1);torch.manual_seed(1234)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    config._attn_implementation='sdpa';llm=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3],tok=SimpleNamespace(convert_tokens_to_ids=lambda _:1))
    engine.forward=lambda ids,cache,length,use_cache=True:llm.model(torch.tensor([ids]),past_key_values=cache,
        position_ids=torch.arange(length,length+len(ids))[None],attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),use_cache=use_cache)
    cfg=dict(seed=1234,memory_tokens=4,adapter_rank=2,adapter_after_layers=[6,12,18,24],replace_layers=list(range(19,31)))
    row=dict(id='cpu_probe',content_ids=[4,5,6,7,8],suffix_ids=[9,10],gold_tokens=[11,12]);L=7
    with torch.no_grad():base=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
    reference=M.RNKV(engine,cfg);initial_prompt=reference.memory.prompt.detach().clone()
    with torch.no_grad():reference_memory=reference.construct(row,base)
    reference.routing.close();checks={}
    for arm in CFG['arms']:
        model=AblationModel(engine,arm,cfg)
        assert torch.equal(model.memory.prompt,initial_prompt)
        if arm=='memory_only':
            assert not model.memory.adapters and list(dict(model.named_parameters()))==['memory.prompt']
            with torch.no_grad():memory=model.construct(row,base)
            assert all(torch.equal(a,b) for a,b in zip(K.flat(memory),K.flat(reference_memory)))
        else:
            for layer in [1,18,19,30,31,36]:
                q=torch.tensor([L+4,L+5]);allowed=model.routing.allowed('answer',layer,2,L,4,q,L+6)
                assert allowed[:,:L].all(),'All body/header positions must remain readable'
                assert bool(allowed[:,L:L+4].all())==(layer in cfg['replace_layers'])
                assert allowed[0,L+4] and not allowed[0,L+5]
        gradients=[];outputs=[]
        for enabled in [False,True]:
            model.activation_checkpointing=enabled;model.zero_grad(set_to_none=True)
            loss,detail=losses(engine,model,row,base,base,'gold',1);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert model.memory.prompt.grad.norm()>0
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone());outputs.append(loss.detach())
        error=float((gradients[0]-gradients[1]).norm()/gradients[0].norm())
        assert torch.allclose(outputs[0],outputs[1],atol=1e-6,rtol=1e-6) and error<1e-5
        assert all(p.grad is None for p in llm.parameters())
        checks[arm]=dict(gradient_checkpoint_relative_error=error,same_initial_prompt=True,frozen_qwen=True)
        model.routing.close()
    assert not torch.cuda.is_initialized()
    C.save(ROOT/'cpu_checks.json',dict(status='completed',checks=checks,device='cpu'))
    print('STRUCTURAL CPU CHECKS PASSED',checks,flush=True)

if __name__=='__main__':main()
