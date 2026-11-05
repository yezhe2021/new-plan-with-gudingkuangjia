"""Actual tiny Qwen: full math target gradient, causal all-layer routing and resume schedules."""
from types import SimpleNamespace
from experiment_common import *
from runtime_new import GemmaMemory,losses,batches
from transformers import Qwen3Config,Qwen3ForCausalLM

def main():
    verify();torch.set_num_threads(1);torch.manual_seed(1234)
    cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    cfg._attn_implementation='sdpa';llm=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3])
    engine.forward=lambda ids,cache,length,use_cache=True:llm.model(torch.tensor([ids]),past_key_values=cache,position_ids=torch.arange(length,length+len(ids))[None],attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),use_cache=use_cache)
    model=GemmaMemory(engine,dict(seed=1234,memory_tokens=4,adapter_rank=2,adapter_after_layers=[6,12,18,24]))
    row=dict(id='math_cpu_probe',task='gsm8k',content_ids=[4,5,6,7,8],suffix_ids=[9,10],gold_tokens=[11,12,13,14]*20+[1])
    with torch.no_grad():base=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
    for layer in range(1,37):
        allowed=model.routing.allowed('answer',layer,2,7,4,torch.tensor([11,12]),13)
        assert allowed[:,:7].all() and bool(allowed[:,7:11].all())==(layer in CFG['replace_layers'])
        assert allowed[0,11] and not allowed[0,12] and allowed[1,12]
    gradients=[];out=[]
    for enabled in [False,True]:
        model.activation_checkpointing=enabled;model.zero_grad(set_to_none=True)
        loss,detail=losses(engine,model,row,base,base,'gold',1);loss.backward()
        assert detail['gold_tokens']==81 and detail['positions']==8 and detail['prefix_tokens']==81
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()) and model.memory.prompt.grad.norm()>0
        gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone());out.append(loss.detach())
    error=float((gradients[0]-gradients[1]).norm()/gradients[0].norm());assert error<1e-5 and torch.allclose(out[0],out[1])
    assert all(p.grad is None for p in llm.parameters());model.routing.close()
    schedules={}
    for epoch in range(1,5):
        bb=batches(epoch);assert len(bb)==384
        flat=[r for batch in bb for r,kind in batch];assert {t:sum(r['task']==t for r in flat) for t in CFG['train_counts']}==CFG['train_counts']
        assert C.digest([[(r['id'],k) for r,k in b] for b in bb])==C.read(ROOT/'manifest.json')['batch_schedules'][str(epoch)]
        schedules[str(epoch)]=len(bb)
    assert not torch.cuda.is_initialized()
    mark('cpu_checks.json',status='completed',actual_tiny_qwen=True,all36_masks_checked=True,memory_visible_layers=CFG['replace_layers'],full_math_gold_tokens=81,checkpoint_gradient_relative_error=error,
        all_frozen_qwen_gradients_none=True,batches_per_epoch=schedules)
    C.log('ALL36 THREE-TASK CPU CHECKS PASSED',error)

if __name__=='__main__':main()
