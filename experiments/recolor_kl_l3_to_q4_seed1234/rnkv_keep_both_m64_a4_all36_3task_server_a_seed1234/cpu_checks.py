"""Actual tiny Qwen: M64 positions, matched M32 initialization, loss/gradient parity."""
from types import SimpleNamespace
from experiment_common import *
from runtime_new import Memory64,losses,batches
from transformers import Qwen3Config,Qwen3ForCausalLM

def main():
    verify();torch.set_num_threads(1);torch.manual_seed(1234)
    cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,max_position_embeddings=256)
    cfg._attn_implementation='sdpa';llm=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    engine=SimpleNamespace(model=llm,prefix_ids=[2,3])
    engine.forward=lambda ids,cache,length,use_cache=True:llm.model(torch.tensor([ids]),past_key_values=cache,position_ids=torch.arange(length,length+len(ids))[None],attention_mask=torch.ones(1,length+len(ids),dtype=torch.long),use_cache=use_cache)
    module=Memory64(engine);assert module.M==64
    master=M.Memory(32,32,16,[6,12,18,24],1234)
    assert torch.equal(module.memory.prompt[:32],master.prompt) and all(torch.equal(a,b) for a,b in zip(module.memory.adapters.state_dict().values(),master.adapters.state_dict().values()))
    for l in range(1,37):
        allowed=module.routing.allowed('answer',l,2,7,64,torch.tensor([71,72]),73)
        assert allowed[:,:71].all() and allowed[0,71] and not allowed[0,72] and allowed[1,72]
    row=dict(id='m64_cpu_math',task='gsm8k',content_ids=[4,5,6,7,8],suffix_ids=[9,10],gold_tokens=[11,12,13,14]*20+[1])
    with torch.no_grad():base=llm.model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=True).past_key_values
    gradients=[];values=[]
    # Original baseline order: construct->CE->teacher->student KL.
    module.zero_grad(set_to_none=True);memory=module.construct(row,base);gold=module.token_logits(row,memory,row['gold_tokens'],rn=True)
    ce=torch.nn.functional.cross_entropy(gold,torch.tensor(row['gold_tokens']));positions=C.positions_for(row['gold_tokens'],row,1);prefix=row['gold_tokens'][:max(positions)]
    with torch.no_grad():teacher=module.logits(row,base,prefix,positions,rn=False)
    loss=ce+.5*C.vocab_kl(teacher,module.logits(row,memory,prefix,positions,rn=True));loss.backward()
    values.append(loss.detach());gradients.append(torch.cat([p.grad.flatten() for p in module.parameters()]).clone());del loss,ce,gold,memory
    for checkpointing in [False,True]:
        module.activation_checkpointing=checkpointing;module.zero_grad(set_to_none=True)
        loss,detail=losses(engine,module,row,base,base,'gold',1);loss.backward()
        assert detail['gold_tokens']==81 and detail['positions']==8 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        values.append(loss.detach());gradients.append(torch.cat([p.grad.flatten() for p in module.parameters()]).clone())
    errors=[float((g-gradients[0]).norm()/gradients[0].norm()) for g in gradients[1:]]
    assert max(errors)<1e-5 and all(torch.allclose(values[0],v,atol=1e-6) for v in values[1:])
    assert all(p.grad is None for p in llm.parameters());module.routing.close()
    for epoch in range(1,5):assert C.digest([[(r['id'],k) for r,k in b] for b in batches(epoch)])==C.read(ROOT/'manifest.json')['batch_schedules'][str(epoch)]
    assert not torch.cuda.is_initialized();mark('cpu_checks.json',status='completed',actual_tiny_qwen=True,memory_tokens=64,all36_masks_checked=True,suffix_start_offset=64,
        first32_prompt_and_adapter_initialization_matches_m32=True,canonical_loss_gradient_relative_errors=errors,full_math_gold_tokens=81,frozen_qwen_gradients_none=True)
    C.log('M64 CPU CHECKS PASSED',errors)
if __name__=='__main__':main()
