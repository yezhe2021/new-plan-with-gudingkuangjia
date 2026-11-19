import torch
from token_alignment import align_tokens
from common import CFG


def render(tok,text,question_start,question_end):
    rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    start=rendered.index(text); end=start+len(text)
    enc=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
    positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and a<end and b>start]
    assert positions==list(range(positions[0],positions[-1]+1))
    n,stop=positions[0],positions[-1]+1
    offsets=[(max(0,a-start),min(len(text),b-start)) for a,b in enc['offset_mapping'][n:stop]]
    probes=[i for i,(a,b) in enumerate(enc['offset_mapping'][:stop]) if b>a and a<start+question_end and b>start+question_start]
    if not probes: raise ValueError('Question span has no tokens')
    return {'ids':enc['input_ids'][:stop],'head_ids':enc['input_ids'][:n],
        'tail_ids':enc['input_ids'][stop:],'offsets':offsets,'head':n,'probe_positions':probes}


def protocol(st,rt,row):
    a=render(st,row['text'],row['question_start'],row['question_end'])
    b=render(rt,row['text'],row['question_start'],row['question_end'])
    if CFG.get('calibration_probe_scope')=='all_body':
        b['probe_positions']=list(range(b['head'],len(b['ids'])))
    return {'source':a,'target':b,'mapping':align_tokens(a['offsets'],b['offsets'])}


@torch.no_grad()
def capture(model,ids,probe_positions=(),storage_dtype=torch.bfloat16):
    """Raw k_proj BEFORE key norm/RoPE; projected Value; native normalized Q."""
    keys,values,queries,outputs,handles={},{},{},{},[]
    for layer,block in enumerate(model.model.layers):
        def pre(module,args,kwargs,layer=layer):
            h=kwargs.get('hidden_states',args[0] if args else None)
            shape=(*h.shape[:-1],-1,module.head_dim)
            keys[layer]=module.k_proj(h).view(shape)[0].detach().to(storage_dtype).cpu()
            values[layer]=module.v_proj(h).view(shape)[0].detach().to(storage_dtype).cpu()
            if probe_positions:
                q=module.q_proj(h[:,probe_positions]).view(1,len(probe_positions),-1,module.head_dim)
                if hasattr(module,'q_norm'): q=module.q_norm(q)
                queries[layer]=q[0].float().cpu()
        def post(module,args,out,layer=layer):
            if probe_positions: outputs[layer]=out[0,probe_positions].float().cpu()
        handles.append(block.self_attn.register_forward_pre_hook(pre,with_kwargs=True))
        handles.append(block.self_attn.o_proj.register_forward_hook(post))
    try:
        t=torch.tensor([ids],device='cuda')
        model.model(input_ids=t,attention_mask=torch.ones_like(t),position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=False)
    finally:
        for h in handles: h.remove()
    return {'k':torch.stack(list(keys.values())),'v':torch.stack(list(values.values())),
        'q':queries,'o':outputs,'key_domain':'pre-k_norm pre-RoPE'}
