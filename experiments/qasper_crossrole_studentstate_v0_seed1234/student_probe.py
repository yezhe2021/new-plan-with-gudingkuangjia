import torch
import torch.nn.functional as F
from common_exp import make_cache, apply_rope, probability, projected, CONFIG


@torch.no_grad()
def collect_student(receiver, item, query, bk, bv, calibrator):
    """Fresh actual Student states. Nothing enters the native probe cache."""
    prefix = len(item['protocol']['prefix_ids']); ids = query['suffix_ids']
    local = [i-prefix for i in query['query_positions']]
    with torch.amp.autocast('cuda',dtype=torch.float16):
        k,v,_,_ = calibrator(bk[None],bv[None])
    pk = torch.cat((item['wrapper_k'].cuda(),k[0]),1)
    pv = torch.cat((item['wrapper_v'].cuda(),v[0]),1)
    state,handles = {},[]
    for index,layer in enumerate(receiver.model.layers):
        def pre(module,args,kwargs,index=index):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            shape=(*hidden.shape[:-1],-1,module.head_dim)
            q=module.q_proj(hidden).view(shape); k=module.k_proj(hidden).view(shape)
            if hasattr(module,'q_norm'): q=module.q_norm(q)
            if hasattr(module,'k_norm'): k=module.k_norm(k)
            state[index]={'q':q[0,local].detach().cpu(), 'suffix_k':k[0].detach().cpu(),
                'suffix_v':module.v_proj(hidden).view(shape)[0].detach().cpu()}
        def post(module,args,output,index=index): state[index]['actual_o']=output[0,local].detach().cpu()
        handles.append(layer.self_attn.register_forward_pre_hook(pre,with_kwargs=True))
        handles.append(layer.self_attn.o_proj.register_forward_hook(post))
    try:
        tensor=torch.tensor([ids],device='cuda',dtype=torch.long)
        cache=make_cache(receiver,pk,pv,torch.arange(prefix,device='cuda'))
        receiver.model(input_ids=tensor,past_key_values=cache,
            attention_mask=torch.ones((1,prefix+len(ids)),device='cuda',dtype=torch.long),
            position_ids=torch.arange(prefix,prefix+len(ids),device='cuda')[None],use_cache=False)
    finally:
        for h in handles: h.remove()
    assert len(state)==36
    return {'layers':state,'positions':query['query_positions'],'prefix':prefix,'source':'fresh_student'}


def local_loss(receiver,item,probe,layer,bk,bv,calibrator):
    """Same detached query/suffix on both sides; substitute evidence ONLY."""
    attn=receiver.model.layers[layer].self_attn; state=probe['layers'][layer]
    positions=torch.tensor(probe['positions'],device='cuda')
    q=apply_rope(receiver,state['q'].cuda(),positions).detach()
    nk=torch.cat((item['native_k'][layer],state['suffix_k']),0).cuda().detach()
    nv=torch.cat((item['native_v'][layer],state['suffix_v']),0).cuda().detach()
    n=len(item['protocol']['head_ids']); end=probe['prefix']
    nk=apply_rope(receiver,nk,torch.arange(nk.shape[0],device='cuda')).detach()
    scale=getattr(attn,'scaling',attn.head_dim**-0.5)
    tp,_=probability(q,nk,positions,scale)
    weight=attn.o_proj.weight.detach()
    assert attn.o_proj.bias is None
    target=projected(tp,nv,weight).detach()
    with torch.amp.autocast('cuda',dtype=torch.float16):
        dk=torch.stack([calibrator.k[layer][h](bk[:,h]) for h in range(8)],1)
        dv=torch.stack([calibrator.v[layer][h](bv[:,h]) for h in range(8)],1)
    body_k=apply_rope(receiver,bk+dk,torch.arange(n,end,device='cuda'))
    student_k=torch.cat((nk[:n],body_k,nk[end:]),0)
    p,logp=probability(q,student_k,positions,scale)
    student_v=torch.cat((nv[:n],bv+dv,nv[end:]),0)
    route=F.kl_div(logp,tp,reduction='sum')/(p.shape[0]*p.shape[1])
    output=(projected(p,student_v,weight)-target).square().mean()/target.square().mean().clamp_min(1e-8)
    anchor=dk.float().square().mean()/bk.float().square().mean().clamp_min(1e-8)
    anchor+=dv.float().square().mean()/bv.float().square().mean().clamp_min(1e-8)
    loss=route+CONFIG['output_weight']*output+CONFIG['anchor_weight']*anchor
    replay=(projected(p.detach(),student_v.detach(),weight)-state['actual_o'].cuda().float()).square().mean()
    replay/=state['actual_o'].cuda().float().square().mean().clamp_min(1e-8)
    return loss,{'route_kl':float(route.detach()),'output_nmse':float(output.detach()),
        'anchor_nmse':float(anchor.detach()),'actual_student_replay_nmse':float(replay.detach())}
