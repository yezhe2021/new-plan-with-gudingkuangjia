import torch
import torch.nn.functional as F
from runtime import make_cache,apply_rope,CFG


@torch.no_grad()
def collect(receiver,item,query):
    qid=query['question_id']
    if qid in item['probes']: return item['probes'][qid]
    prefix=len(item['protocol']['prefix_ids']); ids=query['suffix_ids']
    local=[i-prefix for i in query['query_positions']]
    result,handles={},[]
    for index,layer in enumerate(receiver.model.layers):
        def pre(module,args,kwargs,index=index):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            shape=(*hidden.shape[:-1],-1,module.head_dim)
            q=module.q_proj(hidden).view(shape)
            k=module.k_proj(hidden).view(shape)
            if hasattr(module,'q_norm'): q=module.q_norm(q)
            if hasattr(module,'k_norm'): k=module.k_norm(k)
            result[index]={'q':q[0,local].cpu(),'suffix_k':k[0].cpu(),
                           'suffix_v':module.v_proj(hidden).view(shape)[0].cpu()}
        def post(module,args,output,index=index): result[index]['actual_o']=output[0,local].detach().cpu()
        handles.append(layer.self_attn.register_forward_pre_hook(pre,with_kwargs=True))
        handles.append(layer.self_attn.o_proj.register_forward_hook(post))
    try:
        tensor=torch.tensor([ids],device='cuda',dtype=torch.long)
        cache=make_cache(receiver,item['native_k'].cuda(),item['native_v'].cuda(),torch.arange(prefix,device='cuda'))
        receiver.model(input_ids=tensor,past_key_values=cache,
            attention_mask=torch.ones((1,prefix+len(ids)),device='cuda',dtype=torch.long),
            position_ids=torch.arange(prefix,prefix+len(ids),device='cuda')[None],use_cache=False)
    finally:
        for h in handles: h.remove()
    assert len(result)==len(receiver.model.layers)
    probe={'layers':result,'positions':query['query_positions'],'prefix':prefix}
    item['probes'][qid]=probe
    return probe


def probability(q,k,positions,scale):
    # Input q=[Q,Hq,D], k=[T,Hkv,D], with native RoPE already applied.
    repeat=q.shape[1]//k.shape[1]
    scores=torch.einsum('qhd,thd->hqt',q.float(),k.float().repeat_interleave(repeat,dim=1))*scale
    mask=torch.arange(k.shape[0],device=k.device)[None,:]>positions[:,None]
    scores=scores.masked_fill(mask[None],float('-inf'))
    return scores.softmax(-1),scores.log_softmax(-1).masked_fill(mask[None],0)


def projected(p,v,weight):
    repeat=p.shape[0]//v.shape[1]
    o=torch.einsum('hqt,thd->qhd',p.float(),v.float().repeat_interleave(repeat,dim=1))
    return F.linear(o.flatten(1),weight.float())


def losses(receiver,item,probe,layer,bk,bv,adapter):
    attn=receiver.model.layers[layer].self_attn; native=probe['layers'][layer]
    positions=torch.tensor(probe['positions'],device='cuda',dtype=torch.long)
    q=apply_rope(receiver,native['q'].cuda(),positions).detach()
    nk=torch.cat((item['native_k'][layer],native['suffix_k']),0).cuda()
    nv=torch.cat((item['native_v'][layer],native['suffix_v']),0).cuda().detach()
    all_pos=torch.arange(nk.shape[0],device='cuda')
    n=len(item['protocol']['head_ids']); end=probe['prefix']
    nk=apply_rope(receiver,nk,all_pos).detach()
    teacher_p,_=probability(q,nk,positions,getattr(attn,'scaling',attn.head_dim**-0.5))
    weight=attn.o_proj.weight.detach()
    assert attn.o_proj.bias is None, 'Output bias requires explicit calibration handling'
    target=projected(teacher_p,nv,weight).detach()
    denominator=target.square().mean().clamp_min(1e-8)
    # Separate branches: K loss cannot reach V; V loss cannot reach K.
    with torch.amp.autocast('cuda',dtype=torch.float16):
        dk=torch.stack([adapter.k[layer][h](bk[:,h]) for h in range(adapter.heads)],1)
        dv=torch.stack([adapter.v[layer][h](bv[:,h]) for h in range(adapter.heads)],1)
    translated_k=bk+dk; translated_v=bv+dv
    body_k=apply_rope(receiver,translated_k,torch.arange(n,end,device='cuda'))
    student_k=torch.cat((nk[:n],body_k,nk[end:]),0)
    p,logp=probability(q,student_k,positions,getattr(attn,'scaling',attn.head_dim**-0.5))
    route=F.kl_div(logp,teacher_p,reduction='sum')/(p.shape[0]*p.shape[1])
    key_output=projected(p-teacher_p,nv,weight).square().mean()/denominator
    value_full=torch.cat((nv[:n],translated_v,nv[end:]),0)
    value_output=(projected(p.detach(),value_full,weight)-target).square().mean()/denominator
    actual=native['actual_o'].cuda().float()
    replay=(target-actual).square().mean()/actual.square().mean().clamp_min(1e-8)
    return route+CFG['lambda_output']*key_output,value_output,{
        'route_kl':float(route.detach()),'key_output_nmse':float(key_output.detach()),
        'value_output_nmse':float(value_output.detach()),'native_replay_nmse':float(replay.detach())}

def layer_metrics(receiver,item,probe,layer,bk,bv,adapter):
    # Same formulas as training, returned as separate route/output diagnostics.
    lk,lv,_=losses(receiver,item,probe,layer,bk,bv,adapter)
    return lk,lv
