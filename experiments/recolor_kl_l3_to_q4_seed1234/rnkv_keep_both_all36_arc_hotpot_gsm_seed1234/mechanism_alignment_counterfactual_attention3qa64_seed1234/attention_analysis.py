"""Fixed-query direct effects, fixed nonmemory states, real trajectories and group interventions."""
from common_diag import *
from heterofold.attention import causal_mask,softcap_logits,output_projection

def cosine(a,b):return float(torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=-1).mean())
def rel(a,b):return D.relative_l2(a,b)

@torch.no_grad()
def local(engine,row,native_trace,keep_trace,translated,native,memory):
    L=len(engine.prefix_ids)+len(row['content_ids']);results=[]
    for source,tr in [('native_shifted_q',native_trace),('keep_q',keep_trace)]:
        for layer in range(36):
            paused();att=C.layers_of(engine.model)[layer].self_attn
            # Exclude EOS from mechanism probes. All background suffix/gold KV
            # come from ONE trace and stay identical between the local cases.
            q=tr.q[layer][:,:-1];qpos=tr.qpos[:-1];refk,refv=C._cache_layer(tr.cache,layer)
            fullk=refk.clone();fullv=refv.clone();tk,tv=C._cache_layer(translated,layer);nk,nv=C._cache_layer(native,layer)
            fullk[:,:,:L]=tk;fullv[:,:,:L]=tv
            mk,mv=memory[layer];fullk[:,:,L:L+32]=mk;fullv[:,:,L:L+32]=mv
            nativek=fullk.clone();nativev=fullv.clone();nativek[:,:,:L]=nk;nativev[:,:,:L]=nv
            allowed=causal_mask(att,qpos,fullk.shape[2]);masked=allowed.clone();masked[:,L:L+32]=False
            def probabilities(k,mask):
                logits=torch.einsum('hpd,htd->hpt',q.float(),k[0].repeat_interleave(4,0).float())*float(att.scaling)
                return softcap_logits(att,logits).masked_fill(~mask[None],float('-inf')).softmax(-1)
            def context(p,v):return torch.einsum('hpt,htd->phd',p,v[0].repeat_interleave(4,0).float())
            def project(x):return output_projection(att)(x.reshape(x.shape[0],4096).to(engine.model.dtype)).float()
            pc=probabilities(fullk,masked);pk=probabilities(fullk,allowed);pn=probabilities(nativek,masked)
            oc=context(pc,fullv);ok=context(pk,fullv);on=context(pn,nativev)
            mass=pk[:,:,L:L+32].sum(-1).transpose(0,1) # probe x head
            cm=torch.einsum('hpm,hmd->phd',pk[:,:,L:L+32],fullv[0,:,L:L+32].repeat_interleave(4,0).float())
            expected=(1-mass[...,None])*oc+cm
            error=rel(expected,ok);assert error<2e-5,('Softmax decomposition',source,layer,error)
            # This is the output with unchanged memory K but zero Memory V:
            # only background dilution remains. No additional forward needed.
            diluted=(1-mass[...,None])*oc
            wc,wk,wn=project(oc),project(ok),project(on)
            recon=rel(wn,native_trace.o[layer][:-1].float()) if source=='native_shifted_q' else rel(wk,keep_trace.o[layer][:-1].float())
            assert recon<.03,('Actual attention reconstruction',source,layer,recon)
            # With fixed Q/nonmemory K, all background probabilities change by
            # the same multiplier. Verify this without storing attention maps.
            bg=masked[None].expand_as(pc);factor=(1-mass).transpose(0,1)[:,:,None]
            route_error=float((pk-pc*factor).masked_select(bg).abs().max());assert route_error<2e-5
            results.append(dict(query_source=source,layer=layer+1,memory_mass=float(mass.mean()),head_memory_mass=mass.mean(0).cpu().tolist(),
                wo_cos_C_to_native=cosine(wc,wn),wo_cos_keep_to_native=cosine(wk,wn),wo_nmse_C=rel(wc,wn)**2,wo_nmse_keep=rel(wk,wn)**2,
                wo_change_relative=rel(wk,wc),dilution_only_change_relative=rel(project(diluted),wc),memory_value_contribution_relative=rel(project(cm),wc),
                softmax_decomposition_error=error,background_relative_routing_error=route_error,actual_output_reconstruction_error=recon,
                actual_keep_vs_native_output_cosine=cosine(keep_trace.o[layer][:-1],native_trace.o[layer][:-1]),
                actual_query_difference=rel(keep_trace.q[layer][:,:-1],native_trace.q[layer][:,:-1])))
    return results
