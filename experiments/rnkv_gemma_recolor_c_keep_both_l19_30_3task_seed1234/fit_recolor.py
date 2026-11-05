"""Original HeteroFold moments/fit, RAW projection taps, corpus balancing; resumable every50 texts."""
import gc
from experiment_common import *
from engine_new import models
from heterofold.recolor import RoleStats,FIELDS,balance_corpora,fit_map
from heterofold.alignment import stencil_taps,proportional_layers,exact_end_pairs
from heterofold.capture import capture_kv
from heterofold.modeling import kv_geometry

@torch.no_grad()
def collect(sender,receiver,rr,tag,smoke=False):
    st,sm=sender;rt,rm=receiver;ls,hs,ds=kv_geometry(sm);lr,hr,dr=kv_geometry(rm)
    taps=stencil_taps(ls,lr,CFG['recolor']['span'],CFG['recolor']['stride']);assert ls==34 and lr==36 and hs*ds==hr*dr==1024
    roles={role:RoleStats(3072,1024,lr,'cuda') for role in 'kv'};aligned=0;start=0
    path=ROOT/f'maps/statistics_{tag}.pt'
    if not smoke and path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True);assert old['manifest_sha256']==identity() and old['ids']==[r['id'] for r in rr]
        start=old['texts'];aligned=old['N']
        for role in 'kv':
            for field in FIELDS:
                for a,b in zip(getattr(roles[role],field),old['roles'][role][field]):a.copy_(b)
        del old
    def snapshot(count):
        return dict(manifest_sha256=identity(),ids=[r['id'] for r in rr],roles={role:stats.cpu() for role,stats in roles.items()},N=aligned,texts=count,
            pi=proportional_layers(ls,lr),taps=taps,Ls=ls,Lr=lr,source_kv_width=1024,kv_width=1024,stencil_source_width=3072,span=1,stride=4)
    for i,row in enumerate(rr):
        if i<start:continue
        if paused():
            if not smoke:C.atomic_save(snapshot(i),path)
            raise SystemExit(75)
        # Capture into host RAM to keep long FP32 Gemma projection traces off GPU.
        so,sk,sv=capture_kv(st,sm,row['prompt'],{l for tt in taps for l in tt},device='cpu',dtype=sm.dtype)
        ro,rk,rv=capture_kv(rt,rm,row['prompt'],range(lr),device='cpu',dtype=rm.dtype)
        pairs=exact_end_pairs(so,ro);assert len(pairs)>=8
        si=torch.tensor([p[0] for p in pairs]);ri=torch.tensor([p[1] for p in pairs])
        for l in range(lr):
            for role,sf,rf in [('k',sk,rk),('v',sv,rv)]:
                x=torch.cat([sf[t][si] for t in taps[l]],-1).cuda().float();y=rf[l][ri].cuda().float()
                assert torch.isfinite(x).all() and torch.isfinite(y).all();roles[role].add(l,x,y);del x,y
        aligned+=len(pairs);del sk,sv,rk,rv
        status(status='running',phase='recolor_statistics',corpus=tag,sample=i+1,total=len(rr),aligned=aligned)
        if (i+1)%50==0 or i+1==len(rr):
            if not smoke:C.atomic_save(snapshot(i+1),path)
            C.log('GEMMA RECOLOR STATS',tag,i+1,len(rr),aligned)
    result=snapshot(len(rr));del roles;torch.cuda.empty_cache();return result

def main():
    verify();assert done('cpu_checks.json')
    if done('maps/completed.json'):map_sha();return
    if sys.argv[-1]=='check':
        sender,receiver=models();row=max(cal_rows('train'),key=lambda r:len(r['content_ids']))
        s=collect(sender,receiver,[row],'longest_disposable',smoke=True)
        assert all(torch.isfinite(t).all().item() for role in 'kv' for field in FIELDS for t in s['roles'][role][field]),'Nonfinite Recolor moments'
        mark('maps/smoke.json',status='completed',longest_id=row['id'],all36_moments_finite=all(torch.isfinite(t).all().item() for role in 'kv' for field in FIELDS for t in s['roles'][role][field]),
            source_dtype=str(sender[1].dtype),source_layers=34,target_layers=36,source_capture='pre-k_norm/pre-RoPE RAW',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        return
    assert done('maps/smoke.json');sender,receiver=models();stats={}
    for tag,label in [('openr1','Open-R1 prompt-only'),('hotpot','HotpotQA distractor train full-context')]:
        rr=[r for r in cal_rows('train') if r['dataset']==label];assert len(rr)==800
        stats[label]=collect(sender,receiver,rr,tag)
    del sender,receiver;gc.collect();torch.cuda.empty_cache()
    balanced=balance_corpora(stats);del stats
    result=fit_map(balanced,device='cuda',log=C.log)
    result['sender_model']=CFG['sender'];result['receiver_model']=CFG['receiver'];result['experiment_manifest_sha256']=identity()
    metadata=C.read(CFG['calibration']);result['calibration'].update(train_prompts=1600,calibration_id='openr1_hotpot_n2000',prompt_sha256_sequence=metadata['prompt_sha256_sequence'])
    assert all(torch.isfinite(t).all() for role in 'kv' for name in ['A_'+role,'b_'+role] for t in result[name])
    C.atomic_save(result,ROOT/'maps/recolor.pt')
    mark('maps/completed.json',status='completed',recolor_sha256=C.sha(ROOT/'maps/recolor.pt'),geometry={k:result[k] for k in ['Ls','Lr','source_kv_width','kv_width']},calibration=result['calibration'],finished_at=time.strftime('%F %T'))
    C.log('GEMMA RECOLOR FIT COMPLETED')
if __name__=='__main__':main()
