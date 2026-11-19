import argparse
import torch
from common_exp import ROOT, CONFIG, BASELINE, save, prepared, setup, old_base, adapter, load_state, mapped, collect
from cross_role import RoleResidual, AugmentedMapper
from train_stage_a import validate as reconstruction_validate
from evaluate import run_eval
from student_probe import local_loss


def main(stage):
    folder=ROOT/f'evaluation/{stage}'
    if (folder/'completed.json').exists(): return
    sender,receiver,rt,cache=setup(); data=prepared(); frozen=old_base()
    conditions={'A0_original512':(frozen,None)}
    if stage=='A':
        for mode in CONFIG['stage_a_modes']:
            residual=RoleResidual(mode,CONFIG['cross_role_rank']).cuda().eval()
            load_state(residual,ROOT/f'runs/A_{mode}/best.pt')
            conditions[f'A_{mode}']=(AugmentedMapper(frozen,residual).eval(),None)
    else:
        conditions['B0_original_nativeQ']=(frozen,adapter(BASELINE/'runs/stage_b/best.pt').eval())
        for mode in CONFIG['stage_b_modes']:
            cal=adapter().eval(); load_state(cal,ROOT/f'runs/B_{mode}/best.pt')
            conditions[f'B_{mode}']=(frozen,cal)
    torch.cuda.reset_peak_memory_stats()
    metrics=run_eval(cache,receiver,rt,data['dev'],conditions,folder/'fixed_dev')
    save(folder/'summary.json',metrics)
    if stage=='A':
        reconstruction={name:reconstruction_validate(mapper,cache,data['validation']) for name,(mapper,_) in conditions.items()}
        save(folder/'internal_reconstruction.json',reconstruction)
    # Native-query diagnostics, identical across conditions; not used for selection.
    diagnostics={}
    with torch.no_grad():
        zero=adapter().eval()
        for name,(mapper,cal) in conditions.items():
            values=[]
            for row in data['dev']:
                item=cache.get(row); bk,bv=mapped(mapper,item)
                for query in item['protocol']['queries']:
                    probe=collect(receiver,item,query)
                    for layer in range(36):
                        _,m=local_loss(receiver,item,probe,layer,bk[layer],bv[layer],cal or zero)
                        values.append(m)
            diagnostics[name]={k:sum(x[k] for x in values)/len(values) for k in values[0]}
    save(folder/'attention_diagnostics.json',diagnostics)
    save(folder/'completed.json',{'stage':stage,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'dev_report_only':True,'test_evaluated':False,'conditions':list(conditions)})
    print('FINAL REPORT COMPLETE',stage,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=('A','B')); main(p.parse_args().stage)
