"""Execution-only override: GPU saved tensors and native KV, same scientific code/checkpoints."""
from contextlib import nullcontext
from pathlib import Path
import runpy
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiment_common as E
import engine_offload as EO
import runtime_new as RT
C=E.C;torch=E.torch
ORIGINALS=dict(base=EO.Engine.base,cache=EO.Engine.cache,native_cache=EO.Engine.native_cache,offload_context=RT.offload_context,status=E.status)

@torch.no_grad()
def base_gpu(self,row):return tuple(x.to('cuda') for x in E.saved_base(row))
def resident_status(**v):return ORIGINALS['status'](execution_mode='gpu_resident',**v)

def install(enabled=True):
    EO.Engine.base=base_gpu if enabled else ORIGINALS['base']
    EO.Engine.cache=C.Engine.cache if enabled else ORIGINALS['cache']
    EO.Engine.native_cache=E.R.Engine.native_cache if enabled else ORIGINALS['native_cache']
    RT.offload_context=(lambda enabled=True:nullcontext()) if enabled else ORIGINALS['offload_context']
    E.status=resident_status if enabled else ORIGINALS['status']

if __name__=='__main__':
    E.verify();record=C.read(ROOT/'controls/use_gpu_resident.json')
    assert record['manifest_sha256']==E.identity() and record['entry_sha256']==C.sha(Path(__file__))
    install(True);action=sys.argv[1];args=sys.argv[2:];sys.argv=[str(ROOT/action),*args]
    print(time.strftime('%F %T'),'GPU_RESIDENT EXECUTION',action,'CPU saved-tensor and native-KV offload disabled; keep checkpointing',flush=True)
    try:runpy.run_path(str(ROOT/action),run_name='__main__')
    except torch.cuda.OutOfMemoryError as error:
        E.mark('diagnostics/last_gpu_oom.json',status='waiting_for_more_vram',action=action,args=args,time=time.strftime('%F %T'),execution_mode='gpu_resident',message=str(error),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        traceback.print_exc();raise SystemExit(76)
