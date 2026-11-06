"""Bound GPU allocation and return76 on OOM so the queue can release CUDA and wait."""
import runpy
import sys
import traceback
from experiment_common import *

if __name__=='__main__':
    action=sys.argv[1];args=sys.argv[2:];sys.argv=[str(ROOT/action),*args]
    try:runpy.run_path(str(ROOT/action),run_name='__main__')
    except torch.cuda.OutOfMemoryError as error:
        mark('diagnostics/last_gpu_oom.json',status='waiting_for_more_vram',action=action,args=args,time=time.strftime('%F %T'),message=str(error),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        traceback.print_exc();raise SystemExit(76)
