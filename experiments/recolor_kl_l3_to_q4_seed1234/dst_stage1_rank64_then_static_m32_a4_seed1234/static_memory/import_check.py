"""Check all legacy scorer dependencies on CPU before loading GPU models."""
from experiment_common import *
from score import score
import metrics
import qa_metrics

if __name__=='__main__':
    verify()
    assert Path(metrics.__file__).resolve()==(C.ROOT/'metrics.py').resolve()
    assert Path(qa_metrics.__file__).resolve()==(S.ORIGINAL/'cross_dataset5x64_seed1234/qa_metrics.py').resolve()
    assert callable(score) and not torch.cuda.is_initialized()
    mark('import_check.json',status='completed',scorer_dependencies_resolved=True,cuda_initialized=False)
    C.log('DST PLUS STATIC SCORER IMPORTS PASSED')
