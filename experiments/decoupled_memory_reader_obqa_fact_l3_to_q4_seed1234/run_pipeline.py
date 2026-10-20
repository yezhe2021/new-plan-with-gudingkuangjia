import os,sys,json,time,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
def run(*args):
    print(f'[{time.strftime("%F %T")}] START {args}',flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),*args],cwd=ROOT,check=True)
with (ROOT/'pipeline.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=Path(CFG['prerequisite'])
    if not p.exists():raise RuntimeError('Previous Attention-Output has not completed; do not overlap GPU work')
    run('prepare');run('capture');run('translate');run('smoke')
    # Native/unchanged Writer controls, followed by complete Reader-only arms.
    run('evaluate','baselines')
    for arm in CFG['arms']:
        run('train',arm);run('evaluate',arm)
    run('summarize')
