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
    run('prepare');run('capture');run('translate')
    for arm in ['baselines']+CFG['arms']:run('smoke',arm)
    for arm in ['baselines']+CFG['arms']:run('evaluate',arm)
    run('summarize')
