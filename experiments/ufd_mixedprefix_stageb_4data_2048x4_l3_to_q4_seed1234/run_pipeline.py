import os,sys,json,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
def run(*args):
    print(f'[{time.strftime("%F %T")}] START {args}',flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),*args],cwd=ROOT,check=True)
with (ROOT/'pipeline.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run('prepare');run('capture','smoke');run('smoke');run('capture');run('train')
    for task in ['arc_challenge','openbookqa','hellaswag','mmlu_pro','sciq','gsm8k']:run('evaluate',task)
    run('summarize')
