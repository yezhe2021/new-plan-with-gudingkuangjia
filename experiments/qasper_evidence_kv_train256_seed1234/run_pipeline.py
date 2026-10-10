import subprocess
import sys
from runtime import ROOT,CFG


def run(script,*args):
    print('START',script,*args,flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,check=True)


if __name__=='__main__':
    run('tests_cpu.py')
    if not (ROOT/'prepared.json').exists(): run('prepare_qasper.py')
    if not (ROOT/'smoke/native_precision_completed.json').exists(): run('audit_native_precision.py')
    if not (ROOT/'smoke/completed.json').exists(): run('tests.py')
    for initialization in CFG['stage_a_initializations']:
        if not (ROOT/f'runs/stage_a_{initialization}/completed.json').exists():
            run('train_stage_a.py','--initialization',initialization)
    if not (ROOT/'runs/stage_b/completed.json').exists(): run('train_stage_b.py')
    if not (ROOT/'evaluation/summary.json').exists(): run('evaluate.py')
    print('ALL SMALL-SCALE EXPERIMENTS COMPLETED',flush=True)
