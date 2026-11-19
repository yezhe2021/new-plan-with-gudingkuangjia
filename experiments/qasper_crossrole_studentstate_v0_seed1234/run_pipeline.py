import os
import subprocess
import sys
from common_exp import ROOT, save


def run(script,*args):
    print('PIPELINE START',script,*args,flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,check=True)
    print('PIPELINE FINISH',script,*args,flush=True)


if __name__=='__main__':
    import fcntl
    lock=(ROOT/'.pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run('prepare.py')
    if not (ROOT/'smoke/completed.json').exists(): run('tests.py')
    run('train_a.py','self'); run('train_a.py','cross'); run('report.py','A')
    run('train_b.py','native_matched'); run('train_b.py','student'); run('report.py','B')
    save(ROOT/'completed.json',{'completed':True,'scope':'independent V0 mechanisms; no automatic combination or rollout training'})
    print('ALL CROSS-ROLE AND STUDENT-STATE V0 EXPERIMENTS COMPLETED',flush=True)
