"""Pause via existing control at epoch3 checkpoint, evaluate, then resume full state."""
import fcntl,json,os,subprocess,sys,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
TOKEN='user_requested_drt_stage2_epoch3_eval_20261110'
def read(path):return json.loads(path.read_text())
def save(**v):
    path=HERE/'controller_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(time=time.strftime('%F %T'),pid=os.getpid(),**v),ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def main():
    with (HERE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        flag=ROOT/'controls/pause_requested.json';cp=ROOT/'runs/stage2/epoch3_step1152.pt'
        save(status='waiting_epoch3',target_step=1152)
        while not cp.exists():
            state=read(ROOT/'runner_status.json')
            if state['status']=='failed':raise RuntimeError('Original pipeline failed before epoch3')
            if flag.exists():raise RuntimeError('An independent pause request exists; do not override')
            time.sleep(2)
        # Atomic final checkpoint exists before epoch3 validation/epoch4. Preserve
        # its inode so later resuming/saving the same epoch cannot replace it.
        if not (HERE/'checkpoint.pt').exists():os.link(cp,HERE/'checkpoint.pt')
        if flag.exists():assert read(flag).get('owner')==TOKEN,'Independent pause must not be removed'
        else:
            tmp=HERE/'pause_to_install.json';tmp.write_text(json.dumps(dict(owner=TOKEN,reason='Evaluate epoch3 then resume epoch4',time=time.strftime('%F %T')))+'\n')
            os.link(tmp,flag);tmp.unlink()
        save(status='waiting_training_exit',checkpoint=str(cp))
        with (ROOT/'pipeline.lock').open('a') as pipeline:
            while True:
                try:fcntl.flock(pipeline,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(2)
            state=read(ROOT/'runner_status.json');assert state['status']=='paused',state
            # Inspect complete state on CPU only after training exits; no later
            # accepted update may precede this diagnostic.
            import torch
            last=torch.load(ROOT/'runs/stage2/last.pt',map_location='cpu',weights_only=True)
            assert (last['epoch'],last['step'],last['batch'])==(3,1152,384)
            assert 'optimizer' in last and 'scaler' in last and 'cuda_rng' in last
            del last
            save(status='evaluating_epoch3',step=1152)
            subprocess.run([sys.executable,'-u',str(HERE/'evaluate_epoch3.py')],cwd=ROOT,check=True)
            assert read(HERE/'completed.json')['status']=='completed'
            assert read(flag).get('owner')==TOKEN;flag.unlink()
        # Existing trainer restores optimizer, scaler, RNG and step1152. It
        # completes pending epoch3 validation and continues epoch4, then runs
        # the original fixed-final evaluation. No training hyperparameter edits.
        with (ROOT/'logs/pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'run','train'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(status='evaluation_completed_training_resumed',resume_pid=child.pid,step=1152)

if __name__=='__main__':
    try:main()
    except Exception as error:save(status='failed',error=str(error));raise
