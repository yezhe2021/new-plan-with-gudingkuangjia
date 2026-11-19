"""Detached complete capture/train/evaluation pipeline with failure gates."""
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def save(name,obj):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def launch():
    from runtime import prepare,verify
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():raise RuntimeError('Already launched; inspect existing state before resuming')
        prepare();verify();(ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs/pipeline.log').open('a') as out:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        info=dict(pid=process.pid,started_at=time.strftime('%F %T'),detached=True);save('launch.json',info);print(json.dumps(info))
def run():
    from runtime import verify,done
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify()
        if (ROOT/'failure.json').exists():raise RuntimeError('Previous failure requires review')
        actions=[['-m','unittest','-v','test_projection'],['features.py','trainval'],['train.py','smoke'],['train.py','train'],['evaluate.py']]
        try:
            for action in actions:
                save('runner_status.json',dict(status='running',pid=os.getpid(),action=action,time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START',action,flush=True)
                args=[sys.executable,'-u',*action] if action[0]=='-m' else [sys.executable,'-u',str(ROOT/action[0]),*action[1:]]
                subprocess.run(args,cwd=ROOT,check=True)
        except Exception as e:
            save('failure.json',dict(type=type(e).__name__,message=str(e),action=action));save('runner_status.json',dict(status='failed',pid=os.getpid(),action=action));raise
        assert done('completed.json');save('runner_status.json',dict(status='completed',pid=os.getpid(),completed_at=time.strftime('%F %T')))
if __name__=='__main__':
    if sys.argv[1]=='--launch':launch()
    else:run()
