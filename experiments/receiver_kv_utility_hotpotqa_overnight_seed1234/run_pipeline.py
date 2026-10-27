"""Detached, sequential, resumable overnight experiment. No assistant/API calls."""
import os,sys,json,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def save(name,obj):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)

def run():
 with (ROOT/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save('runner_status.json',dict(status='running',pid=os.getpid(),started_at=time.strftime('%F %T')))
  try:
   phases=[('smoke_completed.json',['collect.py','smoke']),('development_completed.json',['collect.py','development']),('selector.json',['estimators.py']),('test_completed.json',['collect.py','test']),('completed.json',['report.py'])]
   for marker,args in phases:
    if (ROOT/marker).exists():continue
    print(time.strftime('%F %T'),'START',args,flush=True)
    save('runner_status.json',dict(status='running',pid=os.getpid(),phase=args,time=time.strftime('%F %T')))
    subprocess.run([sys.executable,'-u']+[str(ROOT/args[0])]+args[1:],cwd=ROOT,check=True)
   save('runner_status.json',dict(status='completed',pid=os.getpid(),completed_at=time.strftime('%F %T')))
  except Exception as exc:
   save('runner_status.json',dict(status='failed',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),time=time.strftime('%F %T')))
   raise

def launch():
 with (ROOT/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (ROOT/'launch.json').exists():
   old=json.loads((ROOT/'launch.json').read_text());p=Path('/proc')/str(old['pid'])
   if p.exists() and str(ROOT).encode() in (p/'cmdline').read_bytes():print(json.dumps(dict(already_running=True,pid=old['pid'])));return
   assert '--resume' in sys.argv,'Use --resume after reviewing a prior failure'
  assert (ROOT/'manifest.json').exists()
  (ROOT/'logs').mkdir(exist_ok=True)
  with (ROOT/'logs/pipeline.log').open('a') as log:
   p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save('launch.json',dict(pid=p.pid,time=time.strftime('%F %T')));print(json.dumps(dict(launched=True,pid=p.pid)))

if __name__=='__main__':
 if '--run' in sys.argv:run()
 else:launch()
