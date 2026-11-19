"""Detached sequential full experiment; any failed gate stops the pipeline."""
import sys,os,json,time,subprocess,fcntl,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
def save(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def launch():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():raise RuntimeError('Already launched; inspect existing process/status before resuming')
        subprocess.run([sys.executable,'-c','from common import verify; verify()'],cwd=ROOT,check=True)
        with (ROOT/'logs/pipeline.log').open('a') as out:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=p.pid,time=time.strftime('%F %T'),interpreter=sys.executable,detached=True)
        save('launch.json',record);print(json.dumps(record))
def cleanup_train_sources():
    # Only reproducible source files made by this experiment; never touch parent artifacts.
    from common import rows,source_path,identity,done
    for p in ['runs/stage_a/completed.json','runs/stage_b/completed.json','runs/reader/completed.json']:assert done(ROOT/p)
    test_ids={r['id'] for r in rows('test')};count=0;size=0
    for split in ['train','validation']:
        for row in rows(split):
            assert row['id'] not in test_ids
            p=source_path(row);assert p.parent==ROOT/'source'
            if p.exists():size+=p.stat().st_size;p.unlink();count+=1
    save('source_cleanup.json',dict(manifest_sha256=identity(),reason='All training completed; release own regenerable source caches for full evaluation',files=count,bytes=size,time=time.strftime('%F %T')))
def run():
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'failure.json').exists():raise RuntimeError('Failure requires review')
        free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())/1024
        assert free>=CFG['queue_min_free_gpu_gib'],free
        save('runner_status.json',dict(status='running',pid=os.getpid(),started_at=time.strftime('%F %T')))
        actions=[['capture_source.py','trainval'],['stage_a.py','smoke'],['stage_a.py','train'],['stage_b.py','smoke'],['stage_b.py','train'],['stage_reader.py','smoke'],['stage_reader.py','train'],['cleanup'],['capture_source.py','test']]
        actions += [['evaluate.py',t] for t in CFG['reader']['evaluation_tasks']]+[['evaluate.py','summarize']]
        try:
            for args in actions:
                save('runner_status.json',dict(status='running',pid=os.getpid(),action=args,time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START',args,flush=True)
                if args==['cleanup']:cleanup_train_sources()
                else:subprocess.run([sys.executable,'-u',str(ROOT/args[0]),*args[1:]],cwd=ROOT,check=True)
        except Exception as e:
            save('failure.json',dict(type=type(e).__name__,message=str(e),action=args));save('runner_status.json',dict(status='failed',pid=os.getpid(),action=args));raise
        save('runner_status.json',dict(status='completed',pid=os.getpid(),completed_at=time.strftime('%F %T')))
if __name__=='__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    if sys.argv[1]=='--launch':launch()
    else:run()
