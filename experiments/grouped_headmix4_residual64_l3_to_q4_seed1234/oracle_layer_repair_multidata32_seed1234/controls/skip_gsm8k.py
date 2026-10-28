"""User-requested queue reduction; keep original experiment manifest and active child."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
REQUEST=Path(__file__).with_suffix('.json')


def read(path):return json.loads(path.read_text())
def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    try:return p.read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def summarize(E,tasks,request):
    E.verify();results={t:read(ROOT/f'evaluation/{t}/summary.json') for t in tasks}
    assert all(v['manifest_sha256']==E.identity() and v['status']=='completed' for v in results.values())
    macro={a['name']:sum(v['metrics'][a['name']]['accuracy'] for v in results.values())/len(tasks) for a in E.arms()}
    n=sum(results[t]['metrics']['translated']['n'] for t in tasks)
    caveats=[s.replace('192 questions,5184 sample/configuration evaluations',f'{n} questions,{n*27} sample/configuration evaluations')
             for s in E.CFG['caveats'] if not s.startswith('GSM8K keeps formal')]
    caveats+=['GSM8K was skipped at the user request before formal evaluation. Five-task macro excludes GSM8K; it is not scored as zero. Original manifest/config retained, effective task override recorded in controls/skip_gsm8k.json.']
    E.mark('summary.json',status='completed',n=n,configurations=27,tasks=results,macro_accuracy=macro,
           training=False,reader=False,exploratory=True,confirmation_evaluated=False,caveats=caveats,
           effective_tasks=tasks,skipped_tasks=['gsm8k'],user_override=request,user_override_sha256=E.C.sha(REQUEST))
    lines=['# Five-task Oracle layer repair: GSM8K skipped by user','',
           'Original experiment protocol/checkpoints unchanged. GSM8K omitted from execution and macro average.','',
           '| Arm | '+' | '.join(tasks)+' | Macro |','|---|'+'---:|'*(len(tasks)+1)]
    for a in E.arms():
        name=a['name'];lines.append('| '+name+' | '+' | '.join(f'{results[t]["metrics"][name]["accuracy"]*100:.2f}%' for t in tasks)+f' | {macro[name]*100:.2f}% |')
    lines+=['',*['- '+s for s in caveats]]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    E.mark('completed.json',status='completed',finished_at=time.strftime('%F %T'),effective_tasks=tasks,skipped_tasks=['gsm8k'])
    E.status(status='completed',effective_tasks=tasks,skipped_tasks=['gsm8k'])


def run():
    request=read(REQUEST)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save('runner_status.json',dict(status='running',phase='retain_active_evaluation',pid=os.getpid(),skipped_tasks=['gsm8k'],time=time.strftime('%F %T')))
            for child in request['retained_children']:
                print(time.strftime('%F %T'),'RETAIN ACTIVE',child,flush=True)
                while alive(child['pid']):time.sleep(3)
                done=ROOT/f'evaluation/{child["task"]}/completed.json'
                assert done.exists(),('Active evaluation failed',child)
            sys.path.insert(0,str(ROOT))
            import experiment as E
            E.verify()
            tasks=[t for t in E.CFG['tasks'] if t!='gsm8k']
            assert len(tasks)==5
            for task in tasks:
                done=ROOT/f'evaluation/{task}/completed.json'
                if done.exists():
                    assert read(done)['manifest_sha256']==E.identity();continue
                save('runner_status.json',dict(status='running',phase='evaluate '+task,pid=os.getpid(),skipped_tasks=['gsm8k'],time=time.strftime('%F %T')))
                print(time.strftime('%F %T'),'START evaluate',task,'GSM8K SKIPPED',flush=True)
                subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'evaluate',task],cwd=ROOT,check=True)
            summarize(E,tasks,request)
            save('runner_status.json',dict(status='completed',pid=os.getpid(),finished_at=time.strftime('%F %T'),effective_tasks=tasks,skipped_tasks=['gsm8k']))
        except Exception as exc:
            failure=dict(status='failed',type=type(exc).__name__,message=str(exc),time=time.strftime('%F %T'),skipped_tasks=['gsm8k'])
            save('failure.json',failure);save('runner_status.json',failure);save('status.json',failure);raise


if __name__=='__main__':run()
