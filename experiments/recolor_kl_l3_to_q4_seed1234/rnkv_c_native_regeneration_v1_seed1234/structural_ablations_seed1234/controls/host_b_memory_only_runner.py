"""Operational handoff: keep B's existing first arm; never launch the A-delegated arm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent.parent

def save(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return b''
def run():
    h=json.loads((ROOT/'controls/server_a_handoff.json').read_text());pid=h['memory_only_child_pid']
    save('runner_status.json',dict(status='running',action=['ablation_train.py','memory_only'],pid=os.getpid(),
        delegated_arm='keep_both',delegated_host='server-a',time=time.strftime('%F %T')))
    while b'ablation_train.py' in command(pid) and b'memory_only' in command(pid):time.sleep(10)
    assert (ROOT/'arms/memory_only/runs/completed.json').exists(),'First arm stopped before successful completion'
    if not (ROOT/'arms/memory_only/evaluation/completed.json').exists():
        save('runner_status.json',dict(status='running',action=['ablation_evaluate.py','memory_only'],pid=os.getpid(),
            delegated_arm='keep_both',delegated_host='server-a',time=time.strftime('%F %T')))
        subprocess.run([sys.executable,'-u',str(ROOT/'ablation_evaluate.py'),'memory_only'],cwd=ROOT,check=True)
    value=dict(status='completed',host='server-b',arm='memory_only',keep_both_delegated_to='server-a',finished_at=time.strftime('%F %T'))
    save('controls/server_b_memory_only_completed.json',value)
    save('runner_status.json',dict(status='local_arm_completed_remote_arm_delegated',**{k:v for k,v in value.items() if k!='status'}))

if __name__=='__main__':
    try:run()
    except Exception as exc:
        value=dict(status='failed',host='server-b',type=type(exc).__name__,message=str(exc),time=time.strftime('%F %T'))
        save('controls/server_b_failure.json',value);raise
