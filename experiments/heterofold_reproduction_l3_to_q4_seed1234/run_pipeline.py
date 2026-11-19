import argparse
import json
import subprocess
import sys
import hashlib
from common import ROOT,CFG,seed,tokenizers,save
from data import prepare
from collect_statistics import collect_paired_statistics
from train import load_maps,train_output_aware_calibration


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','formal'],default='smoke')
    p.add_argument('--evaluate',choices=['none','hub','full_context','both'],default='none');args=p.parse_args()
    seed();folder=ROOT/'runs'/args.mode;folder.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'tests.py')],check=True,cwd=ROOT)
    st,rt=tokenizers()
    # Validate input provenance/counts before allocating GPU or statistics memory.
    prepared=prepare(st,rt,args.mode=='smoke',write=False)
    fingerprint=hashlib.sha256(json.dumps({'config':CFG,'prepared':prepared},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    fingerprint_path=folder/'run_identity.json'
    if fingerprint_path.exists():
        assert json.loads(fingerprint_path.read_text())['sha256']==fingerprint,'Config/data changed: use a new run directory, never silently reuse maps'
    save(fingerprint_path,{'sha256':fingerprint});save(folder/'effective_config.json',CFG)
    save(folder/'prepared.json',prepared)
    subprocess.run([sys.executable,'-u',str(ROOT/'gpu_gates.py')],check=True,cwd=ROOT)
    maps_path=folder/'maps/recolor.pt'
    maps=load_maps(maps_path) if maps_path.exists() else collect_paired_statistics(prepared,st,rt,folder/'maps')
    for branches in ('k','kv'):
        output=folder/('calibration_'+branches)
        if not (output/'completed.json').exists():
            seed();train_output_aware_calibration(prepared,maps,st,rt,output,branches)
    for interface in ('full_context','hub'):
        if args.evaluate in ('both',interface):
            subprocess.run([sys.executable,'-u',str(ROOT/'evaluate.py'),'--mode',args.mode,'--interface',interface],check=True,cwd=ROOT)
    save(folder/'pipeline_completed.json',{'mode':args.mode,'evaluation':args.evaluate,'smoke_not_performance':args.mode=='smoke'})
    print('HETEROFOLD PIPELINE COMPLETED',flush=True)


if __name__=='__main__':main()
