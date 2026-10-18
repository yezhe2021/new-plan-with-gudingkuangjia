"""Explicit parent identities and independent files for one authorized follow-up."""
from pathlib import Path
import importlib.util
import json
import sys
import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]
CONFIG=json.loads((HERE/'config.json').read_text())
TRAJ=PARENT/CONFIG['previous_directory']
sys.path.insert(0,str(TRAJ))
import trajectory_common as parent
from kv_backend import CFG,FullSyncHFLM,ResidualKVAdapter
log,save,plain,digest,read_rows,write_rows=parent.log,parent.save,parent.plain,parent.digest,parent.read_rows,parent.write_rows


def import_parent(filename):
    spec=importlib.util.spec_from_file_location('immutable_trajectory_'+filename,TRAJ/(filename+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


base_train=import_parent('train')
parent_prepare=import_parent('prepare')


def gpu_check():
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info()
    log(f'Server B {torch.cuda.get_device_name(0)}; free={free/2**30:.2f}/{total/2**30:.2f} GiB')
    assert free/2**30>=CONFIG['min_free_gpu_gib'],'Not enough free memory; unrelated tasks left intact'


def identity():
    parent.verify_manifest()
    source=[p for p in HERE.glob('*.py')]+[HERE/'config.json']
    return {'configuration':CONFIG,'source_sha256':{str(p):digest(p) for p in sorted(source)},
            'parent_manifest_sha256':digest(TRAJ/'manifest.json'),
            'stage_a':{'path':str(PARENT/CONFIG['parent_stage_a']),'sha256':digest(PARENT/CONFIG['parent_stage_a'])},
            'initial_stage_b':{'path':str(TRAJ/'stage_b/best.pt'),'sha256':digest(TRAJ/'stage_b/best.pt')},
            'diagnosis_sha256':digest(HERE/'diagnostic/answer_token_summary.json')}


def freeze_manifest():
    m={'identities':identity(),'prepared_sha256':{str(p.relative_to(HERE)):digest(p) for p in sorted((HERE/'prepared').glob('*.json*'))},
       'smoke_sha256':digest(HERE/'reports/smoke.json')}
    if (HERE/'manifest.json').exists():assert json.loads((HERE/'manifest.json').read_text())==m
    else:save(HERE/'manifest.json',m)
    return m


def verify_manifest():
    m=json.loads((HERE/'manifest.json').read_text());assert m['identities']==identity()
    for rel,sha in m['prepared_sha256'].items():assert digest(HERE/rel)==sha,rel
    assert digest(HERE/'reports/smoke.json')==m['smoke_sha256']
    return m


def load_initial_adapter(trainable=False):
    return parent.load_new_adapter(TRAJ/'stage_b/best.pt',trainable)


def load_new_adapter(path,trainable=False):
    m=verify_manifest();p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['protocol']==CONFIG['protocol']
    assert p['stage_a_sha256']==m['identities']['stage_a']['sha256']
    assert p['initial_stage_b_sha256']==m['identities']['initial_stage_b']['sha256']
    a=ResidualKVAdapter(rank=CONFIG['adapter_rank']).cuda();a.load_state_dict(p['state'],strict=True)
    return a.train(trainable).requires_grad_(trainable)


def model_bundle(trainable=False):
    receiver,sender,base,old,source_tok,hf=parent.model_bundle(False)
    del old
    adapter=load_initial_adapter(trainable)
    return receiver,sender,base,adapter,source_tok,hf
