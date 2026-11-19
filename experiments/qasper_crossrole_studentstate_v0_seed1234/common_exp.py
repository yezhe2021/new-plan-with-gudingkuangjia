import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text())
BASELINE = Path(CONFIG['baseline'])
sys.path.insert(0, str(BASELINE))
import runtime
from runtime import CFG, save, seed, load_models, tokenizers, base, adapter, make_cache, apply_rope
from kv_backend import PaperCache, mapped
from attention_probe import collect, probability, projected


def prepared():
    return json.loads((ROOT/'prepared.json').read_text())


def old_base():
    return base(BASELINE/'runs/stage_a_fresh/best.pt').eval().requires_grad_(False)


def setup():
    seed()
    sender, receiver = load_models()
    st, rt = tokenizers()
    return sender, receiver, rt, PaperCache(sender, receiver, st, rt)


def checkpoint(path, model, step, metric, optimizer=None, scaler=None, branch=None):
    import torch
    obj = {'protocol': CFG['protocol'], 'state': {k:v.detach().cpu() for k,v in model.state_dict().items()},
           'step': step, 'metric': metric, 'branch': branch, 'config': CONFIG}
    if optimizer is not None: obj['optimizer'] = optimizer.state_dict()
    if scaler is not None: obj['scaler'] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def load_state(model, path):
    import torch
    obj = torch.load(path, map_location='cpu', weights_only=True)
    assert obj['protocol'] == CFG['protocol']
    model.load_state_dict(obj['state'], strict=True)
    return obj
