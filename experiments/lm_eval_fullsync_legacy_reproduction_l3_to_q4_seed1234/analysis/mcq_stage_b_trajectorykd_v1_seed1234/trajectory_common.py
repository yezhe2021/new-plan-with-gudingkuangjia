"""Separate follow-up: preserve and verify the immutable scheme-3 experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
PREVIOUS = PARENT / CONFIG['previous_directory']
sys.path.insert(0, str(PREVIOUS))
import runtime as previous_runtime
import train_stage_b as previous_train
from kv_backend import CFG, FullSyncHFLM, load_base, load_model, seed_all


def log(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S UTC] ', time.gmtime()) + message, flush=True)


def digest(path):
    return previous_runtime.digest(path)


def plain(value):
    return value.item() if hasattr(value, 'item') else str(value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=plain) + '\n', encoding='utf-8')
    temp.replace(path)


def read_rows(path):
    return previous_runtime.read_rows(path)


def write_rows(path, rows):
    previous_runtime.write_rows(path, rows)


def identity():
    old = previous_runtime.verify_manifest()
    files = [p for p in HERE.iterdir() if p.suffix in ('.py', '.json') and p.name not in ('manifest.json', 'preflight.json', 'bootstrap.json', 'smoke.json', 'status.json', 'summary.json')]
    files += [PREVIOUS / 'evaluate_stage_b.py', PREVIOUS / 'evaluate_mmlu_pro.py']
    return {'configuration': CONFIG,
            'source_sha256': {str(p): digest(p) for p in sorted(files)},
            'previous_manifest_sha256': digest(PREVIOUS / 'manifest.json'),
            'stage_a': old['identities']['stage_a'],
            'initial_stage_b': {'path': str(PREVIOUS / 'stage_b/best.pt'), 'sha256': digest(PREVIOUS / 'stage_b/best.pt')},
            'models': CFG['models'], 'attention_implementation': CFG['attention_implementation'],
            'data_sha256': {str(PARENT / CONFIG[k]): digest(PARENT / CONFIG[k]) for k in ('parent_training_data', 'parent_validation_data')}}


def freeze_manifest():
    files = sorted((HERE / 'prepared').glob('*.json*'))
    m = {'identities': identity(), 'prepared_sha256': {str(p.relative_to(HERE)): digest(p) for p in files},
         'preflight_sha256': digest(HERE / 'preflight.json'), 'smoke_sha256': digest(HERE / 'smoke.json')}
    if (HERE / 'manifest.json').exists():
        assert json.loads((HERE / 'manifest.json').read_text()) == m, 'Refuse to replace immutable manifest'
    else:
        save(HERE / 'manifest.json', m)
    return m


def verify_manifest():
    m = json.loads((HERE / 'manifest.json').read_text())
    assert m['identities'] == identity(), 'Source/data/parent checkpoint changed'
    for rel, sha in m['prepared_sha256'].items():
        assert digest(HERE / rel) == sha, rel
    assert digest(HERE / 'preflight.json') == m['preflight_sha256']
    assert digest(HERE / 'smoke.json') == m['smoke_sha256']
    return m


def gpu_check():
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    free, total = torch.cuda.mem_get_info()
    log(f'Server B: {torch.cuda.get_device_name(0)}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB')
    assert free / 2**30 >= CONFIG['min_free_gpu_gib'], 'Insufficient free memory; leave unrelated jobs intact'


def load_new_adapter(path, trainable=False):
    import torch
    from kv_backend import ResidualKVAdapter
    payload = torch.load(path, map_location='cpu', weights_only=True)
    m = verify_manifest()
    assert payload['protocol'] == CONFIG['protocol']
    assert payload['stage_a_sha256'] == m['identities']['stage_a']['sha256']
    assert payload['initial_stage_b_sha256'] == m['identities']['initial_stage_b']['sha256']
    adapter = ResidualKVAdapter(rank=CONFIG['adapter_rank']).cuda()
    adapter.load_state_dict(payload['state'], strict=True)
    return adapter.train(trainable).requires_grad_(trainable)


def load_initial_adapter(trainable=True):
    return previous_runtime.load_new_adapter(PREVIOUS / 'stage_b/best.pt', trainable)


def model_bundle(trainable=False):
    seed_all(CONFIG['seed'])
    receiver = load_model(CFG, 'qwen')
    sender = load_model(CFG, 'llama')
    base = load_base(PARENT / CONFIG['parent_stage_a'])
    adapter = load_initial_adapter(trainable)
    from transformers import AutoTokenizer
    source_tok = AutoTokenizer.from_pretrained(CFG['models']['llama'], local_files_only=True)
    hf = FullSyncHFLM(receiver, mode='native')
    return receiver, sender, base, adapter, source_tok, hf


def import_previous(name):
    spec = importlib.util.spec_from_file_location('trajectory_followup_' + name, PREVIOUS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
