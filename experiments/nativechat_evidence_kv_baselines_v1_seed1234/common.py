"""Immutable protocol, atomic results, and bounded single-GPU execution."""
from pathlib import Path
import gc
import hashlib
import json
import random
import time
import shutil
import torch

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def log(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S UTC] ', time.gmtime()) + str(message), flush=True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed():
    random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    torch.set_num_threads(CFG['cpu_threads'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def gpu_check():
    if not torch.cuda.is_available():
        raise RuntimeError('Server B CUDA unavailable')
    free, total = torch.cuda.mem_get_info()
    log(f'Server B {torch.cuda.get_device_name(0)} free={free/2**30:.2f}/{total/2**30:.2f} GiB')
    if free / 2**30 < CFG['min_initial_free_gpu_gib']:
        raise RuntimeError('Insufficient free GPU memory; unrelated tasks left intact')


def store(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def scratch():
    p = Path(CFG['scratch_directory'])
    p.mkdir(parents=True, exist_ok=True)
    return p


def publish(source, name, stage_a=False):
    """Copy finalized weights once; temporary/optimizer states stay in RAM disk."""
    destination = (Path(CFG['durable_stage_a_directory']) if stage_a else ROOT / 'models') / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha = digest(source)
    if destination.exists():
        assert digest(destination) == sha, 'Refuse to replace an existing different final checkpoint'
    else:
        if shutil.disk_usage(destination.parent).free < Path(source).stat().st_size + (256 << 20):
            raise RuntimeError('Insufficient persistent space for finalized weights; no existing weights deleted')
        tmp = destination.with_suffix(destination.suffix + '.tmp')
        shutil.copyfile(source, tmp)
        assert digest(tmp) == sha
        tmp.replace(destination)
    return {'path': str(destination), 'sha256': sha, 'persistent': True}


def freeze():
    sources = [p for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ('__pycache__', 'results'))]
    sources += [ROOT / 'config.json']
    data = sorted((ROOT / 'prepared').glob('*.json*'))
    identities = {'configuration': CFG,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
        'prepared_sha256': {str(p.relative_to(ROOT)): digest(p) for p in data},
        'model_files': {f'{family}/{p.name}': digest(p) for family, folder in CFG['models'].items()
                        for p in Path(folder).iterdir() if p.name in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors.index.json')},
        'startup_report_sha256': digest(ROOT / 'reports/startup.json')}
    identities['model_weights']={}
    for family,folder in CFG['models'].items():
        for p in sorted(Path(folder).glob('*.safetensors')):
            log(f'FREEZE model weights {family}/{p.name}')
            st=p.stat()
            identities['model_weights'][str(p)]={'sha256':digest(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    path = ROOT / 'manifest.json'
    if path.exists():
        assert read(path) == identities
    else:
        save(path, identities)
    return identities


def verify():
    manifest = read(ROOT / 'manifest.json')
    assert manifest['configuration'] == CFG
    for field in ('source_sha256', 'prepared_sha256'):
        for name, expected in manifest[field].items():
            assert digest(ROOT / name) == expected, name
    assert digest(ROOT / 'reports/startup.json') == manifest['startup_report_sha256']
    for name,record in manifest['model_weights'].items():
        st=Path(name).stat()
        assert st.st_size==record['bytes'] and st.st_mtime_ns==record['mtime_ns'],name
    return manifest
