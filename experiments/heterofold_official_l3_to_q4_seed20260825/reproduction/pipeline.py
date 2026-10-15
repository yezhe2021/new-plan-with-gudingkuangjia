import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(*args):
    print('START', ' '.join(args), flush=True)
    subprocess.run([sys.executable, '-u', *args], cwd=ROOT, check=True)
    print('DONE', ' '.join(args), flush=True)


def main():
    import torch
    from heterofold.data.calibration_corpus import PROMPT_DIGEST, load_manifest, digest_text
    from heterofold.registry import resolve_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU training fallback')
    free, total = torch.cuda.mem_get_info()
    print('GPU free GiB', free / 2**30, 'total GiB', total / 2**30, flush=True)
    if free < 25 * 2**30:
        raise RuntimeError('Less than 25 GiB free; not starting alongside other GPU tasks')
    record = {'upstream_commit': 'd8018c9a103094511fd58bb6572d1e4b038e071e',
        'python': sys.version, 'dtype': os.environ['HETEROFOLD_DTYPE'],
        'gpu': torch.cuda.get_device_name(), 'model_config_sha256': {},
        'packages': {p: package_version(p) for p in
          ('torch', 'transformers', 'tokenizers', 'datasets', 'numpy', 'accelerate',
           'huggingface_hub', 'pyarrow', 'safetensors', 'sentencepiece')}}
    for alias in ('llama3', 'qwen4'):
        p = Path(resolve_model(alias))
        record['model_config_sha256'][alias] = {
            f: hashlib.sha256((p / f).read_bytes()).hexdigest()
            for f in ('config.json', 'tokenizer_config.json')}
    (ROOT / 'environment.json').write_text(json.dumps(record, indent=2))
    run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    run('-m', 'heterofold', 'prepare-data', '--data-dir', str(ROOT / 'data'), '--skip-benchmarks')
    manifest = load_manifest(ROOT / 'data/calibration/openr1_hotpot.json')
    rows = manifest['rows']
    actual = digest_text('\n'.join(digest_text(r['prompt']) for r in rows))
    if actual != PROMPT_DIGEST or manifest['prompt_sha256_sequence'] != actual:
        raise AssertionError('Official calibration manifest digest mismatch')
    if sum(r['split'] == 'train' for r in rows) != 1600 or sum(r['split'] == 'validation' for r in rows) != 400:
        raise AssertionError('Wrong calibration split')
    run('reproduction/gpu_smoke.py')
    run('-m', 'heterofold', 'calibrate', '--pair', 'llama3_to_qwen4',
        '--data-dir', str(ROOT / 'data'), '--out', str(ROOT / 'maps/llama3_to_qwen4'),
        '--rank', '16', '--epochs', '4', '--seed', '20260825', '--grad-accum', '8')
    for condition in ('native', 'recolor', 'heterofold'):
        flags = ['--native', '--receiver', 'qwen4'] if condition == 'native' else [
            '--pair', 'llama3_to_qwen4', '--map', str(ROOT / f'maps/llama3_to_qwen4/{condition}.pt')]
        run('-m', 'heterofold', 'evaluate', *flags, '--data-dir', str(ROOT / 'data'),
            '--benchmark', 'arc', 'gsm8k', 'qasper', '--limit', '32',
            '--out', str(ROOT / f'results/{condition}'))
    summary = []
    for p in sorted((ROOT / 'results').glob('*/*/items_*.json')):
        r = json.loads(p.read_text())
        summary.append({'condition': p.relative_to(ROOT / 'results').parts[0],
            **{k: r[k] for k in ('benchmark', 'metric', 'n', 'score_percent', 'elapsed_seconds')}})
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, indent=2))
    print('ALL EXPERIMENTS COMPLETED', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
