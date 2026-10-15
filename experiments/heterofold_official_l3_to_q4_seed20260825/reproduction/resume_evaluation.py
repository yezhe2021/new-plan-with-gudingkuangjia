"""Evaluate saved maps only; never call calibration or overwrite finished runs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate(condition, task, start=0, end=32):
    output = ROOT / f'results/{condition}/{task}/items_{start:05d}_{end:05d}.json'
    complete = ROOT / f'results/{condition}/{task}/items_00000_00032.json'
    if output.exists() or complete.exists():
        print('REUSE', output if output.exists() else complete, flush=True)
        return
    # Recheck after every completed shard: other jobs may start mid-pipeline.
    free, total = torch.cuda.mem_get_info()
    if free < 25 * 2**30:
        raise RuntimeError(
            f'Insufficient free GPU memory before {condition}/{task}/{start}:{end}: '
            f'{free / 2**30:.2f} GiB < 25 GiB; saved results retained, no preemption')
    flags = ['--native', '--receiver', 'qwen4'] if condition == 'native' else [
        '--pair', 'llama3_to_qwen4', '--map', str(ROOT / f'maps/llama3_to_qwen4/{condition}.pt')]
    command = [sys.executable, '-u', '-m', 'heterofold', 'evaluate', *flags,
        '--data-dir', str(ROOT / 'data'), '--benchmark', task,
        '--start', str(start), '--end', str(end), '--out', str(ROOT / f'results/{condition}')]
    print('START', condition, task, start, end, flush=True)
    env = dict(os.environ)
    # Do not change the short-task dispatch relative to completed native baselines.
    env['HETEROFOLD_V100_SDPA'] = '1' if task == 'qasper' else '0'
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    print('DONE', condition, task, start, end, flush=True)


def summarize():
    summary = []
    for condition in ('native', 'recolor', 'heterofold'):
        for task in ('arc', 'gsm8k', 'qasper'):
            rows = {}
            metric, score_key = None, None
            paths = sorted((ROOT / f'results/{condition}/{task}').glob('items_*.json'))
            for path in paths:
                result = json.loads(path.read_text())
                metric, score_key = result['metric'], result['score_key']
                for row in result['rows']:
                    index = int(row['index'])
                    if not 0 <= index < 32:
                        raise ValueError('Unexpected sample index')
                    if index in rows and rows[index] != row:
                        raise ValueError('Conflicting duplicate sample result')
                    rows[index] = row
            if rows:
                summary.append(dict(condition=condition, benchmark=task,
                    metric=metric, n=len(rows), expected=32,
                    score_percent=100 * sum(r[score_key] for r in rows.values()) / len(rows)))
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)
    return summary


def main():
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 25 * 2**30:
        raise RuntimeError('CUDA unavailable or less than 25 GiB free; no preemption')
    audit = json.loads((ROOT / 'smoke/evaluation_memory_adaptation.json').read_text())
    if not all(audit[k] for k in ('native_arc4_predictions_identical',
                                  'cpu_staging_kv_bitwise_identical',
                                  'translated_logits_finite')):
        raise AssertionError('Evaluation memory adaptation has not passed smoke')
    for condition in ('recolor', 'heterofold'):
        if not (ROOT / f'maps/llama3_to_qwen4/{condition}.pt').is_file():
            raise FileNotFoundError(condition)
    # Audit only: no training or calibration entrypoint is called here.
    from datetime import datetime, timezone
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        torch=torch.__version__, dtype=os.environ['HETEROFOLD_DTYPE'],
        explicit_gqa_tasks=['qasper'], sender_capture_device='cpu',
        unchanged=['weights', 'prompt', 'tokenization', 'sampling', 'seeds',
                   'context_limit', 'generation_limit', 'parser', 'metrics', 'test_indices'])
    (ROOT / 'logs/resume_evaluation_protocol.json').write_text(json.dumps(record, indent=2))
    try:
        # Complete short tasks before attempting long-context Qasper.
        for task in ('arc', 'gsm8k'):
            for condition in ('native', 'recolor', 'heterofold'):
                evaluate(condition, task)
                summarize()
        # One official CLI shard per item saves progress and releases CUDA memory.
        for condition in ('native', 'recolor', 'heterofold'):
            for index in range(32):
                evaluate(condition, 'qasper', index, index + 1)
                summarize()
    finally:
        summary = summarize()
    if len(summary) != 9 or any(r['n'] != 32 for r in summary):
        raise AssertionError('Incomplete evaluation grid')
    print('ALL EXPERIMENTS COMPLETED', flush=True)


if __name__ == '__main__':
    main()
