"""Audit saved generations only; preserve all original predictions and summaries."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def normalize_number(value):
    value = value.strip().replace(',', '').replace('$', '').replace('%', '')
    if value.endswith('.'):
        value = value[:-1]
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, '.12g')
    except ValueError:
        return value


def extract(text):
    # Exact historical extraction policy, before the control-token cleanup.
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        return normalize_number(boxed[-1])
    marker = re.findall(r'####\s*([^\n]+)', text)
    if marker:
        return normalize_number(marker[-1])
    numbers = re.findall(r'[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[0-9]+)?%?', text)
    return normalize_number(numbers[-1]) if numbers else ''


def main():
    path = ROOT / 'evaluation/per_sample.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    original = json.loads((ROOT / 'summary.json').read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 544
    for task, data in original['tasks'].items():
        rr = [r for r in rows if r['task'] == task]
        for arm, metric in data['metrics'].items():
            assert len(rr) == metric['n']
            assert sum(r['conditions'][arm]['correct'] for r in rr) == metric['correct']
    gsm = [r for r in rows if r['task'] == 'gsm8k']
    assert len(gsm) == 32
    audited = []
    for row in gsm:
        conditions = {}
        for arm, value in row['conditions'].items():
            assert extract(value['generation']) == value['prediction'], (row['id'], arm)
            assert int(value['prediction'] == normalize_number(row['gold'])) == value['correct']
            clean = re.sub(r'<\|[^>]*\|>', '', value['generation'])
            prediction = extract(clean)
            conditions[arm] = dict(original_prediction=value['prediction'], original_correct=value['correct'],
                                   cleaned_prediction=prediction, cleaned_correct=int(prediction == normalize_number(row['gold'])),
                                   control_tokens=re.findall(r'<\|[^>]*\|>', value['generation']))
        audited.append(dict(id=row['id'], gold=row['gold'], conditions=conditions))
    metrics = {}
    for arm in gsm[0]['conditions']:
        correct = sum(r['conditions'][arm]['cleaned_correct'] for r in audited)
        metrics[arm] = dict(n=32, original_correct=sum(r['conditions'][arm]['original_correct'] for r in audited),
                            cleaned_correct=correct, cleaned_accuracy=correct/32)
    paired = {}
    for arm in ['qwen_native', 'grouped_residual', 'grouped_residual_reader']:
        pairs = [(r['conditions']['task_supervised_reader']['cleaned_correct'], r['conditions'][arm]['cleaned_correct']) for r in audited]
        paired[arm] = dict(both_correct=sum(a and b for a,b in pairs), new_only=sum(a and not b for a,b in pairs),
                           baseline_only=sum(not a and b for a,b in pairs), both_wrong=sum(not a and not b for a,b in pairs))
    summary = dict(policy='Remove literal chat control tokens <|...|> from saved text, then use the exact historical numeric extractor. Apply equally to all six arms; no generation or training rerun.',
                   source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), original_summary_preserved=True,
                   all544_original_counts_verified=True, gsm8k=dict(metrics=metrics, paired=paired),
                   caveat='Only a control-token cleanup. Unit text, rationale content and other historical extraction rules remain unchanged.')
    (OUT/'gsm8k_per_sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audited))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
