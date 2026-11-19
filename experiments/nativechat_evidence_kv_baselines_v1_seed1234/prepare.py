"""Seeded train/validation/pilot split, with document and old-pilot exclusions."""
from collections import Counter
from pathlib import Path
import json
import random
import re
from transformers import AutoTokenizer
from common import ROOT, CFG, save, read, rows, write_rows, digest, log
from protocol import document, audit_row, ProtocolError


def old_ids():
    found = set()
    def walk(value):
        if isinstance(value, str) and re.fullmatch('[0-9a-f]{24}', value):
            found.add(value)
        elif isinstance(value, dict):
            for x in value.values():
                walk(x)
        elif isinstance(value, list):
            for x in value:
                walk(x)
    for filename in CFG['old_pilot_manifests']:
        p = Path(filename)
        if p.exists():
            walk(read(p))
    return found


def normalize(row):
    if row['type'] != 'bridge':
        raise ProtocolError('not_bridge')
    context = row['context']
    context = list(zip(context['title'], context['sentences'])) if isinstance(context, dict) else context
    facts = row['supporting_facts']
    gold = set(facts['title'] if isinstance(facts, dict) else [x[0] for x in facts])
    docs = [(title, sentences) for title, sentences in context if title in gold]
    if len(gold) != 2 or len(docs) != 2:
        raise ProtocolError('not_two_complete_gold_paragraphs')
    return {'id': row.get('_id', row.get('id')), 'question': row['question'], 'answer': row['answer'],
            'titles': [d[0] for d in docs],
            'documents': [document(title, sentences, i + 1) for i, (title, sentences) in enumerate(docs)]}


def main():
    folder = ROOT / 'prepared'
    if (folder / 'data_manifest.json').exists():
        m = read(folder / 'data_manifest.json')
        for name, sha in m['prepared_sha256'].items():
            assert digest(folder / name) == sha
        tok = {k: AutoTokenizer.from_pretrained(v, local_files_only=True) for k, v in CFG['models'].items()}
        for split in ('train','validation','pilot'):
            for row in rows(folder / f'{split}.jsonl'):
                audit_row(tok,row)
        m['protocol_source_sha256']={name:digest(ROOT/name) for name in ('protocol.py','prepare.py','config.json')}
        save(folder / 'data_manifest.json',m)
        log('Prepared split verifies against final native protocol; IDs unchanged')
        return
    tok = {k: AutoTokenizer.from_pretrained(v, local_files_only=True) for k, v in CFG['models'].items()}
    import pyarrow.parquet as pq
    train = pq.read_table(CFG['train_data']).to_pylist()
    dev = read(CFG['dev_data'])
    rng = random.Random(CFG['seed'])
    rng.shuffle(train)
    rng.shuffle(dev)
    excluded = old_ids()
    counts = Counter()
    selected = {}
    reserved_titles, reserved_ids = set(), set()
    audits = []
    # Pilot-dev first, then held-out validation, then training. No prediction or
    # answer content participates in eligibility. Titles are disjoint by split.
    for split, pool, limit in [('pilot', dev, CFG['pilot_count']), ('validation', train, CFG['validation_count']), ('train', train, CFG['train_count'])]:
        chosen = []
        for raw in pool:
            try:
                row = normalize(raw)
                if row['id'] in reserved_ids or (split == 'pilot' and row['id'] in excluded):
                    raise ProtocolError('reused_question_id')
                if set(row['titles']) & reserved_titles:
                    raise ProtocolError('cross_split_document_title')
                audit = audit_row(tok, row)
            except ProtocolError as error:
                counts[split + '/' + str(error)] += 1
                continue
            chosen.append(row)
            audits.append({'split': split, 'id': row['id'], **audit})
            if len(chosen) == limit:
                break
        if len(chosen) != limit:
            save(ROOT / 'reports/preparation_failure.json', {'split': split, 'selected': len(chosen), 'required': limit, 'exclusions': dict(counts)})
            raise RuntimeError(f'Insufficient eligible {split}: {len(chosen)}/{limit}; do not change protocol silently')
        selected[split] = chosen
        reserved_ids.update(r['id'] for r in chosen)
        reserved_titles.update(t for r in chosen for t in r['titles'])
        write_rows(folder / f'{split}.jsonl', chosen)
        log(f'PREPARED {split}: {len(chosen)}')
    write_rows(folder / 'token_audit.jsonl', audits)
    manifest = {'seed': CFG['seed'], 'counts': {k: len(v) for k,v in selected.items()},
        'source_sha256': {p: digest(p) for p in (CFG['train_data'], CFG['dev_data'])},
        'excluded_old_pilot_ids': sorted(excluded), 'exclusion_counts': dict(counts),
        'selection_uses_model_outputs': False, 'gold_annotations_select_complete_documents_only': True,
        'document_titles_disjoint_across_splits': True,
        'prepared_sha256': {p.name: digest(p) for p in sorted(folder.glob('*.jsonl'))},
        'protocol_source_sha256':{name:digest(ROOT/name) for name in ('protocol.py','prepare.py','config.json')}}
    save(folder / 'data_manifest.json', manifest)


if __name__ == '__main__':
    main()
