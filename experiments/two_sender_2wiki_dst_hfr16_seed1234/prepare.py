import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RAW = Path('/hy-tmp/yezhe/data/all_models/datasets/voidful/2WikiMultihopQA/dev.json')


def norm(text):
    return ' '.join(text.casefold().split())


def prepare():
    old = set()
    ids = set()
    for name in ['depth_state_translator_fullsync_l3_to_q4_seed1234',
                 'dst_rank64_gated_dynamic_m32_gate_off_heldout_seed1234']:
        for line in (PROJECT / name / 'prepared/test.jsonl').open():
            row = json.loads(line)
            if row['task'] == '2wiki':
                old.add(norm(row['question']))
                ids.add(row['id'].removeprefix('2wiki_'))
    pool = json.loads(RAW.read_text())
    rng = random.Random(1234)
    rng.shuffle(pool)
    cohort = []
    for row in pool:
        if row['type'] != 'compositional' or norm(row['question']) in old or row['_id'] in ids:
            continue
        edges = row['evidences']
        supports = {title for title, ix in row['supporting_facts']}
        docs = dict(row['context'])
        if len(edges) != 2 or len(supports) != 2 or len(docs) != 10:
            continue
        first, second = edges
        if norm(first[2]) != norm(second[0]):
            first, second = second, first
        if norm(first[2]) != norm(second[0]) or norm(second[2]) != norm(row['answer']):
            continue
        targets = [t for t in supports if norm(t) == norm(second[0])]
        if len(targets) != 1 or not supports <= docs.keys():
            continue
        b = targets[0]
        a, = supports - {b}
        distractors = sorted(set(docs) - supports)
        rng.shuffle(distractors)
        groups = [[a] + distractors[:4], [b] + distractors[4:]]
        for group in groups:
            rng.shuffle(group)
        def render(group):
            return ''.join('Passage: ' + title + '\n' + ' '.join(docs[title]) + '\n\n' for title in group)
        cohort.append(dict(id=row['_id'], question=row['question'], answer=row['answer'],
                           a=render(groups[0]), b=render(groups[1]), titles=groups,
                           support=[a,b], chain=[first,second]))
        old.add(norm(row['question']))
        if len(cohort) == 64:
            break
    assert len(cohort) == 64
    (ROOT / 'cohort.json').write_text(json.dumps(cohort, ensure_ascii=False, indent=2))
    print('Prepared', len(cohort), 'questions', flush=True)


if __name__ == '__main__':
    prepare()
