"""Frozen HeteroFold and A+B+Reader on identical full-distractor HotpotQA rows."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HETEROFOLD_V100_SDPA'] = '1'
os.environ['HETEROFOLD_CAPTURE_DEVICE'] = 'cpu'
import importlib.util
import json
import random
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
REFERENCE = Path(CFG['reference_reader'])
spec = importlib.util.spec_from_file_location('hotpot_frozen_reader', REFERENCE / 'experiment.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
C, U, torch = R.C, R.U, R.torch
from heterofold.maps import load_map, describe
from heterofold.transfer import capture_sender, build_receiver_cache, _cache_layer
from heterofold.evaluation.v100_sdpa import configure
configure()
sys.path.insert(0, str(ROOT))
from metrics import answer_metrics, clean_prediction, paired_interval
ARMS = ['qwen_native', 'heterofold'] + list(CFG['readers'])


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    C.save(ROOT / name, value)


def status(**value):
    save('status.json', dict(time=time.strftime('%F %T'), **value))


def identity():
    return C.sha(ROOT / 'manifest.json')


def mark(name, **value):
    save(name, dict(manifest_sha256=identity(), **value))


def question_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def rows():
    return C.read_jsonl(ROOT / 'prepared/test.jsonl')


def prompt_for(doc):
    passages = ['Passage ' + str(i) + ':\n' + title + '\n' + ''.join(sentences).strip()
                for i, (title, sentences) in enumerate(doc['context'], 1)]
    return ('Answer the question using the passages below. Output only the short answer, without an explanation.\n\n'
            + '\n\n'.join(passages) + '\n\nQuestion: ' + doc['question'] + '\nGive only the answer.')


def prepare():
    if (ROOT / 'manifest.json').exists():
        verify()
        return
    raw = read(CFG['source_json'])
    assert len(raw) == 7405
    indices = sorted(random.Random(CFG['seed']).sample(range(len(raw)), CFG['n']))
    calibration = read(CFG['heterofold_calibration_json'])['rows']
    cal_questions = {question_key(r['question']) for r in calibration}
    cal_ids = {r.get('hotpot_id') for r in calibration if r.get('hotpot_id')}
    selected = [raw[i] for i in indices]
    assert len({d['_id'] for d in selected}) == CFG['n'] == 512
    assert not {question_key(d['question']) for d in selected} & cal_questions
    assert not {d['_id'] for d in selected} & cal_ids
    tok = {f: C.AutoTokenizer.from_pretrained(C.CFG['models'][f], local_files_only=True) for f in ['llama', 'qwen']}
    result = []
    for i, index in enumerate(indices, 1):
        doc = raw[index]
        prompt = prompt_for(doc)
        row = U.make_row('hotpot_' + doc['_id'], 'gsm8k', dict(question=doc['question']),
                         tok['llama'], tok['qwen'], prompt=prompt, candidates=[], gold=doc['answer'], normalize=False, cue='')
        row.update(task='hotpotqa_distractor', source_id=doc['_id'], source_index=index,
                   question=doc['question'], answer=doc['answer'], question_type=doc.get('type'),
                   level=doc.get('level'), paragraph_count=len(doc['context']),
                   context_sha256=C.digest(doc['context']), source_doc_sha256=C.digest(doc))
        end = len(prompt) - len(row['native_tail'])
        row['transfer_text'] = prompt[:end]
        for family in ['llama', 'qwen']:
            length = len(row['frames'][family]['full_ids'])
            assert length + CFG['max_new_tokens'] < CFG['max_context_tokens'], (row['id'], family, length)
        assert row['frames']['qwen']['body_ids'] + row['suffix_ids'] == row['native_context_ids']
        assert row['frames']['qwen']['content_ids'] == tok['qwen'].encode(row['transfer_text'], add_special_tokens=False)
        result.append(row)
        if i % 32 == 0:
            status(status='running', phase='prepare', sample=i, total=CFG['n'])
            print(time.strftime('%F %T'), f'PREPARE {i}/{CFG["n"]}', flush=True)
    C.write_jsonl(ROOT / 'prepared/test.jsonl', result)
    sources = {}
    for path in [Path(CFG['source_json']), Path(CFG['heterofold_calibration_json']), Path(CFG['heterofold_map']),
                 ROOT / 'prepared/test.jsonl', REFERENCE / 'manifest.json',
                 Path(R.CFG['base_parent']) / 'runs/stage_a/best.pt',
                 Path(R.CFG['residual_parent']) / 'runs/stage_b/last.pt']:
        sources[str(path)] = C.sha(path)
    assert R.identity() == CFG['reference_reader_manifest_sha256']
    assert sources[str(Path(R.CFG['base_parent']) / 'runs/stage_a/best.pt')] == R.CFG['base_stage_a_sha256']
    assert sources[str(Path(R.CFG['residual_parent']) / 'runs/stage_b/last.pt')] == R.CFG['frozen_stage_b_sha256']
    for arm, item in CFG['readers'].items():
        parent = Path(item['root'])
        assert C.sha(parent / 'manifest.json') == item['manifest_sha256'], arm
        checkpoint = parent / 'runs/reader/epoch4_step1024.pt'
        assert C.sha(checkpoint) == item['checkpoint_sha256'], arm
        assert read(parent / 'runs/reader/completed.json')['status'] == 'completed'
        sources[str(parent / 'manifest.json')] = item['manifest_sha256']
        sources[str(checkpoint)] = item['checkpoint_sha256']
    for path in [REFERENCE / 'experiment.py', REFERENCE / 'reader.py', REFERENCE / 'grouped_translator.py',
                 REFERENCE / 'residual_translator.py', REFERENCE / 'config.json',
                 U.ROOT / 'experiment.py', U.ROOT / 'config.json', C.ROOT / 'core.py', C.ROOT / 'config.json']:
        sources[str(path)] = C.sha(path)
    for path in (Path(C.CFG['hf_root']) / 'heterofold').rglob('*.py'):
        sources[str(path)] = C.sha(path)
    for family in ['llama', 'qwen']:
        for name in ['config.json', 'tokenizer_config.json']:
            path = Path(C.CFG['models'][family]) / name
            sources[str(path)] = C.sha(path)
    manifest = dict(config=CFG, config_sha256=C.sha(ROOT / 'config.json'),
                    code={p.name: C.sha(p) for p in ROOT.glob('*.py')}, sources=sources,
                    selection='Sorted random sample of 512 original dev indices, random.Random(1234); no length, label or outcome filtering',
                    source_indices=indices, source_ids=[r['source_id'] for r in result],
                    paragraph_counts=dict(Counter(r['paragraph_count'] for r in result)),
                    max_prompt_tokens={f: max(len(r['frames'][f]['full_ids']) for r in result) for f in ['llama', 'qwen']},
                    prompt_truncation=False, discarded_paragraphs=0, calibration_question_overlap=0,
                    training=False, caveats=CFG['caveats'])
    save('manifest.json', manifest)
    mark('prepared/completed.json', status='completed', n=len(result),
         paragraph_counts=manifest['paragraph_counts'], max_prompt_tokens=manifest['max_prompt_tokens'])
    print(time.strftime('%F %T'), 'PREPARE COMPLETED', json.dumps(manifest['max_prompt_tokens']), flush=True)


def verify():
    manifest = read(ROOT / 'manifest.json')
    assert C.sha(ROOT / 'config.json') == manifest['config_sha256']
    for name, digest in manifest['code'].items():
        assert C.sha(ROOT / name) == digest, name
    for path, digest in manifest['sources'].items():
        assert C.sha(Path(path)) == digest, path


def load_states():
    states = {}
    for arm, item in CFG['readers'].items():
        record = torch.load(Path(item['root']) / 'runs/reader/epoch4_step1024.pt', map_location='cpu', weights_only=True)
        assert record['manifest_sha256'] == item['manifest_sha256'] and record['epoch'] == 4 and record['step'] == 1024
        assert record['state_scope'] == 'reader_lora_only'
        assert record['frozen_stage_a_sha256'] == R.CFG['base_stage_a_sha256']
        assert record['frozen_stage_b_sha256'] == R.CFG['frozen_stage_b_sha256']
        states[arm] = record['state']
    return states


@torch.no_grad()
def our_mapped(sender, base, row):
    ids = row['frames']['llama']['body_ids']
    key, value, _ = C.capture(sender, ids, len(ids), [0])
    index = row['source_indices']
    pieces = []
    for start in range(0, len(index), R.CFG['chunk_tokens']):
        ix = index[start:start + R.CFG['chunk_tokens']]
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pieces.append(base(key[:, ix][None].cuda(), value[:, ix][None].cuda()))
    k, v = (torch.cat([pair[j][0] for pair in pieces], dim=1).detach() for j in [0, 1])
    assert k.shape == v.shape == (36, len(row['target_indices']), 8, 128)
    assert torch.isfinite(k).all() and torch.isfinite(v).all()
    return k, v


@torch.no_grad()
def hf_cache(tok, sender_tok, sender, model, mapping, row):
    offsets, keys, values = capture_sender(sender_tok, sender, mapping, row['transfer_text'], device='cpu')
    cache, length = build_receiver_cache(tok, model, mapping, offsets, keys, values,
                                         row['transfer_text'], row['frames']['qwen']['prefix_ids'])
    assert length == len(row['frames']['qwen']['body_ids'])
    assert cache.get_seq_length() == length
    for layer in range(36):
        k, v = _cache_layer(cache, layer)
        assert torch.isfinite(k).all() and torch.isfinite(v).all(), ('HeteroFold nonfinite KV', row['id'], layer)
    return cache


@torch.no_grad()
def generate_value(model, tok, reader, row, cache, enabled):
    assert cache.get_seq_length() == len(row['frames']['qwen']['body_ids'])
    tick = time.monotonic()
    with reader.mode(enabled):
        text, count = C.R.generate(C.runtime(model, tok), torch.tensor([row['suffix_ids']], device='cuda'),
            CFG['max_new_tokens'], past_key_values=cache, past_length=cache.get_seq_length(), temperature=0., seed=CFG['seed'])
    prediction = clean_prediction(text)
    return dict(generation=text, prediction=prediction, generated_tokens=count,
                reached_cap=int(count >= CFG['max_new_tokens']), generation_seconds=time.monotonic()-tick,
                **answer_metrics(prediction, row['answer']))


def frozen_versions(model, excluded=()):
    skip = {id(p) for p in excluded}
    return {name: p._version for name, p in model.named_parameters() if id(p) not in skip}


@torch.no_grad()
def smoke(tok, model, sender_tok, sender, base, reader, states, mapping):
    rr = rows()
    longest = max(rr, key=lambda r: len(r['frames']['qwen']['body_ids']))
    chosen = list({r['id']: r for r in [rr[0], longest]}.values())
    records = []
    for row in chosen:
        with reader.mode(False):
            native = U.native_body(model, row)
            before = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0])
            cache = hf_cache(tok, sender_tok, sender, model, mapping, row)
        # Match the official Runtime.transfer_cache wrapper on the same first row.
        if row['id'] == rr[0]['id']:
            runtime = C.R.Runtime.__new__(C.R.Runtime)
            runtime.model, runtime.tokenizer = model, tok
            runtime.sender_model, runtime.sender_tokenizer, runtime.mapping = sender, sender_tok, mapping
            runtime.receiver_device = 'cuda:0'
            runtime.prefix = row['frames']['qwen']['rendered_prefix']
            with reader.mode(False):
                reference, length, _ = runtime.transfer_cache(row['transfer_text'])
            assert length == cache.get_seq_length()
            assert all(torch.equal(a, b) for layer in range(36)
                       for a, b in zip(_cache_layer(cache, layer), _cache_layer(reference, layer)))
            del reference
        conditions = {'qwen_native': generate_value(model, tok, reader, row, U.clone_cache(native, model), False),
                      'heterofold': generate_value(model, tok, reader, row, cache, False)}
        k, v = our_mapped(sender, base, row)
        for arm in CFG['readers']:
            reader.load_state_dict(states[arm])
            with reader.mode(True):
                logits = U.suffix_logits(model, row, U.translated_cache(model, row, native, k, v), [], [0])
                assert torch.isfinite(logits).all()
            conditions[arm] = generate_value(model, tok, reader, row, U.translated_cache(model, row, native, k, v), True)
        with reader.mode(False):
            restored = U.suffix_logits(model, row, U.clone_cache(native, model), [], [0])
        assert torch.equal(before, restored), 'Reader swap modified native inference'
        record = dict(id=row['id'], conditions=conditions, native_logits_restored_exact=True,
                      full_context=True, qwen_body_tokens=len(row['frames']['qwen']['body_ids']))
        mark('smoke/' + row['id'] + '.json', **record)
        records.append(record)
        del native, k, v, cache, logits, before, restored
        print(time.strftime('%F %T'), 'SMOKE', row['id'], record['qwen_body_tokens'], flush=True)
    assert all(m.enabled_calls > 0 for m in reader.branches.values())
    mark('smoke/completed.json', status='completed', n=len(records),
         official_heterofold_cache_bitwise_parity=True, all_72_reader_branches_used=True,
         longest_selected_prompt_checked=True, native_logits_restored_exact=True)


@torch.no_grad()
def evaluate():
    verify()
    if (ROOT / 'completed.json').exists():
        assert read(ROOT / 'completed.json')['manifest_sha256'] == identity()
        return
    started = time.monotonic()
    tok, model, base, reader = R.load_all()
    sender_tok, sender = C.model('llama')
    mapping = load_map(CFG['heterofold_map'], device='cuda:0', dtype=model.dtype)
    assert int(mapping['Ls']) == 28 and int(mapping['Lr']) == 36
    assert describe(mapping)['best_validation_loss'] == read(Path(CFG['heterofold_map']).parent / 'calibration_log.json')['best_validation_loss']
    states = load_states()
    model.requires_grad_(False)
    original = dict(qwen=frozen_versions(model, reader.parameters()), llama=frozen_versions(sender), base=frozen_versions(base))
    status(status='running', phase='smoke')
    smoke(tok, model, sender_tok, sender, base, reader, states, mapping)
    records = []
    rr = rows()
    for i, row in enumerate(rr, 1):
        path = 'evaluation/per_sample/' + row['id'] + '.json'
        if (ROOT / path).exists():
            record = read(ROOT / path)
            assert record['manifest_sha256'] == identity() and record['source_id'] == row['source_id']
        else:
            with reader.mode(False):
                native = U.native_body(model, row)
            conditions = {}
            for arm in ARMS:
                arm_path = 'evaluation/per_arm/' + arm + '/' + row['id'] + '.json'
                if (ROOT / arm_path).exists():
                    item = read(ROOT / arm_path)
                    assert item['manifest_sha256'] == identity() and item['id'] == row['id']
                    conditions[arm] = item['result']
            if 'qwen_native' not in conditions:
                value = generate_value(model, tok, reader, row, U.clone_cache(native, model), False)
                mark('evaluation/per_arm/qwen_native/' + row['id'] + '.json', id=row['id'], result=value)
                conditions['qwen_native'] = value
            if 'heterofold' not in conditions:
                with reader.mode(False):
                    cache = hf_cache(tok, sender_tok, sender, model, mapping, row)
                value = generate_value(model, tok, reader, row, cache, False)
                mark('evaluation/per_arm/heterofold/' + row['id'] + '.json', id=row['id'], result=value)
                conditions['heterofold'] = value
                del cache
            missing = [arm for arm in CFG['readers'] if arm not in conditions]
            if missing:
                k, v = our_mapped(sender, base, row)
                for arm in missing:
                    reader.load_state_dict(states[arm])
                    value = generate_value(model, tok, reader, row, U.translated_cache(model, row, native, k, v), True)
                    mark('evaluation/per_arm/' + arm + '/' + row['id'] + '.json', id=row['id'], result=value)
                    conditions[arm] = value
                del k, v
            mark(path, id=row['id'], source_id=row['source_id'], source_index=row['source_index'],
                 question=row['question'], gold=row['answer'], paragraph_count=row['paragraph_count'],
                 question_type=row['question_type'], level=row['level'], prompt_sha256=C.digest(row['prompt']),
                 qwen_body_tokens=len(row['frames']['qwen']['body_ids']), conditions=conditions)
            record = read(ROOT / path)
            del native
        records.append(record)
        elapsed = time.monotonic()-started
        status(status='running', phase='evaluation', sample=i, total=len(rr), elapsed_seconds=elapsed,
               estimated_remaining_seconds=elapsed/i*(len(rr)-i))
        if i % 8 == 0 or i == 1:
            print(time.strftime('%F %T'), f'EVALUATION {i}/{len(rr)} elapsed={elapsed:.1f}s', flush=True)
    assert frozen_versions(model, reader.parameters()) == original['qwen']
    assert frozen_versions(sender) == original['llama'] and frozen_versions(base) == original['base']
    assert all(p.grad is None for m in [model, sender, base] for p in m.parameters())
    verify()
    metrics = {}
    for arm in ARMS:
        vv = [r['conditions'][arm] for r in records]
        metrics[arm] = dict(n=len(vv), em=sum(x['em'] for x in vv)/len(vv),
                            f1=sum(x['f1'] for x in vv)/len(vv), em_correct=int(sum(x['em'] for x in vv)),
                            cap_count=sum(x['reached_cap'] for x in vv),
                            mean_generated_tokens=sum(x['generated_tokens'] for x in vv)/len(vv))
    paired = {arm + '_minus_' + comparator: {metric: paired_interval([
        r['conditions'][arm][metric]-r['conditions'][comparator][metric] for r in records], CFG['seed'])
        for metric in ['em', 'f1']} for comparator in ['qwen_native', 'heterofold']
        for arm in ARMS if arm != comparator}
    C.write_jsonl(ROOT / 'evaluation/per_sample.jsonl', records)
    mark('summary.json', status='completed', metrics=metrics, paired=paired, training=False,
         prompt_truncation=False, frozen_models_unchanged=True, elapsed_seconds=time.monotonic()-started,
         caveats=CFG['caveats'])
    lines = ['# HotpotQA distractor512: frozen HeteroFold and Readers', '',
             '512 dev questions sampled with seed1234; every original paragraph preserved. Shared prompt, greedy generation (64 tokens), answer EM/F1. No training.', '',
             '| Method | n | EM | F1 | Cap count |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        item = metrics[arm]
        lines.append(f'| {arm} | {item["n"]} | {item["em"]*100:.2f}% | {item["f1"]*100:.2f}% | {item["cap_count"]} |')
    lines += ['', *['- ' + s for s in CFG['caveats']]]
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    mark('completed.json', status='completed', n=len(records), training=False, finished_at=time.strftime('%F %T'))
    status(status='completed', sample=len(records), total=len(records))
    print(time.strftime('%F %T'), 'COMPLETED', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'evaluate':
        evaluate()
    elif sys.argv[1] == 'verify':
        verify()
    else:
        raise ValueError(sys.argv[1])
