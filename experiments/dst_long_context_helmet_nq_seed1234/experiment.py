"""Paired length sweep on HELMET Natural Questions retrieval passages."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "dst_long_context_ruler_helmet_seed1234"
sys.path.insert(0, str(PREVIOUS))
from build_cohort import COUNT, LENGTHS, SEED, Grid, blocked_questions, file_hash, inject, normalized, save

TASK = "helmet_nq"
SOURCES = PREVIOUS / "sources"
DATA = SOURCES / "helmet_nq_k105.jsonl"
DEPTHS = (0.10, 0.35, 0.65, 0.90)
QA = "Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n"


def passage(ctx):
    return f"Document (Title: {ctx['title']}): {ctx['text'].strip()}"


def contains_answer(text, aliases):
    padded = f" {normalized_answer(text)} "
    return any(f" {alias} " in padded for alias in aliases)


def normalized_answer(text):
    import string

    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def build_example(grid, source, depth):
    question = source["question"]
    answers = sorted(set(source["answers"]))
    aliases = {normalized_answer(value) for value in answers} - {""}
    positives = {passage(ctx) for ctx in source["positive_ctxs"]}
    contexts = [passage(ctx) for ctx in source["ctxs"]]
    # The k105 release marks the inserted positive via retrieval depth.
    inserted = contexts[round(source["depth"] * len(contexts))]
    if inserted not in positives or not aliases or not contains_answer(inserted, aliases):
        return None, "missing_supported_alias"
    distractors = list(dict.fromkeys(p for p in contexts if p not in positives and not contains_answer(p, aliases)))
    if not distractors:
        return None, "empty_distractors"
    text = "\n\n".join(distractors)
    offsets = grid.offsets(text)
    starts, pos = [], 0
    for doc in distractors:
        starts.append(pos)
        pos += len(doc) + 2

    def render(cut):
        body = inject(text, offsets, cut, [depth], [inserted], starts)
        return f"{QA}{body}\n\nQuestion: {question}"

    variants = [grid.fit(render, min(len(offsets), target + 128), target) for target in LENGTHS]
    if not all(variants):
        return None, "does_not_fit_all_lengths"
    if not grid.align(variants):
        return None, "fullsync_incompatible"
    evidence_hash = sha256(inserted.encode()).hexdigest()
    sample_id = "nq_" + sha256(question.encode()).hexdigest()[:20]
    for row in variants:
        assert row["prompt"].count(inserted) == 1
        position = len(grid.encode(row["prompt"].split(inserted, 1)[0]))
        row.update(id=sample_id, task=TASK, question=question, gold=answers[0],
                   helmet_answers=answers, evidence_positions=[position],
                   evidence_sha256=evidence_hash, assigned_depth=depth, retrieval_depth=source["depth"])
    return variants, None


def prepare():
    output = ROOT / "prepared" / TASK
    output.mkdir(parents=True, exist_ok=True)
    path, manifest = output / "rows.jsonl", output / "manifest.json"
    sources = [DATA, Path(__file__), PREVIOUS / "build_cohort.py", PREVIOUS / "evaluate.py"]
    if manifest.exists():
        pins = json.loads(manifest.read_text())
        assert pins["rows_sha256"] == file_hash(path)
        assert all(file_hash(Path(p)) == h for p, h in pins["sources"].items())
        return
    blocked, hashes = blocked_questions()
    grid = Grid()
    # A canonical published variant supplies one candidate per original question.
    pool = {}
    with DATA.open("rb") as stream:
        while True:
            at = stream.tell()
            line = stream.readline()
            if not line:
                break
            source = json.loads(line)
            if source["depth"] == 0.4:
                key = normalized(source["question"])
                assert key not in pool
                pool[key] = at
    assert len(pool) == 993, len(pool)
    order = sorted(pool)
    random.Random(SEED).shuffle(order)
    rejected, selected = Counter(), []
    with DATA.open("rb") as stream:
        for key in order:
            if key in blocked:
                rejected["prior_training_validation_or_test_question"] += 1
                continue
            stream.seek(pool[key])
            variants, reason = build_example(grid, json.loads(stream.readline()), DEPTHS[len(selected) % len(DEPTHS)])
            if variants is None:
                rejected[reason] += 1
                continue
            selected.append(variants)
            if len(selected) % 25 == 0:
                print(f"Prepared {len(selected)}/{COUNT} paired questions", flush=True)
            if len(selected) == COUNT:
                break
    assert len(selected) == COUNT, (len(selected), rejected)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        for group in zip(*selected):
            for row in group:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    hashes.update({str(p): file_hash(p) for p in sources})
    save(manifest, dict(
        task=TASK, seed=SEED, n_per_length=COUNT, lengths=LENGTHS,
        rows_sha256=file_hash(path), public_header_ids=grid.prefix_ids, suffix_ids=grid.suffix_ids,
        nominal_evidence_depths=DEPTHS, rejected=dict(rejected), sources=hashes,
        helmet_commit=json.loads((SOURCES / "HELMET.commit.json").read_text())["sha"],
        data_revision=json.loads((SOURCES / "helmet_hf_meta.json").read_text())["sha"],
        protocol="Zero-shot paired adaptation of the HELMET NQ k105 corpus. Keep the same complete evidence passage and ordered distractor stream. Fit the prompt body on the Qwen token grid.",
    ))
    print(f"HELMET NQ prepared {COUNT} questions × {len(LENGTHS)} lengths", flush=True)


def score(text, row):
    # Match the official QA postprocessor: score raw and parsed outputs against
    # all released aliases, taking the maximum for each metric.
    from evaluate import answer_metrics

    match = re.search(r"Answer:(.*)(?:\n|$)", text, re.IGNORECASE)
    if match is None:
        match = re.match(r"(.*)(?:\n|$)", text)
    prediction = re.sub(r"^Answer:", "", match[1].strip(), flags=re.IGNORECASE).strip()
    answers = row["helmet_answers"]
    pairs = [answer_metrics(pred, answer) for pred in (text, prediction) for answer in answers]
    subem = any(normalized_answer(answer) in normalized_answer(pred)
                for pred in (text, prediction) for answer in answers)
    return prediction, dict(helmet_subem=int(subem), **{key: max(v[key] for v in pairs) for key in ("em", "f1")})


def summarize(task, records, manifest):
    from evaluate import ARMS, intervals

    keys = ("helmet_subem", "f1", "em")
    output = ROOT / "evaluation" / task
    lengths, rows = {}, []
    for length, group in records.items():
        if not group:
            continue
        metrics = {arm: {key: sum(r["conditions"][arm][key] for r in group) / len(group) for key in keys}
                   for arm in ARMS}
        paired = {arm: {key: intervals([r["conditions"]["native"][key] for r in group],
                                       [r["conditions"][arm][key] for r in group]) for key in keys}
                  for arm in ARMS[1:]}
        dst_minus_recolor = intervals([r["conditions"]["recolor_c"]["helmet_subem"] for r in group],
                                      [r["conditions"]["dst_rank64"]["helmet_subem"] for r in group])
        lengths[length] = dict(n=len(group), metrics=metrics, versus_native=paired,
                               dst_minus_recolor_subem=dst_minus_recolor)
        values = " | ".join(f'{metrics[arm]["helmet_subem"]:.2%}' for arm in ARMS)
        rows.append(f"| {length} | {values} |")
    finished = all(len(records[length]) == COUNT for length in LENGTHS)
    if finished:
        ids = [r["id"] for r in records[LENGTHS[0]]]
        assert all([r["id"] for r in records[length]] == ids for length in LENGTHS)
        (output / "RESULTS.md").write_text(
            "# HELMET Natural Questions paired sweep (SubEM)\n\n"
            "| Qwen body tokens | Qwen Native | Recolor+C | DST+Rank64 |\n"
            "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n")
    save(output / "summary.json", dict(task=task, status="completed" if finished else "running",
         manifest_sha256=manifest, primary_metric="helmet_subem", lengths=lengths,
         completed=sum(map(len, records.values())), total=COUNT * len(LENGTHS)))


def evaluate():
    import importlib.util

    # The previous runner imports the cache and generation implementation.
    spec = importlib.util.spec_from_file_location("helmet_nq_long_evaluation", PREVIOUS / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.helmet_score = score
    module.summarize = summarize
    module.main(TASK)


if __name__ == "__main__":
    {"prepare": prepare, "evaluate": evaluate}[sys.argv[1]]()
