"""Prepare paired length sweeps with a fixed question and intact evidence."""

import argparse
from bisect import bisect_right
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys
import unicodedata

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
CORE = ROOT.parent / "recolor_kl_l3_to_q4_seed1234"
BASE = ROOT.parent / "dst_stage1_rank64_then_static_m32_a4_seed1234"
HOT_SOURCE = ROOT.parent / "heterogeneous_multidoc_frozen_writers_v0_seed1234/data/hotpot_dev_distractor_v1.json"
LENGTHS = (1024, 2048, 4096, 8192)
COUNT = 200
SEED = 1234
MAX_SLACK = 8
DST = ROOT.parent / "depth_state_translator_fullsync_l3_to_q4_seed1234"
sys.path.insert(0, str(DST))
from alignment import fullsync


def file_hash(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class Grid:
    def __init__(self):
        core = json.loads((CORE / "config.json").read_text())
        self.path = Path(core["models"]["receiver"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.sender = AutoTokenizer.from_pretrained(core["models"]["sender"], local_files_only=True)
        header = json.loads((CORE / "manifest.json").read_text())
        self.header = header["prefix"]
        self.prefix_ids = header["prefix_ids"]
        self.suffix = header["assistant"]
        self.suffix_ids = self.encode(self.suffix)
        assert self.prefix_ids == [151644, 872, 198] and len(self.suffix_ids) == 9

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def offsets(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]

    def fit(self, render, maximum, target):
        # The target includes the instruction, context and question, excluding
        # the fixed Qwen public header and the answer-side suffix.
        lo, hi, best = 0, maximum, None
        while lo <= hi:
            mid = (lo + hi) // 2
            text = render(mid)
            ids = self.encode(text)
            if len(ids) <= target:
                if best is None or len(ids) > len(best[1]):
                    best = (text, ids, mid)
                if len(ids) == target:
                    break
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None or target - len(best[1]) > MAX_SLACK:
            return None
        text, ids, used = best
        assert self.encode(self.header + text + self.suffix) == self.prefix_ids + ids + self.suffix_ids
        return dict(prompt=text, content_ids=ids, suffix_ids=self.suffix_ids,
                    nominal_length=target, actual_length=len(ids), distractor_token_cut=used)

    def align(self, variants):
        try:
            for row in variants:
                result = fullsync(row["prompt"], self.sender, self.tokenizer, row["content_ids"])
                row["sync_source_indices"] = list(result.source_indices)
                row["sync_audit"] = result.audit
        except ValueError:
            return False
        return True


def inject(text, offsets, cut, positions, evidence, paragraph_starts=None):
    """Insert complete evidence into a prefix of one ordered distractor stream."""
    end = offsets[cut - 1][1] if cut else 0
    parts = []
    left = 0
    for depth, fragment in zip(positions, evidence):
        n = int(depth * cut)
        right = offsets[n - 1][1] if n else 0
        if paragraph_starts is not None:
            right = paragraph_starts[bisect_right(paragraph_starts, right) - 1]
        parts.extend([text[left:right], fragment])
        left = right
    parts.append(text[left:end])
    return "\n\n".join(parts)


def ruler(grid):
    corpus_path = SOURCES / "ruler_essay_corpus.json"
    corpus = json.loads(corpus_path.read_text())
    essay = re.sub(r"\s+", " ", corpus["text"]).strip()
    offsets = grid.offsets(essay)
    rng = random.Random(SEED)
    values = rng.sample(range(1_000_000, 10_000_000), COUNT)
    starts = iter(rng.sample(range(len(offsets) - LENGTHS[-1] - 256), COUNT * 4))
    adjectives = ("amber", "brisk", "cedar", "distant", "ember", "frost", "golden", "hollow", "indigo", "jade")
    nouns = ("anchor", "branch", "cabin", "delta", "engine", "field", "garden", "harbor", "island", "lantern",
             "meadow", "nest", "orchard", "path", "quarry", "river", "summit", "tower", "valley", "willow")
    keys = [f"{a}-{n}" for a in adjectives for n in nouns]
    rng.shuffle(keys)
    rows = []
    for i, (key, value) in enumerate(zip(keys, values)):
        depth = (0.10, 0.35, 0.65, 0.90)[i % 4]
        needle = f"One of the special magic numbers for {key} is: {value}."
        question = f"What is the special magic number for {key} mentioned in the provided text?"
        for start in starts:
            origin = offsets[start][0]
            span = [(a - origin, b - origin) for a, b in offsets[start:start + LENGTHS[-1] + 128]]
            text = essay[origin:origin + span[-1][1]]

            def render(cut):
                body = inject(text, span, cut, [depth], [needle])
                return f"A special magic number is hidden within the following text.\n\n{body}\n\n{question}\nOutput only the number."

            variants = [grid.fit(render, min(len(span), n + 128), n) for n in LENGTHS]
            if all(variants) and all(v["prompt"].count(needle) == 1 for v in variants) and grid.align(variants):
                break
        else:
            raise RuntimeError("RULER ran out of compatible essay windows")
        for fitted in variants:
            position = len(grid.encode(fitted["prompt"].split(needle, 1)[0]))
            rows.append(dict(id=f"ruler_{i:04d}", task="ruler", question=question, gold=str(value),
                             source_essay_start_token=start, needle_depth=depth, evidence_positions=[position], **fitted))
    return rows, dict(
        task_variant="RULER niah_single_2 paired adaptation: one seven-digit needle, essay haystack, 50 questions at each of four depths",
        upstream_commit=json.loads((SOURCES / "RULER.commit.json").read_text())["sha"],
        essay_corpus_revision=corpus["revision"], sources={str(corpus_path): file_hash(corpus_path)},
    )


def blocked_questions():
    paths = [root / "prepared" / (split + ".jsonl") for root in (BASE, CORE)
             for split in ("train", "validation", "test")]
    for path in paths:
        assert path.is_file(), path
    blocked = {normalized(r["question"]) for path in paths for r in jsonl(path) if r.get("question")}
    return blocked, {str(path): file_hash(path) for path in paths}


def helmet(grid):
    data_path = SOURCES / "helmet_hotpotqa_k105.jsonl"
    gold_by_question = {normalized(r["question"]): r for r in json.loads(HOT_SOURCE.read_text())}
    blocked, hashes = blocked_questions()
    offsets = []
    # Store byte offsets so shuffled sampling needs only one retrieval row
    # in memory, even if the full release contains many long contexts.
    with data_path.open("rb") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            offsets.append(offset)
    random.Random(SEED).shuffle(offsets)
    rows, chosen, rejected = [], set(), Counter()
    with data_path.open("rb") as file:
        for offset in offsets:
            file.seek(offset)
            source = json.loads(file.readline())
            question = source["question"]
            key = normalized(question)
            if key in blocked or key in chosen or key not in gold_by_question:
                rejected["excluded_or_no_annotation"] += 1
                continue
            annotated = gold_by_question[key]
            supports = {}
            for title, index in annotated["supporting_facts"]:
                supports.setdefault(title, []).append(index)
            originals = dict(annotated["context"])
            if len(supports) < 2 or not all(title in originals and all(0 <= i < len(originals[title]) for i in idxs)
                                            for title, idxs in supports.items()):
                rejected["missing_support"] += 1
                continue
            if not all(originals[title][i].strip() for title, idxs in supports.items() for i in idxs):
                rejected["empty_support_sentence"] += 1
                continue
            evidence = [f"Document (Title: {t}): {''.join(originals[t]).strip()}" for t in supports]
            depths = [0.22 + 0.50 * j / (len(evidence) - 1) for j in range(len(evidence))]
            support_titles = {normalized(t) for t in supports}
            distractors = []
            used = set()
            for ctx in source["ctxs"]:
                if normalized(ctx["title"]) in support_titles:
                    continue
                passage = f"Document (Title: {ctx['title']}): {ctx['text'].strip()}"
                if passage in used:
                    continue
                used.add(passage)
                distractors.append(passage)
            negative = "\n\n".join(distractors)
            paragraph_starts = [0]
            for passage in distractors[:-1]:
                paragraph_starts.append(paragraph_starts[-1] + len(passage) + 2)
            negative_offsets = grid.offsets(negative)

            def render(cut):
                context = inject(negative, negative_offsets, cut, depths, evidence, paragraph_starts)
                return ("Use the given documents to write a concise and short answer to the question. "
                        "Write your answer in the following format:\nAnswer: [answer]\n\n"
                        f"{context}\n\nQuestion: {question}")

            variants = [grid.fit(render, min(len(negative_offsets), n + 128), n) for n in LENGTHS]
            if not all(variants):
                rejected["does_not_fit_all_lengths"] += 1
                continue
            if any(not all(v["prompt"].count(p) == 1 for p in evidence) for v in variants):
                rejected["ambiguous_evidence_occurrence"] += 1
                continue
            if not grid.align(variants):
                rejected["fullsync_incompatible"] += 1
                continue
            source_id = str(source.get("id", annotated["_id"]))
            for v in variants:
                positions = [len(grid.encode(v["prompt"].split(p, 1)[0])) for p in evidence]
                for title, indices in supports.items():
                    assert all(originals[title][i].strip() in v["prompt"] for i in indices)
                rows.append(dict(id=f"helmet_{annotated['_id']}", task="helmet", question=question, gold=annotated["answer"],
                                 helmet_answers=source["answers"], helmet_id=source_id,
                                 supporting_facts=annotated["supporting_facts"], evidence_positions=positions,
                                 evidence_sha256=[sha256(p.encode()).hexdigest() for p in evidence], **v))
            chosen.add(key)
            if len(chosen) == COUNT:
                break
    assert len(chosen) == COUNT, (len(chosen), dict(rejected))
    hashes.update({str(path): file_hash(path) for path in (data_path, HOT_SOURCE)})
    return rows, dict(
        task_variant="HELMET HotpotQA k105, zero-shot, fixed gold supporting paragraphs restored from HotpotQA distractor dev; ranked HELMET negatives",
        upstream_commit=json.loads((SOURCES / "HELMET.commit.json").read_text())["sha"],
        data_revision=json.loads((SOURCES / "helmet_hf_meta.json").read_text())["sha"],
        candidate_rows=len(offsets), rejected=dict(rejected), sources=hashes,
    )


def main(task):
    root = ROOT / "prepared" / task
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    path = root / "rows.jsonl"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["rows_sha256"] == file_hash(path)
        return
    grid = Grid()
    rows, provenance = {"ruler": ruler, "helmet": helmet}[task](grid)
    by_length = {n: [r for r in rows if r["nominal_length"] == n] for n in LENGTHS}
    first = [r["id"] for r in by_length[LENGTHS[0]]]
    assert len(first) == len(set(first)) == COUNT
    assert all([r["id"] for r in group] == first for group in by_length.values())
    for group in by_length.values():
        assert [(r["question"], r["gold"], r.get("evidence_sha256")) for r in group] == [
            (r["question"], r["gold"], r.get("evidence_sha256")) for r in by_length[LENGTHS[0]]]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as file:
        for n in LENGTHS:
            for row in by_length[n]:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    save(manifest_path, dict(
        task=task, seed=SEED, lengths=list(LENGTHS), n_per_length=COUNT, rows_sha256=file_hash(path),
        prepared_code_sha256=file_hash(Path(__file__)), public_header_ids=grid.prefix_ids,
        suffix_ids=grid.suffix_ids, tokenizer_sha256=file_hash(grid.path / "tokenizer.json"),
        actual_ranges={n: [min(r["actual_length"] for r in group), max(r["actual_length"] for r in group)]
                       for n, group in by_length.items()}, **provenance,
    ))
    print(task, "prepared", len(rows), "rows, paired IDs and intact evidence", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("ruler", "helmet"))
    main(parser.parse_args().task)
