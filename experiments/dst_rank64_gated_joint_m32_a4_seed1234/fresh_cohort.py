"""Freeze a fresh exact-question-disjoint evaluation cohort on the v1 prompts."""

from collections import Counter
import json
import os
from pathlib import Path
import random

import experiment_common as G
from experiment_common import C, CFG, ROOT, SETTINGS


def raw_sources():
    root = ROOT.parent
    original = G.N.ORIGINAL
    source = C.read(original / "cross_dataset5x64_seed1234/config.json")["sources"]
    cache = root / "heterofold_official_l3_to_q4_seed20260825/hf_cache/datasets"
    [arc] = cache.glob("allenai___ai2_arc/ARC-Challenge/*/*/ai2_arc-test.arrow")
    gsm = C.read(G.N.GSM / "config.json")["source"]
    return {**{task: Path(path) for task, path in source.items()},
            "arc": arc, "gsm8k": Path(gsm),
            "hotpotqa": root / "heterogeneous_multidoc_frozen_writers_v0_seed1234/data/hotpot_dev_distractor_v1.json"}


def raw_rows(path):
    if path.suffix == ".arrow":
        from datasets import Dataset
        return Dataset.from_file(str(path))
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return C.read_jsonl(path)
    return C.read(path)


def history():
    seen = set()
    files = {}
    for directory, subdirs, names in os.walk(ROOT.parent):
        path = Path(directory)
        subdirs[:] = [x for x in subdirs if not x.startswith(".") and x not in {
            "hf_cache", "kv_cache", "native_prefix", "runs", "models", "reference", "sources", "__pycache__",
        } and path / x != ROOT]
        for name in names:
            candidate = path / name
            if name.endswith(".jsonl") and ("prepared" in path.parts or name == "per_sample.jsonl"):
                with candidate.open() as stream:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            if isinstance(row, dict) and row.get("question"):
                                seen.add(C.norm(row["question"]))
                files[str(candidate)] = C.sha(candidate)
            elif name.endswith(".json") and "per_sample" in path.parts:
                row = C.read(candidate)
                if isinstance(row, dict) and row.get("question"):
                    seen.add(C.norm(row["question"]))
    if (calibration := Path(C.CFG["calibration"])).exists():
        seen.update(C.norm(row["question"]) for row in C.read(calibration)["rows"])
        files[str(calibration)] = C.sha(calibration)
    return seen, files


def format_row(task, d, question, assistant):
    options = {}
    if task in ("arc", "gsm8k", "hellaswag"):
        prompt, candidates, gold, kind, _ = C.SC.format_item(task, d)
        options = dict(candidates=candidates, gold=gold, kind=kind, normalize=task == "hellaswag")
        suffix = assistant + C.SC.ANSWER_CUE.get(kind, "")
    elif task in ("hotpotqa", "2wiki", "musique"):
        context = d["context"] if task != "musique" else [(p["title"], [p["paragraph_text"]]) for p in d["paragraphs"]]
        answers = list(dict.fromkeys([d["answer"], *d.get("answer_aliases", [])]))
        passages = [f"Passage {i}:\n{title}\n{''.join(sentences).strip()}"
                    for i, (title, sentences) in enumerate(context, 1)]
        prompt = ("Answer the question using the passages below. Output only the short answer, without an explanation.\n\n"
                  + "\n\n".join(passages) + f"\n\nQuestion: {question}\nGive only the answer.")
        options = dict(kind="generation", gold=d["answer"], answers=answers, paragraph_count=len(context))
        suffix = assistant
    else:
        if task == "sciq":
            choices = [d["correct_answer"], d["distractor1"], d["distractor2"], d["distractor3"]]
            permutation = list(range(4))
            seed = int(C.digest(dict(seed=CFG["seed"], question=C.norm(question)))[:16], 16)
            random.Random(seed).shuffle(permutation)
            choices, gold = [choices[i] for i in permutation], permutation.index(0)
            instruction = "Answer the following multiple-choice question using the passage. Select the single correct option.\n\nPassage:\n" + d["support"]
        else:
            choices, gold = d["options"], int(d["answer_index"])
            assert d["answer"] == chr(65 + gold)
            instruction = f"Answer the following multiple-choice question about {d['category'].replace('_', ' ')}. Select the single correct option."
        labels = [chr(65+i) for i in range(len(choices))]
        formatted = "\n".join(f"{label}. {str(choice).strip()}" for label, choice in zip(labels, choices))
        prompt = instruction + f"\n\nQuestion: {question}\n\nChoices:\n{formatted}\n\nAnswer with only the option label."
        options = dict(kind="label", candidates=labels, gold=gold, normalize=False)
        suffix = assistant + C.SC.ANSWER_CUE["arc_label"]
    return dict(prompt=prompt, continuation_suffix=suffix, **options)


def prepare():
    align = G.reference.Z.scoped_module("gated_joint_alignment", G.reference.B.D.ROOT / "alignment.py")
    sender = C.AutoTokenizer.from_pretrained(C.CFG["models"]["sender"], local_files_only=True)
    receiver = C.AutoTokenizer.from_pretrained(C.CFG["models"]["receiver"], local_files_only=True)
    affix = C.read(C.ROOT / "manifest.json")
    sources = raw_sources()
    seen, files = history()
    old = set(seen)
    for split in ("train", "validation", "test"):
        old.update(C.norm(row["question"]) for row in G.rows(split))
    seen.update(old)
    counts, output = {}, []
    count = SETTINGS["evaluation_per_task"]
    for task in CFG["test_counts"]:
        data = raw_rows(sources[task])
        indices = list(range(len(data)))
        rng = random.Random(SETTINGS["evaluation_seed"])
        rng.shuffle(indices)
        quotas = {}
        if task == "mmlu_pro":
            categories = sorted({row["category"] for row in data})
            rng.shuffle(categories)
            quotas = {key: count//len(categories) + (i < count % len(categories)) for i,key in enumerate(categories)}
        accepted, category_counts = 0, Counter()
        for index in indices:
            d = data[index]
            question = d["ctx"] if task == "hellaswag" else d["question"]
            key = C.norm(question)
            if key in seen or (quotas and category_counts[d["category"]] >= quotas[d["category"]]):
                continue
            row = format_row(task, d, question, affix["assistant"])
            try:
                grid = align.fullsync(row["prompt"], sender, receiver)
            except ValueError:
                continue
            body = list(grid.receiver_ids)
            suffix = receiver.encode(row["continuation_suffix"], add_special_tokens=False)
            assert receiver.encode(affix["prefix"]+row["prompt"]+row["continuation_suffix"],
                                   add_special_tokens=False) == affix["prefix_ids"]+body+suffix
            assert len(affix["prefix_ids"])+len(body)+len(suffix)+32+512 < CFG["max_context_tokens"]
            assert task != "musique" or d["answerable"]
            output.append(dict(row, id=f"{task}_fresh_joint_{index}", task=task, question=question,
                source_index=index, source_id=str(d.get("_id", d.get("id", d.get("question_id", index)))),
                source_doc_sha256=C.digest(d), content_ids=body, suffix_ids=suffix,
                sender_ids=list(grid.source_ids), sync_source_indices=list(grid.source_indices), sync_audit=grid.audit))
            seen.add(key)
            accepted += 1
            if quotas:
                category_counts[d["category"]] += 1
            if accepted == count:
                break
        assert accepted == count, (task, accepted, count)
        counts[task] = dict(n=accepted, category_counts=dict(category_counts))
        C.log("FRESH COHORT", task, accepted)
    assert len({C.norm(r["question"]) for r in output}) == len(output)
    assert not any(C.norm(r["question"]) in old for r in output)
    path = ROOT / "prepared/test.jsonl"
    C.write_jsonl(path, output)
    return dict(counts=counts, source_sha256={str(p): C.sha(p) for p in sources.values()},
                exclusion_files=files, excluded_distinct_questions=len(old), cohort_sha256=C.sha(path))
