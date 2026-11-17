"""Freeze full-document QuALITY and L-Eval closed-ended questions."""
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
QUALITY = ROOT.parent / "dst_e4_quality_latency_seed1234" / "quality_dev.jsonl"
LEVAL = "cd34b050269148aed75acbbe4a599873ad0f37e9"
TASKS = ("quality", "coursera", "sfiction")
COUNT = 64
SEED = 1234
LETTERS = "ABCD"


def quality(core):
    expected = core.read(QUALITY.parent / "data_manifest.json")["source_sha256"]
    assert core.sha(QUALITY) == expected
    for source in core.read_jsonl(QUALITY):
        for question in source["questions"]:
            gold = int(question["gold_label"]) - 1
            assert 0 <= gold < len(question["options"]) == 4
            options = "\n".join(f"{label}. {text}" for label, text in zip(LETTERS, question["options"]))
            tail = f"\n\nQuestion:\n{question['question']}\n\nChoices:\n{options}\n\nAnswer with only the option label."
            yield dict(id=str(question["question_unique_id"]), document=str(source["article_id"]),
                       context=source["article"], gold=LETTERS[gold], hard=bool(question["difficult"]),
                       before="Read the following article and answer the multiple-choice question.\n\nArticle:\n",
                       after=tail)


def coursera(rows):
    guide = ("Read the following course material and answer the question. One or more options may be correct. "
             "Answer only with the selected option letters in alphabetical order, such as A or AB.\n\nDocument:\n")
    for i, source in enumerate(rows):
        for j, (question, gold) in enumerate(zip(source["instructions"], source["outputs"], strict=True)):
            assert gold == "".join(sorted(set(gold))) and re.fullmatch(r"[A-D]+", gold)
            yield dict(id=f"coursera_{i}_{j}", document=str(i), context=source["input"], gold=gold,
                       before=guide, after=f"\n\n{question}\n\nAnswer with only the selected option letters.")


def sfiction(rows):
    for i, source in enumerate(rows):
        for j, (question, gold) in enumerate(zip(source["instructions"], source["outputs"], strict=True)):
            match = re.fullmatch(r"(True|False) \[fact: (True|False)\]", gold)
            loyalty, fact = match.groups()
            statement = re.sub(r"\s*Answer this question based on the world described in the document\.?\s*$", "", question)
            yield dict(id=f"sfiction_{i}_{j}", document=str(i), context=source["input"], gold=loyalty, fact=fact,
                       before="Answer based on the world described in the following science-fiction document.\n\nDocument:\n",
                       after=f"\n\n{statement}\n\nAnswer only True or False based on the document.",
                       fact_prompt=(f"{statement}\n\nAnswer only True or False based on real-world facts, "
                                    "using your background knowledge."))


def prepare(core):
    paths = {"quality": QUALITY, "coursera": ROOT / "data/coursera.jsonl", "sfiction": ROOT / "data/sci_fi.jsonl"}
    samples = {"quality": list(quality(core)), "coursera": list(coursera(core.read_jsonl(paths["coursera"]))),
               "sfiction": list(sfiction(core.read_jsonl(paths["sfiction"])))}
    for name, rows in samples.items():
        assert len(rows) == len({r["id"] for r in rows}) and len(rows) >= COUNT
        assert all(r["context"] for r in rows)
        samples[name] = random.Random(SEED).sample(rows, COUNT) if len(rows) > COUNT else rows
    metadata = dict(count=COUNT, seed=SEED, leval_revision=LEVAL, truncation=False,
                    sources_sha256={k: core.sha(p) for k, p in paths.items()},
                    prompts={task: [dict(id=r["id"], document=r["document"], context_sha256=core.digest(r["context"]),
                                         prompt_sha256=core.digest(r["before"]+r["context"]+r["after"]))
                                    for r in rows] for task, rows in samples.items()})
    return samples, metadata
