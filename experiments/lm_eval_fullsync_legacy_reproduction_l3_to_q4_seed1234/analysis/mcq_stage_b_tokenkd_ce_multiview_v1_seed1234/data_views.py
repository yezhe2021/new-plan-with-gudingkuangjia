"""Build training-only augmentation; official benchmark prompts remain unchanged."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from runtime import CONFIG, HERE, PARENT, identities, log, read_rows, save, write_rows


def rng_for(*parts):
    key = "|".join(map(str, (CONFIG["seed"], *parts)))
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def question(row):
    return row["doc"].get("question", row["doc"].get("question_stem")).strip()


def gold_index(row):
    labels = list(map(str, row["doc"]["choices"]["label"]))
    answer = str(row["doc"]["answerKey"])
    if answer not in labels:
        raise RuntimeError("Gold answer does not match choice labels")
    return labels.index(answer)


def permuted_doc(row, rng):
    permutation = list(range(len(row["doc"]["choices"]["text"])))
    rng.shuffle(permutation)
    if len(permutation) > 10:
        raise RuntimeError("Unsupported number of training choices")
    return {"question": question(row),
            "choices": [row["doc"]["choices"]["text"][i] for i in permutation],
            "answer": permutation.index(gold_index(row))}, permutation


def renderer():
    import lm_eval
    import yaml
    from lm_eval.utils import apply_template
    template = yaml.safe_load((Path(lm_eval.__file__).parent / "tasks/mmlu/default/_default_template_yaml").read_text())
    def render(doc):
        if len(doc["choices"]) == 4:
            return apply_template(template["doc_to_text"], doc)
        # A few ARC records may have a different number of choices. Extend the
        # same training-only layout instead of hiding or inventing an option.
        return (doc["question"].strip() + "\n" +
                "\n".join(f"{'ABCDEFGHIJ'[i]}. {text}" for i, text in enumerate(doc["choices"])) +
                "\nAnswer:")
    return render


def text_view(row):
    contexts = {request[0] for request in row["requests"]}
    if contexts != {row["context"]}:
        raise RuntimeError("Original candidate contexts differ")
    return {"base_id": row["id"], "task": row["task"], "view": "text", "fewshot": row["fewshot"],
            "context": row["context"], "continuations": [request[1] for request in row["requests"]],
            "correct_index": gold_index(row), "question": question(row), "demo_ids": [], "permutation": None}


def letter_view(row, pool, split, epoch, shots, render):
    rng = rng_for("letter", split, epoch, row["id"])
    doc, permutation = permuted_doc(row, rng)
    candidates = [item for item in pool if item["id"] != row["id"] and question(item) != question(row)]
    demos = rng.sample(candidates, shots)
    segments = []
    for demo in demos:
        demo_doc, _ = permuted_doc(demo, rng)
        segments.append(render(demo_doc) + " " + "ABCDEFGHIJ"[demo_doc["answer"]] + "\n\n")
    context = CONFIG["letter_description"] + "".join(segments) + render(doc)
    return {"base_id": row["id"], "task": row["task"], "view": "letter", "fewshot": shots,
            "context": context, "continuations": [" " + c for c in "ABCDEFGHIJ"[:len(doc["choices"])]],
            "correct_index": doc["answer"], "question": question(row), "choices": doc["choices"],
            "demo_ids": [item["id"] for item in demos], "permutation": permutation}


def build(train, validation):
    render = renderer()
    pools = defaultdict(list)
    for row in train:
        pools[row["task"]].append(row)
    phases = {}
    for rows in pools.values():
        for index, row in enumerate(sorted(rows, key=lambda item: item["id"])):
            phases[row["id"]] = index % 2
    prepared = {}
    for epoch in range(1, CONFIG["epochs"] + 1):
        views = []
        for row in train:
            if (phases[row["id"]] + epoch) % 2:
                view = text_view(row)
            else:
                first_shot_bit = rng_for("shots", row["id"]).randrange(2)
                visit = (epoch - 1) // 2
                shots = CONFIG["letter_fewshot_counts"][(first_shot_bit + visit) % 2]
                view = letter_view(row, pools[row["task"]], "train", epoch, shots, render)
            view["epoch"] = epoch
            views.append(view)
        prepared[f"train_epoch_{epoch}"] = views
    held_out = []
    for row in validation:
        held_out.append(text_view(row))
        shots = CONFIG["letter_fewshot_counts"][rng_for("validation-shots", row["id"]).randrange(2)]
        held_out.append(letter_view(row, pools[row["task"]], "validation", 0, shots, render))
    prepared["validation"] = held_out
    return prepared


def audit_prepared(train, validation, prepared):
    train_ids, validation_ids = {row["id"] for row in train}, {row["id"] for row in validation}
    if train_ids & validation_ids:
        raise RuntimeError("Train/validation ID overlap")
    for task in sorted({row["task"] for row in train}):
        tq = {question(row) for row in train if row["task"] == task}
        vq = {question(row) for row in validation if row["task"] == task}
        if tq & vq:
            raise RuntimeError("Train/validation question overlap")
    visits = defaultdict(Counter)
    shot_visits = defaultdict(list)
    for split, rows in prepared.items():
        for row in rows:
            if row["base_id"] in row["demo_ids"] or not set(row["demo_ids"]) <= train_ids:
                raise RuntimeError("Unsafe few-shot demonstrations")
            if not 0 <= row["correct_index"] < len(row["continuations"]):
                raise RuntimeError("Incorrect answer remapping")
            if split.startswith("train_"):
                visits[row["base_id"]][row["view"]] += 1
                if row["view"] == "letter":
                    shot_visits[row["base_id"]].append(row["fewshot"])
    if any(count != Counter(text=2, letter=2) for count in visits.values()):
        raise RuntimeError("Each training question must receive both views exactly twice")
    if any(sorted(shots) != [0, 5] for shots in shot_visits.values()):
        raise RuntimeError("Each training question must have one zero-shot and one five-shot letter visit")
    return {split: {"count": len(rows), "views": dict(Counter(row["view"] for row in rows)),
                    "letter_shots": dict(Counter(row["fewshot"] for row in rows if row["view"] == "letter")),
                    "letter_gold": dict(Counter("ABCDEFGHIJ"[row["correct_index"]] for row in rows if row["view"] == "letter"))}
            for split, rows in prepared.items()}


def prepare():
    from transformers import AutoTokenizer
    from kv_backend import CFG
    from runtime import digest
    train = read_rows(PARENT / CONFIG["parent_training_data"])
    validation = read_rows(PARENT / CONFIG["parent_validation_data"])
    if len(train) != CONFIG["training_question_count"] or len(validation) != CONFIG["validation_question_count"]:
        raise RuntimeError("Original training/validation counts changed")
    prepared = build(train, validation)
    audit = audit_prepared(train, validation, prepared)
    tokenizer = AutoTokenizer.from_pretrained(CFG["models"]["qwen"], local_files_only=True)
    for rows in prepared.values():
        for row in rows:
            row["qwen_context_tokens"] = len(tokenizer.encode(row["context"].rstrip(), add_special_tokens=False))
            if row["qwen_context_tokens"] >= CFG["max_length"] - 256:
                raise RuntimeError("Prepared context exceeds the unchanged Full-Sync budget")
    folder = HERE / "prepared"
    if (HERE / "manifest.json").exists():
        raise RuntimeError("Preparation already exists; preserve and inspect it instead of overwriting")
    for split, rows in prepared.items():
        write_rows(folder / f"{split}.jsonl", rows)
    manifest = {"identities": identities(), "audit": audit,
                "training_draws_total": sum(len(rows) for name, rows in prepared.items() if name.startswith("train_")),
                "prepared_sha256": {str(path.relative_to(HERE)): digest(path) for path in sorted(folder.glob("*.jsonl"))},
                "max_context_tokens": max(row["qwen_context_tokens"] for rows in prepared.values() for row in rows),
                "initialization": "Fresh zero-output residual KV adapter; old Stage B is never loaded; frozen original Stage A is reused.",
                "no_mmlu_training": True, "no_kv_tensor_disk_cache": True}
    save(HERE / "manifest.json", manifest)
    log(f"Prepared {manifest['training_draws_total']} view draws; max prompt={manifest['max_context_tokens']} tokens")
    log("View audit: " + json.dumps(audit))


if __name__ == "__main__":
    prepare()
