"""Use installed official task configs/functions; only route dataset bytes locally."""
import hashlib
import importlib.metadata
import json
import random
import zipfile
from pathlib import Path

import datasets
from lm_eval.tasks import get_task_dict

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
ORIGINAL_LOAD = datasets.load_dataset
DATASETS = {}


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_id(task, doc):
    # Task/split independent content key, so different IDs cannot conceal leakage.
    q = doc.get("question_stem", doc.get("question", doc.get("ctx", str(doc))))
    return task + ":" + hashlib.sha256(str(q).strip().encode()).hexdigest()


def convert_mcq(row, arc=False):
    source = row["question"] if isinstance(row.get("question"), dict) else row
    choices = source["choices"]
    if isinstance(choices, list):
        choices = {"label": [x["label"] for x in choices], "text": [x["text"] for x in choices]}
    return {"id": str(row.get("id", "")),
            "question" if arc else "question_stem": source.get("stem", row.get("question_stem", row.get("question"))),
            "choices": choices, "answerKey": str(row["answerKey"])}


def local_dataset(path, name=None, **kwargs):
    key = (path, name)
    if key in DATASETS:
        result = DATASETS[key]
    else:
        paths = CFG["datasets"]
        if path == "allenai/openbookqa":
            result = datasets.DatasetDict({split: datasets.Dataset.from_list([
                convert_mcq(x) for x in jsonl(Path(paths["openbookqa_root"]) / filename)])
                for split, filename in (("train", "train.jsonl"), ("validation", "dev.jsonl"), ("test", "test.jsonl"))})
        elif path == "allenai/ai2_arc" and name == "ARC-Challenge":
            result = datasets.DatasetDict()
            with zipfile.ZipFile(paths["arc_zip"]) as z:
                for split, suffix in (("train", "Train"), ("validation", "Dev"), ("test", "Test")):
                    filename = next(x for x in z.namelist() if x.endswith(f"ARC-Challenge-{suffix}.jsonl"))
                    rows = [convert_mcq(json.loads(line), True) for line in z.read(filename).decode().splitlines() if line.strip()]
                    result[split] = datasets.Dataset.from_list(rows)
        elif path == "openai/gsm8k":
            result = datasets.DatasetDict({split: datasets.Dataset.from_list(jsonl(paths[f"gsm8k_{split}"]))
                                           for split in ("train", "test")})
        elif path == "Rowan/hellaswag":
            # Official default is0-shot. The unused train split is not substituted with eval answers.
            result = datasets.DatasetDict(train=datasets.Dataset.from_list([]),
                                          validation=datasets.Dataset.from_list(jsonl(paths["hellaswag"])))
        elif path == "TIGER-Lab/MMLU-Pro":
            import pyarrow.parquet as pq
            val = ROOT / paths["mmlu_pro_validation"]
            if not val.exists():
                raise FileNotFoundError("Official MMLU-Pro validation demonstrations missing; do not substitute test answers")
            result = datasets.DatasetDict({"test": datasets.Dataset.from_list(pq.read_table(paths["mmlu_pro_test"]).to_pylist()),
                                           "validation": datasets.Dataset.from_list(pq.read_table(val).to_pylist())})
        else:
            return ORIGINAL_LOAD(path, name, **kwargs)
        DATASETS[key] = result
    return result[kwargs["split"]] if kwargs.get("split") else datasets.DatasetDict(dict(result))


def install_local_data():
    if importlib.metadata.version("lm_eval") != CFG["lm_eval_version"]:
        raise RuntimeError("harness version changed")
    datasets.load_dataset = local_dataset


def tasks(names):
    install_local_data()
    return get_task_dict(list(names))


def examples(task_name, task, split, count, train_pool=None):
    if train_pool is not None:
        task.dataset["train"] = datasets.Dataset.from_list(train_pool)
        # ConfigurableTask creates the sampler at initialization.
        task.sampler.replace_df(list(train_pool))
    task.set_fewshot_seed(CFG["seed"])
    docs = list(task.dataset[split])
    order = list(range(len(docs)))
    random.Random(CFG["seed"]).shuffle(order)
    rows = []
    for index in order[:count]:
        doc = docs[index]
        shots = int(task.get_config("num_fewshot") or 0)
        if train_pool is not None and shots:
            # Training must not accidentally place this very question's gold rationale in a demo.
            key = stable_id(task_name, doc)
            task.sampler.replace_df([d for d in train_pool if stable_id(task_name,d) != key])
        context = task.fewshot_context(doc, num_fewshot=shots)
        requests = task.construct_requests(doc, context, metadata=(task_name, index, 1))
        if not isinstance(requests, list): requests = [requests]
        row = {"id": stable_id(task_name, doc), "task": task_name, "doc": doc,
               "context": context, "fewshot": shots,
               "requests": [list(r.args) for r in requests],
               "output_type": task.get_config("output_type")}
        if task.get_config("output_type") == "generate_until":
            row["target"] = task.doc_to_target(doc)
            row["target_delimiter"] = task.get_config("target_delimiter") or ""
        rows.append(row)
    return rows


def prepare(group, smoke=False):
    names = CFG["groups"][group]
    objects = tasks(names)
    output = {"train": [], "validation": []}
    audit = {}
    for name in names:
        task = objects[name]
        train_docs = list(task.dataset["train"])
        test_key = task.get_config("test_split") or task.get_config("validation_split")
        official_test_ids = {stable_id(name, x) for x in task.dataset[test_key]}
        if name == "gsm8k":
            order = list(range(len(train_docs))); random.Random(CFG["seed"]).shuffle(order)
            unique = {}; excluded = 0
            for i in order:
                key = stable_id(name, train_docs[i])
                if key in official_test_ids or key in unique: excluded += 1; continue
                unique[key] = train_docs[i]
            safe = list(unique.values()); n = CFG["train_counts"][name]
            train_pool, val_docs = safe[:n], safe[n:n+CFG["validation_count"]]
            if len(train_pool) != n or len(val_docs) != CFG["validation_count"]:
                raise RuntimeError("Insufficient clean GSM8K train/validation rows")
            task.dataset["validation"] = datasets.Dataset.from_list(val_docs)
        else:
            blocked = official_test_ids | {stable_id(name,x) for x in task.dataset["validation"]}
            unique = {}; excluded = 0
            for doc in train_docs:
                key = stable_id(name,doc)
                if key in blocked or key in unique: excluded += 1; continue
                unique[key] = doc
            train_pool = list(unique.values())
            if len(train_pool) < CFG["train_counts"][name]: raise RuntimeError("Insufficient clean MCQ training rows")
        audit[name] = {"official_train_count":len(train_docs),"conservative_excluded_train_count":excluded,
                       "train_pool_count":len(train_pool),"test_split_unchanged":True}
        train = examples(name, task, "train", CFG["train_counts"][name], train_pool)
        # Explicitly use held-out original train subset as a new pool before selecting GSM rows.
        val = examples(name, task, "validation", CFG["validation_count"], train_pool if name == "gsm8k" else None)
        train_ids, val_ids = {r["id"] for r in train}, {r["id"] for r in val}
        test_ids = {stable_id(name, x) for x in task.dataset[test_key]}
        if train_ids & val_ids or train_ids & test_ids:
            raise RuntimeError(f"Training overlap with validation/test: {name}")
        if smoke: train, val = train[:2], val[:1]
        output["train"].extend(train); output["validation"].extend(val)
    folder = ROOT / ("runs_smoke" if smoke else "runs") / group
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return output


def save_prepared(group, smoke=False):
    result = prepare(group, smoke)
    folder = ROOT / ("runs_smoke" if smoke else "runs") / group
    folder.mkdir(parents=True, exist_ok=True)
    for split, rows in result.items():
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
