"""Official MMLU likelihood evaluation of the existing MCQ Full-Sync checkpoints.

The installed harness owns prompts, candidate tokenization, scores, and aggregation.
Only dataset byte routing and the existing model-side KV provenance are changed.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENT))
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S UTC] ", time.gmtime()) + message, flush=True)


def plain(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=plain) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def flattened(values):
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update(flattened(value))
        elif hasattr(value, "get_config"):
            result[key] = value
        elif isinstance(value, tuple) and hasattr(value[-1], "get_config"):
            result[key] = value[-1]
    return result


def task_objects():
    import datasets
    import pyarrow.parquet as pq
    from lm_eval.tasks import get_task_dict
    from tasks_native import install_local_data
    if importlib.metadata.version("lm_eval") != CONFIG["lm_eval_version"]:
        raise RuntimeError("Installed harness version changed")
    install_local_data()
    previous = datasets.load_dataset
    source = {split: pq.read_table(HERE / "datasets/mmlu" /
                                  f"{split}-00000-of-00001.parquet").to_pylist()
              for split in ("dev", "test")}
    subjects = {row["subject"] for row in source["test"]}
    if len(subjects) != 57 or {row["subject"] for row in source["dev"]} != subjects:
        raise RuntimeError("Official MMLU subject coverage changed")
    cache = {}

    def local(path, name=None, **kwargs):
        if path != CONFIG["dataset_repo"]:
            return previous(path, name, **kwargs)
        if name not in subjects:
            raise RuntimeError(f"Unexpected MMLU dataset config: {name}")
        if name not in cache:
            cache[name] = datasets.DatasetDict({
                split: datasets.Dataset.from_list([row for row in rows if row["subject"] == name])
                for split, rows in source.items()})
            if len(cache[name]["dev"]) != 5:
                raise RuntimeError(f"Expected the five official dev examples for {name}")
        return cache[name][kwargs["split"]] if kwargs.get("split") else cache[name]

    datasets.load_dataset = local
    objects = get_task_dict([CONFIG["task"]])
    leaves = flattened(objects)
    if len(leaves) != 57:
        raise RuntimeError(f"Expected 57 official subject tasks, got {len(leaves)}")
    return objects, leaves


def allocation(leaves):
    names = sorted(leaves)
    count = CONFIG["test_count_total"]
    return {name: list(range(count // len(names) + (index < count % len(names))))
            for index, name in enumerate(names)}


def identities():
    import lm_eval
    from kv_backend import CFG
    source_paths = [HERE / "config.json", HERE / "run_mmlu.py",
                    EXPERIMENT / "config.json", EXPERIMENT / "kv_backend.py",
                    EXPERIMENT / "tasks_native.py"]
    source_paths += [EXPERIMENT / "base_code" / name
                     for name in ("common.py", "protocol.py", "translator.py")]
    task_directory = Path(lm_eval.__file__).parent / "tasks/mmlu/default"
    source_paths += sorted(path for path in task_directory.iterdir() if path.is_file())
    checkpoint_paths = [EXPERIMENT / CONFIG[key]
                        for key in ("stage_a_checkpoint", "stage_b_checkpoint")]
    data_directory = HERE / "datasets/mmlu"
    provenance = json.loads((data_directory / "provenance.json").read_text())
    if provenance["revision"] != CONFIG["dataset_revision"]:
        raise RuntimeError("Dataset revision mismatch")
    for item in provenance["files"]:
        if digest(data_directory / item["file"]) != item["sha256"]:
            raise RuntimeError("Dataset checksum mismatch")
    return {"configuration": CONFIG, "original_protocol": CFG["protocol"],
            "models": CFG["models"], "max_length": CFG["max_length"],
            "attention_implementation": CFG["attention_implementation"],
            "dataset": provenance,
            "source_sha256": {str(path): digest(path) for path in source_paths},
            "checkpoint_sha256": {str(path): digest(path) for path in checkpoint_paths},
            "selection": {stage: json.loads((EXPERIMENT / f"runs/mcq/stage_{stage}/selection.json").read_text())["best"]
                          for stage in ("a", "b")},
            "versions": {name: importlib.metadata.version(name)
                         for name in ("lm_eval", "torch", "transformers", "datasets")}}


def verify_manifest():
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["identities"] != identities():
        raise RuntimeError("Experiment code, data, or checkpoints changed since preparation")
    return manifest


def prepare():
    from transformers import AutoTokenizer
    from kv_backend import CFG
    _, leaves = task_objects()
    samples = allocation(leaves)
    tokenizer = AutoTokenizer.from_pretrained(CFG["models"]["qwen"], local_files_only=True)
    selected = []
    for name in sorted(leaves):
        task = leaves[name]
        if task.get_config("output_type") != "multiple_choice":
            raise RuntimeError("MMLU must use official multiple-choice likelihood scoring")
        task.set_fewshot_seed(CONFIG["seed"])
        for index in samples[name]:
            doc = task.test_docs()[index]
            context = task.fewshot_context(doc, num_fewshot=CONFIG["num_fewshot"])
            requests = task.construct_requests(doc, context, metadata=(name, index, 1))
            candidates = [request.args[1] for request in requests]
            if len(candidates) != 4:
                raise RuntimeError("MMLU must have four official candidate requests")
            selected.append({"task": name, "doc_id": index, "doc": doc,
                             "context": context, "candidates": candidates,
                             "prompt_sha256": hashlib.sha256(context.encode()).hexdigest(),
                             "qwen_context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
    manifest = {"identities": identities(), "samples": samples,
                "selected_questions_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                "n_subjects": len(leaves), "n_questions": len(selected),
                "pilot_subset": True, "full_official_test": False,
                "context_tokens_min_max": [min(row["qwen_context_tokens"] for row in selected),
                                           max(row["qwen_context_tokens"] for row in selected)]}
    if (HERE / "manifest.json").exists():
        if json.loads((HERE / "manifest.json").read_text()) != manifest:
            raise RuntimeError("Refusing to overwrite a different existing manifest")
    save(HERE / "manifest.json", manifest)
    with (HERE / "selected_questions.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"Prepared {len(selected)} questions across {len(leaves)} subjects; "
        f"Qwen prompt token range={manifest['context_tokens_min_max']}")


def gpu_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    log(f"GPU {torch.cuda.get_device_name(0)}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB")
    if free / 2**30 < CONFIG["min_free_gpu_gib"]:
        raise RuntimeError("Insufficient free GPU memory; unrelated jobs were not changed")


def smoke():
    import torch
    from lm_eval.api.instance import Instance
    from kv_backend import CFG, FullSyncHFLM, load_model, load_base, load_adapter, seed_all
    verify_manifest()
    gpu_check()
    seed_all(CONFIG["seed"])
    rows = [json.loads(line) for line in (HERE / "selected_questions.jsonl").read_text().splitlines()]
    chosen = [min(rows, key=lambda row: row["qwen_context_tokens"]),
              max(rows, key=lambda row: row["qwen_context_tokens"])]
    receiver = load_model(CFG, "qwen")
    native = FullSyncHFLM(receiver, mode="native")
    oracle = FullSyncHFLM(receiver, mode="oracle")
    records = []
    for row in chosen:
        requests = [Instance("loglikelihood", row["doc"], (row["context"], candidate), index,
                             metadata=(row["task"], row["doc_id"], 1))
                    for index, candidate in enumerate(row["candidates"])]
        expected = native.loglikelihood(requests)
        actual = oracle.loglikelihood(requests)
        lengths = [len(native._encode_pair(row["context"], candidate)[1]) for candidate in row["candidates"]]
        error = max(abs(a[0] - b[0]) / n for a, b, n in zip(expected, actual, lengths))
        agreement = max(range(4), key=lambda i: expected[i][0]) == max(range(4), key=lambda i: actual[i][0])
        if error > CONFIG["parity_max_per_token_ll_error"] or not agreement:
            raise RuntimeError(f"Native KV injection parity failed: error={error}; agreement={agreement}")
        records.append({"task": row["task"], "doc_id": row["doc_id"],
                        "context_tokens": row["qwen_context_tokens"], "native": expected,
                        "oracle": actual, "max_per_token_ll_error": error,
                        "prediction_agreement": agreement})
        log(f"Native/oracle parity passed: {row['task']}; error={error:.6f}")
    oracle._cached_kv = None
    del oracle, native
    gc.collect()
    torch.cuda.empty_cache()
    sender = load_model(CFG, "llama")
    base = load_base(EXPERIMENT / CONFIG["stage_a_checkpoint"])
    adapter = load_adapter(EXPERIMENT / CONFIG["stage_b_checkpoint"])
    translated = FullSyncHFLM(receiver, "translated", sender, base, adapter)
    row = chosen[-1]
    requests = [Instance("loglikelihood", row["doc"], (row["context"], candidate), index,
                         metadata=(row["task"], row["doc_id"], 1))
                for index, candidate in enumerate(row["candidates"])]
    scores = translated.loglikelihood(requests)
    if len(scores) != 4 or not all(torch.isfinite(torch.tensor(score[0])) for score in scores):
        raise RuntimeError("Translated MMLU likelihoods are not finite")
    save(HERE / "smoke.json", {"SMOKE_ONLY_NOT_BENCHMARK_RESULTS": True, "passed": True,
                               "native_oracle_parity": records,
                               "stage_b_longest_prompt": {"task": row["task"], "doc_id": row["doc_id"],
                                                          "scores": scores},
                               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
    log("MMLU native/oracle and translated longest-prompt smoke checks passed")


def evaluate(condition):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from kv_backend import CFG, FullSyncHFLM, load_model, load_base, load_adapter, seed_all
    manifest = verify_manifest()
    gpu_check()
    seed_all(CONFIG["seed"])
    task_objects()
    if condition == "llama_native":
        model = load_model(CFG, "llama")
        lm = HFLM(pretrained=model, tokenizer=CFG["models"]["llama"], batch_size=1,
                  backend="causal", max_length=CFG["max_length"], logits_cache=False)
    else:
        receiver = load_model(CFG, "qwen")
        sender = base = adapter = None
        if condition.startswith("stage_"):
            sender = load_model(CFG, "llama")
            base = load_base(EXPERIMENT / CONFIG["stage_a_checkpoint"])
            if condition == "stage_b":
                adapter = load_adapter(EXPERIMENT / CONFIG["stage_b_checkpoint"])
        lm = FullSyncHFLM(receiver, "translated" if sender is not None else "native", sender, base, adapter)
    result = evaluator.simple_evaluate(
        model=lm, tasks=[CONFIG["task"]], num_fewshot=CONFIG["num_fewshot"],
        samples=manifest["samples"], limit=None, bootstrap_iters=CONFIG["bootstrap_iters"],
        log_samples=True, apply_chat_template=CONFIG["apply_chat_template"],
        random_seed=CONFIG["seed"], numpy_random_seed=CONFIG["seed"],
        torch_random_seed=CONFIG["seed"], fewshot_random_seed=CONFIG["seed"])
    payload = json.loads(json.dumps(result, default=plain))
    rows = [row for samples in payload["samples"].values() for row in samples]
    if len(rows) != CONFIG["test_count_total"]:
        raise RuntimeError(f"Unexpected evaluated sample count: {len(rows)}")
    correct = sum(row["acc"] for row in rows)
    official_acc = payload["results"]["mmlu"]["acc,none"]
    if abs(official_acc - correct / len(rows)) > 1e-8:
        raise RuntimeError("Official weighted MMLU aggregate does not match logged samples")
    selected = {(row["task"], row["doc_id"]): row for row in
                [json.loads(line) for line in (HERE / "selected_questions.jsonl").read_text().splitlines()]}
    predictions = []
    for name, samples in payload["samples"].items():
        for row in samples:
            reference = selected[(name, row["doc_id"])]
            if row["arguments"][0][0] != reference["context"] or row["doc"] != reference["doc"]:
                raise RuntimeError("Evaluated prompt or question differs from the fixed selection")
            scores = [float(response[0]) for response in row["filtered_resps"]]
            predicted = max(range(4), key=lambda index: scores[index])
            gold = int(row["doc"]["answer"])
            if int(predicted == gold) != row["acc"]:
                raise RuntimeError("Logged candidate scores do not reproduce official accuracy")
            predictions.append({"condition": condition, "task": name, "doc_id": row["doc_id"],
                                "question": row["doc"]["question"], "choices": row["doc"]["choices"],
                                "gold": "ABCD"[gold], "prediction": "ABCD"[predicted],
                                "choice_loglikelihood": dict(zip("ABCD", scores)), "acc": row["acc"],
                                "prompt_sha256": reference["prompt_sha256"]})
    payload["experiment"] = {"condition": condition, "checkpoint_group": "mcq",
                             "manifest_sha256": digest(HERE / "manifest.json"),
                             "pilot_subset": True, "full_official_test": False,
                             "n_questions": len(rows), "n_correct": int(correct),
                             "scoring": "official MMLU four-letter loglikelihood; no generated-answer parser"}
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    for name, samples in payload["samples"].items():
        with (output / f"{condition}__{name}__samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in samples:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / f"{condition}__predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(output / f"{condition}.json", payload)
    log(f"{condition} MMLU accuracy={official_acc:.6f} ({int(correct)}/{len(rows)})")


def summarize():
    results = []
    manifest_hash = digest(HERE / "manifest.json")
    for condition in CONFIG["conditions"]:
        path = HERE / "results" / f"{condition}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result["experiment"]["manifest_sha256"] != manifest_hash:
            raise RuntimeError("Result belongs to another experiment manifest")
        results.append({"condition": condition, "n": result["experiment"]["n_questions"],
                        "correct": result["experiment"]["n_correct"],
                        "acc": result["results"]["mmlu"]["acc,none"],
                        "acc_stderr": result["results"]["mmlu"].get("acc_stderr,none")})
    save(HERE / "summary.json", {"pilot_subset": True, "n_subjects": 57, "num_fewshot": 5,
                                 "results": results, "completed": len(results) == len(CONFIG["conditions"])})
    log("Summary: " + json.dumps(results))


def pipeline():
    import fcntl
    lock = (HERE / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_manifest()
    if not json.loads((HERE / "smoke.json").read_text()).get("passed"):
        raise RuntimeError("Smoke gate has not passed")
    try:
        for condition in CONFIG["conditions"]:
            status = {"phase": condition, "pid": os.getpid(), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            save(HERE / "status.json", status)
            path = HERE / "results" / f"{condition}.json"
            if not path.exists():
                log("START " + condition)
                subprocess.run([sys.executable, "-u", str(HERE / "run_mmlu.py"), "evaluate",
                                "--condition", condition], cwd=EXPERIMENT, check=True)
            summarize()
        save(HERE / "status.json", {"phase": "completed", "pid": os.getpid(),
                                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log("ALL MMLU MCQ TRANSFER CONDITIONS COMPLETED")
    except Exception as error:
        save(HERE / "status.json", {"phase": "failed", "error": repr(error), "pid": os.getpid(),
                                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "evaluate", "summarize", "pipeline"))
    parser.add_argument("--condition", choices=CONFIG["conditions"])
    args = parser.parse_args()
    if args.action == "evaluate":
        if args.condition is None:
            parser.error("evaluate requires --condition")
        evaluate(args.condition)
    else:
        globals()[args.action]()
