"""Official harness evaluation of the newly selected Stage B checkpoint."""
import argparse
import importlib.util
import json

from runtime import CONFIG, HERE, PARENT, digest, gpu_check, load_new_adapter, log, plain, save, verify_manifest, write_rows
from kv_backend import CFG, FullSyncHFLM, load_base, load_model, seed_all


def evaluate(family):
    from lm_eval import evaluator
    from tasks_native import install_local_data
    manifest = verify_manifest()
    gpu_check()
    seed_all(CONFIG["seed"])
    baseline_directory = PARENT / CONFIG["mmlu_baseline_directory"]
    if family == "mcq":
        install_local_data()
        names = CFG["groups"]["mcq"]
        samples, limit, shots = None, CFG["test_count"], None
    else:
        specification = importlib.util.spec_from_file_location("fixed_mmlu_dataset_router", baseline_directory / "run_mmlu.py")
        router = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(router)
        router.task_objects()
        old_manifest = json.loads((baseline_directory / "manifest.json").read_text())
        names, samples, limit, shots = ["mmlu"], old_manifest["samples"], None, 5
    receiver = load_model(CFG, "qwen")
    sender = load_model(CFG, "llama")
    base = load_base(PARENT / CONFIG["parent_stage_a"])
    adapter = load_new_adapter(HERE / "stage_b/best.pt")
    lm = FullSyncHFLM(receiver, "translated", sender, base, adapter)
    result = evaluator.simple_evaluate(
        model=lm, tasks=names, num_fewshot=shots, samples=samples, limit=limit,
        bootstrap_iters=1000, log_samples=True, apply_chat_template=False,
        random_seed=CONFIG["seed"], numpy_random_seed=CONFIG["seed"],
        torch_random_seed=CONFIG["seed"], fewshot_random_seed=CONFIG["seed"])
    result = json.loads(json.dumps(result, default=plain))
    reference_path = (PARENT / "runs/mcq/evaluation/stage_b.json" if family == "mcq"
                      else baseline_directory / "results/stage_b.json")
    reference = json.loads(reference_path.read_text())
    predictions = []
    for task, rows in result["samples"].items():
        old = {row["doc_id"]: row for row in reference["samples"][task]}
        if {row["doc_id"] for row in rows} != set(old):
            raise RuntimeError("Official evaluation sample IDs differ from the old baseline")
        for row in rows:
            previous = old[row["doc_id"]]
            if row["doc"] != previous["doc"] or row["arguments"] != previous["arguments"]:
                raise RuntimeError("Official benchmark question, prompt, or candidates changed")
            scores = [float(value[0]) for value in row["filtered_resps"]]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            if family == "mmlu":
                gold = int(row["doc"]["answer"])
                choices = row["doc"]["choices"]
            else:
                labels = list(map(str, row["doc"]["choices"]["label"]))
                gold = labels.index(str(row["doc"]["answerKey"]))
                choices = row["doc"]["choices"]["text"]
            if int(prediction == gold) != row["acc"]:
                raise RuntimeError("Official accuracy disagrees with the logged candidate scores")
            predictions.append({"task": task, "doc_id": row["doc_id"],
                                "question": row["doc"].get("question", row["doc"].get("question_stem")),
                                "choices": choices, "gold": "ABCDEFGHIJ"[gold],
                                "prediction": "ABCDEFGHIJ"[prediction], "scores": scores, "acc": row["acc"]})
        write_rows(HERE / f"evaluation/{family}/new_stage_b__{task}__samples.jsonl", rows)
    expected = 128 if family == "mmlu" else 256
    if len(predictions) != expected:
        raise RuntimeError("Incorrect number of official evaluation questions")
    result["experiment"] = {"protocol": CONFIG["protocol"], "condition": "new_stage_b", "family": family,
                            "stage_a_sha256": manifest["identities"]["stage_a"]["sha256"],
                            "stage_b_sha256": digest(HERE / "stage_b/best.pt"),
                            "selected_on_validation_only": True, "pilot_subset": True,
                            "full_official_test": False, "unchanged_baseline_requests": True}
    write_rows(HERE / f"evaluation/{family}/predictions.jsonl", predictions)
    save(HERE / f"evaluation/{family}/new_stage_b.json", result)
    log(f"Official {family} evaluation completed: " + json.dumps(result["results"]))


def summarize():
    from collections import Counter
    summary = {"protocol": CONFIG["protocol"], "pilot_subset": True,
               "selection": json.loads((HERE / "stage_b/selection.json").read_text())["best"], "families": {}}
    for family in CONFIG["evaluation_families"]:
        path = HERE / f"evaluation/{family}/new_stage_b.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        folder = (PARENT / "runs/mcq/evaluation" if family == "mcq" else
                  PARENT / CONFIG["mmlu_baseline_directory"] / "results")
        comparisons = {arm: json.loads((folder / f"{arm}.json").read_text())["results"]
                       for arm in ("qwen_native", "llama_native", "stage_a", "stage_b")}
        predictions = json.loads("[]")
        from runtime import read_rows
        predictions = read_rows(HERE / f"evaluation/{family}/predictions.jsonl")
        summary["families"][family] = {"new_stage_b": result["results"], "baselines": comparisons,
                                        "predicted_letters": dict(Counter(row["prediction"] for row in predictions))}
    summary["completed"] = len(summary["families"]) == len(CONFIG["evaluation_families"])
    save(HERE / "summary.json", summary)
    log("RESULT SUMMARY SAVED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "summarize"))
    parser.add_argument("--family", choices=CONFIG["evaluation_families"])
    args = parser.parse_args()
    if args.action == "evaluate":
        if args.family is None:
            parser.error("evaluate requires --family")
        evaluate(args.family)
    else:
        summarize()
