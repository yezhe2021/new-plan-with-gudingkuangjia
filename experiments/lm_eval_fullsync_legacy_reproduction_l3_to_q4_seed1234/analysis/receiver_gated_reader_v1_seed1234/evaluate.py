"""Official harness evaluation of the newly selected Stage B checkpoint."""
import argparse
import importlib.util
import json

from experiment import CONFIG, HERE, PARENT, digest, gpu_check, load_new_adapter, log, plain, save, verify_manifest, write_rows
from kv_backend import CFG, load_base, load_model, seed_all
from experiment import ReaderHFLM as FullSyncHFLM


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
    adapter = load_new_adapter(HERE / "reader/best.pt")
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
        write_rows(HERE / f"evaluation/{family}/new_reader__{task}__samples.jsonl", rows)
    expected = 128 if family == "mmlu" else 256
    if len(predictions) != expected:
        raise RuntimeError("Incorrect number of official evaluation questions")
    result["experiment"] = {"protocol": CONFIG["protocol"], "condition": "new_reader", "family": family,
                            "stage_a_sha256": manifest["identities"]["stage_a"]["sha256"],
                            "reader_sha256": digest(HERE / "reader/best.pt"),
                            "selected_on_validation_only": True, "pilot_subset": True,
                            "full_official_test": False, "unchanged_baseline_requests": True}
    write_rows(HERE / f"evaluation/{family}/predictions.jsonl", predictions)
    save(HERE / f"evaluation/{family}/new_reader.json", result)
    log(f"Official {family} evaluation completed: " + json.dumps(result["results"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("family",choices=["mcq","mmlu"])
    evaluate(parser.parse_args().family)
