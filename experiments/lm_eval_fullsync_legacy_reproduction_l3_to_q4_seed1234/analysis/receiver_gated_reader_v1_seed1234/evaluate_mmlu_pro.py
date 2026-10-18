"""Evaluate the selected external reader on the unchanged MMLU-Pro pilot.

Official lm-eval owns prompt formatting, generation, stops, filters, and scoring.
Persist each completed generation for progress reporting and exact-input resume.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from experiment import (CONFIG, HERE, PARENT, digest, gpu_check, load_new_adapter,
                     log, plain, save, verify_manifest, write_rows)
from kv_backend import CFG, load_base, load_model, seed_all
from experiment import ReaderHFLM as FullSyncHFLM

OUTPUT = HERE / "evaluation/mmlu_pro"
BASELINES = PARENT / "runs/mcq/evaluation_ood"


def task_leaves(values):
    found = {}
    for name, value in values.items():
        if isinstance(value, dict):
            found.update(task_leaves(value))
        elif hasattr(value, "get_config"):
            found[name] = value
        elif isinstance(value, tuple) and hasattr(value[-1], "get_config"):
            found[name] = value[-1]
    return found


def argument_key(arguments):
    return hashlib.sha256(json.dumps(list(arguments), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reference_rows(result):
    return {name: rows for name, rows in result["samples"].items() if name.startswith("mmlu_pro_")}


def input_identity():
    import lm_eval
    task_root = Path(lm_eval.__file__).parent / "tasks/mmlu_pro"
    paths = [Path(__file__), HERE / "reader/best.pt", HERE / "manifest.json",
             BASELINES / "stage_b.json", PARENT / CFG["datasets"]["mmlu_pro_validation"],
             Path(CFG["datasets"]["mmlu_pro_test"])]
    paths += [path for path in sorted(task_root.iterdir()) if path.is_file()]
    return {str(path): digest(path) for path in paths}


def prepare():
    from transformers import AutoTokenizer
    from tasks_native import tasks
    verify_manifest()
    previous = json.loads((BASELINES / "stage_b.json").read_text())
    rows = reference_rows(previous)
    leaves = task_leaves(tasks(["mmlu_pro"]))
    if set(rows) != set(leaves) or sum(map(len, rows.values())) != 128:
        raise RuntimeError("Original MMLU-Pro pilot must contain 128 questions across the same 14 subjects")
    tokenizer = AutoTokenizer.from_pretrained(CFG["models"]["qwen"], local_files_only=True)
    samples, selected = {}, []
    for name in sorted(rows):
        task = leaves[name]
        if task.get_config("output_type") != "generate_until" or task.get_config("num_fewshot") != 5:
            raise RuntimeError("MMLU-Pro must use the original official 5-shot generation task")
        task.set_fewshot_seed(CFG["seed"])
        samples[name] = []
        for reference in rows[name]:
            index = reference["doc_id"]
            doc = task.test_docs()[index]
            context = task.fewshot_context(doc, num_fewshot=5)
            requests = task.construct_requests(doc, context, metadata=(name, index, 1))
            request = requests[0] if isinstance(requests, list) else requests
            arguments = json.loads(json.dumps(list(request.args)))
            if doc != reference["doc"] or arguments != reference["arguments"][0]:
                raise RuntimeError("Official MMLU-Pro prompt, generation kwargs, or question differs from the baseline")
            kwargs = arguments[1]
            if kwargs != {"until": ["Question:"], "max_gen_toks": 2048, "do_sample": False, "temperature": 0.0}:
                raise RuntimeError("Original 2048-token greedy generation protocol changed")
            samples[name].append(index)
            selected.append({"task": name, "doc_id": index, "arguments": arguments,
                             "request_key": argument_key(arguments),
                             "context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
    if len({row["request_key"] for row in selected}) != len(selected):
        raise RuntimeError("Generation requests must have unique exact-input identities")
    manifest = {"parent_protocol": CONFIG["protocol"], "condition": "gated_external_reader",
                "input_sha256": input_identity(), "samples": samples, "n_questions": len(selected),
                "selected": selected, "pilot_subset": True, "full_official_test": False,
                "chat_template": False, "max_gen_toks": 2048, "num_fewshot": 5,
                "smoke_max_gen_toks": 8, "checkpoint_selection": "held-out reference and fixed-student-prefix loss only; see reader/selection.json",
                "context_tokens_min_max": [min(row["context_tokens"] for row in selected),
                                           max(row["context_tokens"] for row in selected)]}
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Refusing to overwrite an existing different MMLU-Pro manifest")
    save(path, manifest)
    log(f"Prepared unchanged MMLU-Pro: {len(selected)} questions; prompt range={manifest['context_tokens_min_max']}")


def verified_inputs():
    verify_manifest()
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    if manifest["input_sha256"] != input_identity():
        raise RuntimeError("MMLU-Pro source, checkpoint, or data changed after preparation")
    return manifest


def model():
    seed_all(CFG["seed"])
    receiver = load_model(CFG, "qwen")
    sender = load_model(CFG, "llama")
    base = load_base(PARENT / CONFIG["parent_stage_a"])
    adapter = load_new_adapter(HERE / "reader/best.pt")
    return receiver, sender, base, adapter


def smoke():
    from lm_eval.api.instance import Instance
    manifest = verified_inputs()
    gpu_check()
    receiver, sender, base, adapter = model()
    lm = FullSyncHFLM(receiver, "translated", sender, base, adapter)
    row = max(manifest["selected"], key=lambda item: item["context_tokens"])
    context, kwargs = row["arguments"]
    kwargs = dict(kwargs, max_gen_toks=manifest["smoke_max_gen_toks"])
    request = Instance("generate_until", {}, (context, kwargs), 0,
                       metadata=(row["task"], row["doc_id"], 1))
    response = lm.generate_until([request], disable_tqdm=True)
    if len(response) != 1 or not isinstance(response[0], str):
        raise RuntimeError("MMLU-Pro generation smoke failed")
    save(OUTPUT / "smoke.json", {"SMOKE_ONLY_NOT_BENCHMARK_RESULTS": True, "passed": True,
                                 "request_key": row["request_key"], "task": row["task"],
                                 "doc_id": row["doc_id"], "context_tokens": row["context_tokens"],
                                 "smoke_max_gen_toks": manifest["smoke_max_gen_toks"],
                                 "official_eval_max_gen_toks": 2048, "response": response[0],
                                 "manifest_sha256": digest(OUTPUT / "manifest.json")})
    log("Longest-prompt generation smoke passed; full evaluation retains 2048 tokens")


class RecordedHFLM(FullSyncHFLM):
    def __init__(self, receiver, sender, base, adapter, manifest):
        super().__init__(receiver, "translated", sender, base, adapter)
        self.reference = {row["request_key"]: row for row in manifest["selected"]}
        self.run_hash = digest(OUTPUT / "manifest.json")
        self.raw_path = OUTPUT / "raw_generations.jsonl"
        self.completed = 0
        self.cached = {}
        if self.raw_path.exists():
            for line in self.raw_path.read_text().splitlines():
                row = json.loads(line)
                if row["manifest_sha256"] != self.run_hash or row["request_key"] not in self.reference:
                    raise RuntimeError("Cached generation belongs to another checkpoint or input")
                if argument_key(row["arguments"]) != row["request_key"]:
                    raise RuntimeError("Cached generation exact-input hash mismatch")
                self.cached[row["request_key"]] = row

    def generate_until(self, requests, disable_tqdm=False):
        responses = []
        for request in requests:
            key = argument_key(request.args)
            if key not in self.reference:
                raise RuntimeError("Harness requested a different MMLU-Pro prompt or generation setting")
            reference = self.reference[key]
            if key in self.cached:
                response = self.cached[key]["response"]
                elapsed = 0.
            else:
                log(f"START MMLU-Pro [{self.completed + 1}/{len(self.reference)}] "
                    f"{reference['task']} doc={reference['doc_id']} max_gen_toks=2048")
                started = time.monotonic()
                generated = super().generate_until([request], disable_tqdm=True)
                if len(generated) != 1:
                    raise RuntimeError("Expected exactly one greedy response")
                response = generated[0]
                elapsed = time.monotonic() - started
                record = {"task": reference["task"], "doc_id": reference["doc_id"],
                          "request_key": key, "manifest_sha256": self.run_hash,
                          "arguments": list(request.args), "response": response, "elapsed_seconds": elapsed}
                with self.raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                self.cached[key] = record
            responses.append(response)
            self.completed += 1
            save(OUTPUT / "status.json", {"phase": "evaluating", "completed_questions": self.completed,
                                          "expected_questions": len(self.reference),
                                          "task": reference["task"], "doc_id": reference["doc_id"],
                                          "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            log(f"MMLU-Pro [{self.completed}/{len(self.reference)}] {reference['task']} "
                f"doc={reference['doc_id']} chars={len(response)} elapsed={elapsed:.1f}s")
        return responses


def run():
    from lm_eval import evaluator
    from tasks_native import install_local_data
    manifest = verified_inputs()
    if not json.loads((OUTPUT / "smoke.json").read_text()).get("passed"):
        raise RuntimeError("Generation smoke has not passed")
    if json.loads((OUTPUT / "smoke.json").read_text())["manifest_sha256"] != digest(OUTPUT / "manifest.json"):
        raise RuntimeError("Generation smoke belongs to a different manifest")
    lock = (OUTPUT / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "new_reader.json").exists():
        log("MMLU-Pro result already exists; no duplicate evaluation launched")
        return
    save(OUTPUT / "status.json", {"phase": "starting", "completed_questions": 0, "expected_questions": 128})
    try:
        gpu_check()
        install_local_data()
        receiver, sender, base, adapter = model()
        lm = RecordedHFLM(receiver, sender, base, adapter, manifest)
        result = evaluator.simple_evaluate(
            model=lm, tasks=["mmlu_pro"], num_fewshot=None, samples=manifest["samples"], limit=None,
            bootstrap_iters=1000, log_samples=True, apply_chat_template=False,
            random_seed=CFG["seed"], numpy_random_seed=CFG["seed"],
            torch_random_seed=CFG["seed"], fewshot_random_seed=CFG["seed"])
        result = json.loads(json.dumps(result, default=plain))
        old = reference_rows(json.loads((BASELINES / "stage_b.json").read_text()))
        all_rows = []
        for task, rows in reference_rows(result).items():
            reference = {row["doc_id"]: row for row in old[task]}
            if {row["doc_id"] for row in rows} != set(reference):
                raise RuntimeError("Scored MMLU-Pro question IDs differ from baseline")
            for row in rows:
                previous = reference[row["doc_id"]]
                if row["doc"] != previous["doc"] or row["arguments"] != previous["arguments"]:
                    raise RuntimeError("Scored MMLU-Pro inputs differ from baseline")
            write_rows(OUTPUT / f"new_reader__{task}__samples.jsonl", rows)
            all_rows.extend(rows)
        correct = int(sum(row["exact_match"] for row in all_rows))
        if len(all_rows) != 128 or abs(result["results"]["mmlu_pro"]["exact_match,custom-extract"] - correct / 128) > 1e-8:
            raise RuntimeError("Official aggregate differs from the 128 scored generation records")
        result["experiment"] = {"protocol": CONFIG["protocol"], "condition": "gated_external_reader",
                                "manifest_sha256": digest(OUTPUT / "manifest.json"), "pilot_subset": True,
                                "full_official_test": False, "unchanged_baseline_requests": True,
                                "correct": correct, "n_questions": len(all_rows)}
        save(OUTPUT / "new_reader.json", result)
        baselines = {arm: json.loads((BASELINES / f"{arm}.json").read_text())["results"]["mmlu_pro"]
                     for arm in ("qwen_native", "llama_native", "stage_a", "stage_b")}
        valid = sum(row["filtered_resps"][0] in list("ABCDEFGHIJ") for row in all_rows)
        save(OUTPUT / "summary.json", {"condition": "gated_external_reader", "n_questions": 128,
                                       "correct": correct, "valid_extracted_answers": valid,
                                       "official_metrics": result["results"]["mmlu_pro"], "baselines": baselines,
                                       "pilot_subset": True, "max_gen_toks": 2048, "num_fewshot": 5})
        save(OUTPUT / "status.json", {"phase": "completed", "completed_questions": 128,
                                      "expected_questions": 128, "correct": correct,
                                      "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log(f"MMLU-PRO COMPLETED: {correct}/128; valid extracted answers={valid}/128")
    except Exception as error:
        save(OUTPUT / "status.json", {"phase": "failed", "error": repr(error),
                                      "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "run"))
    args = parser.parse_args()
    globals()[args.action]()
