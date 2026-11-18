"""Replay the frozen short and full-document test cohorts with the new Stage B."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE_LONG = ROOT.parent / "long_qa_quality_leval_full64_seed1234"
ARM = "dst_rank64_route_output"


def checkpoint(core, torch, engine):
    completion = core.read(ROOT / "runs/completed.json")
    training_pin = core.sha(ROOT / "manifest.json")
    assert completion["status"] == "completed" and completion["manifest_sha256"] == training_pin
    path = ROOT / f"runs/epoch4_step{completion['steps']}.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert core.sha(path) == completion["checkpoint_sha256"]
    assert data["manifest_sha256"] == training_pin and data["step"] == 1536
    engine.correction.load_state_dict(data["state"], strict=True)
    engine.correction.eval().requires_grad_(False)
    engine.mode = "test"
    return completion["checkpoint_sha256"]


def check_pause():
    if (ROOT / "controls/pause_requested.json").exists():
        raise SystemExit(75)


def status(core, name, task, sample, total):
    core.save(ROOT / "status.json", dict(phase=name, task=task, sample=sample, total=total, step=1536, total_steps=1536))


def write_report(core, split, pin, records, metrics, baseline):
    arms = records[0]["conditions"]
    aggregate = {arm: {key: sum(r["conditions"][arm][key] for r in records) / len(records) for key in metrics}
                 for arm in arms}
    delta = {key: core.paired([r["conditions"][ARM][key] - r["conditions"][baseline][key] for r in records])
             for key in metrics}
    result = dict(n=len(records), scores=aggregate, paired_minus_original=delta, checkpoint_sha256=pin)
    core.save(ROOT / "evaluation" / split / records[0]["task"] / "summary.json", result)
    return result


def short():
    from objective import C, CFG, G, REFERENCE, BodyReader, Engine, score, torch

    engine = Engine()
    pin = checkpoint(C, torch, engine)
    reader = BodyReader(engine)
    old_pin = G.identity()
    rows = G.rows("test")
    tasks = {}
    with torch.no_grad():
        for task, total in CFG["test_counts"].items():
            saved = []
            for i, row in enumerate((r for r in rows if r["task"] == task), 1):
                check_pause()
                name = Path(task) / "per_sample" / (row["id"] + ".json")
                prior = C.read(REFERENCE / "evaluation" / name)
                assert prior["manifest_sha256"] == old_pin and prior["id"] == row["id"] and prior["gold"] == row["gold"]
                path = ROOT / "evaluation/short" / name
                if path.exists():
                    record = C.read(path)
                    assert record["checkpoint_sha256"] == pin and record["id"] == row["id"]
                else:
                    result = score(reader, row, engine.translated(row, engine.correction, checkpointed=False))
                    conditions = {key: prior["conditions"][key] for key in ("native", "B_rank64", "C")}
                    conditions[ARM] = result
                    if task == "hellaswag":
                        for value in conditions.values():
                            raw = value["raw_sum_scores"]
                            value["raw_correct"] = int(max(range(len(raw)), key=raw.__getitem__) == row["gold"])
                    record = dict(checkpoint_sha256=pin, task=task, id=row["id"], gold=row["gold"], conditions=conditions)
                    C.save(path, record)
                assert engine.student_test_native_prefills == 0
                saved.append(record)
                status(C, "eval_short", task, i, total)
            assert len(saved) == total
            keys = {"gsm8k": ("correct", "legacy_correct"), "hellaswag": ("correct", "raw_correct"),
                    "hotpotqa": ("em", "f1"), "2wiki": ("em", "f1"), "musique": ("em", "f1")}.get(task, ("correct",))
            tasks[task] = write_report(C, "short", pin, saved, keys, "B_rank64")
            print("SHORT COMPLETED", task, tasks[task]["scores"][ARM], flush=True)
    C.save(ROOT / "short_summary.json", dict(status="completed", checkpoint_sha256=pin, tasks=tasks))


def long_prediction(longer, engine, row, task, gold):
    longer.B.SENDER_CHUNK = longer.PREFILL
    sender_ids, source = longer.B.capture(engine, row)
    assert row["sender_ids"] == sender_ids
    index = longer.B.fullsync(row["prompt"], engine.sender_tok, engine.tok, row["content_ids"])
    assert list(index.source_ids) == sender_ids
    length = len(row["content_ids"])
    body = longer.transferred(engine, length, source, longer.torch.tensor(index.source_indices, device="cpu"),
                              None, longer.CHUNK)
    return longer.answer(engine, task, row, body, gold)


def long():
    sys.path.insert(0, str(REFERENCE_LONG))
    import evaluate as longer
    from cohort import COUNT, TASKS, prepare

    C, torch = longer.C, longer.torch
    cohorts, info = prepare(C)
    parent = REFERENCE_LONG / "manifest.json"
    prior = C.read(parent)
    assert info == prior["cohort"] and C.read(REFERENCE_LONG / "summary.json")["status"] == "completed"
    old_pin = C.sha(parent)
    engine = longer.B.Engine()
    pin = checkpoint(C, torch, engine)
    header = C.read(C.ROOT / "manifest.json")
    results = {}
    with torch.no_grad():
        for task in TASKS:
            records = []
            for i, item in enumerate(cohorts[task], 1):
                check_pause()
                name = Path(task) / "per_sample" / (item["id"] + ".json")
                old = C.read(REFERENCE_LONG / "evaluation" / name)
                assert old["manifest_sha256"] == old_pin and not old["clipped"] and old["gold"] == item["gold"]
                assert old["context_sha256"] == C.digest(item["context"]) and old["id"] == item["id"]
                path = ROOT / "evaluation/long" / name
                if path.exists():
                    record = C.read(path)
                    assert record["checkpoint_sha256"] == pin and record["id"] == item["id"]
                else:
                    prompt = item["before"] + item["context"] + item["after"]
                    assert C.digest(prompt) == old["prompt_sha256"]
                    row = longer.encode_row(engine, header, task, prompt)
                    assert len(row["content_ids"]) == old["receiver_body_tokens"]
                    prediction = long_prediction(longer, engine, row, task, item["gold"])
                    if task == "sfiction":
                        fact = longer.encode_row(engine, header, task, item["fact_prompt"])
                        assert old["fact_gold"] == item["fact"]
                        factual = long_prediction(longer, engine, fact, task, item["fact"])
                        prediction["loyalty_correct"] = prediction["correct"]
                        prediction["fact_correct"] = factual["correct"]
                        prediction["fact"] = factual
                        prediction["correct"] = .5*(prediction["loyalty_correct"]+prediction["fact_correct"])
                    assert engine.student_test_native_prefills == 0
                    record = dict(id=item["id"], task=task, checkpoint_sha256=pin, gold=item["gold"],
                                  body_tokens=len(row["content_ids"]), clipped=False,
                                  conditions=dict(old["conditions"], **{ARM: prediction}))
                    C.save(path, record)
                records.append(record)
                status(C, "eval_long", task, i, COUNT)
            assert len(records) == COUNT
            keys = ("correct", "loyalty_correct", "fact_correct") if task == "sfiction" else ("correct",)
            results[task] = write_report(C, "long", pin, records, keys, "dst_rank64")
            print("LONG COMPLETED", task, results[task]["scores"][ARM], flush=True)
    C.save(ROOT / "long_summary.json", dict(status="completed", checkpoint_sha256=pin, tasks=results))
    status(C, "completed", None, 192, 192)


if __name__ == "__main__":
    {"short": short, "long": long}[sys.argv[1]]()
