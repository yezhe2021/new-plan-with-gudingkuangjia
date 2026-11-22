"""Swap M32 banks between matched same-task rows, preserving recipient body KV."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "dst_rank64_gated_dynamic_m64_a4_server_c_seed1234/reference_m32"
CHECKPOINT = ROOT.parent / "dst_rank64_gated_joint_m32_a4_seed1234/initial/runs/epoch4_step1536.pt"
CHECKPOINT_SHA = "4547b751d7d486efb803ce2b7df65ecad5df0b91739d3aa6830831be8ca914f6"
COHORT_SHA = "de54be31b2d6830e3e09e70d056c2011d2ad3c81a62109513ceb90aee3f4b625"
RANK64 = "B_rank64"
CORRECT = "gated_dynamic"
WRONG = "wrong_memory"
ARMS = (RANK64, CORRECT, WRONG)
QA = {"hotpotqa", "2wiki", "musique"}
DIFFERENT_FIELDS = ("question", "prompt", "context_sha256", "source_doc_sha256", "support_sha256")

sys.path.insert(0, str(REFERENCE))
import experiment as original
from gated_memory import AnswerCache

C, G, torch = original.C, original.G, original.torch


def status(phase, **fields):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T %z"), phase=phase, **fields))


def length(row):
    return len(row["content_ids"])


def matches(task_rows):
    """Pair nearby body lengths without using answers, predictions, or scores."""
    pending = sorted(task_rows, key=lambda row: (length(row), row["id"]))
    result = []
    while pending:
        left = pending.pop(0)
        right_index = next(i for i, right in enumerate(pending)
                           if all(not left.get(key) or left[key] != right.get(key) for key in DIFFERENT_FIELDS))
        result.append((left, pending.pop(right_index)))
    return result


def prepare():
    old_manifest = REFERENCE / "manifest.json"
    checkpoint_sha = C.sha(CHECKPOINT)
    cohort_sha = C.sha(G.INPUTS / "prepared/test.jsonl")
    rank64_sha = C.sha(G.INPUTS / "rank64.pt")
    assert checkpoint_sha == CHECKPOINT_SHA and cohort_sha == COHORT_SHA
    assert rank64_sha == original.SETTINGS["inputs_sha256"]["rank64.pt"]
    assert C.sha(G.INPUTS / "dst.pt") == original.SETTINGS["inputs_sha256"]["dst.pt"]
    rows = G.rows("test")
    assert len(rows) == len({row["id"] for row in rows}) == 512
    assert Counter(row["task"] for row in rows) == original.CFG["test_counts"]
    groups = {task: matches([row for row in rows if row["task"] == task]) for task in original.CFG["test_counts"]}
    donor_map = {item["id"]: other["id"] for pairs in groups.values() for left, right in pairs
                 for item, other in ((left, right), (right, left))}
    manifest = dict(protocol="dst-fullbody-chat-v1", source_manifest_sha256=C.sha(old_manifest),
                    rank64_sha256=rank64_sha, checkpoint_sha256=checkpoint_sha, test_sha256=cohort_sha,
                    evaluator_sha256=C.sha(Path(__file__)), scorer_sha256=C.sha(G.INPUTS / "score.py"),
                    donor_rule="reciprocal within-task pairs, greedy nearest body lengths, distinct question and available context fingerprints",
                    donor_map=donor_map)
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)
    return groups, manifest, C.sha(path)


def load(manifest):
    for role in ("sender", "receiver"):
        path = Path(C.CFG["models"][role])
        if not path.is_dir():
            path = Path("/hy-tmp/yezhe/data/all_models") / path.relative_to("/hy-tmp/yezhe/all_models")
        assert path.is_dir(), path
        C.CFG["models"][role] = str(path)
    engine = original.Engine()
    memory = original.GatedDynamicMemory(engine, original.CFG)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["manifest_sha256"] == manifest["source_manifest_sha256"] and checkpoint["step"] == 1536
    memory.load_state_dict(checkpoint["state"], strict=True)
    memory.eval().requires_grad_(False)
    assert memory.M == 32 and memory.gates.numel() == 36
    assert torch.count_nonzero(memory.gates)
    engine.mode = "test"
    return engine, memory


def bank_difference(correct, wrong, offset):
    left = torch.stack(correct.bank[offset::2]).float()
    right = torch.stack(wrong.bank[offset::2]).float()
    return float((left - right).norm() / left.norm().clamp_min(1e-12))


def archived(row, manifest):
    path = REFERENCE / "evaluation" / row["task"] / "per_sample" / (row["id"] + ".json")
    record = C.read(path)
    assert record["manifest_sha256"] == manifest["source_manifest_sha256"]
    assert all(record[key] == row[key] for key in ("id", "question", "gold"))
    return record["conditions"]


def evaluate_row(engine, memory, row, donor, own, other, manifest, pin):
    prior = archived(row, manifest)
    correct = original.score(memory, row, own, rn=True)
    # construct() already unrotates each donor key to the gate's canonical coordinates.
    # The independent readout does not consume donor absolute positions.
    wrong_cache = AnswerCache(body=own.body, bank=other.bank)
    wrong = original.score(memory, row, wrong_cache, rn=True)
    assert engine.student_test_native_prefills == 0
    engine.unchanged()
    scores = {RANK64: prior[RANK64], CORRECT: correct, WRONG: wrong}
    if row["task"] == "hellaswag":
        for result in scores.values():
            raw = result["raw_sum_scores"]
            result["raw_correct"] = int(max(range(len(raw)), key=raw.__getitem__) == row["gold"])
    parity = {key: correct[key] == prior[CORRECT][key]
              for key in ("prediction", "correct", "em", "f1", "legacy_correct") if key in correct}
    return dict(manifest_sha256=pin, id=row["id"], donor_id=donor["id"], gold=row["gold"],
                body_tokens=length(row), donor_body_tokens=length(donor), conditions=scores,
                relative_key_l2=bank_difference(own, other, 0),
                relative_value_l2=bank_difference(own, other, 1), correct_m32_archive_parity=parity)


def sample_path(row):
    return ROOT / "evaluation" / row["task"] / "per_sample" / (row["id"] + ".json")


def checked_record(row, donor, pin):
    record = C.read(sample_path(row))
    assert record["manifest_sha256"] == pin and record["id"] == row["id"] and record["donor_id"] == donor["id"]
    return record


def aggregate(task, record_pairs):
    records = [record for pair in record_pairs for record in pair]
    keys = ("em", "f1") if task in QA else ("correct", "legacy_correct") if task == "gsm8k" else (
        ("correct", "raw_correct") if task == "hellaswag" else ("correct",))
    metrics = {arm: {key: statistics.mean(record["conditions"][arm][key] for record in records) for key in keys}
               for arm in ARMS}
    deltas = {}
    for baseline in (RANK64, CORRECT):
        comparisons = {}
        for key in keys:
            def difference(record):
                return record["conditions"][WRONG][key] - record["conditions"][baseline][key]

            comparisons[key] = dict(
                pair_bootstrap=C.paired([statistics.mean(map(difference, pair)) for pair in record_pairs]),
                improved=sum(difference(record) > 0 for record in records),
                declined=sum(difference(record) < 0 for record in records),
            )
        deltas[baseline] = comparisons
    return dict(n=len(records), metrics=metrics, wrong_minus=deltas,
                median_relative_key_l2=statistics.median(r["relative_key_l2"] for r in records),
                median_relative_value_l2=statistics.median(r["relative_value_l2"] for r in records),
                archive_disagreements=sum(not all(r["correct_m32_archive_parity"].values()) for r in records))


def save_summary(tasks, pin, complete):
    output = dict(status="completed" if complete else "running", manifest_sha256=pin, tasks=tasks)
    if complete:
        output["macro_acc_em"] = {
            arm: statistics.mean(v["metrics"][arm]["em" if task in QA else "correct"] for task, v in tasks.items())
            for arm in ARMS
        }
        lines = ["# Wrong-Memory Intervention on Gated Dynamic M32", "",
                 "| Task | Metric | Rank64 (archived) | Correct M32 | Wrong M32 | Wrong − Correct |",
                 "|---|---|---:|---:|---:|---:|"]
        for task, item in tasks.items():
            for key in item["metrics"][CORRECT]:
                values = [item["metrics"][arm][key] for arm in ARMS]
                lines.append(f"| {task} | {key} | " + " | ".join(f"{v:.2%}" for v in values)
                             + f" | {(values[-1]-values[-2])*100:+.2f} pp |")
        values = list(output["macro_acc_em"].values())
        lines.append("| **Macro** | Acc / EM | " + " | ".join(f"{v:.2%}" for v in values)
                     + f" | {(values[-1]-values[-2])*100:+.2f} pp |")
        (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    C.save(ROOT / "summary.json", output)


@torch.no_grad()
def run():
    summary = ROOT / "summary.json"
    if summary.exists() and C.read(summary)["status"] == "completed":
        return
    status("prepare")
    groups, manifest, pin = prepare()
    while True:
        if (ROOT / "controls/pause_requested.json").exists():
            status("paused")
            return
        free_mib = torch.cuda.mem_get_info()[0] // 2**20
        if free_mib >= 18_500:
            break
        status("waiting_gpu", free_mib=free_mib)
        time.sleep(30)
    engine, memory = load(manifest)
    stats = {}
    completed = 0
    # Run the three more evidence-dependent tasks first.
    order = ("hotpotqa", "2wiki", "musique", "arc", "sciq", "mmlu_pro", "hellaswag", "gsm8k")
    assert set(order) == set(groups)
    started = time.perf_counter()
    for task in order:
        task_pairs = []
        for left, right in groups[task]:
            if (ROOT / "controls/pause_requested.json").exists():
                status("paused", task=task, completed=completed)
                return
            pair = ((left, right), (right, left))
            if not all(sample_path(row).exists() for row, _ in pair):
                banks = {}
                for row in (left, right):
                    body = engine.cache(row, engine.base(row), engine.correction)
                    banks[row["id"]] = memory.construct(row, body)
                    assert banks[row["id"]].body is body
                    assert len(banks[row["id"]].bank) == 72 and all(x.shape[-2] == 32 for x in banks[row["id"]].bank)
                for row, donor in pair:
                    if not sample_path(row).exists():
                        record = evaluate_row(engine, memory, row, donor, banks[row["id"]], banks[donor["id"]], manifest, pin)
                        C.save(sample_path(row), record)
                del banks, body
            task_pairs.append([checked_record(row, donor, pin) for row, donor in pair])
            completed += 2
            status("evaluation", task=task, task_completed=len(task_pairs)*2, completed=completed,
                   total=512, elapsed_seconds=time.perf_counter()-started)
            if len(task_pairs) == 1 or len(task_pairs) % 4 == 0:
                print("WRONG MEMORY", task, len(task_pairs)*2, "/ 64", "overall", completed, "/ 512", flush=True)
        stats[task] = aggregate(task, task_pairs)
        C.save(ROOT / "evaluation" / task / "summary.json", stats[task])
        save_summary(stats, pin, complete=(completed == 512))
        print("TASK COMPLETE", task, stats[task]["metrics"], flush=True)
    status("completed", completed=completed, total=512)


def main():
    if sys.argv[1] == "launch":
        (ROOT / "logs").mkdir(exist_ok=True)
        with (ROOT / "logs/pipeline.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(__file__), "run"], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        C.save(ROOT / "launch.json", dict(pid=child.pid, server="server-b", time=time.strftime("%F %T %z")))
        print("Launched", child.pid, flush=True)
    else:
        with (ROOT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run()
            except Exception as error:
                status("failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
