"""Run the full fixed DST suite for the two translators and nested exact oracles."""

import time
from statistics import mean

from train import ROOT, C, CFG, fixed, torch, Experiment, new_models

reader_tools = fixed.load_module("rgst32_evaluation_tools", ROOT.parent /
                                 "fullsync_value_subspace_oracle_seed1234/run.py")
SCORER_TASKS = reader_tools.ref.probe.CHOICE
ARMS = ("ridge500", "rgst16", "rgst32", "oracle16", "oracle32", "dst_rank64", "native")


def summarize(rows):
    fields = ("correct", "em", "f1", "reached_cap", "generated_tokens")
    summary = {}
    for name in ARMS:
        entries = [r["results"][name] for r in rows]
        summary[name] = {f: mean(e[f] for e in entries) for f in fields if all(f in e for e in entries)}
    pairs = (("rgst32","rgst16"), ("oracle32","oracle16"),
             ("rgst16","ridge500"), ("rgst32","ridge500"), ("rgst32","dst_rank64"))
    return dict(n=len(rows), metrics=summary, paired={
        a+"_minus_"+b: {f:C.paired([r["results"][a][f] - r["results"][b][f] for r in rows])
                        for f in fields if f in summary[a] and f in summary[b]}
        for a,b in pairs})


@torch.no_grad()
def run():
    torch.set_float32_matmul_precision("highest")
    experiment = Experiment()
    experiment.prepare()
    engine, reader = experiment.engine, experiment.reader
    assert C.read(ROOT/"stage2/completed.json")["updates_per_rank"] == CFG["updates_per_epoch"] * CFG["stage2"]["epochs"]
    models = new_models()
    for n, model in models.items():
        checkpoint = torch.load(ROOT/"stage2"/f"final_r{n}.pt", map_location="cuda", weights_only=True)
        assert checkpoint["manifest_sha256"] == experiment.identity
        model.load_state_dict(checkpoint["model"])
        model.eval().requires_grad_(False)
    engine.mode = "test"
    tasks = {}
    total = 0
    for task, count in CFG["test_counts"].items():
        tasks[task] = []
        rows = [r for r in fixed.rows("test") if r["task"] == task]
        assert len(rows) == count
        policy = []
        for i, row in enumerate(rows, 1):
            start = time.monotonic()
            sample = row["id"]
            path = ROOT / "per_sample" / task / (sample+".json")
            if path.exists():
                record = C.read(path)
                policy = record["batch_policy"]
            else:
                source = experiment.source(row)
                teacher = experiment.target(row)
                caches = {"ridge500": experiment.make_cache(experiment.predict(source))}
                caches.update({f"rgst{n}": experiment.make_cache(experiment.predict(source, model))
                               for n,model in models.items()})
                caches.update({f"oracle{n}": experiment.make_cache(experiment.predict(source, target=teacher, rank=n))
                               for n in (16,32)})
                if i == 1 and task not in SCORER_TASKS:
                    policy = reader_tools.batch_policy(reader, engine, row, caches)
                result = reader_tools.score(engine, reader, row, caches, policy)
                previous = C.read(fixed.ROOT/"evaluation"/task/"per_sample"/(sample+".json"))
                assert previous["manifest_sha256"] == fixed.identity()
                assert previous["id"] == sample and previous["question"] == row["question"] and previous["gold"] == row["gold"]
                result.update({n: previous["conditions"][n] for n in ("dst_rank64", "native")})
                record = dict(id=sample, task=task, question=row["question"], gold=row["gold"],
                              batch_policy=policy, results=result)
                C.save(path, record)
            tasks[task].append(record)
            total += 1
            C.save(ROOT/"status.json", dict(stage="evaluation", task=task, task_completed=i,
                   completed=total, total=sum(CFG["test_counts"].values()), seconds=time.monotonic()-start))
            if i == 1 or i % 8 == 0:
                C.log("RGST evaluation", task, i, len(rows))
        C.save(ROOT/"tasks"/(task+".json"), summarize(tasks[task]))
    C.save(ROOT/"summary.json", dict(status="completed", n=total, manifest_sha256=experiment.identity,
        tasks={t:summarize(v) for t,v in tasks.items()},
        choice=summarize([r for t in sorted(SCORER_TASKS) for r in tasks[t]]),
        multi_hop=summarize([r for t in ("hotpotqa","2wiki","musique") for r in tasks[t]])))
    C.save(ROOT/"status.json", dict(stage="completed", evaluated=total))


if __name__ == "__main__":
    run()
