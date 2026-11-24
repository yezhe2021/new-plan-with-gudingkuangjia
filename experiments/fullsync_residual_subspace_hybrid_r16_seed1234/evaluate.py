"""Evaluate trained RGST stages, subspace oracles, and the locked references."""

from statistics import mean

from train import (ROOT, C, K, SPECS, fixed, load_context, initialize_models, make_cache,
                   pull_raw, translate_body, torch)

PROJECT = ROOT.parent
workflow = fixed.load_module("rgst_eval_decoding", PROJECT / "fullsync_value_subspace_oracle_seed1234/run.py")
readout_tools = fixed.load_module("rgst_attention_reference", PROJECT /
                                  "dst_rank64_joint_conditional_statistics_seed1234/train.py")
BASELINE = PROJECT / "fullsync_ab_vs_joint_statistics_seed1234"
STAT_BASELINE = PROJECT / "fullsync_residual_generation_seed1234"
NEW = ("E0", "oracle16", "oracle32", *(f"{n}_{p}" for p in ("phase1", "phase2") for n in SPECS))
REFERENCES = ("native", "old_fullsync_ab", "dst_rank64", "recolor")
GENERATE_GROUPS = (("E0", "oracle16", "oracle32"),
                   *(tuple(f"{n}_{p}" for n in SPECS) for p in ("phase1", "phase2")))


def layer_readout(cache, probes, header):
    bodies = K.pairs(cache)
    key, value = (torch.cat([p[i][:, :, header:, :] for p in bodies], 0).float().unsqueeze(2) for i in (0, 1))
    q, outside_z, outside, target = probes
    scale = q.shape[-1] ** -0.5
    s = (q @ key.transpose(-1, -2)) * scale
    mass = (s.logsumexp(-1) - outside_z).sigmoid().unsqueeze(-1)
    actual = mass * (s.softmax(-1) @ value) + (1 - mass) * outside
    return float(((actual - target).square().mean((1,2,3,4)) /
                 target.square().mean((1,2,3,4)).clamp_min(1e-5)).mean())


def summary(rows):
    metrics = {}
    fields = ("correct", "em", "f1", "reached_cap", "logit_kl", "attention_nmse")
    for n in (*NEW, *REFERENCES):
        v = [r["results"][n] for r in rows]
        metrics[n] = {f: mean(x[f] for x in v) for f in fields if all(f in x for x in v)}
    comparisons = [("oracle16", "E0"), ("oracle32", "E0"), ("E3_phase1", "E1_phase1"),
                   ("E3_phase1", "E2_phase1"), ("E3_phase1", "E5_phase1"),
                   ("E4_phase1", "E3_phase1"), *((f"{n}_phase2", f"{n}_phase1") for n in SPECS),
                   ("E3_phase2", "old_fullsync_ab")]
    paired = {a+"_minus_"+b: {
        f: C.paired([r["results"][a][f] - r["results"][b][f] for r in rows])
        for f in fields if f in metrics[a] and f in metrics[b]}
        for a,b in comparisons}
    return dict(n=len(rows), metrics=metrics, paired=paired)


def previous(task, sample, row):
    a = C.read(BASELINE / "per_sample" / task / (sample + ".json"))
    b = C.read(STAT_BASELINE / "per_sample" / task / (sample + ".json"))
    assert a["gold"] == b["gold"] == row["gold"] and a["question"] == b["question"] == row["question"]
    return dict(native=a["results"]["native"], old_fullsync_ab=a["results"]["fullsync_ab"],
                dst_rank64=a["results"]["dst_rank64"], recolor=b["results"]["recolor"])


@torch.no_grad()
def run():
    torch.set_float32_matmul_precision("highest")
    p = ROOT / "manifest.json"
    manifest = C.read(p)
    assert all(C.sha(ROOT / n)==h for n,h in manifest["code"].items())
    assert (ROOT / "phase1/completed.json").exists() and (ROOT / "phase2/completed.json").exists()
    engine, legacy, assembler, taps, _ = load_context()
    engine.mode = "test"
    reader = fixed.BodyReader(engine)
    stats = torch.load(ROOT/"statistics.pt", map_location="cuda", weights_only=True)
    models = {}
    for phase in ("phase1", "phase2"):
        for n,model in initialize_models(stats).items():
            checkpoint = torch.load(ROOT / phase / (n+".pt"), weights_only=True, map_location="cuda")
            model.load_state_dict(checkpoint["state"])
            models[f"{n}_{phase}"] = model.eval().requires_grad_(False)
    task_rows = {r["id"]: r for r in fixed.rows("test")}
    results = {}
    policies = {}
    for task, ids in manifest["test"].items():
        is_choice = task in workflow.ref.probe.CHOICE
        batched = {}
        results[task] = []
        for i, sample in enumerate(ids, 1):
            out = ROOT / "per_sample" / task / (sample + ".json")
            row = task_rows[sample]
            if out.exists():
                record = C.read(out)
                batched = record["batches"]
            else:
                source = pull_raw(engine, legacy, row["prompt"], row["content_ids"])
                raw_native = legacy.capture(engine.model, engine.prefix_ids + row["content_ids"])
                start = len(engine.prefix_ids)
                body = [z[:, start:] for z in raw_native]
                unmodified = engine.native_cache(row, baseline=True)
                target_tokens = workflow.ref.probe.target_tokens(engine, row)
                teacher = reader.token_logits(row, unmodified, target_tokens, rn=False).log_softmax(-1)
                q_readout = readout_tools.TeacherReadout(engine, unmodified.get_seq_length())
                clone = C.DynamicCache(ddp_cache_data=K.pairs(unmodified), config=engine.model.config)
                suffix = row["suffix_ids"] + target_tokens
                with workflow.ref.probe.attention(engine.model, q_readout):
                    engine.forward(suffix, clone, clone.get_seq_length(), use_cache=False)
                probes = tuple(z.cuda() for z in q_readout.result())
                values = previous(task, sample, row)
                for j, group in enumerate(GENERATE_GROUPS):
                    cache = {}
                    for arm in group:
                        if arm == "E0":
                            mapped = translate_body(source, taps, stats)
                        elif arm in ("oracle16", "oracle32"):
                            mapped = translate_body(source, taps, stats, native=body, oracle_rank=int(arm[6:]))
                        else:
                            mapped = translate_body(source, taps, stats, model=models[arm])
                        cache[arm] = make_cache(engine, assembler, mapped)
                    if i == 1:
                        batched[str(j)] = [] if is_choice else workflow.batch_policy(reader, engine, row, cache)
                    scored = workflow.score(engine, reader, row, cache, batched[str(j)])
                    for name, prefix in cache.items():
                        logits = reader.token_logits(row, prefix, target_tokens, rn=False)
                        scored[name]["logit_kl"] = float(torch.nn.functional.kl_div(
                            logits.log_softmax(-1), teacher, log_target=True, reduction="batchmean"))
                        scored[name]["attention_nmse"] = layer_readout(prefix, probes, start)
                    values.update(scored)
                    del cache
                record = dict(id=sample, task=task, gold=row["gold"], question=row["question"],
                              batches=batched, results=values)
                C.save(out, record)
            results[task].append(record)
            if i == 1 or i % 4 == 0:
                f = "correct" if is_choice or task == "gsm8k" else "f1"
                C.log("RGST test", task, i, len(ids), {k: record["results"][k][f]
                     for k in ("E0", "E3_phase1", "E3_phase2", "oracle16", "oracle32")})
            C.save(ROOT/"status.json", dict(stage="evaluation", task=task, completed=i, total=len(ids)))
        C.save(ROOT/"tasks"/(task + ".json"), summary(results[task]))
        policies[task] = batched
        engine.unchanged()
    assert engine.student_test_native_prefills == 0
    C.save(ROOT/"summary.json", dict(status="completed", n=sum(map(len,results.values())),
        manifest_sha256=C.sha(p), policies=policies, tasks={t:summary(v) for t,v in results.items()},
        choice=summary([r for t in sorted(workflow.ref.probe.CHOICE) for r in results[t]]),
        multi_hop=summary([r for t in ("hotpotqa", "2wiki", "musique") for r in results[t]])))
    C.save(ROOT/"status.json", dict(stage="completed"))


if __name__ == "__main__":
    run()
