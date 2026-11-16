"""Paired downstream evaluation and offline Native-referenced diagnostics."""
import fcntl
import math
import sys

import numpy as np
from torch.nn import functional as F

import experiment as G

C, K, torch = G.C, G.K, G.torch
QA = {"hotpotqa", "2wiki", "musique"}
TASKS = tuple(G.CFG["test_counts"])
METRICS = ("k_nmse", "v_nmse", "k_cosine", "v_cosine", "attention_output_cosine", "logit_kl", "gold_ce")


def load():
    models = {}
    for arm in G.ARMS:
        for stage in G.STAGES:
            if arm == "no_carry":
                path = G.checkpoint_path(arm, stage)
                assert G.done(f"runs/{stage}/completed.json")
                assert C.sha(path) == C.read(path.parent / "completed.json")["checkpoint_sha256"]
        base = G.load_weights(G.stage1_model(arm, device="cuda"), arm, "stage1")
        rank = G.load_weights(G.rank64_model(base), arm, "stage2")
        models[arm, "stage1"] = base.eval().requires_grad_(False)
        models[arm, "stage2"] = rank.eval().requires_grad_(False)
    return models


def location(stage, phase, row):
    return f"{phase}/{stage}/{row['task']}/per_sample/{row['id']}.json"


def missing(stage, phase, row, arm):
    return not G.done(location(stage, phase, row), arm)


def progress(phase, row, index, total):
    G.status(status="running", phase=phase, task=row["task"], sample=index, total=total)
    if index == 1 or index % 16 == 0 or index == total:
        C.log("DEPTH ABLATION", phase, index, total, row["task"])


def cache_metrics(engine, model, row, source, target):
    positions = row["sync_source_indices"]
    n = len(positions)
    chunk = G.CFG["token_chunk"]
    prefix = len(engine.prefix_ids)
    error, energy, cosine = [torch.zeros(2, 36, device="cuda") for _ in range(3)]
    pieces = []
    engine.norm_counts = [0] * 36
    for begin in range(0, n, chunk):
        end = min(begin + chunk, n)
        predicted = model(*source, positions[begin:end])
        for i, (p, t) in enumerate(zip(predicted, target)):
            native = t[:, begin:end].to(device=p.device, dtype=torch.float32)
            error[i] += (p - native).square().sum((1, 2))
            energy[i] += native.square().sum((1, 2))
            cosine[i] += F.cosine_similarity(p, native, dim=-1).sum(1)
        positions_q = torch.arange(prefix + begin, prefix + end, device="cuda")
        pieces.append(G.engine_module.finalize_raw(engine.model, *predicted, positions_q))
    assert engine.norm_counts == [math.ceil(n / chunk)] * 36
    engine.norm_counts = None
    pairs = []
    for layer, pair in enumerate(K.pairs(engine.header)):
        pairs.append(tuple(torch.cat([value, *(piece[i][layer][None] for piece in pieces)], dim=2)
                           for i, value in enumerate(pair)))
    cache = C.DynamicCache(ddp_cache_data=pairs, config=engine.model.config)
    nmse = (error / energy.clamp_min(1e-6 * n * target[0].shape[-1])).mean(1).tolist()
    cosine = (cosine / n).mean(1).tolist()
    return cache, dict(k_nmse=nmse[0], v_nmse=nmse[1], k_cosine=cosine[0], v_cosine=cosine[1])


def readout(attn, query, key, value):
    window = getattr(attn, "sliding_window", None)
    if window:
        key, value = key[:, :, -window:], value[:, :, -window:]
    output = F.scaled_dot_product_attention(
        query, key.repeat_interleave(attn.num_key_value_groups, dim=1),
        value.repeat_interleave(attn.num_key_value_groups, dim=1), scale=attn.scaling,
    )
    return attn.o_proj(output.transpose(1, 2).reshape(1, 1, -1)).flatten().float()


def native_queries(engine, row, native):
    model = engine.model
    boundary = native.get_seq_length()
    suffix = row["suffix_ids"]
    end = boundary + len(suffix)
    tables = C._rotary_tables(model, torch.arange(end - 1, end, device="cuda"))
    queries = {}
    def hook(layer):
        def capture(_module, _inputs, output):
            raw = output[0, -1]
            cos, sin = tables(layer)
            left, right = raw.chunk(2, dim=-1)
            rotated = torch.cat((-right, left), dim=-1)
            queries[layer] = (raw * cos + rotated * sin)[None, :, None, :]
        return capture
    attention = [block.self_attn for block in C.layers_of(model)]
    handles = [attn.q_norm.register_forward_hook(hook(i)) for i, attn in enumerate(attention)]
    try:
        result = engine.forward(suffix, K.clone(native, model, detach=True), boundary, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    native_pairs = K.pairs(result.past_key_values)
    assert len(queries) == len(native_pairs) == 36
    assert all(key.shape[2] == end for key, _ in native_pairs)
    return [(attn, queries[i], readout(attn, queries[i], *native_pairs[i]),
             tuple(x[:, :, boundary:] for x in native_pairs[i])) for i, attn in enumerate(attention)]


def diagnose_row(engine, reader, models, row, write=True):
    source = engine.source(row)
    tk, tv, native = engine.teacher(row)
    tokens = row["gold_tokens"]
    gold = torch.tensor(tokens, device="cuda")
    teacher = reader.token_logits(row, native, tokens, rn=False)
    query = native_queries(engine, row, native)
    for (arm, stage), model in models.items():
        if write and not missing(stage, "diagnostics", row, arm):
            continue
        cache, stats = cache_metrics(engine, model, row, source, (tk, tv))
        similarity = []
        for (attn, q, native_out, native_suffix), pair in zip(query, K.pairs(cache)):
            hybrid = [torch.cat((value, suffix), dim=2) for value, suffix in zip(pair, native_suffix)]
            ours = readout(attn, q, *hybrid)
            similarity.append(float(F.cosine_similarity(ours, native_out, dim=0)))
        student = reader.token_logits(row, cache, tokens, rn=False)
        stats.update(attention_output_cosine=sum(similarity) / len(similarity),
                     logit_kl=float(C.vocab_kl(teacher, student)), gold_ce=float(F.cross_entropy(student, gold)))
        assert all(math.isfinite(v) for v in stats.values()), (arm, stage, row["id"], stats)
        if write:
            G.mark(location(stage, "diagnostics", row), arm=arm, status="completed",
                   id=row["id"], task=row["task"], metrics=stats, layer_attention_output_cosine=similarity)
        else:
            C.log("DIAGNOSTIC SMOKE", arm, stage, row["id"], stats)
        del cache


@torch.no_grad()
def diagnostics():
    G.verify()
    engine = G.Engine()
    reader = G.BodyReader(engine)
    models = load()
    rows = G.reference.rows("validation")
    for i, row in enumerate(rows, 1):
        if G.paused():
            raise SystemExit(75)
        if any(missing(stage, "diagnostics", row, arm) for arm, stage in models):
            diagnose_row(engine, reader, models, row)
            engine.unchanged()
        progress("diagnostics", row, i, len(rows))


def scorer():
    original = G.reference.N.ORIGINAL
    sys.path.extend(map(str, (C.ROOT, original, original / "cross_dataset5x64_seed1234")))
    return G.load_module("depth_ablation_scorer", G.RANK / "score.py", experiment_common=G.scope).score


@torch.no_grad()
def test(smoke=False):
    G.verify()
    if smoke and G.done("checks/evaluation_completed.json"):
        return
    score = scorer()
    engine = G.Engine()
    reader = G.BodyReader(engine)
    engine.mode = "test"
    records = C.read_jsonl(G.HELDOUT)
    if smoke:
        models = {("carry", "stage1"): G.load_weights(G.stage1_model("carry", device="cuda"), "carry", "stage1")}
        records = [next(row for row in records if row["task"] == task) for task in ("arc", "hellaswag")]
    else:
        models = load()
    before = {key: C.versions(model) for key, model in models.items()}
    for i, row in enumerate(records, 1):
        if G.paused():
            raise SystemExit(75)
        pending = [key for key in models if smoke or missing(key[1], "evaluation", row, key[0])]
        if pending:
            source = engine.source(row)
            for arm, stage in pending:
                engine.norm_counts = [0] * 36
                cache = engine.translated(row, models[arm, stage], source, checkpointed=False)
                assert engine.norm_counts == [math.ceil(len(row["content_ids"]) / G.CFG["token_chunk"])] * 36
                engine.norm_counts = None
                result = score(reader, row, cache)
                assert engine.student_test_native_prefills == 0
                if smoke:
                    C.log("TEST SMOKE", row["task"], row["id"], result)
                else:
                    G.mark(location(stage, "evaluation", row), arm=arm, status="completed", id=row["id"],
                           task=row["task"], gold=row["gold"], result=result, student_native_body_tokens=0)
                del cache
            del source
            engine.unchanged()
            assert all(C.versions(model) == before[key] for key, model in models.items())
        if not smoke:
            progress("evaluation", row, i, len(records))
    if smoke:
        engine.mode = "train"
        diagnose_row(engine, reader, models, G.reference.rows("validation")[0], write=False)
        G.mark("checks/evaluation_completed.json", status="completed", native_body_prefills=0,
               k_norm_once_per_chunk=True, tasks=[row["task"] for row in records])


def read(stage, phase, row, arm, field):
    path = location(stage, phase, row)
    assert G.done(path, arm), (arm, path)
    value = C.read(G.folder(arm) / path)
    assert value["id"] == row["id"] and value["task"] == row["task"]
    return value[field]


def metric_keys(task):
    if task in QA:
        return ("f1", "em")
    if task == "gsm8k":
        return ("correct", "legacy_correct")
    if task == "hellaswag":
        return ("correct", "raw_sum_correct")
    return ("correct",)


def metrics_for(task, result, gold):
    if task == "hellaswag":
        result = {**result, "raw_sum_correct": int(np.argmax(result["raw_sum_scores"]) == gold)}
    return {k: result[k] for k in metric_keys(task)}


def intervals(a, b, indices):
    sampled_a = a[indices].mean(1)
    sampled_b = b[indices].mean(1)
    return {
        "carry": (float(a.mean()), sampled_a),
        "no_carry": (float(b.mean()), sampled_b),
        "carry_minus_no_carry": (float((a - b).mean()), sampled_a - sampled_b),
    }


def estimate(values):
    return {key: dict(mean=point, ci95=np.quantile(samples, [.025, .975]).tolist())
            for key, (point, samples) in values.items()}


def summarize_stage(stage, heldout, validation):
    rng = np.random.default_rng(G.CFG["seed"])
    repeats = G.SETTINGS["bootstrap_samples"]
    tasks, macro = {}, []
    for task in TASKS:
        rows = [row for row in heldout if row["task"] == task]
        pairs = []
        for row in rows:
            pair = {arm: metrics_for(task, read(stage, "evaluation", row, arm, "result"), row["gold"])
                    for arm in G.ARMS}
            differences = {k: pair["carry"][k] - pair["no_carry"][k] for k in metric_keys(task)}
            pairs.append(dict(id=row["id"], **pair, carry_minus_no_carry=differences))
        indices = rng.integers(0, len(pairs), size=(repeats, len(pairs)))
        primary = metric_keys(task)[0]
        tasks[task] = {}
        for key in metric_keys(task):
            a, b = (np.array([p[arm][key] for p in pairs]) for arm in G.ARMS)
            result = intervals(a, b, indices)
            tasks[task][key] = estimate(result)
            if key == primary:
                macro.append(result)
        for arm in G.ARMS:
            G.mark(f"evaluation/{stage}/{task}/summary.json", arm=arm, status="completed", n=len(pairs),
                   metrics={key: val[arm] for key, val in tasks[task].items()})
        C.save(G.ROOT / f"paired/{stage}/{task}.json", dict(n=len(pairs), scores=tasks[task], per_sample=pairs))
    macro = estimate({label: (float(np.mean([result[label][0] for result in macro])),
                              np.mean([result[label][1] for result in macro], axis=0)) for label in macro[0]})
    diag = {}
    for arm in G.ARMS:
        items = [read(stage, "diagnostics", row, arm, "metrics") for row in validation]
        diag[arm] = {k: float(np.mean([item[k] for item in items])) for k in METRICS}
        G.mark(f"diagnostics/{stage}/summary.json", arm=arm, status="completed", n=len(items), metrics=diag[arm])
    return dict(tasks=tasks, eight_task_macro=macro, diagnostics=diag)


def markdown(stages):
    def pct(v):
        return f"{100*v['mean']:.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]"
    lines = ["# DST Depth-Recurrence Ablation",
             "", "Llama3.2-3B → Qwen3-4B, seed 1234, final step 1536 per stage. Full-Sync, no Memory.",
             "", "Offline diagnostics use the same 192 validation rows. NMSE and cosine use raw projected K/V. "
             "The attention probe uses the Native Qwen query at the final suffix token, keeps the Native suffix K/V "
             "fixed, and measures cosine after the attention output projection. Logit KL and CE use the full gold target.",
             "", "| Stage | Arm | K NMSE | V NMSE | K cosine | V cosine | Attention output cosine | Logit KL | Gold CE | 8-task mean (%) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stage, result in stages.items():
        for arm in G.ARMS:
            diag = " | ".join(f"{result['diagnostics'][arm][key]:.5f}" for key in METRICS)
            lines.append(f"| {stage} | {arm} | {diag} | {pct(result['eight_task_macro'][arm])} |")
    for stage, result in stages.items():
        lines += ["", f"## {stage}", "",
                  "| Task | Metric | Carry (%) | No-Carry (%) | Carry − No-Carry (pp) |",
                  "|---|---|---:|---:|---:|"]
        for task, metrics in {**result["tasks"], "Macro": {"primary": result["eight_task_macro"]}}.items():
            for key, vals in metrics.items():
                lines.append(f"| {task} | {key} | " + " | ".join(pct(v) for v in vals.values()) + " |")
    lines += ["", "Each test task uses 128 rows. Intervals are paired question bootstrap 95% intervals "
              "with 10,000 replicates and within-task resampling for the macro. Macro uses QA token F1 and "
              "accuracy elsewhere. HellaSwag token-mean scores and GSM8K strict numeric extraction are primary.",
              "", "The held-out cohort previously served Gated Joint evaluation and is disjoint from this training, "
              "validation and original DST test cohort by normalized question. Both students use zero Native body tokens. "
              "The two arms have equal allocated parameters. No-Carry can leave some previous-state parameters without effective gradients."]
    return "\n".join(lines) + "\n"


def append_index(section):
    index = G.ROOT.parent.parent / "EXPERIMENT_RESULTS.md"
    with index.with_name(".experiment_results.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = index.read_text() if index.is_file() else ""
        if section.splitlines()[0] not in old.splitlines():
            updated = index.with_name(".experiment_results.depth_ablation.tmp")
            updated.write_text(old + ("\n\n" if old else "# Experiment Results\n\n") + section)
            updated.replace(index)


def report():
    G.verify()
    heldout, validation = C.read_jsonl(G.HELDOUT), G.reference.rows("validation")
    stages = {stage: summarize_stage(stage, heldout, validation) for stage in G.STAGES}
    (G.ROOT / "RESULTS.md").write_text(markdown(stages))
    section = ["## DST depth-recurrence ablation (Llama→Qwen, seed 1234)", "",
               f"[Full results and intervals](experiments/{G.ROOT.name}/RESULTS.md)", "",
               "| Stage | Carry macro | No-Carry macro | Carry − No-Carry (95% CI, pp) |",
               "|---|---:|---:|---:|"]
    for stage, result in stages.items():
        a, b, d = result["eight_task_macro"].values()
        lo, hi = d["ci95"]
        section.append(f"| {stage} | {100*a['mean']:.2f}% | {100*b['mean']:.2f}% | {100*d['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    append_index("\n".join(section) + "\n")
    C.save(G.ROOT / "summary.json", dict(status="completed", manifest_sha256=G.identity(), stages=stages,
           validation_n=len(validation), heldout_n=len(heldout), student_native_body_tokens=0))
    G.status(status="completed")


if __name__ == "__main__":
    {"diagnostics": diagnostics, "test": test, "smoke": lambda: test(smoke=True), "report": report}[sys.argv[1]]()
