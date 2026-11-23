"""Calibrate frozen Full-Sync Stage-A Value outputs and test receiver readout."""

import gzip
import json
import random
import sys
from pathlib import Path
from statistics import mean

from correction import ErrorMoments, TargetMoments, apply_map, bins, gate_score, prepare_maps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "dst_kvridge_fullbody_v1_seed1234"))
import evaluate_fixed as fixed
from experiment_common import C, CALIBRATION, K, Z, torch
from transformers.models.qwen3 import modeling_qwen3 as qwen

REFERENCE = ROOT.parent / "ridge_rank64_attention_mass_oracle_seed1234"
PROBE = fixed.load_module("frozen_receiver_probe", REFERENCE / "run.py")
METHODS = ("fixed_lowrank", "adaptive_diagonal", "adaptive_lowrank")
BASE = "mlp"
RANK = 16
GATES = 4
STRENGTHS = (0.01, 0.03, 0.1, 0.3, 1.0)
SPLIT = {"native_covariance": 64, "bucket_edges": 12, "residuals": 64, "strength_validation": 20}
CONTEXT, SUFFIX = 384, 24


def calibration(engine):
    with gzip.open(CALIBRATION, "rt", encoding="utf-8") as stream:
        all_rows = [json.loads(line) for line in stream]
    random.Random(1234).shuffle(all_rows)
    selected = []
    for document in all_rows:
        text = document["text"].lstrip()
        target = engine.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens, offsets = target["input_ids"], target["offset_mapping"]
        if len(tokens) < CONTEXT + SUFFIX or offsets[CONTEXT - 1][1] > offsets[CONTEXT][0]:
            continue
        body = text[:offsets[CONTEXT - 1][1]]
        if engine.tok.encode(body, add_special_tokens=False) != tokens[:CONTEXT]:
            continue
        selected.append(dict(id=document["id"], prompt=body, content_ids=tokens[:CONTEXT],
                             suffix_ids=tokens[CONTEXT:CONTEXT + SUFFIX]))
        if len(selected) == sum(SPLIT.values()):
            break
    assert len(selected) == sum(SPLIT.values()) and len({r["id"] for r in selected}) == len(selected)
    partition = {}
    cursor = 0
    for name, count in SPLIT.items():
        partition[name] = selected[cursor:cursor + count]
        cursor += count
    return partition


def body_values(cache, start, end):
    values = torch.cat([v[..., start:end, :] for _, v in K.pairs(cache)], dim=0)
    return values.flatten(0, 1).float()


def training_values(engine, writer, row, paired=True):
    start = len(engine.prefix_ids)
    end = start + len(row["content_ids"])
    target = body_values(engine.native_cache(row, baseline=True), start, end)
    if paired:
        prediction = body_values(writer.cache(engine, row), start, end)
        return prediction, target
    return target


def fit(engine, writer, data):
    target = TargetMoments()
    for i, row in enumerate(data["native_covariance"], 1):
        target.add(training_values(engine, writer, row, paired=False))
        if i % 16 == 0:
            C.log("native covariance", i, SPLIT["native_covariance"])
    center, variance, cov = target.finish(RANK)
    scores = []
    for row in data["bucket_edges"]:
        translated = writer.cache(engine, row)
        start = len(engine.prefix_ids)
        values = body_values(translated, start, translated.get_seq_length())
        scores.append(gate_score(values, center, variance))
    quantiles = torch.arange(1, GATES, device=center.device) / GATES
    edges = torch.quantile(torch.cat(scores, 1), quantiles, dim=1).T.contiguous()

    residuals = ErrorMoments(center, variance, edges)
    for i, row in enumerate(data["residuals"], 1):
        residuals.add(*training_values(engine, writer, row))
        if i % 16 == 0:
            C.log("MLP residual calibration", i, SPLIT["residuals"])
    fixed_error, adaptive = residuals.finish()
    C.save(ROOT / "residual_calibration.json", dict(
        tokens_per_head=int(residuals.counts.sum(1)[0]),
        average_bucket_fraction=(residuals.counts / residuals.counts.sum(1, keepdim=True)).mean(0).tolist(),
        mean_relative_mse_by_bucket=(adaptive / fixed_error[:, None]).mean((0, 2)).tolist()))
    return center, variance, cov, edges, fixed_error, adaptive


def caches_from_values(engine, translated, values, start, end):
    layers = K.pairs(translated)
    heads = layers[0][1].shape[1]
    assert values.shape[1] == end - start == translated.get_seq_length() - start
    vals = values.unflatten(0, (len(layers), heads))
    return C.DynamicCache(ddp_cache_data=[
        (key, torch.cat((value[..., :start, :], vals[i:i + 1].to(value.dtype)), dim=2))
        for i, (key, value) in enumerate(layers)
    ], config=engine.model.config)


class LocalReadout:
    """Compare V-only and full-native readout targets at the same translated query."""

    def __init__(self, engine, native, variants, start, end):
        self.stock = qwen.ALL_ATTENTION_FUNCTIONS[engine.model.config._attn_implementation]
        self.native = K.pairs(native)
        self.variants = variants
        self.start, self.end = start, end
        self.stats = []

    def __call__(self, module, query, key, value, attention_mask, scaling, **kwargs):
        i = module.layer_idx
        assert len(self.stats) == i and query.shape[0] == 1 and key.shape[0] == 1
        h, length, d = key.shape[1:]
        g = query.shape[1] // h
        count = min(8, query.shape[-2])
        inputs = query[0, :, -count:].float().reshape(h, g, count, d)
        k = key[0].float()
        v = value[0].float().unsqueeze(1)
        start, end = self.start, self.end
        teacher_k, teacher_v = self.native[i]
        teacher_k = teacher_k[0, :, start:end].float()
        teacher_v = teacher_v[0, :, start:end].float().unsqueeze(1)

        scores = (inputs @ k.unsqueeze(1).transpose(-1, -2)) * scaling
        # All probes are in the unpadded suffix and can see the entire body.
        visible = torch.arange(length, device=key.device) <= torch.arange(length - count, length,
                                                                           device=key.device)[:, None]
        assert length - count >= end
        scores.masked_fill_(~visible, -torch.inf)
        mapped = scores.softmax(-1)
        scores[..., start:end] = (inputs @ teacher_k.unsqueeze(1).transpose(-1, -2)) * scaling
        native = scores.softmax(-1)
        pb = mapped[..., start:end]

        def complement(p):
            return p[..., :start] @ v[..., :start, :] + p[..., end:] @ v[..., end:, :]

        target = pb @ teacher_v
        full_target = complement(native) + native[..., start:end] @ teacher_v
        mapped_rest = complement(mapped)
        metrics = {"energy": target.square().sum(), "full_energy": full_target.square().sum()}
        for name, variant in self.variants.items():
            body = variant[i * h:(i + 1) * h].unsqueeze(1)
            read = pb @ body
            metrics[name] = (read - target).square().sum()
            metrics[name, "full"] = (mapped_rest + read - full_target).square().sum()
        self.stats.append(metrics)
        return self.stock(module, query, key, value, attention_mask, scaling=scaling, **kwargs)

    def finish(self):
        results = {}
        for name in self.variants:
            numerator = torch.stack([s[name] for s in self.stats]).sum()
            denominator = torch.stack([s["energy"] for s in self.stats]).sum().clamp_min(1e-8)
            full_num = torch.stack([s[name, "full"] for s in self.stats]).sum()
            full_den = torch.stack([s["full_energy"] for s in self.stats]).sum().clamp_min(1e-8)
            results[name] = dict(value_readout_nmse=float(numerator / denominator),
                                 full_readout_nmse=float(full_num / full_den))
        return results


def create_variants(prediction, parameters, maps):
    center, variance, _, edges, _, _ = parameters
    groups = bins(gate_score(prediction, center, variance), edges)
    values = {BASE: prediction}
    for key, weights in maps.items():
        # Account for the FP16 cache store before comparing readouts.
        values[key] = apply_map(prediction, center, weights, groups).half().float()
    return values


def readout(engine, native, translated, values, row, start, end):
    probe = LocalReadout(engine, native, values, start, end)
    clone = C.DynamicCache(ddp_cache_data=K.pairs(translated), config=engine.model.config)
    with PROBE.attention(engine.model, probe):
        engine.forward(row["suffix_ids"], clone, end, use_cache=False)
    assert len(probe.stats) == engine.model.config.num_hidden_layers
    return probe.finish()


def select_strengths(engine, writer, parameters, maps, rows):
    data = []
    for i, row in enumerate(rows, 1):
        translated = writer.cache(engine, row)
        native = engine.native_cache(row, baseline=True)
        start, end = len(engine.prefix_ids), native.get_seq_length()
        pred = body_values(translated, start, end)
        values = create_variants(pred, parameters, maps)
        data.append(readout(engine, native, translated, values, row, start, end))
        if i % 5 == 0:
            C.log("strength validation", i, len(rows))
    metrics = {key: {field: mean(record[key][field] for record in data) for field in data[0][key]} for key in data[0]}
    selected = {name: min(STRENGTHS, key=lambda strength: metrics[name, strength]["value_readout_nmse"])
                for name in METHODS}
    C.save(ROOT / "strength_validation.json", dict(
        n=len(data), selected=selected, scores={str(k): v for k, v in metrics.items()}))
    return selected


def dst_score(reader, row, cache):
    tokens = PROBE.target_tokens(reader.engine, row)
    result = dict(nll=float(PROBE.nll(reader, row, cache, tokens).mean()))
    if row["task"] in PROBE.CHOICE:
        score = fixed.score_locked(reader, row, cache)
        result.update(correct=score["correct"], prediction=score["prediction"], scores=score["scores"])
    return result


def trial(engine, writer, row, parameters, maps, scorer):
    translated = writer.cache(engine, row)
    native = engine.native_cache(row, baseline=True)
    start, end = len(engine.prefix_ids), native.get_seq_length()
    assert translated.get_seq_length() == end == start + len(row["content_ids"])
    prediction, target = (body_values(cache, start, end) for cache in (translated, native))
    values = create_variants(prediction, parameters, maps)
    results = readout(engine, native, translated, values, row, start, end)
    target_energy = target.square().mean().clamp_min(1e-8)
    for name, variant in values.items():
        results[name]["value_nmse"] = float((variant - target).square().mean() / target_energy)
        cache = translated if name == BASE else caches_from_values(engine, translated, variant, start, end)
        # The generation scorer extends its input cache; keep the common prefix reusable.
        cache = C.DynamicCache(ddp_cache_data=K.pairs(cache), config=engine.model.config)
        results[name].update(scorer(cache))
    results["native"] = scorer(native)
    return dict(id=row["id"], question=row["question"], task=row["task"], gold=row["gold"],
                body_tokens=end - start, results=results)


def summary(rows):
    names = rows[0]["results"]
    all_metrics = {"nll", "correct", "exact", "set_f1", "recall", "capped", "value_nmse",
                   "value_readout_nmse", "full_readout_nmse"}
    table = {}
    for name in names:
        fields = all_metrics.intersection(*(r["results"][name].keys() for r in rows))
        table[name] = {field: mean(r["results"][name][field] for r in rows) for field in sorted(fields)}
    pairs = {}
    for name, control in (("fixed_lowrank", BASE), ("adaptive_lowrank", BASE),
                          ("adaptive_lowrank", "fixed_lowrank"), ("adaptive_lowrank", "adaptive_diagonal"),
                          ("adaptive_diagonal", BASE)):
        pairs[name + "_minus_" + control] = {field: C.paired(
            [r["results"][name][field] - r["results"][control][field] for r in rows])
            for field in table[name] if field in table[control]}
    return dict(n=len(rows), metrics=table, paired=pairs)


@torch.no_grad()
def run():
    fixed.verify()
    assert C.sha(fixed.HERB / "cohort.json") == PROBE.COHORT_SHA
    engine = Z.DE.Engine()
    engine.mode = "test"
    reader = fixed.BodyReader(engine)
    herb = fixed.load_module("fullsync_covariance_herb", fixed.HERB / "evaluate.py")
    writer = fixed.HistoricalMLP(herb)
    data = calibration(engine)
    old = C.read(REFERENCE / "manifest.json")
    manifest = dict(
        stage_a_sha256=herb.PINS["fullsync_ab"]["stage_a"], source_manifest_sha256=fixed.identity(),
        stage="Stage-A only", calibration_source_sha256=C.sha(CALIBRATION),
        code_sha256={name: C.sha(ROOT / name) for name in ("run.py", "correction.py")},
        splits={split: [r["id"] for r in rows] for split, rows in data.items()},
        context=CONTEXT, suffix=SUFFIX, rank=RANK, gate="four per-head bins of normalized predicted V energy",
        strengths=STRENGTHS, seed=1234, dst_probe_manifest_sha256=C.sha(REFERENCE / "manifest.json"),
        herb_cohort_sha256=PROBE.COHORT_SHA, eval_ids=old["tasks"])
    path = ROOT / "manifest.json"
    if path.exists():
        assert json.loads(json.dumps(manifest)) == C.read(path)
    else:
        C.save(path, manifest)
    state = ROOT / "statistics.pt"
    if state.exists():
        saved = torch.load(state, map_location="cuda", weights_only=True)
        assert saved["manifest_sha256"] == C.sha(path)
        parameters = saved["parameters"]
    else:
        parameters = fit(engine, writer, data)
        tmp = state.with_suffix(".tmp")
        torch.save(dict(manifest_sha256=C.sha(path), parameters=parameters), tmp)
        tmp.replace(state)
    validation = ROOT / "strength_validation.json"
    if validation.exists():
        chosen = C.read(validation)["selected"]
    else:
        chosen = select_strengths(engine, writer, parameters, prepare_maps(parameters, STRENGTHS),
                                  data["strength_validation"])
    indexed = prepare_maps(parameters, sorted(set(chosen.values())))
    maps = {name: indexed[name, strength] for name, strength in chosen.items()}
    del indexed
    C.log("selected strengths", chosen)

    dst = {r["id"]: r for r in fixed.rows("test")}
    herb_rows = {r["id"]: r for r in herb.render_rows(C.read(fixed.HERB / "cohort.json"), engine.tok)}
    results = {}
    for task, ids in manifest["eval_ids"].items():
        results[task] = []
        for i, sample in enumerate(ids, 1):
            output = ROOT / "per_sample" / task / (sample + ".json")
            if output.exists():
                record = C.read(output)
            elif task == "herb":
                item = herb_rows[sample]
                engine.prefix_ids = item["header_ids"]
                engine.header = herb.model_forward(engine.model, engine.prefix_ids).past_key_values
                row = dict(id=sample, task=task, prompt=item["segment"]["text"], question=item["question"],
                           content_ids=item["segment"]["ids"], suffix_ids=item["suffix_ids"],
                           gold=item["ground_truth"])
                record = trial(engine, writer, row, parameters, maps,
                               lambda cache: herb.generate(engine.model, engine.tok, row["suffix_ids"],
                                                           cache, 192, item))
            else:
                row = dst[sample]
                assert row["task"] == task
                record = trial(engine, writer, row, parameters, maps, lambda cache: dst_score(reader, row, cache))
            C.save(output, record)
            results[task].append(record)
            if task == "herb" or i in {1, 8, 16}:
                metric = "set_f1" if task == "herb" else "value_readout_nmse"
                C.log(task, i, len(ids), sample, {key: round(val[metric], 4) for key, val in
                                                record["results"].items() if metric in val})
        assert engine.student_test_native_prefills == 0
        engine.unchanged()
        C.save(ROOT / "tasks" / (task + ".json"), summary(results[task]))
    all_dst = [r for name, rows in results.items() if name != "herb" for r in rows]
    choice = [r for name, rows in results.items() if name in PROBE.CHOICE for r in rows]
    C.save(ROOT / "summary.json", dict(status="completed", manifest_sha256=C.sha(path),
           selected_strengths=chosen, dst=summary(all_dst), dst_choice=summary(choice),
           tasks={name: summary(rows) for name, rows in results.items()}))


if __name__ == "__main__":
    run()
