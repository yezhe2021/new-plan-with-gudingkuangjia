"""Frozen-cache counterfactual: correct translated body attention mass only."""

import random
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "dst_kvridge_fullbody_v1_seed1234"))
import evaluate_fixed as fixed
from experiment_common import C, K, torch
from transformers.models.qwen3 import modeling_qwen3 as qwen

ARMS = ("native", "translated", "mass_oracle")
CHOICE = {"arc", "sciq", "mmlu_pro", "hellaswag"}
COHORT_SHA = "1f5d84fad280e7bdb94141243b9a3156d1f6e5e67918844aee14160a83412393"


class BodyAttention:
    def __init__(self, start, end, reference=None):
        self.start, self.end, self.reference = start, end, reference
        self.stats = {}
        self.collect = True

    def __call__(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        assert dropout == 0.0 and kwargs.get("sliding_window") is None
        batch, heads, queries, width = query.shape
        kv_heads, length = key.shape[1:3]
        group = heads // kv_heads
        q = query.float().reshape(batch, kv_heads, group, queries, width)
        scores = ((q @ key.float().unsqueeze(2).transpose(-1, -2)) * scaling)
        scores = scores.reshape(batch, heads, queries, length)
        if attention_mask is None and queries > 1:
            positions = torch.arange(length, device=query.device)
            attention_mask = positions <= positions[-queries:, None]
        if attention_mask is not None:
            mask = attention_mask[..., :length]
            if mask.dtype == torch.bool:
                scores.masked_fill_(~mask, -torch.inf)
            else:
                scores.add_(mask)

        layer = module.layer_idx
        probabilities = self.distribution(q, scores, layer, scaling)
        output = probabilities.reshape(batch, kv_heads, group, queries, length) @ value.float().unsqueeze(2)
        output = output.reshape(batch, heads, queries, width).transpose(1, 2).contiguous()
        return output.to(query.dtype), None

    def distribution(self, q, scores, layer, scaling):
        if self.reference is None:
            return scores.softmax(-1)
        start, end = self.start, self.end
        body = scores[..., start:end]
        native_key = self.reference[layer][..., start:end, :]
        reference = ((q @ native_key.float().unsqueeze(2).transpose(-1, -2)) * scaling).reshape_as(body)
        shift = reference.logsumexp(-1) - body.logsumexp(-1)
        if self.collect:
            complement = torch.logaddexp(scores[..., :start].logsumexp(-1),
                                         scores[..., end:].logsumexp(-1))
            old = (body.logsumexp(-1) - complement).sigmoid()
            target = (reference.logsumexp(-1) - complement).sigmoid()
        body.add_(shift.unsqueeze(-1))
        probabilities = scores.softmax(-1)
        if self.collect:
            realized = probabilities[..., start:end].sum(-1)
            assert layer not in self.stats
            self.stats[layer] = torch.stack((old.mean(), target.mean(), (target - old).abs().mean(),
                                             shift.mean(), shift.abs().mean(),
                                             (realized - target).abs().max()))
        return probabilities


@contextmanager
def attention(model, replacement):
    registry = qwen.ALL_ATTENTION_FUNCTIONS
    implementation = model.config._attn_implementation
    missing = object()
    previous = registry._local_mapping.get(implementation, missing)
    registry[implementation] = replacement
    try:
        yield
    finally:
        if previous is missing:
            del registry[implementation]
        else:
            registry[implementation] = previous


def target_tokens(engine, row):
    gold = row["gold"]
    if row["task"] in CHOICE:
        gold = row["candidates"][gold]
    elif row["task"] == "herb":
        gold = ", ".join(gold)
    tokens = engine.tok.encode(gold, add_special_tokens=False)
    assert tokens
    return tokens


def nll(reader, row, cache, tokens):
    logits = reader.token_logits(row, cache, tokens, rn=False)
    labels = torch.tensor(tokens, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")


def evaluate(engine, reader, adapter, row, parity=False):
    native = engine.native_cache(row, baseline=True)
    translated = engine.translated(row, adapter, checkpointed=False)
    start = len(engine.prefix_ids)
    end = start + len(row["content_ids"])
    assert native.get_seq_length() == translated.get_seq_length() == end and 0 < start < end
    tokens = target_tokens(engine, row)
    reference = [key for key, _ in K.pairs(native)]
    arms = {}
    diagnostics = {}
    for name in ARMS:
        cache = native if name == "native" else translated
        controller = BodyAttention(start, end, reference if name == "mass_oracle" else None)
        with attention(engine.model, controller):
            losses = nll(reader, row, cache, tokens)
            controller.collect = False
            values = {"nll": float(losses.mean()), "first_token_nll": float(losses[0])}
            if row["task"] in CHOICE:
                choice = fixed.score_locked(reader, row, cache)
                values.update(correct=choice["correct"], prediction=choice["prediction"],
                              scores=choice["scores"])
        if parity and name != "mass_oracle":
            stock = nll(reader, row, cache, tokens)
            diagnostics[name + "_max_abs_token_nll_delta_vs_stock"] = float((stock - losses).abs().max())
        if name == "mass_oracle":
            assert set(controller.stats) == set(range(engine.model.config.num_hidden_layers))
            stats = torch.stack([controller.stats[layer] for layer in range(len(reference))]).cpu()
            fields = ("translated_mass", "reference_mass", "abs_mass_gap", "logit_shift",
                      "abs_logit_shift", "max_mass_verification_error")
            diagnostics.update(zip(fields, stats.mean(0).tolist()))
            diagnostics["max_mass_verification_error"] = float(stats[:, -1].max())
            assert diagnostics["max_mass_verification_error"] < 1e-5, diagnostics
            diagnostics["per_layer"] = [dict(zip(fields, values)) for values in stats.tolist()]
        arms[name] = values
    assert engine.student_test_native_prefills == 0
    return dict(id=row["id"], task=row["task"], question=row["question"], gold=row["gold"],
                body_tokens=end - start, target_tokens=tokens, arms=arms, diagnostics=diagnostics)


def summarize(records):
    fields = [key for key in ("nll", "first_token_nll", "correct")
              if all(key in r["arms"]["native"] for r in records)]
    summary = {name: {field: mean(r["arms"][name][field] for r in records) for field in fields}
               for name in ARMS}
    improvements = {}
    for field in fields:
        sign = 1 if field == "correct" else -1
        improvements[field] = C.paired([sign * (r["arms"]["mass_oracle"][field] -
                                                r["arms"]["translated"][field]) for r in records])
    gap = summary["translated"]["nll"] - summary["native"]["nll"]
    return dict(n=len(records), arms=summary, oracle_improvement=improvements,
                fraction_of_native_nll_gap_closed=improvements["nll"]["mean"] / gap if gap > 0 else None,
                mean_abs_mass_gap=mean(r["diagnostics"]["abs_mass_gap"] for r in records),
                max_mass_verification_error=max(r["diagnostics"]["max_mass_verification_error"] for r in records))


@torch.no_grad()
def run():
    fixed.verify()
    assert C.sha(fixed.HERB / "cohort.json") == COHORT_SHA
    engine = fixed.Engine()
    reader = fixed.BodyReader(engine)
    adapter = fixed.load_ridge_b(engine)
    engine.mode = "test"
    rng = random.Random(1234)
    seen = {}
    all_rows = fixed.rows("test")
    herb = fixed.load_module("attention_mass_locked_herb", fixed.HERB / "evaluate.py")
    herb_rows = list(herb.render_rows(C.read(fixed.HERB / "cohort.json"), engine.tok))
    assert len(herb_rows) == 20
    selected = {}
    for task, expected in fixed.CFG["test_counts"].items():
        options = [row for row in all_rows if row["task"] == task]
        assert len(options) == expected
        selected[task] = rng.sample(options, 16)
    selected["herb"] = herb_rows
    C.save(ROOT / "manifest.json", dict(
        code_sha256=C.sha(__file__), ridge_experiment_manifest_sha256=fixed.identity(),
        ridge_sha256=engine.ridge_sha256,
        rank64_sha256=C.read(fixed.ROOT / "runs/stage2/completed.json")["checkpoint_sha256"],
        herb_cohort_sha256=COHORT_SHA, oracle="per actual query and head; native K logsumexp only",
        gold_targets="official correct candidate, canonical open answer, or HERB IDs in stored order",
        tasks={task: [row["id"] for row in rows] for task, rows in selected.items()}))

    for task, samples in selected.items():
        seen[task] = []
        for i, row in enumerate(samples):
            if task == "herb":
                segment = row["segment"]
                aligned = fixed.alignment.fullsync(segment["text"], engine.sender_tok, engine.tok, segment["ids"])
                engine.prefix_ids = row["header_ids"]
                engine.header = herb.model_forward(engine.model, engine.prefix_ids).past_key_values
                row = dict(id=row["id"], task=task, prompt=segment["text"], question=row["question"],
                           gold=row["ground_truth"], sender_ids=list(aligned.source_ids),
                           content_ids=list(aligned.receiver_ids), sync_source_indices=list(aligned.source_indices),
                           suffix_ids=row["suffix_ids"])
            record = evaluate(engine, reader, adapter, row, parity=i == 0)
            C.save(ROOT / "per_sample" / task / (row["id"] + ".json"), record)
            seen[task].append(record)
            if i == 0 or (i + 1) % 4 == 0:
                C.log(task, i + 1, len(samples), row["id"],
                      {name: round(v["nll"], 4) for name, v in record["arms"].items()},
                      {key: value for key, value in record["diagnostics"].items() if "vs_stock" in key})
        engine.unchanged()
        C.save(ROOT / "tasks" / (task + ".json"), summarize(seen[task]))
    dst_records = [record for task, records in seen.items() if task != "herb" for record in records]
    choice_records = [record for task, records in seen.items() if task in CHOICE for record in records]
    C.save(ROOT / "summary.json", dict(status="completed", dst=summarize(dst_records),
                                      dst_choice=summarize(choice_records),
                                      tasks={task: summarize(records) for task, records in seen.items()}))
    C.log("Attention mass oracle completed", len(dst_records), len(seen["herb"]))


if __name__ == "__main__":
    run()
