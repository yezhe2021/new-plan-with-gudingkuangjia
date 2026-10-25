"""Frozen-model attention alignment and direct KV-access intervention pilot."""
import os
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
from metrics import text_units, projection, normalize, to_token_scores, ranked_positions, answer_normalize

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def identity():
    return sha(ROOT / "manifest.json")


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def status(**fields):
    save(ROOT / "status.json", dict(time=time.strftime("%F %T"), **fields))


def rows():
    return [json.loads(line) for line in (ROOT / "prepared/questions.jsonl").read_text().splitlines()]


def tokenizer(family):
    return AutoTokenizer.from_pretrained(CFG["models"][family]["path"], local_files_only=True, use_fast=True)


def layout(tok, evidence, question, answer):
    content = ("Use only the supplied evidence to answer the question. Give only the short answer, "
               "without explanation.\n\nEvidence:\n" + evidence + "\n\nQuestion: " + question)
    rendered = tok.apply_chat_template([dict(role="user", content=content)], tokenize=False,
        add_generation_prompt=True, enable_thinking=False, date_string=CFG["chat_date"])
    # Locate by the entire content, not by a potentially repeated question/evidence fragment.
    start = rendered.index(content) + content.index("Evidence:\n") + len("Evidence:\n")
    full = rendered + CFG["answer_prefix"]
    encoded = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    prefix_start = len(rendered)
    prefix_indices = [i for i, (a, b) in enumerate(offsets) if b > prefix_start and a < len(full)]
    assert prefix_indices
    cut = min(prefix_indices)
    assert cut > 0 and len(ids) <= CFG["max_prompt_tokens"]
    combined = tok(full + " " + answer, add_special_tokens=False)["input_ids"]
    assert combined[:len(ids)] == ids, "Answer boundary changes prompt tokenization"
    gold_ids = combined[len(ids):]
    assert 0 < len(gold_ids) <= CFG["max_gold_tokens"]
    units = text_units(evidence)
    matrix = projection(offsets[:cut], start, units)
    assert np.all(matrix.sum(0) > 0), "Uncovered common text unit"
    assert projection(offsets[cut:], start, units).sum() == 0, "Evidence overlaps query segment"
    query_weights = [max(0, min(b, len(full)) - max(a, prefix_start)) for a, b in offsets[cut:]]
    assert sum(query_weights) == len(CFG["answer_prefix"])
    return dict(body_ids=ids[:cut], suffix_ids=ids[cut:], gold_ids=gold_ids,
        offsets=offsets[:cut], evidence_start=start, query_weights=query_weights,
        prefix_last_index=max(i for i, w in enumerate(query_weights) if w),
        rendered_prompt=full, rendered_prompt_sha256=hashlib.sha256(full.encode()).hexdigest(),
        total_prompt_tokens=len(ids))


def prepare():
    assert not list((ROOT / "probes").glob("**/*.npz")), "Do not redefine an experiment after capture"
    tokenizers = {family: tokenizer(family) for family in CFG["models"]}
    assert all(t.is_fast for t in tokenizers.values())
    docs = read(CFG["dataset"])
    random.Random(CFG["seed"]).shuffle(docs)
    accepted, rejected, seen = [], Counter(), set()
    needed = CFG["calibration_questions"] + CFG["evaluation_questions"]
    for doc in docs:
        if len(accepted) == needed:
            break
        answer, question = str(doc["answer"]).strip(), doc["question"].strip()
        key = " ".join(question.casefold().split())
        if key in seen or not 0 < len(answer) <= CFG["max_answer_characters"] or answer.lower() in ["yes", "no"]:
            rejected["duplicate_or_answer_filter"] += 1
            continue
        supports = {(title, int(i)) for title, i in doc["supporting_facts"]}
        facts, distractions = [], []
        for title, sentences in doc["context"]:
            for i, sentence in enumerate(sentences):
                line = f"{title}: {sentence.strip()}"
                (facts if (title, i) in supports else distractions).append((line, (title, i) in supports))
        if len(facts) != len(supports) or not facts:
            rejected["missing_support"] += 1
            continue
        rng = random.Random(int(digest([CFG["seed"], doc["_id"]])[:12], 16))
        rng.shuffle(distractions)
        lines = facts + distractions[:CFG["distractor_sentences"]]
        rng.shuffle(lines)
        evidence = "\n".join(line for line, _ in lines)
        if answer.casefold() not in evidence.casefold():
            rejected["answer_not_in_evidence"] += 1
            continue
        try:
            layouts = {family: layout(tok, evidence, question, answer) for family, tok in tokenizers.items()}
        except (AssertionError, ValueError) as exc:
            rejected["length_or_boundary_audit"] += 1
            continue
        spans, cursor = [], 0
        for line, support in lines:
            if support:
                spans.append([cursor, cursor + len(line)])
            cursor += len(line) + 1
        accepted.append(dict(id=doc["_id"], question=question, answer=answer, evidence=evidence,
            units=text_units(evidence), support_spans=spans,
            split="calibration" if len(accepted) < CFG["calibration_questions"] else "evaluation", layouts=layouts))
        seen.add(key)
    assert len(accepted) == needed, (len(accepted), rejected)
    assert len({r["id"] for r in accepted}) == needed
    output = ROOT / "prepared/questions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
    models = {}
    for family, entry in CFG["models"].items():
        path = Path(entry["path"])
        small = {str(f.relative_to(path)): sha(f) for f in path.iterdir() if f.is_file() and
                 f.name in ["config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json", "model.safetensors.index.json"]}
        models[family] = dict(path=str(path), dtype=entry["dtype"], files_sha256=small,
            weights=[dict(name=f.name, bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
                     for f in sorted(path.glob("*.safetensors"))])
    manifest = dict(config=CFG, config_sha256=sha(ROOT / "config.json"), dataset_sha256=sha(CFG["dataset"]),
        prepared_sha256=sha(output), code={p.name: sha(p) for p in ROOT.glob("*.py")}, models=models,
        versions=dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__),
        selection_audit=dict(accepted=needed, rejected=dict(rejected), official_benchmark=False),
        model_weight_identity_policy="Index/config/tokenizer hashes plus weight file size/mtime; full multi-GB weight hashes not computed",
        references=["https://aclanthology.org/N19-1357/", "https://aclanthology.org/D19-1002/"])
    save(ROOT / "manifest.json", manifest)
    save(ROOT / "prepared/completed.json", dict(manifest_sha256=identity(), n=needed))
    status(status="prepared", calibration=CFG["calibration_questions"], evaluation=CFG["evaluation_questions"])
    log(f"PREPARED {needed} shared questions; {dict(rejected)}")


def verify():
    m = read(ROOT / "manifest.json")
    assert sha(ROOT / "config.json") == m["config_sha256"]
    assert sha(ROOT / "prepared/questions.jsonl") == m["prepared_sha256"]
    for name, expected in m["code"].items():
        assert sha(ROOT / name) == expected, ("Code changed", name)
    for entry in m["models"].values():
        path = Path(entry["path"])
        for name, expected in entry["files_sha256"].items():
            assert sha(path / name) == expected, (path, name)
        for weight in entry["weights"]:
            st = (path / weight["name"]).stat()
            assert st.st_size == weight["bytes"] and st.st_mtime_ns == weight["mtime_ns"]


def load_model(family):
    assert torch.cuda.is_available()
    free = torch.cuda.mem_get_info()[0] / 2**30
    assert free >= CFG["free_gpu_gib"], ("GPU busy", free)
    assert shutil.disk_usage(ROOT).free / 2**30 >= CFG["disk_reserve_gib"]
    torch.manual_seed(CFG["seed"])
    entry = CFG["models"][family]
    cls = Gemma3ForConditionalGeneration if family == "gemma" else AutoModelForCausalLM
    log(f"LOAD {family} {entry['dtype']}, {free:.1f} GiB free")
    model = cls.from_pretrained(entry["path"], dtype=getattr(torch, entry["dtype"]), device_map="cuda",
        local_files_only=True, attn_implementation="eager").eval().requires_grad_(False)
    modules = [(name, module) for name, module in model.named_modules() if name.endswith(".self_attn") and
               (("language_model.layers." in name) if family == "gemma" else bool(re.match(r"model\.layers\.\d+\.self_attn$", name)))]
    text_config = model.config.get_text_config()
    assert len(modules) == text_config.num_hidden_layers, [name for name, _ in modules]
    assert all(module.config._attn_implementation == "eager" for _, module in modules)
    assert not any(p.requires_grad for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    return tokenizer(family), model, modules


def tensors(cache):
    return [tensor for layer in cache.layers for tensor in [layer.keys, layer.values] if tensor is not None]


@torch.inference_mode()
def prefill(model, item):
    ids = torch.tensor([item["body_ids"]], device="cuda")
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(ids.shape[1], device="cuda")[None],
                use_cache=True, logits_to_keep=1)
    assert torch.isfinite(out.logits).all(), "Nonfinite prefill logits"
    assert all(t.shape[-2] == len(item["body_ids"]) for t in tensors(out.past_key_values))
    return out.past_key_values


def visibility(body_length, continuation_length, blocked):
    assert len(set(blocked)) == len(blocked)
    assert all(0 <= p < body_length for p in blocked)
    mask = torch.ones((1, body_length + continuation_length), device="cuda", dtype=torch.long)
    if blocked:
        mask[0, blocked] = 0
    assert int((mask == 0).sum()) == len(blocked)
    return mask


@torch.inference_mode()
def continuation(model, tok, item, body, blocked=(), generate=False, capture_modules=None):
    cache = copy.deepcopy(body)
    b, suffix = len(item["body_ids"]), item["suffix_ids"]
    collected, handles, blocked_mass = [], [], []
    if capture_modules is not None:
        def hook(module, args, output):
            weights = output[1]
            assert weights is not None and weights.ndim == 4, "Missing eager attention probabilities"
            assert weights.shape[-2] == len(suffix)
            collected.append(weights[0].detach().float().cpu().numpy())
            if blocked:
                blocked_mass.append(float(weights[..., list(blocked)].abs().max()))
        handles = [module.register_forward_hook(hook) for _, module in capture_modules]
    try:
        out = model(input_ids=torch.tensor([suffix], device="cuda"), past_key_values=cache,
            attention_mask=visibility(b, len(suffix), blocked), position_ids=torch.arange(b, b + len(suffix), device="cuda")[None],
            use_cache=True, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    first_logits = out.logits[0, -1].float()
    assert torch.isfinite(first_logits).all(), "Nonfinite continuation logits"
    first = first_logits.detach().cpu().numpy().copy()
    gold = item["gold_ids"]
    first_logp = F.log_softmax(first_logits, dim=-1)[gold[0]]
    total_logp = float(first_logp)
    if len(gold) > 1:
        score_cache = copy.deepcopy(out.past_key_values)
        teacher = model(input_ids=torch.tensor([gold[:-1]], device="cuda"), past_key_values=score_cache,
            attention_mask=visibility(b, len(suffix) + len(gold) - 1, blocked),
            position_ids=torch.arange(b + len(suffix), b + len(suffix) + len(gold) - 1, device="cuda")[None],
            use_cache=True, logits_to_keep=0)
        logits = teacher.logits[0].float()
        assert torch.isfinite(logits).all(), "Nonfinite gold-answer logits"
        total_logp += float(F.log_softmax(logits, -1).gather(1, torch.tensor(gold[1:], device="cuda")[:, None]).sum())
        del teacher, score_cache, logits
    result = dict(gold_tokens=len(gold), gold_nll_sum=-total_logp, gold_nll_per_token=-total_logp / len(gold))
    if generate:
        config_eos = model.generation_config.eos_token_id
        eos = set(config_eos if isinstance(config_eos, list) else [config_eos]) | {tok.eos_token_id}
        decoded_ids, current_logits = [], first_logits
        for step in range(CFG["max_new_tokens"]):
            token = int(current_logits.argmax())
            if token in eos:
                break
            decoded_ids.append(token)
            decoded = tok.decode(decoded_ids, skip_special_tokens=True)
            if "\n" in decoded.strip():
                break
            if step + 1 == CFG["max_new_tokens"]:
                break
            pos = b + len(suffix) + len(decoded_ids) - 1
            next_output = model(input_ids=torch.tensor([[token]], device="cuda"), past_key_values=out.past_key_values,
                attention_mask=visibility(b, len(suffix) + len(decoded_ids), blocked),
                position_ids=torch.tensor([[pos]], device="cuda"), use_cache=True, logits_to_keep=1)
            out = next_output
            current_logits = out.logits[0, -1].float()
            assert torch.isfinite(current_logits).all()
        result.update(generation=tok.decode(decoded_ids, skip_special_tokens=True), generated_tokens=len(decoded_ids),
                      reached_cap=int(len(decoded_ids) >= CFG["max_new_tokens"]))
    if capture_modules is not None:
        assert len(collected) == len(capture_modules)
        result["maximum_blocked_attention"] = max(blocked_mass, default=0.)
    return result, first, collected


def complete_record(result, row):
    result["gold_nll_per_character"] = result["gold_nll_sum"] / len(" " + row["answer"])
    if "generation" in result:
        result["exact_match"] = int(answer_normalize(result["generation"]) == answer_normalize(row["answer"]))
    return result


@torch.inference_mode()
def smoke(family):
    tok, model, modules = load_model(family)
    row = rows()[0]
    item = row["layouts"][family]
    body = prefill(model, item)
    originals = [t.clone() for t in tensors(body)]
    baseline, logits, _ = continuation(model, tok, item, body, generate=True)
    noop, again, raw = continuation(model, tok, item, body, capture_modules=modules)
    noop_error = float(np.max(np.abs(logits - again)))
    assert noop_error == 0, ("Null-mask replay differs", noop_error)
    full_ids = item["body_ids"] + item["suffix_ids"]
    full = model(input_ids=torch.tensor([full_ids], device="cuda"), attention_mask=torch.ones((1, len(full_ids)), device="cuda", dtype=torch.long),
                 use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().numpy()
    parity = float(np.max(np.abs(full - logits)))
    assert full.argmax() == logits.argmax(), "Segmented-cache vs full prompt top1 differs"
    assert parity < (.15 if CFG["models"][family]["dtype"] == "float16" else .002), parity
    matrix = projection(item["offsets"], item["evidence_start"], row["units"])
    eligible = np.flatnonzero(matrix.sum(1) > 0)
    blocked = eligible[:max(1, len(eligible) // 5)].tolist()
    altered, _, _ = continuation(model, tok, item, body, blocked=blocked, capture_modules=modules)
    assert altered["maximum_blocked_attention"] == 0
    assert all(torch.equal(old, now) for old, now in zip(originals, tensors(body))), "Source cache mutated"
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    weights = np.asarray(item["query_weights"], float)
    weights /= weights.sum()
    projected = np.stack([np.einsum("hqk,q,ku->hu", layer, weights, np.pad(matrix, ((0, len(item["suffix_ids"])), (0, 0)))) for layer in raw])
    assert np.isfinite(projected).all() and np.all(projected.sum(-1) > 0)
    record = dict(status="passed", family=family, manifest_sha256=identity(), source_cache_bitwise_unchanged=True,
        no_op_max_logit_error=noop_error, full_prompt_max_logit_error=parity,
        maximum_masked_probability=altered["maximum_blocked_attention"], layers=len(modules),
        query_heads=raw[0].shape[0], common_units=len(row["units"]), all_probabilities_finite=True,
        baseline=complete_record(baseline, row), dtype=CFG["models"][family]["dtype"],
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    save(ROOT / f"smoke/{family}.json", record)
    log(f"SMOKE PASSED {family}: {record}")


@torch.inference_mode()
def capture(family):
    done = ROOT / f"probes/{family}/completed.json"
    if done.exists():
        assert read(done)["manifest_sha256"] == identity()
        return
    assert read(ROOT / f"smoke/{family}.json")["manifest_sha256"] == identity()
    tok, model, modules = load_model(family)
    for index, row in enumerate(rows(), 1):
        folder = ROOT / f"probes/{family}"
        path = folder / (row["id"] + ".npz")
        if path.exists():
            assert read(path.with_suffix(".json"))["manifest_sha256"] == identity()
            continue
        status(status="running", phase="attention_capture", model=family, sample=index, total=len(rows()))
        item = row["layouts"][family]
        body = prefill(model, item)
        result, _, raw = continuation(model, tok, item, body, capture_modules=modules, generate=True)
        matrix = projection(item["offsets"], item["evidence_start"], row["units"])
        padded = np.pad(matrix, ((0, len(item["suffix_ids"])), (0, 0)))
        weights = np.array(item["query_weights"], dtype=float)
        weights /= weights.sum()
        last = np.stack([layer[:, item["prefix_last_index"]] @ padded for layer in raw])
        mean = np.stack([np.einsum("hqk,q,ku->hu", layer, weights, padded) for layer in raw])
        folder.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, last_prefix=last.astype(np.float32), prefix_character_mean=mean.astype(np.float32))
        save(path.with_suffix(".json"), dict(id=row["id"], family=family, split=row["split"], manifest_sha256=identity(),
            baseline=complete_record(result, row), layers=last.shape[0], heads=last.shape[1], evidence_units=len(row["units"]),
            mean_evidence_attention_mass=float(last.sum(-1).mean()), min_evidence_attention_mass=float(last.sum(-1).min())))
        temp.replace(path)
        log(f"CAPTURE {family} {index}/{len(rows())} EM={complete_record(result,row)['exact_match']}")
        del body, raw
        torch.cuda.empty_cache()
    save(done, dict(status="completed", manifest_sha256=identity(), n=len(rows())))


def load_probe(family, row, view="last_prefix"):
    path = ROOT / f"probes/{family}/{row['id']}.npz"
    with np.load(path) as values:
        return values[view].astype(np.float64)


def selectors(family, row):
    own = normalize(load_probe(family, row))
    choices = {"own_global": (own.mean((0, 1)), True)}
    for band, indices in enumerate(np.array_split(np.arange(own.shape[0]), CFG["layer_bands"])):
        choices[f"own_layer_band_{band}"] = (own[indices].mean((0, 1)), False)
    matching = read(ROOT / "matching.json")
    for other in CFG["models"]:
        if other == family:
            continue
        source = normalize(load_probe(other, row))
        choices[f"cross_global_{other}"] = (source.mean((0, 1)), True)
        pair = next(p for p in matching["pairs"] if set(p["models"]) == {family, other})
        representative = pair["functional_head_pair"]
        source_side, own_side = pair["models"].index(other), pair["models"].index(family)
        sl, sh = representative["indices"][source_side]
        tl, th = representative["indices"][own_side]
        choices[f"cross_matched_head_{other}"] = (source[sl, sh], False)
        choices[f"own_matched_head_for_{other}"] = (own[tl, th], False)
    return choices


@torch.inference_mode()
def ablate(family):
    done = ROOT / f"ablations/{family}/completed.json"
    if done.exists():
        assert read(done)["manifest_sha256"] == identity()
        return
    tok, model, _ = load_model(family)
    selected_rows = [r for r in rows() if r["split"] == "evaluation"]
    for index, row in enumerate(selected_rows, 1):
        path = ROOT / f"ablations/{family}/{row['id']}.json"
        if path.exists():
            assert read(path)["manifest_sha256"] == identity()
            continue
        item = row["layouts"][family]
        body = prefill(model, item)
        baseline, _, _ = continuation(model, tok, item, body, generate=True)
        baseline = complete_record(baseline, row)
        previous = read(ROOT / f"probes/{family}/{row['id']}.json")["baseline"]
        assert abs(previous["gold_nll_sum"] - baseline["gold_nll_sum"]) < 1e-5
        assert previous["generation"] == baseline["generation"]
        matrix = projection(item["offsets"], item["evidence_start"], row["units"])
        eligible = np.flatnonzero(matrix.sum(1) > 0)
        choices = selectors(family, row)
        interventions = []
        for fraction in CFG["ablation_fractions"]:
            budget = max(1, math.ceil(len(eligible) * fraction))
            for name, (unit_scores, decode) in choices.items():
                scores, valid = to_token_scores(unit_scores, matrix)
                positions = ranked_positions(scores, valid, budget, seed=CFG["seed"])
                interventions.append((name, fraction, positions, decode))
            scores, _ = to_token_scores(choices["own_global"][0], matrix)
            interventions.append(("own_bottom", fraction, ranked_positions(scores, eligible, budget, reverse=True), True))
            for replicate in range(CFG["random_replicates"]):
                rng = np.random.default_rng(int(digest([CFG["seed"], row["id"], family, fraction, replicate])[:12], 16))
                interventions.append((f"random_{replicate}", fraction, sorted(rng.choice(eligible, budget, replace=False).tolist()), True))
        interventions.append(("all_evidence_access", 1., eligible.tolist(), True))
        records = []
        for ci, (name, fraction, positions, decode) in enumerate(interventions, 1):
            assert set(positions) <= set(eligible.tolist())
            if fraction < 1:
                assert len(positions) == max(1, math.ceil(len(eligible) * fraction))
            status(status="running", phase="kv_ablation", model=family, sample=index, total=len(selected_rows),
                   condition=name, condition_index=ci, conditions=len(interventions))
            value, _, _ = continuation(model, tok, item, body, blocked=positions, generate=decode)
            complete_record(value, row)
            records.append(dict(selector=name, fraction=fraction, blocked_positions=positions,
                blocked_count=len(positions), evidence_tokens=len(eligible), **value,
                delta_nll_per_token=value["gold_nll_per_token"] - baseline["gold_nll_per_token"],
                delta_nll_per_character=value["gold_nll_per_character"] - baseline["gold_nll_per_character"]))
        save(path, dict(id=row["id"], family=family, manifest_sha256=identity(), baseline=baseline, interventions=records))
        log(f"ABLATION {family} {index}/{len(selected_rows)} {len(records)} conditions baseline_EM={baseline['exact_match']}")
        del body
        torch.cuda.empty_cache()
    save(done, dict(status="completed", manifest_sha256=identity(), n=len(selected_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify", "smoke", "capture", "ablate"])
    parser.add_argument("family", nargs="?", choices=list(CFG["models"]))
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare()
        else:
            verify()
            if args.action != "verify":
                assert args.family
                globals()[args.action](args.family)
    except Exception as exc:
        import traceback
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        save(ROOT / "failure.json", error)
        status(status="failed", phase=args.action, **error)
        raise
