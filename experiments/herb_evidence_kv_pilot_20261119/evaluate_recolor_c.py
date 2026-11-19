"""Add frozen Recolor+C to the same HERB questions and receiver continuation."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RECOLOR_ROOT = ROOT.parent / "recolor_kl_l3_to_q4_seed1234"
RECOLOR_MANIFEST_SHA = "f6b176c9915d2e968ce560fb2f981f6e2cefdd9405a9e2beeac9afa0a739b4c6"
MAP_SHA = "3009cc3434170e4806d064e0e65f49b04f108fa22c5cb8e2b5deaf58e38c0f32"
CORRECTION_SHA = "14a4c5f712d9c8440710555071a0f5b83aee9301493a6a7abe06222c25757425"


def read(path):
    return json.loads(path.read_text())


def engine_module():
    sys.path.insert(0, str(RECOLOR_ROOT))
    spec = importlib.util.spec_from_file_location("herb_archived_recolor_core", RECOLOR_ROOT / "core.py")
    recolor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recolor)
    return recolor


def freeze_inputs():
    from evaluate import C, prepare

    prior = read(ROOT / "cap192/summary.json")
    C.CFG["models"].update({family: info["path"] for family, info in prior["model_files"].items()})
    samples_path = ROOT / "samples.json"
    assert C.sha(samples_path) == prior["samples_sha256"]
    tok = C.AutoTokenizer.from_pretrained(C.CFG["models"]["qwen"], local_files_only=True)
    rows, _ = prepare(read(samples_path)["samples"], tok)
    C.save(ROOT / "recolor_inputs.json", dict(
        samples_sha256=C.sha(samples_path), baseline_summary_sha256=C.sha(ROOT / "cap192/summary.json"),
        rows=[dict(id=item["id"], prompt=row["prompt"], content_ids=row["frames"]["qwen"]["content_ids"],
                   suffix_ids=row["suffix_ids"], full_ids=full) for item, row, full, _ in rows],
    ))


def load_recolor(recolor, recolor_map, correction, model_paths):
    manifest_path = RECOLOR_ROOT / "manifest.json"
    assert recolor.sha(manifest_path) == RECOLOR_MANIFEST_SHA
    manifest = read(manifest_path)
    assert recolor.sha(RECOLOR_ROOT / "config.json") == manifest["config_sha256"]
    for file in ("core.py", "residuals.py"):
        assert recolor.sha(RECOLOR_ROOT / file) == manifest["code"][file]
    assert recolor.sha(recolor_map) == MAP_SHA
    assert recolor.sha(correction) == CORRECTION_SHA
    source_root = Path(recolor.HC.__file__).resolve().parent
    for file in ("calibration.py", "transfer.py", "modeling.py", "maps.py"):
        assert recolor.sha(source_root / file) == manifest["references"]["reference/heterofold/" + file]
    recolor.CFG["recolor"] = str(recolor_map)
    recolor.CFG["models"] = {"receiver": model_paths["qwen"], "sender": model_paths["llama"]}
    engine = recolor.Engine()
    engine.mode = "test"
    checkpoint = torch.load(correction, map_location="cpu", weights_only=True)
    assert checkpoint["manifest_sha256"] == RECOLOR_MANIFEST_SHA
    assert checkpoint["recolor_sha256"] == MAP_SHA
    assert checkpoint["arm"] == "C" and checkpoint["step"] == 800 and checkpoint["epoch"] == 4
    module = recolor.fresh("C").eval()
    module.load_state_dict(checkpoint["state"], strict=True)
    return engine, module.requires_grad_(False)


def logits(engine, ids, cache, past_length, start):
    hidden = engine.forward(ids, cache, past_length, use_cache=False).last_hidden_state[0, start:]
    return engine.model.lm_head(hidden).float()


def run_arm(recolor, metrics, engine, item, row, cache, max_new_tokens):
    tokens = engine.rollout(row, recolor.clone_cache(cache, engine.model), max_new_tokens)
    prediction = engine.tok.decode(tokens, skip_special_tokens=True).strip()
    answer_ids = engine.tok.encode(item["ground_truth"], add_special_tokens=False)
    suffix = row["suffix_ids"]
    scores = logits(engine, suffix + answer_ids[:-1], cache, cache.get_seq_length(), len(suffix) - 1)
    labels = torch.tensor(answer_ids, device=scores.device)
    eos = engine.model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    stops.update((engine.tok.eos_token_id, engine.tok.convert_tokens_to_ids("<|im_end|>")))
    stopped = tokens[-1] in stops
    return dict(tokens=tokens, prediction=prediction, stopped_eos=stopped, reached_cap=not stopped,
                reference_token_nll=float(torch.nn.functional.cross_entropy(scores, labels)),
                **metrics.answer_metrics(prediction, item["ground_truth"]))


@torch.no_grad()
def run(recolor_map, correction_path, models_root, output):
    recolor = engine_module()
    metrics_module = importlib.util.spec_from_file_location("herb_recolor_metrics", ROOT / "herb_metrics.py")
    metrics = importlib.util.module_from_spec(metrics_module)
    metrics_module.loader.exec_module(metrics)
    baseline = ROOT / "cap192"
    reference = read(baseline / "summary.json")
    samples = ROOT / "samples.json"
    items = {item["id"]: item for item in read(samples)["samples"]}
    inputs = read(ROOT / "recolor_inputs.json")
    assert recolor.sha(samples) == reference["samples_sha256"] == inputs["samples_sha256"]
    assert recolor.sha(baseline / "summary.json") == inputs["baseline_summary_sha256"]
    assert len(items) == reference["n"] == len(inputs["rows"])
    model_paths = {}
    for family, info in reference["model_files"].items():
        path = Path(info["path"])
        if models_root is not None:
            path = models_root.joinpath(*path.parts[-2:])
        for name, expected in info["sha256"].items():
            assert recolor.sha(path / name) == expected, (family, name)
        model_paths[family] = str(path)
    engine, correction = load_recolor(recolor, recolor_map, correction_path, model_paths)
    model, tok = engine.model, engine.tok
    assert model.config._attn_implementation == "sdpa" and model.dtype == torch.float16
    max_new_tokens = reference["max_new_tokens"]
    probe_ids = tok.encode("The issue was", add_special_tokens=False)
    results = []
    for row in inputs["rows"]:
        item = items[row["id"]]
        full = row["full_ids"]
        suffix = row["suffix_ids"]
        assert engine.prefix_ids + row["content_ids"] + suffix == full
        length = len(engine.prefix_ids) + len(row["content_ids"])
        # The translated path uses RAW Llama and the public Qwen chat header only.
        base = engine.base(row)
        translated = engine.cache(row, base, correction)
        assert translated.get_seq_length() == length
        score = run_arm(recolor, metrics, engine, item, row, translated, max_new_tokens)

        # Native cache and generation are computed separately as an endpoint control.
        native = engine.native_cache(row, baseline=True)
        full_logits = logits(engine, full + probe_ids, None, 0, len(full) - 1)
        native_probe = recolor.clone_cache(native, model)
        native_logits = logits(engine, suffix + probe_ids, native_probe, length, len(suffix) - 1)
        check = dict(max_logit_error=float((full_logits - native_logits).abs().max()),
                     kl=float(recolor.vocab_kl(full_logits, native_logits)),
                     argmax_matches=bool(torch.equal(full_logits.argmax(-1), native_logits.argmax(-1))))
        assert check["kl"] < 0.001 and check["argmax_matches"] and check["max_logit_error"] < 0.3, check
        prev = read(baseline / "per_sample" / (item["id"] + ".json"))
        assert native.get_seq_length() == length
        native_score = run_arm(recolor, metrics, engine, item, row, native, max_new_tokens)
        matches = native_score["tokens"] == prev["conditions"]["native_kv"]["tokens"]

        result = dict(id=item["id"], recolor_c=score, native_kv=native_score, native_logits_parity=check,
                      native_matches_original_tokens=matches, evidence_prefill_tokens=length,
                      recolor_translated_tokens=len(row["content_ids"]))
        recolor.save(output / "per_sample" / (item["id"] + ".json"), result)
        results.append(result)
        print(f"COMPLETED {len(results)}/{len(items)} {item['id']}", flush=True)
    engine.unchanged()
    assert engine.student_test_native_prefills == 0
    metrics = {key: sum(x["recolor_c"][key] for x in results) / len(results)
               for key in ("em", "f1", "reference_token_nll")}
    recolor.save(output / "summary.json", dict(
        n=len(results), max_new_tokens=max_new_tokens, metrics=metrics,
        hit_cap=sum(x["recolor_c"]["reached_cap"] for x in results),
        native_matches_original=sum(x["native_matches_original_tokens"] for x in results),
        recolor_sha256=MAP_SHA, correction_sha256=CORRECTION_SHA, correction_step=800,
        baseline_summary_sha256=recolor.sha(baseline / "summary.json"), samples_sha256=recolor.sha(samples),
        prepared_sha256=recolor.sha(ROOT / "recolor_inputs.json"),
        student_test_native_prefills=engine.student_test_native_prefills,
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--recolor-map", type=Path, default=ROOT / "recolor_c_assets/recolor.pt")
    parser.add_argument("--correction", type=Path, default=ROOT / "recolor_c_assets/epoch4_step800.pt")
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "recolor_c_cap192")
    args = parser.parse_args()
    if args.prepare_only:
        freeze_inputs()
    else:
        run(args.recolor_map, args.correction, args.models_root, args.output)
