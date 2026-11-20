"""Freeze the HERB inputs and evaluate two independent senders on the same Qwen grid."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from herb_metrics import answer_metrics


ROOT = Path(__file__).resolve().parent
METHODS = ("fullsync_ab", "dst_rank64", "recolor_c")
MODEL_DIRS = dict(llama="LLM-Research/Llama-3.2-3B-Instruct",
                  gemma="LLM-Research/gemma-3-4b-it", qwen="Qwen/Qwen3-4B")
ASSIGNMENTS = dict(llama_slack_gemma_pr=("llama", "gemma"), gemma_slack_llama_pr=("gemma", "llama"),
                   llama_both_independent=("llama", "llama"), gemma_both_independent=("gemma", "gemma"))


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def split_evidence(text, tok, expected):
    marker = "\n\nDocument 2:\n"
    start = text.index(marker)
    # Prefer assigning the blank separator to Slack. Shift only within that separator if BPE requires it.
    for cut in (start + 2, start, start + 1):
        pieces = (text[:cut], text[cut:])
        ids = [tok.encode(piece, add_special_tokens=False) for piece in pieces]
        if ids[0] + ids[1] == expected:
            return tuple(dict(text=piece, ids=tokens) for piece, tokens in zip(pieces, ids))
    raise ValueError("Cannot split documents without changing the frozen Qwen token sequence")


def checkpoint(path, expected, stage, step, manifest, torch):
    assert sha(path) == expected, path
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert state["manifest_sha256"] == manifest and state["step"] == step, path
    assert state.get("stage", state.get("arm")) == stage, path
    return state


def make_writer(args, method, family, sender_tok, sender, target, legacy, torch):
    project = args.project
    if method == "fullsync_ab":
        artifacts = read(args.fullsync / "manifest.json")["artifacts"]
        pins = {}
        for asset in ("stage_a", "residual", "module"):
            record = artifacts[f"{family}/{asset}"]
            file = Path(record["path"])
            assert sha(file) == record["sha256"], file
            pins[asset] = record["sha256"]
        writer = legacy.load_writer(family)

        def translate(segment):
            src = legacy.unit(sender_tok, segment["text"])
            dst = legacy.unit(target, segment["text"])
            assert dst["body_ids"] == segment["ids"]
            index = legacy.align(src, dst)
            return tuple(x.flatten(2) for x in legacy.translate(writer, legacy.capture(sender, src["ids"]), index))

        return translate, pins

    if method == "dst_rank64":
        from heterofold.capture import capture_kv
        from heterofold.modeling import layers_of

        gemma = project / "gemma_dst_rank64_stage12_server_a_seed1234"
        if family == "llama":
            origin = project / "dst_rank64_gated_dynamic_m32_a4_server_a_seed1234"
            inputs = origin / "inputs"
            pins = read(origin / "config.json")["inputs_sha256"]
            base_path, residual_path = inputs / "dst.pt", inputs / "rank64.pt"
            base_manifest, rank_manifest = inputs / "dst_manifest.json", inputs / "rank64_manifest.json"
            assert sha(base_manifest) == pins["dst_manifest.json"]
            assert sha(rank_manifest) == pins["rank64_manifest.json"]
            base_sha, rank_sha = pins["dst.pt"], pins["rank64.pt"]
            code = inputs
        else:
            base_path = gemma / "runs/stage1/epoch4_step1536.pt"
            residual_path = gemma / "runs/stage2/epoch4_step1536.pt"
            base_manifest = rank_manifest = gemma / "manifest.json"
            base_sha = read(gemma / "runs/stage1/completed.json")["checkpoint_sha256"]
            rank_sha = read(gemma / "runs/stage2/completed.json")["checkpoint_sha256"]
            code = gemma
        sys.path.insert(0, str(gemma))
        align = load_module("herb_dst_alignment", gemma / "alignment.py").fullsync
        mod = load_module(f"herb_{family}_dst", code / "translator.py")
        residual = load_module(f"herb_{family}_residual", code / "residual.py").ResidualTranslator
        cfg = read(base_manifest)["config"]
        base = mod.DepthStateTranslator(mod.DSTConfig(**cfg["model"]), cfg["seed"])
        cp = checkpoint(base_path, base_sha, "stage1", 1536, sha(base_manifest), torch)
        base.load_state_dict(cp["state"], strict=True, assign=True)
        del cp
        base.cuda().eval().requires_grad_(False)
        writer = residual(base, rank=64, seed=cfg["seed"])
        cp = checkpoint(residual_path, rank_sha, "stage2", 1536, sha(rank_manifest), torch)
        writer.load_state_dict(cp["state"], strict=True, assign=True)
        del cp
        writer.cuda().eval().requires_grad_(False)

        def translate(segment):
            aligned = align(segment["text"], sender_tok, target, segment["ids"])
            source_layers = range(len(layers_of(sender)))
            _, keys, values = capture_kv(sender_tok, sender, segment["text"], source_layers, device="cpu")
            source = tuple(tuple(group[l] for l in source_layers) for group in (keys, values))
            indices = aligned.source_indices
            chunk = cfg["token_chunk"]
            pieces = [writer(*source, indices[start:start + chunk]) for start in range(0, len(indices), chunk)]
            return tuple(torch.cat(parts, dim=1).cpu() for parts in zip(*pieces))

        return translate, dict(stage1=base_sha, rank64=rank_sha)

    from heterofold.calibration import ExampleBuilder
    from heterofold.maps import load_map

    parent = project / "recolor_kl_l3_to_q4_seed1234"
    residuals = load_module("herb_recolor_residuals", parent / "residuals.py")
    if family == "llama":
        reference = parent
        map_path = Path(read(parent / "config.json")["recolor"])
        map_sha = read(parent / "config.json")["recolor_sha256"]
    else:
        reference = project / "rnkv_gemma_recolor_c_keep_both_l19_30_3task_seed1234"
        map_path = reference / "maps/recolor.pt"
        map_sha = read(reference / "maps/completed.json")["recolor_sha256"]
    correction = reference / "runs/C/epoch4_step800.pt"
    c_sha = read(reference / "runs/C/completed.json")["checkpoint_sha256"]
    assert sha(map_path) == map_sha
    cp = checkpoint(correction, c_sha, "C" if family == "llama" else None, 800,
                    sha(reference / "manifest.json"), torch)
    assert cp["recolor_sha256"] == map_sha
    writer = residuals.make_corrections("linear", 64)
    writer.load_state_dict(cp["state"], strict=True)
    del cp
    writer.eval().requires_grad_(False)
    builder = ExampleBuilder((sender_tok, sender), (target, SimpleNamespace(dtype=torch.float16)),
                             load_map(map_path), "cuda")

    def translate(segment):
        out = target(segment["text"], add_special_tokens=False, return_offsets_mapping=True)
        assert out["input_ids"] == segment["ids"]
        k, v = builder.recolor_outputs(segment["text"], out["offset_mapping"])
        return (torch.stack([layer.key(block.float()) for layer, block in zip(writer, k)]).cpu(),
                torch.stack([layer.value(block.float()) for layer, block in zip(writer, v)]).cpu())

    return translate, dict(recolor=map_sha, c=c_sha)


def load_model(path, family, method, torch):
    from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration

    model_class = Gemma3ForConditionalGeneration if family == "gemma" and method != "fullsync_ab" else AutoModelForCausalLM
    dtype = torch.float32 if family == "gemma" else torch.float16
    attention = "eager" if method == "fullsync_ab" and family != "qwen" else "sdpa"
    print(f"Loading {family} ({dtype}, {attention})", flush=True)
    return model_class.from_pretrained(path, dtype=dtype, attn_implementation=attention,
                                      device_map={"": "cuda:0"}, local_files_only=True).eval().requires_grad_(False)


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", HETEROFOLD_V100_SDPA="1")
    sys.path.insert(0, str(args.project / "recolor_kl_l3_to_q4_seed1234/reference"))
    import torch
    from transformers import AutoTokenizer, DynamicCache, __version__ as transformers_version
    from heterofold.attention import rotate_half
    from heterofold.capture import capture_kv
    from heterofold.evaluation.v100_sdpa import configure
    from heterofold.modeling import chat_affixes, decoder
    from heterofold.transfer import _cache_layer, _rotary_tables, finalize_layer

    configure()
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    with torch.no_grad():
        reference_path = ROOT / "cap192/summary.json"
        reference = read(reference_path)
        samples_path = ROOT / "samples.json"
        input_path = ROOT / "recolor_inputs.json"
        inputs = read(input_path)
        assert sha(samples_path) == inputs["samples_sha256"] == reference["samples_sha256"]
        assert sha(reference_path) == inputs["baseline_summary_sha256"]
        items = {row["id"]: row for row in read(samples_path)["samples"]}
        paths = {k: args.models / v for k, v in MODEL_DIRS.items()}
        identity = {k: {f: sha(path / f) for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                    "model.safetensors.index.json")} for k, path in paths.items()}
        assert all(identity[k] == ref["sha256"] for k, ref in reference["model_files"].items())
        tgt = AutoTokenizer.from_pretrained(paths["qwen"], local_files_only=True)
        prefix, _ = chat_affixes(tgt, thinking=False)
        header_ids = tgt.encode(prefix, add_special_tokens=False)
        rows = inputs["rows"]
        for row in rows:
            row["segments"] = split_evidence(row["prompt"], tgt, row["content_ids"])
            assert header_ids + row["content_ids"] + row["suffix_ids"] == row["full_ids"]
            assert all(tgt.encode(prefix + s["text"], add_special_tokens=False) == header_ids + s["ids"]
                       for s in row["segments"])
        legacy = None
        if args.method == "fullsync_ab":
            sys.path.insert(0, str(args.fullsync))
            legacy = load_module("herb_legacy_multidoc", args.fullsync / "experiment.py")
        mapped, pins = {}, {}
        for family in ("llama", "gemma"):
            sender_tok = AutoTokenizer.from_pretrained(paths[family], local_files_only=True)
            sender = load_model(paths[family], family, args.method, torch)
            translate, pins[family] = make_writer(args, args.method, family, sender_tok, sender, tgt, legacy, torch)
            mapped[family] = {}
            for row in rows:
                segments = tuple(translate(segment) for segment in row["segments"])
                for segment, target in zip(segments, row["segments"]):
                    assert segment[0].shape == segment[1].shape == (36, len(target["ids"]), 1024)
                    assert all(torch.isfinite(tensor).all() for tensor in segment)
                mapped[family][row["id"]] = segments
            print(f"Translated both evidence segments with {family} for {len(rows)} questions", flush=True)
            del sender, translate
            gc.collect()
            torch.cuda.empty_cache()

        model = load_model(paths["qwen"], "qwen", args.method, torch)
        cap = reference["max_new_tokens"]

        def forward(ids, cache=None, *, use_cache=True):
            length = 0 if cache is None else cache.get_seq_length()
            x = torch.tensor([ids], device="cuda")
            return decoder(model)(input_ids=x, past_key_values=cache, use_cache=use_cache,
                                  attention_mask=torch.ones(1, length + len(ids), device="cuda", dtype=torch.long),
                                  position_ids=torch.arange(length, length + len(ids), device="cuda")[None])

        header = forward(header_ids).past_key_values
        n_header = len(header_ids)

        def assemble(pieces, *, normalized=False):
            keys, values = [torch.cat(parts, dim=1).cuda() for parts in zip(*pieces)]
            tables = _rotary_tables(model, torch.arange(n_header, n_header + keys.shape[1], device="cuda"))
            layers = []
            for l, (k, v) in enumerate(zip(keys, values)):
                cos, sin = tables(l)
                if normalized:
                    k = k.to(model.dtype).reshape(-1, 8, 128).transpose(0, 1)
                    k = k * cos[None] + rotate_half(k) * sin[None]
                    v = v.to(model.dtype).reshape(-1, 8, 128).transpose(0, 1)
                else:
                    k, v = finalize_layer(model, l, k, v, cos, sin)
                hk, hv = _cache_layer(header, l)
                layers.append((torch.cat((hk, k[None]), dim=2), torch.cat((hv, v[None]), dim=2)))
            return DynamicCache(ddp_cache_data=layers, config=model.config)

        def native_raw(text):
            layers = range(model.config.num_hidden_layers)
            _, keys, values = capture_kv(tgt, model, prefix + text, layers, device="cpu")
            return tuple(torch.stack([group[l][n_header:] for l in layers]) for group in (keys, values))

        def score(ids, answer, cache):
            gold = tgt.encode(answer, add_special_tokens=False)
            clone = DynamicCache(ddp_cache_data=[tuple(t.clone() for t in _cache_layer(cache, l))
                                 for l in range(36)], config=model.config)
            scored = forward(ids + gold[:-1], clone, use_cache=False)
            logits = model.lm_head(scored.last_hidden_state[0, len(ids) - 1:]).float()
            nll = torch.nn.functional.cross_entropy(logits, torch.tensor(gold, device="cuda"))
            state = forward(ids, cache)
            eos = model.generation_config.eos_token_id
            stops = set(eos if isinstance(eos, list) else [eos])
            stops.update((tgt.eos_token_id, tgt.convert_tokens_to_ids("<|im_end|>")))
            tokens = []
            for step in range(cap):
                token = int(model.lm_head(state.last_hidden_state[0, -1]).float().argmax())
                tokens.append(token)
                if token in stops or step + 1 == cap:
                    break
                state = forward([token], state.past_key_values)
            prediction = tgt.decode(tokens, skip_special_tokens=True).strip()
            return dict(tokens=tokens, prediction=prediction, reached_cap=tokens[-1] not in stops,
                        reference_token_nll=float(nll), **answer_metrics(prediction, answer))

        output = ROOT / "multisender" / args.method
        results = []
        for row in rows:
            rid = row["id"]
            ids = row["suffix_ids"]
            gold = items[rid]["ground_truth"]
            joint = forward(header_ids + row["content_ids"]).past_key_values
            prior = read(ROOT / "cap192/per_sample" / (rid + ".json"))["conditions"]["native_kv"]["tokens"]
            if not results:
                reconstructed = assemble((native_raw(row["prompt"]),))
                exact = model.lm_head(forward(ids, forward(header_ids + row["content_ids"]).past_key_values)
                                      .last_hidden_state[0, -1]).float()
                reconstruction = model.lm_head(forward(ids, reconstructed).last_hidden_state[0, -1]).float()
                kl = torch.nn.functional.kl_div(reconstruction.log_softmax(-1), exact.log_softmax(-1),
                                                 log_target=True, reduction="sum")
                assert kl < 0.001 and exact.argmax() == reconstruction.argmax(), float(kl)
                save(output / "preflight.json", dict(id=rid, native_roundtrip_kl=float(kl), top1_parity=True))
            conditions = {"native_joint": score(ids, gold, joint),
                          "native_independent": score(ids, gold, assemble(tuple(native_raw(s["text"])
                                                                               for s in row["segments"])))}
            assert conditions["native_joint"]["tokens"] == prior, rid
            for arm, (left, right) in ASSIGNMENTS.items():
                pieces = (mapped[left][rid][0], mapped[right][rid][1])
                cache = assemble(pieces, normalized=args.method == "fullsync_ab")
                conditions[arm] = score(ids, gold, cache)
            result = dict(id=rid, segment_token_counts=[len(s["ids"]) for s in row["segments"]],
                          conditions=conditions, native_joint_matches_baseline=True)
            save(output / "per_sample" / (rid + ".json"), result)
            results.append(result)
            print(f"COMPLETED {args.method} {len(results)}/{len(rows)} {rid}", flush=True)
        fields = ("em", "f1", "reference_token_nll", "reached_cap")
        metrics = {arm: {field: sum(row["conditions"][arm][field] for row in results) / len(results)
                         for field in fields} for arm in results[0]["conditions"]}
        save(output / "summary.json", dict(method=args.method, n=len(results), max_new_tokens=cap,
             metrics=metrics, assignments=ASSIGNMENTS, checkpoints=pins, model_files=identity,
             samples_sha256=sha(samples_path), input_sha256=sha(input_path), script_sha256=sha(Path(__file__)),
             native_joint_reproduced=len(results), gpu=torch.cuda.get_device_name(),
             torch_version=torch.__version__, transformers_version=transformers_version))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--fullsync", type=Path)
    run(parser.parse_args())
