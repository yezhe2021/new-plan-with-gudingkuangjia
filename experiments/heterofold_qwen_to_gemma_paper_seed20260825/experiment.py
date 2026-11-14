"""Apply the paper's Recolor and calibration recipe with Gemma as receiver."""

from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
REFERENCE = ROOT.parent / CFG["evaluation_reference"]
sys.path.insert(0, str(REFERENCE))
import reverse_common as G
import reverse_engine as E

C, torch = G.C, G.torch
from heterofold import attention, calibration, capture, maps, recolor, transfer
from heterofold.alignment import exact_end_pairs

CORPUS = ROOT.parent / CFG["calibration_manifest"]
RECOLOR = ROOT / "maps/recolor.pt"
HETEROFOLD = ROOT / "maps/heterofold.pt"


def status(phase, **details):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), phase=phase, **details))


def pause():
    if (ROOT / "controls/pause_requested.json").exists():
        raise SystemExit(75)


def mark(name, **details):
    C.save(ROOT / name, dict(manifest_sha256=C.sha(ROOT / "manifest.json"), **details))


def completed(name, artifact=None):
    path = ROOT / name
    if not path.exists():
        return False
    record = C.read(path)
    assert record["manifest_sha256"] == C.sha(ROOT / "manifest.json")
    assert record["status"] == "completed"
    if artifact is not None:
        assert C.sha(artifact) == record["sha256"]
    return True


def verify():
    for path, expected in C.read(ROOT / "manifest.json")["pins"].items():
        assert C.sha(path) == expected, path


def prepare():
    assert C.sha(CORPUS) == CFG["calibration_sha256"]
    assert G.identity() == CFG["evaluation_manifest_sha256"]
    test_path = REFERENCE / "prepared/test.jsonl"
    assert C.sha(test_path) == CFG["evaluation_rows_sha256"]
    if (ROOT / "manifest.json").exists():
        verify()
        return
    corpus = C.read(CORPUS)
    rows = corpus["rows"]
    counts = Counter((row["dataset"], row["split"]) for row in rows)
    assert len(counts) == 4 and sorted(counts.values()) == [200, 200, 800, 800]
    assert len({row["identity"] for row in rows}) == len(rows)
    sender = C.AutoTokenizer.from_pretrained(G.CFG["sender"], local_files_only=True)
    receiver = C.AutoTokenizer.from_pretrained(G.CFG["receiver"], local_files_only=True)
    ref_manifest = C.read(REFERENCE / "manifest.json")
    header, suffix = C.chat_affixes(receiver, thinking=False)
    assert receiver.encode(header, add_special_tokens=False) == ref_manifest["prefix_ids"]
    assert suffix == ref_manifest["suffix"]
    test = G.rows("test")
    assert Counter(row["task"] for row in test) == G.CFG["test_counts"]
    test_questions = {C.norm(row["question"]) for row in test if row["task"] == "hotpotqa"}
    train_questions = {C.norm(row["question"]) for row in rows if row["dataset"].startswith("HotpotQA")}
    assert not (test_questions & train_questions), "HotpotQA calibration/evaluation overlap"
    min_pairs, longest = None, (0, None)
    for index, row in enumerate(rows):
        text = row["prompt"]
        begin, end = row["question_char_start"], row["question_char_end"]
        assert text[begin:end] == row["question"]
        target_offsets = receiver(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        calibration.question_token_positions(target_offsets, begin, end)
        longest = max(longest, (len(target_offsets), index))
        if row["split"] == "train":
            source_offsets = sender(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            n = len(exact_end_pairs(source_offsets, target_offsets))
            assert n >= recolor.MIN_ALIGNED_TOKENS, row["identity"]
            min_pairs = n if min_pairs is None else min(n, min_pairs)
    pins = [CORPUS, test_path, REFERENCE / "manifest.json", C.ROOT / "manifest.json",
            *ROOT.glob("*.py"), ROOT / "config.json",
            *(REFERENCE / file for file in ("experiment.py", "reverse_common.py", "reverse_engine.py"))]
    pins += [Path(module.__file__) for module in (attention, calibration, capture, maps, recolor, transfer)]
    pins += [REFERENCE / f'evaluation/{row["task"]}/per_sample/{row["id"]}.json' for row in test]
    C.save(ROOT / "manifest.json", dict(
        config=CFG, models={key: G.CFG[key] for key in ("sender", "receiver", "source_dtype", "receiver_dtype")},
        train=1600, validation=400, test_counts=G.CFG["test_counts"], corpus_id=corpus["calibration_id"],
        prefix_ids=ref_manifest["prefix_ids"], minimum_aligned_pairs=min_pairs, longest_calibration=longest,
        pins={str(path): C.sha(path) for path in pins},
    ))
    C.log("PREPARED", "1600 train, 400 held-out, 512 paired test", "longest calibration", longest)


class CalibrationGemma(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.model = text_model.model
        self.config = text_model.config

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, *, use_cache, logits_to_keep=None):
        return self.model(input_ids, use_cache=use_cache)


@contextmanager
def gemma_capture_layout():
    """The Gemma norm hooks return [batch, heads, tokens, dim]."""
    original = capture._token_major

    def token_major(output, heads, head_dim):
        if torch.is_tensor(output) and output.ndim == 4:
            assert output.shape[0] == 1 and output.shape[1] == heads and output.shape[-1] == head_dim
            return output[0].transpose(0, 1)
        return original(output, heads, head_dim)

    capture._token_major = token_major
    try:
        yield
    finally:
        capture._token_major = original


def setup():
    engine = E.Engine()
    return engine, CalibrationGemma(engine.model).eval()


def report(message):
    C.log(message)
    status("pipeline", detail=message)
    pause()


@torch.no_grad()
def native_attention_parity(engine, model):
    row = next(row for row in G.rows("test") if row["task"] == "hotpotqa"
               and 1050 <= len(row["content_ids"]) <= 1400)
    ids = torch.tensor([engine.prefix_ids + row["content_ids"]], device="cuda")
    probes = torch.arange(ids.shape[1] - 8, ids.shape[1], device="cuda")
    actual, handles = {}, []
    for index, block in enumerate(C.layers_of(model)):
        def save(_module, _args, output, layer=index):
            actual[layer] = output[0, probes].detach()
        handles.append(block.self_attn.o_proj.register_forward_hook(save))
    try:
        with gemma_capture_layout():
            captured = capture.capture_native_qkv(model, ids, probes)
    finally:
        for hook in handles:
            hook.remove()
    errors = {}
    positions = torch.arange(ids.shape[1], device="cuda")
    for index, block in enumerate(C.layers_of(model)):
        attn = block.self_attn
        q, k, v = (captured[index][role] for role in "qkv")
        cos, sin = attention.rotary_cos_sin(model, positions, index, model.dtype)
        q = attention.apply_rope(q, cos[probes], sin[probes])
        k = attention.apply_rope(k, cos, sin)
        mask = attention.causal_mask(attn, probes, ids.shape[1])
        groups = attn.num_key_value_groups
        weights = attention.attention_probabilities(attn, q, k, mask, groups)
        context = torch.einsum("hpt,thd->phd", weights, v.repeat_interleave(groups, dim=1))
        predicted = attn.o_proj(context.flatten(1))
        relative = float((predicted - actual[index]).norm() / actual[index].norm().clamp_min(1e-8))
        kind = attn.layer_type
        errors[kind] = max(relative, errors.get(kind, 0.))
        assert relative < 0.002, (index, kind, relative)
    assert set(errors) == {"sliding_attention", "full_attention"}
    engine.unchanged()
    mark("checks/native_attention.json", status="completed", max_relative_error=errors, tokens=ids.shape[1])
    C.log("GEMMA NATIVE ATTENTION PARITY", errors)


def fit():
    verify()
    if completed("maps/recolor_completed.json", RECOLOR):
        return
    pause()
    engine, model = setup()
    native_attention_parity(engine, model)
    sender, receiver = (engine.sender_tok, engine.sender), (engine.tok, model)
    corpus = C.read(CORPUS)
    rows = [row for row in corpus["rows"] if row["split"] == "train"]
    stats = {}
    for label in sorted({row["dataset"] for row in rows}):
        prompts = [row["prompt"] for row in rows if row["dataset"] == label]
        stats[label] = recolor.collect_statistics(sender, receiver, prompts, device="cuda",
                                                  log=report, tag=f" {label}:", **CFG["recolor"])
    engine.unchanged()
    del engine, model, sender, receiver
    torch.cuda.empty_cache()
    balanced = recolor.balance_corpora(stats)
    del stats
    base = recolor.fit_map(balanced, device="cuda", log=report)
    base.update(sender="qwen4", receiver="gemma3_4b_it")
    base["calibration"].update(calibration_id=corpus["calibration_id"],
                               prompt_sha256_sequence=corpus["prompt_sha256_sequence"], train_prompts=len(rows))
    assert all(torch.isfinite(t).all() for role in "kv" for field in ("A", "b") for t in base[f"{field}_{role}"])
    maps.atomic_save(base, RECOLOR)
    mark("maps/recolor_completed.json", status="completed", sha256=C.sha(RECOLOR), calibration=base["calibration"])
    C.log("RECOLOR COMPLETE", RECOLOR)


def train():
    verify()
    assert completed("maps/recolor_completed.json", RECOLOR)
    if completed("maps/calibration_completed.json", HETEROFOLD):
        return
    pause()
    engine, model = setup()
    base = maps.load_map(RECOLOR)
    rows = C.read(CORPUS)["rows"]
    sender, receiver = (engine.sender_tok, engine.sender), (engine.tok, model)
    check_row = rows[C.read(ROOT / "manifest.json")["longest_calibration"][1]]
    with gemma_capture_layout():
        if not completed("checks/calibration.json"):
            builder = calibration.ExampleBuilder(sender, receiver, base, "cuda")
            assert builder.prefix_ids.tolist() == engine.prefix_ids
            prefixes = calibration.prefix_cache(model, builder.prefix_ids, "cuda")
            corrections = torch.nn.ModuleList([
                calibration.LayerCorrection(base["kv_width"], CFG["calibration"]["rank"])
                for _ in range(base["Lr"])
            ]).cuda()
            example = builder.build(check_row)
            metrics = calibration.example_objective(model, corrections, example, *prefixes,
                lambda_k_output=CFG["calibration"]["lambda_k_output"], backward=True)
            assert all(torch.isfinite(p.grad).all() for p in corrections.parameters())
            assert all(p.grad.abs().sum() > 0 for n, p in corrections.named_parameters() if "right_" in n)
            engine.unchanged()
            del builder, prefixes, corrections, example
            torch.cuda.empty_cache()
            mark("checks/calibration.json", status="completed", metrics=metrics)
            C.log("CALIBRATION GRADIENT CHECK", metrics)
        result = calibration.calibrate(sender, receiver, base, rows, device="cuda",
            out_path=HETEROFOLD, checkpoint_path=ROOT / "maps/train_checkpoint.pt",
            log=report, **CFG["calibration"])
    engine.unchanged()
    details = result["functional_correction"]
    assert len(details["history"]) == CFG["calibration"]["epochs"]
    best = min(details["history"], key=lambda entry: entry["validation"]["loss"])
    assert best["validation"]["loss"] == details["best_validation_loss"]
    mark("maps/calibration_completed.json", status="completed", sha256=C.sha(HETEROFOLD),
         best_epoch=best["epoch"], updates=800, calibration=details)
    C.log("CALIBRATION COMPLETE", "best held-out epoch", best["epoch"])


def cache(engine, mapping, source, content_ids):
    keys, values, indices = source
    length = len(engine.prefix_ids) + len(content_ids)
    positions = torch.arange(len(engine.prefix_ids), length, device="cuda")
    rotary = transfer._rotary_tables(engine.model, positions)
    output = E.copy_cache(engine.header)
    for layer in range(mapping["Lr"]):
        key, value = transfer.map_layer(mapping, keys, values, indices, layer)
        key, value = transfer.finalize_layer(engine.model, layer, key, value, *rotary(layer))
        output.update(key[None], value[None], layer)
    assert output.get_seq_length() == length
    for layer in output.layers:
        expected = min(length, engine.model.config.sliding_window - 1) if layer.is_sliding else length
        assert layer.keys.shape[-2] == layer.values.shape[-2] == expected
    return output


@torch.no_grad()
def evaluate():
    verify()
    assert completed("maps/recolor_completed.json", RECOLOR)
    assert completed("maps/calibration_completed.json", HETEROFOLD)
    if completed("summary.json"):
        return
    pause()
    scorer = G.scoped("heterofold_reverse_scorer", REFERENCE / "experiment.py",
                       reverse_common=G, reverse_engine=E)
    metrics = scorer.evaluation_metrics()
    engine = E.Engine()
    engine.mode = "test"
    reader = E.BodyReader(engine)
    mappings = {name: maps.load_map(path, device="cuda", dtype=torch.float32)
                for name, path in (("Recolor_only", RECOLOR), ("HeteroFold_R16", HETEROFOLD))}
    taps = maps.sender_layers(mappings["Recolor_only"])
    assert taps == maps.sender_layers(mappings["HeteroFold_R16"])
    arms = ("native_gemma", "DST_stage1", "DST_rank64", *mappings)
    rows = G.rows("test")
    summaries = {}
    for task, count in G.CFG["test_counts"].items():
        records = []
        for i, row in enumerate((r for r in rows if r["task"] == task), 1):
            pause()
            rel = f'evaluation/{task}/per_sample/{row["id"]}.json'
            path = ROOT / rel
            if not path.exists():
                reference = C.read(REFERENCE / rel)
                assert reference["manifest_sha256"] == G.identity()
                assert all(reference[k] == row[k] for k in ("id", "task", "question", "gold"))
                enc = engine.tok(row["prompt"], add_special_tokens=False, return_offsets_mapping=True)
                assert enc["input_ids"] == row["content_ids"]
                offsets, keys, values = capture.capture_kv(engine.sender_tok, engine.sender, row["prompt"],
                                                           taps, device="cuda")
                source = keys, values, transfer.source_indices(offsets, enc["offset_mapping"], "cuda")
                result = dict(reference["conditions"])
                for name, mapping in mappings.items():
                    result[name] = scorer.score(reader, row, cache(engine, mapping, source, row["content_ids"]), metrics)
                del source, keys, values
                assert engine.student_test_native_prefills == 0
                engine.unchanged()
                mark(rel, id=row["id"], task=task, question=row["question"], gold=row["gold"], conditions=result)
            record = C.read(path)
            assert record["manifest_sha256"] == C.sha(ROOT / "manifest.json") and record["id"] == row["id"]
            values = record["conditions"]
            if task == "hellaswag":
                for result in values.values():
                    raw = result["raw_sum_scores"]
                    result["raw_correct"] = int(max(range(len(raw)), key=raw.__getitem__) == row["gold"])
            records.append(values)
            if i == 1 or i % 8 == 0:
                status("evaluation", task=task, sample=i, total=count)
                C.log("EVAL", task, i, count)
        assert len(records) == count
        keys = ("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else ("correct",)
        keys += {"hellaswag": ("raw_correct",), "gsm8k": ("legacy_correct",)}.get(task, ())
        scores = {arm: {k: sum(row[arm][k] for row in records) / count for k in keys} for arm in arms}
        pairs = {arm: {k: C.paired([row["HeteroFold_R16"][k] - row[arm][k] for row in records]) for k in keys}
                 for arm in arms[:-1]}
        summaries[task] = dict(n=count, metrics=scores, heterofold_minus=pairs)
        mark(f"evaluation/{task}/summary.json", status="completed", **summaries[task])
    lines = ["# Qwen3-4B → Gemma3-4B: HeteroFold paper calibration", "",
             "| Task | Metric | Gemma Native | DST Stage 1 | DST+Rank64 | Recolor | HeteroFold R16 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task, summary in summaries.items():
        scores = summary["metrics"]
        for metric in scores["HeteroFold_R16"]:
            values = " | ".join(f"{scores[arm][metric]:.2%}" for arm in arms)
            lines.append(f"| {task} | {metric} | {values} |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    mark("summary.json", status="completed", tasks=summaries)
    status("evaluation", status="completed")


if __name__ == "__main__":
    {"prepare": prepare, "fit": fit, "train": train, "evaluate": evaluate}[sys.argv[1]]()
