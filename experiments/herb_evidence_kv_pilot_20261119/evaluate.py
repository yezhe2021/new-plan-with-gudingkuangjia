"""Four-arm HERB inference pilot using the frozen HotpotQA Same128 translator."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "hotpot_same128_interface_ablation_seed1234"
sys.path.insert(0, str(PARENT))
from common import C, H, P, U, torch


CHECKPOINT_HASHES = {
    "stage_a": "d5e04f7e3e12444ac15b7f9b511f8aa91130ab8e5137180b349c216209ebd9c0",
    "stage_b": "811ccfad968c4c0a49361746aa5af899680b380eb4896937d336f325c2ec119e",
}
ARMS = ("q_only", "text_oracle", "native_kv", "translated_kv")
PROBE = "The issue was"


def prepare(items, tok):
    lt = C.AutoTokenizer.from_pretrained(C.CFG["models"]["llama"], local_files_only=True)
    with (PARENT / "prepared/questions.jsonl").open() as previous:
        archived = json.loads(next(previous))["variants"]["D_evidence_only"]
    replay = U.make_row(archived["id"], "gsm8k", {"question": archived["prompt"]}, lt, tok,
                        prompt=archived["prompt"], candidates=[], cue="")
    for family in ("llama", "qwen"):
        assert replay["frames"][family]["full_ids"] == archived["frames"][family]["full_ids"]
    assert replay["source_indices"] == archived["source_indices"]
    prefix, assistant = C.chat_affixes(tok, thinking=False)
    prepared = []
    shared_anchor = None
    for item in items:
        evidence = "Reference information:\n" + "\n\n".join(
            f"Document {i}:\n{source['text']}" for i, source in enumerate(item["evidence"], 1)
        ) + "\nEnd of reference"
        query = ("Use the reference when available. Give a concise answer that covers every part of the question. "
                 "If there is insufficient information, say so.\n\nQuestion: " + item["question"])
        # The translation input contains evidence and fixed structure only.
        row = U.make_row(item["id"], "gsm8k", {"question": item["question"]}, lt, tok,
                         prompt=evidence, candidates=[], cue="")
        assert row["native_tail"] == ""
        frame = row["frames"]["qwen"]
        body = frame["body_ids"]
        anchor = body[:frame["native_prefix_length"]]
        content = evidence + "\n\n" + query
        rendered = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        assert rendered == prefix + content + assistant
        full = tok.encode(rendered, add_special_tokens=False)
        assert full[:len(body)] == body, item["id"]
        compact = tok.encode(prefix + query + assistant, add_special_tokens=False)
        row["suffix_ids"] = full[len(body):]
        assert row["suffix_ids"]
        if shared_anchor is None:
            shared_anchor = anchor
        assert shared_anchor == anchor
        prepared.append((item, row, full, compact))
    return prepared, shared_anchor


def backbone_logits(model, ids, cache=None, past_length=0, start=-1):
    tensor = torch.tensor([ids], device="cuda")
    out = model.model(tensor, past_key_values=cache,
                      attention_mask=torch.ones(1, past_length + len(ids), device="cuda", dtype=torch.long),
                      position_ids=torch.arange(past_length, past_length + len(ids), device="cuda")[None],
                      use_cache=False)
    return model.lm_head(out.last_hidden_state[0, start:]).float()


def parity(reference, comparison):
    return dict(max_logit_error=float((reference - comparison).abs().max()),
                kl=float(U.full_kl(reference, comparison)),
                argmax_matches=bool(torch.equal(reference.argmax(dim=-1), comparison.argmax(dim=-1))))


def preflight(model, tok, prepared, anchor, output):
    probe_ids = tok.encode(PROBE, add_special_tokens=False)
    checks = []
    for item, row, full, _ in prepared:
        body = row["frames"]["qwen"]["body_ids"]
        suffix = row["suffix_ids"]
        native_cache = U.native_body(model, row)
        k, v, _ = C.capture(model, body, len(body), [0])
        anchor_length = row["frames"]["qwen"]["native_prefix_length"]
        native_injected = U.translated_cache(model, row, anchor, k[:, anchor_length:].cuda(),
                                            v[:, anchor_length:].cuda())
        reference = backbone_logits(model, full + probe_ids, start=len(full) - 1)
        split = backbone_logits(model, suffix + probe_ids, native_cache, len(body), len(suffix) - 1)
        injected = backbone_logits(model, suffix + probe_ids, native_injected, len(body), len(suffix) - 1)
        checks.append(dict(id=item["id"], text_vs_native=parity(reference, split),
                           native_vs_native_injected=parity(split, injected)))
    kl_max = U.CFG["native_parity_kl_max"]
    passed = all(check[key]["kl"] < kl_max and check[key]["argmax_matches"]
                 and check[key]["max_logit_error"] < 0.3
                 for check in checks for key in ("text_vs_native", "native_vs_native_injected"))
    C.save(output / "preflight.json", dict(passed=passed, kl_max=kl_max, max_logit_error=0.3, checks=checks))
    assert passed, "Native cache parity failed. See preflight.json."
    print(f"PREFLIGHT PASSED {len(checks)} questions", flush=True)


def run_arm(model, tok, item, ids, max_new_tokens, past_length=0, cache=None):
    # U.rollout needs only the number of cached positions and the continuation IDs.
    rollout_row = {"frames": {"qwen": {"body_ids": range(past_length)}}, "suffix_ids": ids}
    generation_cache = U.clone_cache(cache, model) if cache is not None else None
    result = U.rollout(model, tok, rollout_row, generation_cache, max_new_tokens)
    prediction = tok.decode(result["tokens"], skip_special_tokens=True).strip()
    answer_ids = tok.encode(item["ground_truth"], add_special_tokens=False)
    assert answer_ids, item["id"]
    logits = backbone_logits(model, ids + answer_ids[:-1], cache, past_length, len(ids) - 1)
    labels = torch.tensor(answer_ids, device=logits.device)
    nll = float(torch.nn.functional.cross_entropy(logits, labels))
    return dict(**result, prediction=prediction, reference_token_nll=nll,
                **H.answer_metrics(prediction, item["ground_truth"]))


def load_translator():
    manifest = json.loads((PARENT / "manifest.json").read_text())
    assert manifest["parent_manifest_sha256"] == P.identity()
    checkpoints = manifest["checkpoints"]
    for stage, expected in CHECKPOINT_HASHES.items():
        assert C.sha(Path(checkpoints[stage]["path"])) == expected
    translator = P.fresh_b().eval().requires_grad_(False)
    checkpoint = torch.load(checkpoints["stage_b"]["path"], map_location="cpu", weights_only=True)
    assert checkpoint["manifest_sha256"] == P.identity()
    assert checkpoint["frozen_stage_a_sha256"] == CHECKPOINT_HASHES["stage_a"]
    translator.residual.load_state_dict(checkpoint["state"], strict=True)
    return translator


def nmse(prediction, target):
    prediction, target = prediction.float(), target.float().to(prediction.device)
    return float((prediction - target).square().sum() / target.square().sum())


@torch.inference_mode()
def run(models_root, output, max_new_tokens):
    P.verify()
    if models_root is not None:
        for family, path in C.CFG["models"].items():
            C.CFG["models"][family] = str(models_root.joinpath(*Path(path).parts[-2:]))
    model_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    identities = {family: dict(path=path, sha256={name: C.sha(Path(path) / name) for name in model_files})
                  for family, path in C.CFG["models"].items()}
    items = json.loads((ROOT / "samples.json").read_text())["samples"]
    C.seed()
    tok, model = C.model("qwen")
    prepared, shared_anchor_ids = prepare(items, tok)
    anchor = U.native_body(model, {"frames": {"qwen": {"body_ids": shared_anchor_ids}}})
    preflight(model, tok, prepared, anchor, output)
    _, sender = C.model("llama")
    translator = load_translator()
    assert all(not p.requires_grad for module in (model, sender, translator) for p in module.parameters())
    results = []
    for item, row, full, compact in prepared:
        length = len(row["frames"]["qwen"]["body_ids"])
        suffix = row["suffix_ids"]
        conditions = {"q_only": run_arm(model, tok, item, compact, max_new_tokens),
                      "text_oracle": run_arm(model, tok, item, full, max_new_tokens),
                      "native_kv": run_arm(model, tok, item, suffix, max_new_tokens, length, U.native_body(model, row))}
        # Only this offline metric path constructs native evidence targets.
        source, native_target = P.capture_pair(sender, model, row, target=True)
        k, v = P.map_source(translator, source)
        cache = U.translated_cache(model, row, anchor, k, v)
        conditions["translated_kv"] = run_arm(model, tok, item, suffix, max_new_tokens, length, cache)
        result = dict(id=item["id"], conditions=conditions, evidence_prefill_tokens=length,
                      sender_prefill_tokens=len(row["frames"]["llama"]["body_ids"]),
                      native_generation_matches=conditions["text_oracle"]["tokens"] == conditions["native_kv"]["tokens"],
                      key_nmse=nmse(k, native_target["key"]), value_nmse=nmse(v, native_target["value"]))
        results.append(result)
        C.save(output / "per_sample" / (item["id"] + ".json"), result)
        print(f"COMPLETED {len(results)}/{len(items)} {item['id']}", flush=True)
    metrics = {arm: {metric: sum(x["conditions"][arm][metric] for x in results) / len(results)
                     for metric in ("em", "f1", "reference_token_nll")} for arm in ARMS}
    C.save(output / "summary.json", dict(n=len(results), arms=metrics, max_new_tokens=max_new_tokens,
                                      checkpoint_sha256=CHECKPOINT_HASHES, samples_sha256=C.sha(ROOT / "samples.json"),
                                      native_matching_generations=sum(r["native_generation_matches"] for r in results),
                                      model_files=identities, archived_tokenizer_replay=True,
                                      gpu=torch.cuda.get_device_name(), torch_version=torch.__version__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    run(args.models_root, args.output, args.max_new_tokens)
