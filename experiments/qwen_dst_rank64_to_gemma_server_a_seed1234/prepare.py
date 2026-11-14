"""Retokenize the frozen v1 cohort for Gemma and prepare Qwen Full-Sync."""

from pathlib import Path

from reverse_common import ALIGN, ARCH, C, CFG, N, REFERENCE, ROOT, TEMPLATE, mark, verify


def main():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    N.verify()
    source = C.AutoTokenizer.from_pretrained(CFG["sender"], local_files_only=True)
    target = C.AutoTokenizer.from_pretrained(CFG["receiver"], local_files_only=True)
    marker = "DST_TEXT_SLOT_PLACEHOLDER"
    template = target.apply_chat_template([{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True)
    header, assistant = template.split(marker)

    def encode(text):
        return target.encode(text, add_special_tokens=False)

    prefix_ids = encode(header)
    stop = target.convert_tokens_to_ids("<end_of_turn>")
    assert prefix_ids == [2, 105, 2364, 107] and stop == 106
    previous_assistant = C.read(C.ROOT / "manifest.json")["assistant"]
    pins = list(ROOT.glob("*.py")) + [
        TEMPLATE / name for name in ("config.json", "translator.py", "residual.py", "runtime.py", "train.py", "alignment.py")
    ]
    pins += list((TEMPLATE / "reference").glob("*.py"))
    pins += [REFERENCE / "manifest.json", C.ROOT / "manifest.json"]
    for path in map(Path, (CFG["sender"], CFG["receiver"])):
        pins.extend(f for f in path.iterdir() if f.is_file() and (
            f.name in {"config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt"}
            or f.name.startswith("tokenizer")
        ))
    report = {}
    for split in ("train", "validation", "test"):
        converted = []
        for old in N.rows(split):
            body = old["prompt"]
            assert old["continuation_suffix"].startswith(previous_assistant)
            suffix = assistant + old["continuation_suffix"][len(previous_assistant):]
            aligned = ALIGN.fullsync(body, source, target)
            ids, suffix_ids = list(aligned.receiver_ids), encode(suffix)
            assert encode(header + body + suffix) == prefix_ids + ids + suffix_ids, old["id"]
            row = dict(old, content_ids=ids, suffix_ids=suffix_ids, continuation_suffix=suffix,
                       sender_ids=list(aligned.source_ids), sync_source_indices=list(aligned.source_indices),
                       sync_audit=aligned.audit)
            if split != "test":
                if old["task"] == "arc":
                    target_ids = encode(old["candidates"][old["gold"]])
                elif old["task"] == "hotpotqa":
                    target_ids = encode(old["gold"]) + [stop]
                else:
                    target_ids = encode(old["gold_target_text"]) + [stop]
                assert target_ids
                row["gold_tokens"] = target_ids
            assert len(prefix_ids) + len(ids) + len(suffix_ids) + max(512, len(row.get("gold_tokens", []))) < CFG["max_context_tokens"]
            converted.append(row)
        assert len(converted) == sum(CFG[split + "_counts"].values())
        path = ROOT / "prepared" / (split + ".jsonl")
        C.write_jsonl(path, converted)
        pins += [path, REFERENCE / "prepared" / path.name]
        report[split] = dict(samples=len(converted), tokens=sum(len(r["content_ids"]) for r in converted),
                             lookahead=sum(r["sync_audit"]["intra_unit_lookahead_tokens"] for r in converted))
    model = ARCH.DepthStateTranslator(ARCH.DSTConfig(**CFG["model"]), CFG["seed"]).specification()
    C.save(ROOT / "config.json", CFG)
    C.save(ROOT / "manifest.json", dict(config=CFG, model=model, header=header, prefix_ids=prefix_ids,
           suffix=assistant, stop_token=stop, alignment=report, pins={str(p): C.sha(p) for p in pins}))
    mark("prepared/completed.json", status="completed")
    print("Prepared reverse DST:", report, flush=True)


if __name__ == "__main__":
    main()
