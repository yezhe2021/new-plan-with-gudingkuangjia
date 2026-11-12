"""Freeze QuALITY articles and nested Qwen-length prefixes."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CORE = PROJECT / "recolor_kl_l3_to_q4_seed1234"
URL = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev"
SOURCE_SHA256 = "99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef"
LENGTHS = (1024, 2048, 4096, 8192)
SEED = 1234
ARTICLES = 8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fit_prefixes(text, tokenizer, header, suffix, header_ids, suffix_ids):
    tokenization = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenization["offset_mapping"]
    if len(offsets) < LENGTHS[-2]:
        return None
    rows = []
    for length in LENGTHS:
        if length > len(offsets):
            break
        ends = list(dict.fromkeys(end for _, end in offsets[max(0, length - 12):length + 12]))
        ends.sort(key=lambda end: abs(end - offsets[length - 1][1]))
        for end in ends:
            body = text[:end]
            ids = tokenizer.encode(body, add_special_tokens=False)
            if len(ids) != length:
                continue
            if tokenizer.encode(header + body + suffix, add_special_tokens=False) == header_ids + ids + suffix_ids:
                rows.append(dict(prompt=body, length=length, end_character=end, content_ids=ids, suffix_ids=suffix_ids))
                break
        else:
            if length == LENGTHS[-1]:
                break
            return None
    assert all(y["prompt"].startswith(x["prompt"]) for x, y in zip(rows, rows[1:]))
    return rows


def main(source):
    assert sha(source) == SOURCE_SHA256, "QuALITY source does not match the official pinned file"
    cfg = json.loads((CORE / "config.json").read_text())
    header = json.loads((CORE / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["receiver"], local_files_only=True)
    tokenizer_path = Path(cfg["models"]["receiver"])
    suffix_ids = tokenizer.encode(header["assistant"], add_special_tokens=False)
    assert len(header["prefix_ids"]) == 3 and len(suffix_ids) == 9
    texts = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            article = item["article"]
            key = hashlib.sha256(article.encode("utf-8")).hexdigest()
            texts.setdefault(key, (str(item["article_id"]), article))
    order = sorted(texts)
    random.Random(SEED).shuffle(order)
    candidates = []
    for digest in order:
        article_id, text = texts[digest]
        fitted = fit_prefixes(text, tokenizer, header["prefix"], header["assistant"], header["prefix_ids"], suffix_ids)
        if fitted is None:
            continue
        candidates.append((digest, article_id, fitted))
    # Preserve genuine article prefixes at 8K. Every selected 8K article is
    # also included at the shorter lengths so that its curve remains paired.
    long = [item for item in candidates if len(item[2]) == len(LENGTHS)]
    short = [item for item in candidates if len(item[2]) != len(LENGTHS)]
    chosen = (long + short)[:ARTICLES]
    assert len(chosen) == ARTICLES and long, "Not enough articles for the agreed workload"
    rows = []
    selected = []
    for digest, article_id, fitted in chosen:
        selected.append(dict(article_id=article_id, article_sha256=digest, lengths=[row["length"] for row in fitted]))
        rows.extend(dict(id=f"{digest[:16]}_{row['length']}", article_sha256=digest, **row) for row in fitted)
    save(ROOT / "inputs.json", rows)
    save(ROOT / "data_manifest.json", dict(
        source_url=URL, source_sha256=SOURCE_SHA256, source_distinct_articles=len(texts),
        eligible_8k_articles=len(long), articles_per_length={n: sum(r["length"] == n for r in rows) for n in LENGTHS},
        seed=SEED, lengths=list(LENGTHS), selected=selected, input_sha256=sha(ROOT / "inputs.json"),
        prefix_ids=header["prefix_ids"], suffix_ids=suffix_ids,
        receiver_tokenizer_sha256={name: sha(tokenizer_path / name) for name in ("tokenizer.json", "tokenizer_config.json")},
        purpose="Latency only, paired nested article prefixes",
    ))
    print("QuALITY prepared:", len(selected), "articles and", len(rows), "prefixes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    main(parser.parse_args().source)
