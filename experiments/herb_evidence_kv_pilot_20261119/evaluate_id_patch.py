"""Swap only the affected-customer KV positions in the eight frozen HERB examples."""

import argparse
import gc
import os
import re
import sys
from pathlib import Path

from evaluate_multisender import MODEL_DIRS, ROOT, load_model, make_writer, read, save, sha, split_evidence


METHODS = ("dst_rank64", "recolor_c")
CUSTOMER = re.compile(r"\bCUST-\d+\b", re.IGNORECASE)
LABEL = "Affected customer ID:"


def select_slots(tokenizer, slack, customer, header_length):
    offset = slack["text"].index(customer)
    assert slack["text"].count(customer) == 1 and customer.startswith("CUST-")
    tokens = tokenizer(slack["text"], add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokens["offset_mapping"]
    assert tokens["input_ids"] == slack["ids"]
    end = offset + len(customer)
    slots = [i for i, (a, b) in enumerate(offsets) if a < end and b > offset]
    assert len(slots) == 7 and slots == list(range(slots[0], slots[-1] + 1))
    assert offsets[slots[0]][0] == offset - 1 and offsets[slots[-1]][1] == end
    assert slack["text"][offset - 1] == " "
    digit_start = offset + len("CUST-")
    assert [offsets[i] for i in slots[-4:]] == [(p, p + 1) for p in range(digit_start, end)]
    zeros = slots[-4:-2]
    # The control is another seven-token Slack body window, with seven tokens
    # between the end of the window and the start of the customer ID.
    control = list(range(slots[0] - 14, slots[0] - 7))
    assert control[0] >= 0 and not set(control) & set(slots)
    distractor = slack["text"][offsets[control[0]][0]:offsets[control[-1]][1]]
    assert not CUSTOMER.search(distractor)
    return dict(id_slots=[header_length + i for i in slots],
                id_tokens=tokenizer.convert_ids_to_tokens([slack["ids"][i] for i in slots]),
                leading_zero_slots=[header_length + i for i in zeros],
                control_slots=[header_length + i for i in control], control_text=distractor)


def select_split_span(align, sender, target, slack, customer, header_length):
    text = slack["text"]
    sender_encoding = sender(text, add_special_tokens=False, return_offsets_mapping=True)
    target_encoding = target(text, add_special_tokens=False, return_offsets_mapping=True)
    assert target_encoding["input_ids"] == slack["ids"]
    units = align.synchronized_units(text, align.spans(text, sender_encoding),
                                     align.spans(text, target_encoding))
    start = text.index(customer) + len("CUST-")
    end = text.index(customer) + len(customer)
    matches = [u for u in units if len(u.llama_indices) == 1 and len(u.qwen_indices) > 1
               and start <= u.char_start < u.char_end <= end]
    assert len(matches) == 1, (customer, matches)
    unit, = matches
    return dict(text=unit.raw_text, source_token=unit.llama_indices[0] - 1,
                slots=[header_length + i - 1 for i in unit.qwen_indices])


def heldout_rewrites(rows, items, target, corpus):
    triples = [digits for doc in corpus["heldout"] for _, _, digits in doc["spans"] if len(digits) == 3]
    pools = ([x for x in triples if x.startswith("0") and not x.startswith("00")],
             [x for x in triples if not x.startswith("0")])
    train_numbers = {digits for doc in corpus["train"] for _, _, digits in doc["spans"]}
    assert not train_numbers.intersection(triples)
    rewritten = []
    for category, pool in zip(("one_zero", "nonzero"), pools):
        assert len(pool) >= len(rows)
        for row, digits in zip(rows, pool):
            org_id = row["id"]
            old = items[org_id]["required_facts"]["customer"]
            customer = "CUST-" + digits + old[-1]
            slack, pr = row["pieces"]
            assert slack["text"].count(old) == 1 and old not in pr["text"] and old not in items[org_id]["question"]
            text = slack["text"].replace(old, customer)
            ids = target.encode(text, add_special_tokens=False)
            assert len(ids) == len(slack["ids"])
            assert target.encode(text + pr["text"], add_special_tokens=False) == ids + pr["ids"]
            rewritten.append(dict(id=org_id + "_" + category, source_case=org_id, customer=customer,
                                  pieces=(dict(text=text, ids=ids), pr), content_ids=ids + pr["ids"],
                                  suffix_ids=row["suffix_ids"]))
    return rewritten


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", HETEROFOLD_V100_SDPA="1")
    sys.path.insert(0, str(args.project / "recolor_kl_l3_to_q4_seed1234/reference"))
    import torch
    from transformers import AutoTokenizer, DynamicCache
    from heterofold.capture import capture_kv
    from heterofold.evaluation.v100_sdpa import configure
    from heterofold.modeling import chat_affixes, decoder
    from heterofold.transfer import _cache_layer, _rotary_tables, finalize_layer

    configure()
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    sample_path = ROOT / "samples.json"
    input_path = ROOT / "recolor_inputs.json"
    ref_path = ROOT / "cap192/summary.json"
    earlier = args.reference or ROOT / "multisender" / args.method
    manifest = read(earlier / "summary.json")
    inputs, reference = read(input_path), read(ref_path)
    assert sha(sample_path) == inputs["samples_sha256"] == manifest["samples_sha256"]
    assert sha(ref_path) == inputs["baseline_summary_sha256"]
    assert sha(input_path) == manifest["input_sha256"]
    items = {item["id"]: item for item in read(sample_path)["samples"]}
    assert len(items) == len(inputs["rows"]) == manifest["n"] == 8
    model_paths = {family: args.models / MODEL_DIRS[family] for family in ("llama", "qwen")}
    for family, directory in model_paths.items():
        for file, expected in manifest["model_files"][family].items():
            assert sha(directory / file) == expected, (family, file)
    tgt = AutoTokenizer.from_pretrained(model_paths["qwen"], local_files_only=True)
    st = AutoTokenizer.from_pretrained(model_paths["llama"], local_files_only=True)
    prefix, _ = chat_affixes(tgt, thinking=False)
    header_ids = tgt.encode(prefix, add_special_tokens=False)
    h = len(header_ids)
    for row in inputs["rows"]:
        row["pieces"] = split_evidence(row["prompt"], tgt, row["content_ids"])
        assert header_ids + row["content_ids"] + row["suffix_ids"] == row["full_ids"]
        assert all(tgt.encode(prefix + s["text"], add_special_tokens=False) == header_ids + s["ids"]
                   for s in row["pieces"])
    rows = inputs["rows"]
    if args.heldout_rewrites:
        rows = heldout_rewrites(rows, items, tgt, read(args.mapper_training / "corpus.json"))
        assert all(tgt.encode(prefix + slack["text"], add_special_tokens=False) == header_ids + slack["ids"]
                   for slack, _ in (r["pieces"] for r in rows))

    with torch.no_grad():
        llama = load_model(model_paths["llama"], "llama", args.method, torch)
        translate, pins = make_writer(args, args.method, "llama", st, llama, tgt, None, torch)
        assert pins == manifest["checkpoints"]["llama"]
        translated_slack = {r["id"]: translate(r["pieces"][0]) for r in rows}
        digit_handoffs = {}
        if args.receiver_prefill:
            from receiver_prefill import assemble_digit_handoff, assert_only_local_slots_changed, digit_handoff

            align = sys.modules["herb_dst_alignment"]
            for row in rows:
                rid = row["id"]
                slack = row["pieces"][0]
                customer, = CUSTOMER.findall(slack["text"])
                audit = select_slots(tgt, slack, customer, h)
                digit_slots = audit["id_slots"][-4:]
                split_span = select_split_span(align, st, tgt, slack, customer, h)
                assert split_span["slots"] == digit_slots[:3]
                arms = {"split_span": digit_slots[:3], "all_digits": digit_slots}
                digit_handoffs[rid] = {
                    name: (slots, digit_handoff(translated_slack[rid], slack["ids"], slots, h))
                    for name, slots in arms.items()
                }
        del translate, llama
        gc.collect()
        torch.cuda.empty_cache()
        mappers, mapper_hashes = {}, {}
        if args.mapper_training:
            from train_span_mapper import SpanMapper, position_features

            training = read(args.mapper_training / "training_summary.json")
            for name in ("shared", "span"):
                path = args.mapper_training / (name + ".pt")
                mapper_hashes[name] = sha(path)
                assert mapper_hashes[name] == training["metrics"][name]["sha256"]
                payload = torch.load(path, map_location="cpu", weights_only=True)
                assert payload["arm"] == name and payload["checkpoints"] == pins
                assert payload["training_data_sha256"] == sha(args.mapper_training / "corpus.json")
                mapper = SpanMapper(payload["state"]["scales"], payload["rank"])
                mapper.load_state_dict(payload["state"])
                mappers[name] = mapper.cuda().eval().requires_grad_(False)
        model = load_model(model_paths["qwen"], "qwen", args.method, torch)
        assert reference["max_new_tokens"] == manifest["max_new_tokens"]
        cap = reference["max_new_tokens"]
        number_ids = [tgt.encode(str(i), add_special_tokens=False) for i in range(10)]
        assert all(len(ids) == 1 for ids in number_ids)
        number_ids = torch.tensor([ids[0] for ids in number_ids], device="cuda")

        def forward(ids, cache=None, *, use_cache=True):
            length = 0 if cache is None else cache.get_seq_length()
            x = torch.tensor([ids], device="cuda")
            return decoder(model)(input_ids=x, past_key_values=cache, use_cache=use_cache,
                                  attention_mask=torch.ones(1, length + len(ids), device="cuda", dtype=torch.long),
                                  position_ids=torch.arange(length, length + len(ids), device="cuda")[None])

        header = forward(header_ids).past_key_values
        layers = range(model.config.num_hidden_layers)
        eos = model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        stops.update((tgt.eos_token_id, tgt.convert_tokens_to_ids("<|im_end|>")))

        def native_raw(text):
            _, keys, values = capture_kv(tgt, model, prefix + text, layers, device="cpu")
            return tuple(torch.stack([group[l][h:] for l in layers]) for group in (keys, values))

        def assemble(left, right):
            keys, values = [torch.cat((a, b), dim=1).cuda() for a, b in zip(left, right)]
            table = _rotary_tables(model, torch.arange(h, h + keys.shape[1], device="cuda"))
            pairs = []
            for l in layers:
                k, v = finalize_layer(model, l, keys[l], values[l], *table(l))
                hk, hv = _cache_layer(header, l)
                pairs.append((torch.cat((hk, k[None]), dim=2), torch.cat((hv, v[None]), dim=2)))
            return DynamicCache(ddp_cache_data=pairs, config=model.config)

        def cache_copy(base, donor=None, positions=()):
            selected = torch.tensor(positions, device="cuda", dtype=torch.long)
            pairs = []
            if donor is not None:
                assert base.get_seq_length() == donor.get_seq_length() and len(positions) > 0
                other = torch.ones(base.get_seq_length(), device="cuda", dtype=torch.bool)
                other[selected] = False
                unselected = other.nonzero().flatten()
            for l in layers:
                original = _cache_layer(base, l)
                pair = tuple(t.clone() for t in original)
                if donor is not None:
                    for target, base_view, source in zip(pair, original, _cache_layer(donor, l)):
                        assert target.shape == source.shape
                        replacement = source.index_select(2, selected)
                        target.index_copy_(2, selected, replacement)
                        assert torch.equal(target.index_select(2, selected), replacement)
                        assert torch.equal(target.index_select(2, unselected), base_view.index_select(2, unselected))
                pairs.append(pair)
            return DynamicCache(ddp_cache_data=pairs, config=model.config)

        def readout(ids, cache, customer):
            probe = LABEL + " " + customer + "."
            tok = tgt(probe, add_special_tokens=False, return_offsets_mapping=True)
            label = tgt.encode(LABEL, add_special_tokens=False)
            assert tok["input_ids"][:len(label)] == label
            gold = tok["input_ids"][len(label):]
            prefix_ids = ids + label
            trace = forward(prefix_ids + gold[:-1], cache_copy(cache), use_cache=False).last_hidden_state
            scores = model.lm_head(trace[0, len(prefix_ids) - 1:]).float().log_softmax(-1)
            target = scores.gather(-1, torch.tensor(gold, device="cuda")[:, None]).flatten()
            digits = []
            char_start = len(probe) - 5
            for n, (a, b) in enumerate(tok["offset_mapping"][len(label):]):
                if char_start <= a < b <= char_start + 4:
                    assert b - a == 1
                    digits.append(dict(digit=probe[a:b], probability=float(target[n].exp()),
                                       top1=tgt.decode([int(scores[n].argmax())])))
            assert [x["digit"] for x in digits] == list(customer[-4:])
            state = forward(ids, cache_copy(cache))
            tokens = []
            for step in range(cap):
                token = int(model.lm_head(state.last_hidden_state[0, -1]).float().argmax())
                tokens.append(token)
                if token in stops or step + 1 == cap:
                    break
                state = forward([token], state.past_key_values)
            answer = tgt.decode(tokens, skip_special_tokens=True).strip()
            found = list(dict.fromkeys(CUSTOMER.findall(answer)))
            return dict(prediction=answer, tokens=tokens, reached_cap=tokens[-1] not in stops,
                        customer_ids=found, customer_exact=found == [customer],
                        id_and_period_logprob=float(target.sum()), digits=digits,
                        period_probability=float(target[-1].exp()),
                        following_digit_probability=float(scores[-1, number_ids].exp().sum()))

        results = []
        output = args.output or ROOT / "id_kv_patch" / args.method
        native_token_parity = 0
        for row in rows:
            rid = row["id"]
            slack, pr = row["pieces"]
            customer = row["customer"] if args.heldout_rewrites else items[rid]["required_facts"]["customer"]
            if not args.heldout_rewrites:
                assert customer.startswith("CUST-00")
            audit = select_slots(tgt, slack, customer, h)
            raw_slack, raw_pr = native_raw(slack["text"]), native_raw(pr["text"])
            t_slack = translated_slack[rid]
            assert all(x.shape == y.shape == (36, len(slack["ids"]), 1024)
                       for x, y in zip(raw_slack, t_slack))
            native, translated = assemble(raw_slack, raw_pr), assemble(t_slack, raw_pr)
            assert native.get_seq_length() == translated.get_seq_length() == h + len(row["content_ids"])
            # The entire header and PR region must remain bitwise identical.
            end_slack = h + len(slack["ids"])
            for l in layers:
                for a, b in zip(_cache_layer(native, l), _cache_layer(translated, l)):
                    assert torch.equal(a[:, :, :h], b[:, :, :h])
                    assert torch.equal(a[:, :, end_slack:], b[:, :, end_slack:])
            arms = {
                "A_native": native,
                "B_translated": translated,
            }
            if not args.heldout_rewrites and not args.receiver_prefill:
                arms["D_restore_id"] = cache_copy(translated, native, audit["id_slots"])
                arms["D_restore_zeros"] = cache_copy(translated, native, audit["leading_zero_slots"])
            if args.heldout_rewrites:
                audit.pop("leading_zero_slots")
            if not args.span_oracle and not args.receiver_prefill:
                arms["C_damage_id"] = cache_copy(native, translated, audit["id_slots"])
                arms["D_restore_non_id"] = cache_copy(translated, native, audit["control_slots"])
            if args.receiver_prefill:
                audit["receiver_prefill"] = {}
                for name, (slots, handoff) in digit_handoffs[rid].items():
                    text = tgt.decode(handoff.token_ids)
                    assert text == customer[-4:][:len(slots)]
                    sent_slots = handoff.before[0].shape[1] + handoff.after[0].shape[1]
                    assert sent_slots + len(handoff.token_ids) == len(slack["ids"])
                    filled = assemble_digit_handoff(model, forward, header, handoff, raw_pr)
                    change = assert_only_local_slots_changed(translated, filled, slots, layers)
                    assert all(delta > 0 for delta in change.values())
                    donor = cache_copy(translated, native, slots)
                    difference = assert_only_local_slots_changed(donor, filled, slots, layers)
                    audit["receiver_prefill"][name] = dict(
                        text=text, local_token_ids=handoff.token_ids, local_slots=slots,
                        sender_kv_slots=sent_slots, max_abs_delta_from_translated=change,
                        max_abs_delta_from_native_swap=difference,
                    )
                    arms["local_prefill_" + name] = filled
                    arms["native_swap_" + name] = donor
            raw_errors = {}
            if args.span_oracle:
                align = sys.modules["herb_dst_alignment"]
                span = select_split_span(align, st, tgt, slack, customer, h)
                audit["split_span"] = span
                assert span["slots"] == audit["id_slots"][-4:-1], (rid, span)
                local_slots = [slot - h for slot in span["slots"]]
                i, *others = local_slots
                audit["repeat_max_abs_pre_rope"] = {
                    role: max(float((group[:, i] - group[:, j]).abs().max()) for j in others)
                    for role, group in zip(("k", "v"), t_slack)
                }
                arms["D_restore_split_span"] = cache_copy(translated, native, span["slots"])
                if mappers:
                    base_kv = torch.stack([part[:, local_slots] for part in t_slack]).cuda().float()
                    native_kv = torch.stack([part[:, local_slots] for part in raw_slack]).cuda().float()
                    scales = mappers["span"].scales
                    assert torch.equal(scales, mappers["shared"].scales)

                    def per_role_nmse(kv):
                        return ((kv - native_kv) / scales).square().mean(dim=(1, 2, 3)).tolist()

                    raw_errors["B_translated"] = per_role_nmse(base_kv)
                    for name, mapper in mappers.items():
                        features = torch.tensor(position_features([len(local_slots)], shared=name == "shared"),
                                                device="cuda")
                        refined = mapper(base_kv, features)
                        assert torch.isfinite(refined).all()
                        corrected = tuple(t.clone() for t in t_slack)
                        for buffer, learned in zip(corrected, refined.cpu()):
                            buffer[:, local_slots] = learned
                        arm = "mapper_" + name
                        raw_errors[arm] = per_role_nmse(refined)
                        arms[arm] = cache_copy(translated, assemble(corrected, raw_pr), span["slots"])
            conditions = {k: readout(row["suffix_ids"], v, customer) for k, v in arms.items()}
            if not args.heldout_rewrites:
                prior = read(earlier / "per_sample" / (rid + ".json"))
                parity = conditions["A_native"]["tokens"] == prior["conditions"]["native_independent"]["tokens"]
                native_token_parity += parity
                if not args.span_oracle:
                    assert parity, rid
                else:
                    assert conditions["A_native"]["customer_exact"], (rid, conditions["A_native"]["customer_ids"])
            result = dict(id=rid, expected_customer=customer, slots=audit, conditions=conditions)
            if args.heldout_rewrites:
                result["source_case"] = row["source_case"]
            if raw_errors:
                result["split_span_kv_nmse"] = raw_errors
            save(output / "per_sample" / (rid + ".json"), result)
            results.append(result)
            print(f"COMPLETED {args.method} {len(results)}/{len(rows)} {rid}", flush=True)
        metrics = {}
        for arm in arms:
            records = [r["conditions"][arm] for r in results]
            metrics[arm] = dict(exact=sum(r["customer_exact"] for r in records),
                                casefold_exact=sum(
                                    {s.casefold() for s in r["conditions"][arm]["customer_ids"]}
                                    == {r["expected_customer"].casefold()} for r in results),
                                capped=sum(r["reached_cap"] for r in records),
                                mean_id_and_period_logprob=sum(r["id_and_period_logprob"] for r in records) / len(records),
                                mean_period_probability=sum(r["period_probability"] for r in records) / len(records),
                                mean_following_digit_probability=sum(r["following_digit_probability"] for r in records) / len(records),
                                mean_digit_probabilities=[sum(r["digits"][j]["probability"] for r in records) / len(records)
                                                          for j in range(4)])
            if mappers and arm in results[0]["split_span_kv_nmse"]:
                scores = [r["split_span_kv_nmse"][arm] for r in results]
                metrics[arm]["split_span_kv_nmse"] = [sum(values) / len(values) for values in zip(*scores)]
        eligible = [r for r in results if r["conditions"]["A_native"]["customer_exact"]
                    and not r["conditions"]["B_translated"]["customer_exact"]]
        save(output / "summary.json", dict(method=args.method, n=len(results), metrics=metrics, cap=cap,
             eligible=[r["id"] for r in eligible], checkpoints=pins, probe_label=LABEL,
             native_prior_token_parity=None if args.heldout_rewrites else native_token_parity,
             heldout_id_rewrites=args.heldout_rewrites, receiver_prefill=args.receiver_prefill,
             earlier_summary_sha256=sha(earlier / "summary.json"),
             input_sha256=sha(input_path), samples_sha256=sha(sample_path), gpu=torch.cuda.get_device_name(),
             mapper_sha256=mapper_hashes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--span-oracle", action="store_true")
    parser.add_argument("--canonical-dst", action="store_true")
    parser.add_argument("--mapper-training", type=Path)
    parser.add_argument("--heldout-rewrites", action="store_true")
    parser.add_argument("--receiver-prefill", action="store_true")
    args = parser.parse_args()
    if args.span_oracle and (args.method != "dst_rank64" or args.output is None):
        parser.error("--span-oracle requires dst_rank64 and --output")
    if args.mapper_training and not args.span_oracle:
        parser.error("--mapper-training requires --span-oracle")
    if args.heldout_rewrites and not args.mapper_training:
        parser.error("--heldout-rewrites requires --mapper-training")
    if args.receiver_prefill and (args.method != "dst_rank64" or args.output is None or args.span_oracle):
        parser.error("--receiver-prefill requires dst_rank64 and --output, and runs separately from --span-oracle")
    run(args)
