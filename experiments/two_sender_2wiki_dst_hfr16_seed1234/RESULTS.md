# Splitting the evidence across two frozen senders costs DST 1.8 F1 points and native 7.0, but this cohort's answer is literally in the second hop so nothing here tests composition

The DST line so far reads one sender's documents into one translated cache. A two-hop question is the
natural place to ask whether that still works when the evidence is divided: hop one and hop two in
different caches, translated independently, then assembled back into the receiver. All writers are
frozen from earlier experiments, so nothing is trained here; the only variable is how the same ten
documents are handed to the senders.

**Protocol.** 64 compositional items from the official 2WikiMultihopQA dev split, seed 1234, filtered
to exactly two supporting documents among ten, with `edge1.target == edge2.source` and
`edge2.target == answer`; questions and ids used by `depth_state_translator_fullsync_l3_to_q4` and
`dst_rank64_gated_dynamic_m32_gate_off_heldout` are excluded, so no item was seen during earlier
tuning. Sender A receives the hop-one document plus four distractors, sender B the hop-two document
plus four distractors, both shuffled. Only the Qwen receiver sees the question; document order and the
answer suffix (`Output only the short answer, without an explanation.`) are identical across all four
conditions, and the joint condition is the exact concatenation A+B. Greedy decoding, 64-token cap,
EM / F1 in percent. Writers: native (no translation), DST stage-1 + Rank-64, HeteroFold-R16. Native
cache assembly is parity-checked on every item (`inputs` replay: first-token KL 2.1e-06, top-1 always
matching).

| Arm | A only | B only | Joint, one sender | Split, two senders |
|---|---:|---:|---:|---:|
| Native | 1.56 / 9.00 | 26.56 / 35.34 | 14.06 / 28.50 | 9.38 / 21.50 |
| DST + Rank-64 | 4.69 / 10.40 | 20.31 / 30.91 | 15.62 / 26.86 | 14.06 / 25.09 |
| HeteroFold-R16 | 3.12 / 4.75 | 4.69 / 11.24 | 3.12 / 8.28 | 1.56 / 2.53 |

Paired F1 differences in points, 95 % intervals from a per-item bootstrap over the same 64 items:

| Contrast | Native | DST + Rank-64 | HeteroFold-R16 |
|---|---:|---:|---:|
| split − joint | −7.00 [−14.8, +0.1] | −1.77 [−11.5, +8.1] | −5.75 [−11.5, −0.9] |
| split − B only | −13.84 [−24.4, −3.5] | −5.81 [−15.0, +2.8] | −8.70 [−15.7, −2.8] |
| joint − B only | −6.84 [−19.0, +5.4] | −4.05 [−15.2, +6.5] | −2.95 [−9.0, +2.5] |

Three things follow, and only the first is about DST.

**1. DST is indistinguishable from native in both reading modes.** Joint: DST − native −1.64
[−11.7, +8.3]. Split: +3.59 [−6.0, +13.6]. Both intervals straddle zero by a wide margin, so the
seemingly better tolerance (splitting costs DST 1.77 against native's 7.00) is not a detectable
advantage at n = 64 — the difference of differences is about 5 points with an interval several times
that wide. What the data do support is a bound: dividing the evidence does not damage the translated
cache *more* than it damages the native one.

**2. HeteroFold-R16 is the only arm that actually breaks.** 2.53 F1 and 1/64 EM in split mode, 22.6
points [13.9, 31.9] below DST. That matches its standing in the single-sender protocols (41.80 macro
against Rank-64's 50.59); with two independently translated caches the same weakness compounds. It
should not be used as the "structured" baseline in split-evidence comparisons without that caveat.

**3. The cohort does not require composition, and that is the real finding.** B alone beats the joint
reading for every arm — native 35.34 against 28.50, DST 30.91 against 26.86 — and for native the gap
against B alone is the largest deficit on the board (−13.84 when split, interval excluding zero). The
filter that guarantees `edge2.target == answer` also places the answer string inside the hop-two
document, so a reader that opens only B can answer without ever using hop one; A's five documents then
act as added distraction rather than added evidence. Anything built on this cohort measures *cache
assembly under noise*, not multi-hop composition. A cohort that genuinely needs both hops has to
select items whose answer appears in neither document alone, which none of the filters above guarantee.

**What this rules out.** Dividing the evidence between two frozen senders is not by itself what breaks
a translated cache: the split costs DST and native comparable amounts within noise, and the collapse
seen with HeteroFold-R16 is a property of that translator rather than of the split. **What it does not
settle.** Whether DST can carry a genuinely compositional two-hop read — that needs a cohort where B
alone is insufficient, and the present numbers cannot speak to it, because on this cohort the single
strongest reading for all three arms is B alone.

Archived: code, cohort, manifest with checkpoint hashes, `per_sample/` for 64 items × 4 conditions and
the run log. `inputs/heterofold.pt` (865 MB writer artifact) is excluded per archive policy.
