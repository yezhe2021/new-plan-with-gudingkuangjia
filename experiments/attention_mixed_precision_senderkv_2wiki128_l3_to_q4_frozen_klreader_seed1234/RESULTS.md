# Attention-Aware INT8 Protection Points the Right Way and Does Not Prove It

Llama3.2-3B → grouped Stage A + rank-64 residual Stage B → Qwen3-4B with the original KL-only
Q/O-LoRA Reader, all frozen, no training. The sender KV at the transmitted positions is quantised
and the receiver dequantises it in full; this measures precision and payload, not a prefill-free or
low-memory path.

Seven conditions on 128 new controlled 2Wiki evidence questions (128 rows, one question each, with
the previous 24 excluded by id and question text):

| Arm | What it protects | Acc | ΔNLL/token | 95% CI | KL | wire MB | eff bits |
|---|---|---:|---:|---|---:|---:|---:|
| fp16 | nothing, reference | 39.06 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | 46.0 | 16.00 |
| int4 | nothing, all INT4 | 39.84 | +0.0126 | [+0.0021, +0.0246] | 0.0085 | 13.0 | 4.52 |
| attention_top | top 10 % by attention → INT8 | 37.50 | +0.0051 | [−0.0037, +0.0139] | 0.0052 | 14.2 | 4.93 |
| random_0 | 10 % random → INT8 | 40.62 | +0.0075 | [−0.0033, +0.0192] | 0.0078 | 14.2 | 4.93 |
| random_1 | 10 % random → INT8 | 39.06 | +0.0150 | [+0.0050, +0.0266] | 0.0081 | 14.2 | 4.93 |
| random_2 | 10 % random → INT8 | 39.06 | +0.0151 | [+0.0047, +0.0266] | 0.0080 | 14.2 | 4.93 |
| attention_bottom | bottom 10 % by attention → INT8 | 39.84 | +0.0124 | [+0.0019, +0.0243] | 0.0084 | 14.2 | 4.93 |
| random_mean | mean of the three draws | 39.58 | +0.0125 | [+0.0028, +0.0240] | 0.0080 | 14.2 | 4.93 |

Wire bytes are the real serialised packet length, not tensor element counts, and the four protected
arms are asserted byte-identical per question (14.2 MB mean, 4.93 effective bits/element against
4.52 for all-INT4 — so the protected arms are *not* an equal-budget comparison with INT4, and the
INT4 row is the cheaper reference, not a matched control).

Paired on the same question, bootstrapped by question:

| Contrast | ΔNLL/token | 95% CI |
|---|---:|---|
| attention_top − random_mean | −0.0074 | [−0.0154, +0.0010] |
| attention_top − attention_bottom | −0.0073 | [−0.0155, +0.0013] |

## Reading

The ranking points the way the hypothesis wants. Protecting the top-attention 10 % gives
+0.0051 nats/token against +0.0125 for random positions and +0.0126 for all-INT4 — roughly a 2.5×
smaller perturbation than random at the same payload, and it is the only arm whose interval reaches
below zero. Top versus bottom is the cleanest possible contrast here: same number of protected
positions, same bytes, same metadata, so the only difference is *which* positions, and top beats
bottom by the same 0.007 nats/token.

And it still does not reach significance. Both intervals include zero, and one of the three random
draws (random_0, +0.0075) lands as close to the FP16 baseline as the attention arm does. With
n=128 and an interval half-width of 0.008, an effect of 0.007 is simply underneath the resolution of
this run. The accuracy column agrees: 37.50 to 40.62 across seven arms is coin-flip territory and
should be read as diagnostic only.

**The reader path is not usable in this run.** Every Qwen arm, including the FP16 reference,
generates to the cap on 100 % of questions and scores 1.56 % EM. Whatever the frozen KL-only Reader
was doing on 2Wiki, it was already capped before quantisation, so the second path carries no signal
about quantisation at all — its paired contrasts (+0.000008 and +0.0007) are the difference between
two degenerate distributions. Any conclusion here rests on the Llama direct-swap path alone.

What would settle it: more rows (n≈500 would halve the interval to the effect size), or a task where
the FP16 baseline is not itself at the cap. As it stands the honest statement is that attention-aware
position selection is the better prior by a consistent but unproven margin, and that this run cannot
separate it from a lucky random draw.

Caveats from the design:

- the ranking comes from the unquantised Llama's last forced-prefix query and is never recomputed
  after quantisation;
- 2Wiki transfer makes the FP16 baseline low (39 %), so accuracy is diagnostic;
- the quantiser borrows KIVI's asymmetric min/range axis but is not an unmodified KIVI/KVTuner
  reproduction;
- K groups lie on the original-order transmitted positions and do not reconstruct a contiguous
  window across untransmitted ones.
