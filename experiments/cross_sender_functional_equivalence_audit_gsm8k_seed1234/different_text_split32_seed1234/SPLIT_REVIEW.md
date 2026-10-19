# Pre-generation split review

All 32 proposed splits were read before running any model generation for this
experiment. The initial automatic split placed test_30 at "If their | total age".
The rule was corrected BEFORE GPU generation to allow a complete sentence ending
in a numeric token ("ratio of 7:11."). Numeric adjacency remains prohibited for
fallback clause/word cuts; decimal points inside a number are not sentence cuts.
The initial preflight files are retained under attempts/001_pre_generation_review.

No questions were replaced and no answer/correctness was used to choose a cut.
Some original questions contain dispensable background. In particular test_11's
A contains a shopping introduction, while B already states the quantities,
prices and query. Keep the fixed sample; do not assume both fragments are needed
on every row. Interpret omission controls per question and inspect raw text.
