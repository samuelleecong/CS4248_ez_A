# Saturated TACO Significance Report

All results in this report come from fresh HF replay of the six fully saturated checkpoints listed in the experiment README. The source of truth is the replayed per-query dataset in [`saturated_per_query.csv`](./saturated_per_query.csv) and the run summary in [`saturated_run_summary.csv`](./saturated_run_summary.csv).

## Setup

The analysis replays six final saturated student checkpoints on the fixed TACO test split of 1000 paired queries and code snippets. Every significance test is paired at the query level because every model is evaluated on the same exact test questions. That pairing is the critical design choice: it is what makes small differences in `MRR`, `Recall@1`, and margin interpretable rather than just noisy decimals.

Saturated seed-level significance is out of scope here. There are no matched saturated multi-seed reruns in local artifacts, so the defensible evidence for the saturated set is query-level paired testing, not seed-level variance estimation.

## What Was Tested And Why

`Paired permutation on reciprocal rank` is the primary test for retrieval quality because `MRR` is literally the mean of per-query reciprocal rank. If one method has a better MRR, that should appear as a consistently positive per-query reciprocal-rank difference, not just a better final scalar.

`Exact McNemar` is the right test for `Recall@1` and `Recall@10` because those metrics reduce to paired binary outcomes on each query: either the model retrieved the correct item within the cutoff or it did not. McNemar tests whether one model gets significantly more unique wins on the same paired questions.

`Wilcoxon signed-rank on margin` is used for the hardest-negative margin because that distribution is skewed, noisy, and not well modeled as Gaussian. This test asks whether one method is consistently better on local positive-vs-hardest-negative separation.

`Wilcoxon signed-rank on document cosine` directly tests the mechanism claim behind BiMGA. If BiMGA is better because it learns teacher-like document geometry, then its per-query document cosine should be consistently higher than the baselines.

`Bootstrap confidence intervals` are reported alongside every paired delta so the report does not collapse to a p-value checklist. The interval gives the direction and practical scale of the effect, not just whether the null was rejected.

The saturated HF repos publish aggregate `Asym MRR` and `Doc Cosine` inside each repo's `metrics.json`, but they do not publish the per-query fine-tuned teacher embeddings needed for paired significance on those teacher-space diagnostics. Accordingly, this report treats those teacher-space values as aggregate supporting diagnostics. Only the student-only retrieval metrics and hardest-negative margins receive full paired significance testing from public artifacts alone.

## Replay Sanity

| Run | Method | Replay MRR | README MRR | Delta | Replay Asym MRR | Replay Doc Cosine | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s7_control_bs32 | control | 0.2052 | 0.2050 | 0.0002 | — | — | ok |
| s7_embed_dw100_aw10 | embed_distill | 0.3026 | 0.3030 | -0.0004 | 0.3101 | 0.6794 | ok |
| s8_A2_bimga_uniform | bimga_uniform | 0.3133 | 0.3130 | 0.0003 | 0.3158 | 0.8560 | ok |
| s8_hnp_dw100_pw10 | hard_neg_pair | 0.3022 | 0.3020 | 0.0002 | 0.0072 | 0.0009 | ok |
| s9_score_dw100 | score_distill | 0.3008 | 0.3010 | -0.0002 | 0.0058 | -0.0002 | ok |
| s10_bimga_dw100_aw10 | bimga | 0.3248 | 0.3250 | -0.0002 | 0.3213 | 0.8810 | ok |


## Primary Claim Table

| Comparison | MRR Delta | Adj p | 95% CI | R@1 | R@10 | Margin | Doc Cosine | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bimga vs control | 0.1196 | 0.0001 | [0.0975, 0.1418] | significant_better | significant_better | significant_better | not_applicable | BiMGA is significantly better on retrieval; paired document-alignment testing is not applicable because the fine-tuned teacher targets were not published. |
| bimga vs embed_distill | 0.0222 | 0.0306 | [0.0074, 0.0370] | not_significant | not_significant | significant_better | not_applicable | BiMGA is significantly better on retrieval. Paired document-alignment testing is not applicable because the fine-tuned teacher targets were not published, but the HF aggregate doc cosine still favors BiMGA. |
| bimga vs bimga_uniform | 0.0115 | 0.2816 | [0.0002, 0.0227] | not_significant | not_significant | significant_better | not_applicable | BiMGA and the comparison method are not significantly different on retrieval, and the published aggregate doc-cosine gap is also small, so bidirectional alignment is supported but margin guidance remains unproven. |
| bimga vs hard_neg_pair | 0.0226 | 0.0444 | [0.0065, 0.0389] | not_significant | not_significant | significant_worse | not_applicable | BiMGA is significantly better on retrieval. Paired document-alignment testing is not applicable because the fine-tuned teacher targets were not published, but the HF aggregate doc cosine still favors BiMGA. |
| bimga vs score_distill | 0.0240 | 0.0306 | [0.0082, 0.0402] | not_significant | not_significant | significant_worse | not_applicable | BiMGA is significantly better on retrieval. Paired document-alignment testing is not applicable because the fine-tuned teacher targets were not published, but the HF aggregate doc cosine still favors BiMGA. |


## Symmetric Vs Asymmetric Within-Run Tests

| Run | Metric | Delta | Adj p | Status |
| --- | --- | --- | --- | --- |
| s7_control_bs32 | sym_vs_asym_rr | — | — | not_applicable |
| s7_control_bs32 | sym_vs_asym_correct_at_1 | — | — | not_applicable |
| s7_control_bs32 | sym_vs_asym_correct_at_10 | — | — | not_applicable |
| s7_embed_dw100_aw10 | sym_vs_asym_rr | — | — | not_applicable |
| s7_embed_dw100_aw10 | sym_vs_asym_correct_at_1 | — | — | not_applicable |
| s7_embed_dw100_aw10 | sym_vs_asym_correct_at_10 | — | — | not_applicable |
| s8_A2_bimga_uniform | sym_vs_asym_rr | — | — | not_applicable |
| s8_A2_bimga_uniform | sym_vs_asym_correct_at_1 | — | — | not_applicable |
| s8_A2_bimga_uniform | sym_vs_asym_correct_at_10 | — | — | not_applicable |
| s8_hnp_dw100_pw10 | sym_vs_asym_rr | — | — | not_applicable |
| s8_hnp_dw100_pw10 | sym_vs_asym_correct_at_1 | — | — | not_applicable |
| s8_hnp_dw100_pw10 | sym_vs_asym_correct_at_10 | — | — | not_applicable |
| s9_score_dw100 | sym_vs_asym_rr | — | — | not_applicable |
| s9_score_dw100 | sym_vs_asym_correct_at_1 | — | — | not_applicable |
| s9_score_dw100 | sym_vs_asym_correct_at_10 | — | — | not_applicable |
| s10_bimga_dw100_aw10 | sym_vs_asym_rr | — | — | not_applicable |
| s10_bimga_dw100_aw10 | sym_vs_asym_correct_at_1 | — | — | not_applicable |
| s10_bimga_dw100_aw10 | sym_vs_asym_correct_at_10 | — | — | not_applicable |


If an exact within-run symmetric-vs-asymmetric paired test is unavailable, that is a publication limitation rather than a model limitation: the public HF repos expose the aggregate asymmetric metrics, but not the query-level fine-tuned teacher targets required to test them properly.

## Interpretation

The report uses one fixed interpretation rule set:

- When `MRR` and `doc_cosine` are both significantly better in the same direction, the result supports a geometry-driven gain.
- When `margin` is significantly better but `MRR` is not, the result supports a local hardest-negative improvement without a matching global retrieval improvement.
- When `BiMGA` and `BiMGA-uniform` are not significantly different on both `MRR` and `doc_cosine`, the evidence supports bidirectional alignment but does not prove that margin guidance is necessary.
- When score-only or pairwise baselines approach the alignment methods on `MRR` but remain significantly worse on `doc_cosine`, that means they can mimic ranking behavior without actually reproducing teacher document geometry.

## All-Pairs Appendix

| Comparison | MRR Delta | Adj p | 95% CI | Status |
| --- | --- | --- | --- | --- |
| bimga vs bimga_uniform | 0.0115 | 0.2816 | [0.0002, 0.0227] | not_significant |
| bimga vs embed_distill | 0.0222 | 0.0306 | [0.0074, 0.0370] | significant_better |
| bimga vs hard_neg_pair | 0.0226 | 0.0444 | [0.0065, 0.0389] | significant_better |
| bimga vs score_distill | 0.0240 | 0.0306 | [0.0082, 0.0402] | significant_better |
| bimga vs control | 0.1196 | 0.0001 | [0.0975, 0.1418] | significant_better |
| bimga_uniform vs embed_distill | 0.0107 | 0.7738 | [-0.0035, 0.0246] | not_significant |
| bimga_uniform vs hard_neg_pair | 0.0111 | 0.7738 | [-0.0052, 0.0271] | not_significant |
| bimga_uniform vs score_distill | 0.0125 | 0.7738 | [-0.0033, 0.0285] | not_significant |
| bimga_uniform vs control | 0.1080 | 0.0001 | [0.0851, 0.1320] | significant_better |
| embed_distill vs hard_neg_pair | 0.0004 | 1.0000 | [-0.0148, 0.0157] | not_significant |
| embed_distill vs score_distill | 0.0018 | 1.0000 | [-0.0136, 0.0173] | not_significant |
| embed_distill vs control | 0.0974 | 0.0001 | [0.0761, 0.1186] | significant_better |
| hard_neg_pair vs score_distill | 0.0014 | 1.0000 | [-0.0097, 0.0125] | not_significant |
| hard_neg_pair vs control | 0.0970 | 0.0001 | [0.0770, 0.1175] | significant_better |
| score_distill vs control | 0.0956 | 0.0001 | [0.0761, 0.1154] | significant_better |


## Excluded Tests

The following tests are intentionally excluded from the evidence chain:

- unpaired t-tests on final scalar MRR values
- z-tests on aggregate metrics
- Pearson or Spearman significance across only six saturated runs
- seed-level significance claims for the saturated set

These are excluded because they either use the wrong unit of analysis, are underpowered, or are unavailable for the saturated checkpoints.
