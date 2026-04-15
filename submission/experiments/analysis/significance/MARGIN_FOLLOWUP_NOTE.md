# Margin Follow-up Note

This note adds query-level evidence to the mixed margin result discussed in the saturated significance package.

## What We Checked

We compared BiMGA against `score_distill`, `hard_neg_pair`, `embed_distill`, and `bimga_uniform` on the same 1000 test queries and asked four follow-up questions:

1. Does a method with better hardest-negative margin usually also win on reciprocal rank?
2. When ScoreDistill or PairDistill beats BiMGA on margin, how often does BiMGA still win on retrieval?
3. Are the mixed results just noise, or do they show a consistent disagreement pattern?
4. Does BiMGA trade a slightly weaker strict margin for a better global rank profile?

## Method-level evidence

From [`margin_followup_method_summary.csv`](./margin_followup_method_summary.csv):

- `bimga` has the best overall reciprocal rank: `0.3248`
- `score_distill` and `hard_neg_pair` have better mean hardest-negative margin than `bimga`:
  - `score_distill`: `-0.1120`
  - `hard_neg_pair`: `-0.1092`
  - `bimga`: `-0.1272`
- Even so, `bimga` has the best top-1 and top-10 retrieval rates:
  - `Recall@1 = 0.241`
  - `Recall@10 = 0.486`
- `bimga` also has fewer very-bad failures than `hard_neg_pair` and `score_distill` when we look at `rank >= 51`.

This supports the interpretation that the strict hardest-negative margin is not the same thing as best full-ranking behaviour.

## Pairwise evidence: BiMGA vs ScoreDistill

From [`margin_followup_pairwise.csv`](./margin_followup_pairwise.csv):

- `score_distill` has better margin on `557 / 1000` queries.
- But even on those queries, `bimga` still gets better reciprocal rank on `133` of them, with `86` ties.
- The margin-delta and reciprocal-rank-delta are related, but not identical:
  - Spearman correlation: `0.6633`
  - Pearson correlation: `0.5295`
- The non-tied agreement rate is `0.8000`, which means about `20%` of non-tied queries show disagreement between “who has the better margin” and “who has the better reciprocal rank.”

So margin matters, but it is not a perfect proxy for retrieval quality.

## Pairwise evidence: BiMGA vs PairDistill

- `hard_neg_pair` has better margin on `578 / 1000` queries.
- But even on those queries, `bimga` still gets better reciprocal rank on `158` of them, with `82` ties.
- Correlation is again only moderate:
  - Spearman: `0.6572`
  - Pearson: `0.5198`
- The disagreement rate is `0.2208` on non-tied queries.

This is the clearest evidence that strong local hardest-negative shaping does not automatically produce the best overall ranking.

## Why this strengthens the interpretation

The mixed result is now more grounded:

- Margin and retrieval are clearly related.
- But they are not interchangeable.
- ScoreDistill and PairDistill are better at the strict local hardest-negative metric.
- BiMGA is still better at the overall ranking objective that MRR measures.

That is exactly the pattern we would expect if BiMGA's main advantage comes from improving the broader geometry of the retrieval space rather than only sharpening the single hardest local comparison.

## What would prove it further

The strongest next analyses would be:

1. **Top-k negative margin**
   Replace the single hardest negative with the average of the top-k negatives. If BiMGA improves global ranking structure, it may look better on a less brittle top-k margin than on strict top-1 hardest-negative margin.

2. **Whole-ranking diagnostics**
   Evaluate diagnostics that depend on more of the ranked list, not just the top wrong code. For example: average positive-vs-top-10 negative gap, rank-bucket transitions, or cumulative gain contributions.

3. **Teacher-space paired tests**
   If the fine-tuned teacher targets become available, run paired significance on per-query document cosine and exact symmetric-vs-asymmetric differences. That would directly test the geometry hypothesis rather than inferring it from published aggregates.

4. **Category-conditioned margin analysis**
   Repeat this same comparison by problem type or difficulty tier. It is possible that hard-negative margin helps most on one category, while BiMGA's geometry gains help more broadly.
