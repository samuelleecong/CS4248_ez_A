# Quantitative Analysis

We replayed all 31 uploaded TACO sweep checkpoints from the `cs4248-nlp` Hugging Face organization on the fixed 1000-query TACO test split. All reported values in this section come from the replay outputs in [run_metrics.csv](./csv/run_metrics.csv), [run_diagnostics.csv](./csv/run_diagnostics.csv), [method_summary.csv](./csv/method_summary.csv), and [seed_summary.csv](./csv/seed_summary.csv).

Our two quantitative questions were: `(1)` whether better document embedding quality explains stronger symmetric retrieval, and `(2)` whether stronger KD methods create cleaner positive-vs-hardest-negative score separation.

## Document Embedding Quality

Document embedding quality is measured as the mean cosine similarity between the student document embeddings and the teacher document embeddings on the TACO test set. This metric is strongly associated with final symmetric retrieval quality. Across replay-compatible runs, the correlation between document cosine and symmetric test MRR is `r = 0.814` ([Figure 2](./figures/fig02_doc_cosine_vs_sym_mrr.png)). The same trend appears in the representative subset: `score_distill` has document cosine `0.0062` and symmetric test MRR `0.2664`, `embed_distill` improves to document cosine `0.2324` and MRR `0.2818`, while `bimga` reaches document cosine `0.2845` and MRR `0.2973`. The best representative run by symmetric MRR is `s3_A2_bimga_uniform`, which achieves MRR `0.2978` and the highest document cosine in the subset at `0.2873`.

The symmetric-versus-asymmetric comparison reinforces the same conclusion. The correlation between document cosine and the symmetric-minus-asymmetric MRR gap is `r = -0.960` ([Figure 3](./figures/fig03_doc_cosine_vs_sym_asym_gap.png)). Methods with better document alignment lose much less when the student must encode documents itself. For example, `s1_embed_dw100_aw10` has a symmetric-asymmetric gap of `0.1390`, while `s1_score_dw100` has a much larger gap of `0.2593`. These results support the main symmetric-retrieval hypothesis: methods that align only the query side are not enough, because document representation quality remains a major bottleneck.

## Score Separation

For score separation, we define a per-query margin as `score(query, positive) - max(score(query, hardest negative))` over the full TACO candidate pool. This is a strict statistic, and the average margin remains negative for all representative methods because each positive is compared against the single strongest negative in the entire retrieval set. The meaningful comparison is therefore which method is `less negative` and which method reduces the fraction of negative-margin queries.

Under this definition, `hard_neg_pair` gives the best average hardest-negative separation: in the representative subset, `s1_hnp_dw100_pw10` has mean margin `-0.1246` and median margin `-0.1223`, better than `score_distill` (`-0.1369`) and `embed_distill` (`-0.1465`) ([Figure 4](./figures/fig04_margin_distribution_best_methods.png), [Figure 5](./figures/fig05_margin_summary_best_methods.png)). However, this does not make it the strongest retrieval model. `bimga` still performs better on symmetric MRR (`0.2973` vs `0.2683`) and also achieves a lower negative-margin rate (`0.789` vs `0.820`). This means score separation is informative, but not sufficient on its own to explain the final method ranking. Hard-negative pairwise training sharpens local ranking pressure, but without strong document alignment it does not produce the best symmetric retriever.

Teacher-student margin tracking shows the same pattern. `s1_hnp_dw100_pw10` has the strongest teacher-student margin correlation among the representative runs at `r = 0.562` ([Figure 8](./figures/fig08_teacher_margin_vs_student_margin.png)), which suggests that it learns the teacher’s local ranking structure well. Even so, BiMGA remains stronger overall because it combines useful ranking information with much better document-space alignment.

## Robustness

The method ranking is stable across seeds. In the seed summary, `bimga` has the highest mean test MRR at `0.3008 ± 0.0051`, ahead of `embed_distill` (`0.2783 ± 0.0074`), `hard_neg_pair` (`0.2628 ± 0.0046`), and `score_distill` (`0.2584 ± 0.0073`) ([Figure 7](./figures/fig07_seed_stability.png)). It also has the highest and most stable document cosine, `0.2854 ± 0.0007`. The representative training curves show that these improvements are not just lucky endpoints; `s3_A2_bimga_uniform` has the strongest final validation MRR in the representative subset at `0.4838` ([Figure 6](./figures/fig06_training_curves_best_runs.png)).

## Quantitative Takeaway

Taken together, these results show that document embedding quality is the strongest quantitative explanation for why BiMGA performs best in symmetric TACO retrieval. Score separation does matter, but it does not by itself determine the final method ranking. The strongest overall methods are the ones that improve document alignment most effectively while still maintaining useful ranking behavior. This supports the core claim that bidirectional alignment is the main reason BiMGA is strongest in the symmetric code-search setting.
