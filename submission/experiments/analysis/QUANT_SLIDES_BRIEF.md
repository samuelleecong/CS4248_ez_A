# Quantitative Analysis Slide Brief

This brief packages the replayed HF results into a 4-slide quantitative story for the presentation.

## Slide 1: Quantitative Setup

### On-slide text
- 31 HF-uploaded TACO checkpoints replayed on fixed 1000-query TACO test split
- Symmetric retrieval = student encodes both query and code
- Main metrics: `MRR`, `Recall@1`, `Recall@10`
- Analysis targets:
  - document embedding quality
  - score separation against hardest negatives

### Visual
- Small table or callout from [method_summary.csv](./csv/method_summary.csv)
- Optional setup reminder from [fig01_core_sweep_mrr.png](./figures/fig01_core_sweep_mrr.png)

### Speaker notes
We are not using training logs as the main evidence here. Everything in this section comes from replaying the actual uploaded checkpoints on the same fixed TACO test split. That matters because it makes the quantitative section checkpoint-verified instead of report-note-only.

## Slide 2: Document Embedding Quality Explains Symmetric Retrieval

### On-slide text
- Document cosine vs symmetric MRR: `r = 0.814`
- Document cosine vs symmetric-asymmetric gap: `r = -0.960`
- Best representative methods:
  - `bimga_uniform`: MRR `0.2978`, doc cosine `0.2873`
  - `bimga`: MRR `0.2973`, doc cosine `0.2845`
  - `embed_distill`: MRR `0.2818`, doc cosine `0.2324`

### Visual
- [fig02_doc_cosine_vs_sym_mrr.png](./figures/fig02_doc_cosine_vs_sym_mrr.png)
- [fig03_doc_cosine_vs_sym_asym_gap.png](./figures/fig03_doc_cosine_vs_sym_asym_gap.png)

### Speaker notes
This is the most important quantitative result. Once the student has to encode documents itself, document quality becomes a first-class bottleneck. The stronger the document alignment, the better the symmetric retrieval quality and the smaller the penalty from moving away from teacher-encoded documents.

## Slide 3: Score Separation Helps, But Does Not Fully Explain the Winner

### On-slide text
- Strict hardest-negative margin is still negative for all methods
- `hard_neg_pair` gets best average margin:
  - mean margin `-0.1246`
  - median margin `-0.1223`
- But `bimga` still wins on retrieval:
  - MRR `0.2973` vs `0.2683`
  - negative-margin rate `0.789` vs `0.820`

### Visual
- [fig04_margin_distribution_best_methods.png](./figures/fig04_margin_distribution_best_methods.png)
- [fig05_margin_summary_best_methods.png](./figures/fig05_margin_summary_best_methods.png)
- Optional mention of [fig08_teacher_margin_vs_student_margin.png](./figures/fig08_teacher_margin_vs_student_margin.png)

### Speaker notes
This is the nuance slide. If we only looked at mean hardest-negative margin, we might incorrectly say hard-negative pairwise KD is strongest. It is not. It sharpens local separation, but it does not create the best symmetric retriever. BiMGA still wins because document alignment matters more than margin shaping alone.

## Slide 4: Final Quantitative Takeaway

### On-slide text
- Multi-seed mean test MRR:
  - `bimga`: `0.3008 ± 0.0051`
  - `embed_distill`: `0.2783 ± 0.0074`
  - `hard_neg_pair`: `0.2628 ± 0.0046`
  - `score_distill`: `0.2584 ± 0.0073`
- Best explanation of the final ranking:
  - document alignment is the main driver
  - score separation is secondary

### Visual
- [fig07_seed_stability.png](./figures/fig07_seed_stability.png)
- Optional support: [fig06_training_curves_best_runs.png](./figures/fig06_training_curves_best_runs.png)

### Speaker notes
The final claim should be modest and clean. We are not saying score separation is irrelevant. We are saying it is not the primary reason the best method wins. The most defensible conclusion from the replayed checkpoints is that BiMGA is strongest because it improves document embeddings enough to support symmetric retrieval, and that result remains stable across seeds.
