# CS4248 IU Slides (Template-Aligned, Slide-Fit Version)

Use this as concise, paste-ready content for `assignment_details/cs4248-iu-template.pdf`.
Keep each slide to <= 6 bullets and <= 12 words per bullet where possible.

---

## Slide 1 - Title

`Team XX - Intermediate Update: Distilled Embeddings for Efficient MBPP Code Search`

- Members: `<Name 1>`, `<Name 2>`, `<Name 3>`, `<Name 4>`
- Emails: `<e0xxxxxxx@u.nus.edu>`, `<e0yyyyyyy@u.nus.edu>`, `<e0zzzzzzz@u.nus.edu>`, `<e0wwwwwww@u.nus.edu>`
- Mentor: `<Mentor Name> <email@u.nus.edu>`
- Date: `5 March 2026`

---

## Slide 2 - Abstract

Goal summary (1-2 sentences):
- We build an MBPP text-to-code retriever that ranks Python snippets for natural-language queries.
- We distill a strong teacher into a smaller student to reduce latency with minimal quality loss.

Sprint progress (2-4 sentences):
- Baseline and teacher fine-tuning are complete with reproducible evaluation on 500 MBPP test queries.
- Teacher MRR improved from `0.587` to `0.731`; Recall@1 improved from `0.458` to `0.620`.
- Next sprint: run four parallel lanes for KD baseline, ranking KD, hard negatives, and unified benchmarking.

---

## Slide 3 - Motivation

- Fast, accurate code retrieval supports assistants, RAG, and developer search.
- Large embedding models are strong but expensive at inference time.
- Distillation can preserve quality while improving deployment efficiency.
- MBPP offers realistic query-code pairs and quick experiment cycles.

Suggested visual:
- Ready chart: `assignment_details/visuals/iu/s03_quality_efficiency_tradeoff.png`
- Chart note: MRR and Recall@1 vs estimated latency index (teacher=1.0).

---

## Slide 4 - Task Statement

- Input: MBPP natural-language problem statement (`text`).
- Output: ranked MBPP Python candidates (`code`) by semantic relevance.
- Basic deliverable: compact student retriever with quality retention and speedup.
- Optional extension: use retrieved examples for RAG generation and measure pass@k change.

---

## Slide 5 - Proposed Method

Pipeline:
1. Fine-tune teacher retriever on MBPP query-code pairs. (Done)
2. Produce teacher embeddings/similarity targets. (Done)
3. Train student with distillation objectives. (In progress)
4. Compare quality, latency, throughput, memory. (In progress)
5. Select best model and freeze final config. (Planned)

Model choices:
- Teacher: `microsoft/unixcoder-base` (125M).
- Students: `all-MiniLM-L6-v2` (22M), `gte-small` (33M).

---

## Slide 6 - Progress (1/2): Results + Member Work

Current retrieval results (MBPP test, n=500):

| Model | MRR | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| UniXcoder (pretrained) | 0.587 | 0.458 | 0.742 | 0.814 |
| UniXcoder (fine-tuned) | 0.731 | 0.620 | 0.866 | 0.912 |
| Delta | +0.144 | +0.162 | +0.124 | +0.098 |

Completed work by member:
- `<Member A>`: MBPP preprocessing + split/eval pipeline.
- `<Member B>`: baseline dual-encoder retrieval + metric scripts.
- `<Member C>`: teacher contrastive fine-tuning + checkpoint tracking.
- `<Member D>`: error-case analysis + experiment documentation.

Team self-assessment:
- Current status: `Satisfactory`, trending to `Excellent` if KD retention + speed targets are met.

---

## Slide 7 - Progress (2/2): Insights, Risks, Open Questions

Technical insights:
- Fine-tuning on 374 train pairs still produced strong top-1 gains.
- Common misses are near-duplicate utilities with wrong intent.

Challenges and hazards:
- Small training set can overfit quickly.
- KD loss balance can hurt ranking if alignment dominates.
- Multiple ablations may become compute/time bottlenecks.

Open questions for next sprint:
- Which loss mix best improves Recall@1 without hurting Recall@10?
- Do hard negatives improve validation generalization or overfit near-duplicates?

---

## Slide 8 - Proposed Evaluation

Quality metrics:
- `MRR`, `Recall@1`, `Recall@5`, `Recall@10`.

Efficiency metrics:
- Mean latency (ms/query), throughput (queries/s), model size, peak memory.

KD-specific metrics:
- Quality retention: `MRR_student / MRR_teacher`.
- Speedup: `latency_teacher / latency_student`.

Outcome criteria:
- Satisfactory target: `>=90%` teacher MRR retention and `>=2x` speedup.
- Excellent target: `>=95%` retention and `>=3x` speedup, with ablation evidence.

Evaluation protocol:
- Tune on validation only; lock config; report final test once.
- Justification grounded in retrieval distillation literature ([5], [6]).

---

## Slide 9 - Resources

Data / corpora:
- MBPP (`google-research-datasets/mbpp`) for query-code retrieval.

Compute:
- Colab or local GPU (`>=8GB VRAM`) for teacher/student training.

Software:
- Python, PyTorch, Transformers, Sentence-Transformers, scikit-learn.
- Team codebase includes reproducible scripts, configs, and result logs.

Human resources:
- 4-member team for parallel experiment lanes + mentor review.

---

## Slide 10 - Schedule and Role Assignment

Note for final slide formatting:
- Grey out past rows and mark whether goals were met.
- Keep owner continuity with Slide 6 (each member extends their completed lane).

| Date (2026) | Milestone | Owner | Status |
|---|---|---|---|
| Feb 24 - Mar 4 | MBPP preprocess + baseline + teacher FT + eval | `<Member A/B/C/D>` | Met |
| Mar 5 - Mar 8 | KD setup: teacher target export + configs + run tracking | `<Member C>` | In progress |
| Mar 5 - Mar 10 | Student baseline KD (`L_align + L_contrastive`) on MiniLM/GTE | `<Member B>` | In progress |
| Mar 5 - Mar 10 | Hard-negative pool + failure-case refresh | `<Member D>` | In progress |
| Mar 5 - Mar 11 | Latency/throughput/memory benchmark harness | `<Member A>` | In progress |
| Mar 11 - Mar 14 | Ranking-aware KD (`L_KL/rank`) + loss-mix sweep | `<Member B/C>` | Planned |
| Mar 15 - Mar 19 | Joint ablation review + checkpoint freeze (val only) | `<All>` | Planned |
| Mar 20 - Apr 6 | Final test run + analysis + report/slides | `<All>` | Planned |

Internal team status:
- No teammate dropout/unresponsiveness observed as of `4 March 2026`.

---

## Slide 11 - Acknowledgements

- CS4248 instructors and TAs for project feedback.
- Open-source contributors (HuggingFace and Sentence-Transformers).

---

## Slide 12 - References (Harvard-Style Strings, Square Brackets)

[1] Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q. and Sutton, C. (2021) *Program Synthesis with Large Language Models*. arXiv:2108.07732.

[2] Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., Zhou, M. and Yin, M. (2020) *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*. arXiv:2002.08155.

[3] Guo, D., Lu, S., Duan, N., Wang, Y., Zhou, M. and Yin, J. (2022) *UniXcoder: Unified Cross-Modal Pre-training for Code Representation*. arXiv:2203.03850.

[4] Husain, H., Wu, H.-H., Gazit, T., Allamanis, M. and Brockschmidt, M. (2019) *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*. arXiv:1909.09436.

[5] Pan, X., Liu, H., He, J. and Chen, X. (2023) *EmbedDistill: Geometric Knowledge Distillation for Information Retrieval*. arXiv:2301.12005.

[6] Reimers, N. and Gurevych, I. (2019) *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.

[7] Zhang, Y., Gao, T., Choi, J., Sennrich, R. and Awadalla, H. (2023) *DistilCSE: Bootstrapping Dense Retrieval with Contrastive Distillation*. arXiv:2112.05638.

---

## Optional Backup Slide - Distillation Design Decision

| Option | Loss | Benefit | Risk | Use |
|---|---|---|---|---|
| A. Alignment baseline | `L_align + L_contrastive` | Stable start | weaker ranking fidelity | First |
| B. Ranking-aware KD | `L_align + L_KL` | better top-k ranking | more tuning | Second |
| C. Hybrid | `L_align + L_KL + L_contrastive` | best final candidate | widest ablation space | Final |

One-line decision:
- Implement A first, then adopt C if validation MRR improves consistently.
