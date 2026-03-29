# BiMGA: Bidirectional Margin-Guided Alignment for Retrieval Knowledge Distillation

## Background: Why Existing Methods Only Align Queries

Embedding-alignment KD methods for retrieval (EmbedDistill [1], QED [2], LEAF [3]) were designed for **asymmetric deployment**: the teacher's document encoder is frozen and shared at inference time, so the student only needs a small query encoder.

```
Asymmetric architecture (EmbedDistill, QED):
  Query:  student_q_encoder(query)  →  small, trainable, aligned to teacher
  Doc:    teacher_d_encoder(doc)    →  large, frozen, reused from teacher
```

In this setup, document alignment is unnecessary — the doc side *is* the teacher. This design makes sense for web search where you can pre-compute document embeddings offline with the large model and only need the small model for real-time queries.

## The Gap: Symmetric Bi-Encoders

In our setting (and many practical deployments), we need a **fully self-contained small model** that encodes both queries and documents at inference time:

```
Symmetric architecture (our setup):
  Query:  student_q_encoder(query)  →  trainable, aligned to teacher  ✓
  Doc:    student_d_encoder(doc)    →  trainable, NO teacher signal   ✗
```

The student's document encoder learns only through the contrastive loss (`one_hot`), which provides indirect gradient signal via `student_q @ student_d.T`. It never receives direct supervision on *where* document embeddings should live in the embedding space. Yet at evaluation time, it is the student's document embeddings (not the teacher's) that determine retrieval quality:

```python
# Evaluation — student encodes BOTH sides
query_embs = student.encode(queries)
code_embs  = student.encode(codes)     # ← no teacher supervision during training
scores     = query_embs @ code_embs.T  # ← this determines MRR, Recall@k
```

The teacher has learned a rich representation of code structure, but existing methods discard this knowledge entirely for the document side.

## Motivation from Experiments

Our hyperparameter sweep (TinyBERT-4L student, MiniLM-L6 teacher, TACO dataset) confirmed that embedding alignment is the single most impactful KD signal:

| Setting | Best method | MRR | vs control |
|---------|------------|----:|----:|
| No KD | control_supervised | 0.1983 | — |
| Score KD only (dw=100) | score_distill | 0.2279 | +0.030 |
| Query align (aw=10, dw=50) | embed_distill | 0.2513 | +0.053 |

Query alignment alone gives +0.053 MRR — nearly double the impact of score KD. Yet half the student's parameters (the document encoder) receive none of this alignment signal. BiMGA closes this gap.

## When Is Document Alignment Useful?

The asymmetric approach (freeze teacher docs, only train student queries) is a valid design choice. Whether document alignment adds value depends on the deployment scenario:

**Against — teacher can pre-compute documents:**
- In web search, documents are indexed offline — run the large teacher once, cache embeddings
- Only queries arrive in real-time, so only the query encoder needs to be small/fast
- Reusing teacher doc embeddings gives the best possible doc representations
- This is the dominant paradigm in production search (EmbedDistill [1], QED [2], LEAF [3])

**For — symmetric bi-encoder is needed:**
- **Dynamic corpora**: new code/documents arrive continuously (e.g. new repositories, API updates) — re-running a large teacher on every change is expensive
- **Edge/on-device deployment**: cannot store a large teacher model or its precomputed embeddings
- **Resource constraints**: teacher inference cost scales linearly with corpus size — impractical for large or frequently updated collections
- **Deployment simplicity**: one small self-contained model vs maintaining separate teacher (docs) and student (queries) systems
- **Our evaluation setting**: the student encodes both sides at test time, so document embedding quality directly determines MRR and Recall

In the code search setting specifically, codebases change frequently and users may want local/offline search. A symmetric bi-encoder that handles both sides well is the practical target.

## Method

BiMGA extends embedding alignment in two ways:

### 1. Bidirectional Alignment

We align both the student query and document embeddings to their teacher counterparts:

```
L_align = (1/B) * sum_i [ w_i * (||q_s_i - q_t_i||_2 + ||d_s_i - d_t_i||_2) ]
```

where `q_s, d_s` are student embeddings and `q_t, d_t` are finetuned teacher embeddings.

EmbedDistill [1] defines a document alignment loss (Eq. 8) for the cross-encoder → bi-encoder setting, where the teacher is a cross-encoder that lacks separate doc embeddings and requires dual-pooling to extract them. However, in their bi-encoder → bi-encoder setting (Fig. 1a), they freeze the teacher's document encoder and reuse it directly as the student's doc encoder — doc alignment is never activated. QED [2] and LEAF [3] are explicitly query-only. CLIP-KD [4] aligns both towers but in the vision-language domain, not text retrieval.

BiMGA's contribution on this axis is applying doc alignment in the **symmetric bi-encoder setting** where the student must encode documents itself. This is a straightforward extension of EmbedDistill's Eq. 8 to a scenario they did not evaluate — the novelty here is in the application, not the formulation.

### 2. Margin-Guided Weighting

Not all examples benefit equally from alignment. We weight each example by the teacher's ranking confidence, measured as the margin between the positive score and the hardest in-batch negative:

```
m_i = s_t(q_i, d_i+) - max_{j != i} s_t(q_i, d_j)

w_i = sigmoid(m_i / tau)
```

where `tau` is the distillation temperature and `sigmoid` maps the margin to [0, 1].

**Intuition:** When the teacher is confident (large margin), the alignment signal is reliable and should be enforced strongly. When the teacher is uncertain (small margin), forcing alignment risks copying the teacher's mistakes.

This differs from existing confidence-weighted approaches:
- ADAM [5] weights a **score-level KL loss** by teacher confidence, not an embedding alignment loss.
- Margin-MSE [6] uses margins as the **distillation target** (MSE between student and teacher margins), not as a **weight** on alignment.

### 3. Full Loss

BiMGA combines bidirectional margin-guided alignment with standard contrastive and score KD:

```
L_total = L_one_hot + dw * L_distill_kl + aw * L_bimga_align
```

where `dw` (`distill_weight`) and `aw` (`align_weight`) control the contribution of each component.

## Relation to Existing Methods

### Architecture and Signal Comparison

| Method | Architecture | Query align | Doc align | Weighting | Score KD |
|--------|:------------:|:-----------:|:---------:|-----------|:--------:|
| EmbedDistill [1] | asymmetric (frozen docs) | L2 | N/A (frozen) | uniform | KL |
| QED [2] | asymmetric (frozen docs) | L2 | N/A (frozen) | uniform | -- |
| LEAF [3] | asymmetric (frozen docs) | L2 | N/A (frozen) | uniform | -- |
| ADAM [5] | symmetric | -- | -- | confidence on score KD | KL |
| Margin-MSE [6] | symmetric | -- | -- | margin as target | MSE |
| CLIP-KD [4] | symmetric (vision-language) | MSE | MSE | uniform | contrastive |
| **BiMGA (ours)** | **symmetric** | **L2** | **L2** | **margin-guided** | **KL** |

Key distinction: EmbedDistill, QED, and LEAF were designed for asymmetric deployment where the teacher's document encoder is reused directly. In our symmetric bi-encoder setting, the student must encode documents itself, making document alignment both possible and necessary. CLIP-KD [4] aligns both towers but in the vision-language domain and without confidence weighting.

### Loss Equations and Training Effects

There are two fundamentally different KD signal types: **score-based** (match the teacher's ranking) and **embedding-based** (match the teacher's vector space). Each method uses one or both.

#### Score-based KD methods

These tell the student *which documents are more relevant than others*, but not *where to place embeddings*. The student can satisfy the loss with completely different embedding geometry as long as relative rankings are preserved.

| Method | Loss equation | Training effect |
|--------|--------------|-----------------|
| **score_distill** | `KL(softmax(S_s / tau) \|\| softmax(S_t / tau)) * tau^2` | Matches full similarity matrix distribution. Scaled by T^2, so very small for cosine similarities — needs high `distill_weight`. |
| **margin_mse** | `(1/N) * sum_{i, j!=i} (m_s_ij - m_t_ij)^2` where `m_ij = s_ii - s_ij` | Matches pairwise score *margins* (how much better is the positive vs each negative). Not affected by T^2 scaling. Uses `pair_weight`. |
| **hard_neg_pair_distill** | `KL + BCE(sigmoid((s_s+ - s_s-) / tau), sigmoid((s_t+ - s_t-) / tau))` | Score KL + binary preference on teacher's top-k hardest negatives. Focuses KD signal on the most confusable documents. Note: paper [4] uses `KL(P_pair_teacher \|\| P_pair_student)` over reranker pairs; our implementation approximates this with BCE on in-batch margins. |
| **all_pairs_distill** | `KL(softmax[s_t+, s_t_j] \|\| softmax[s_s+, s_s_j])` for all `j != i` | Binary KL on every (positive, negative) pair independently. Paper [4] operates on top-k reranked docs; our adaptation uses in-batch negatives. |
| **adam_lite** | `KL_dark * sigmoid(m_t / tau)` on synthetic "dark" docs | Creates hard negatives by mixing positive + negative doc embeddings. Weights KD by teacher confidence. Uses `distill_weight`. |
| **pointwise** | `MSE(S_s, S_t)` | Matches absolute scores, not just rankings. Simple but doesn't account for score distribution shape. |

#### Embedding-based KD methods

These tell the student *exactly where to place each embedding* in vector space. A stronger signal than score KD because it transfers the teacher's entire geometric structure, not just pairwise rankings. Each example gets a direct target independently of batch composition.

| Method | Loss equation | Training effect |
|--------|--------------|-----------------|
| **qed_align** | `(1/B) * sum_i \|\|q_s_i - q_t_i\|\|_2` | Pulls student query embeddings toward teacher query space. Doc encoder learns only through contrastive loss. Uses `align_weight`. |
| **embed_distill** | `KL(S_s \|\| S_t) + \|\|q_s - q_t\|\|_2` | Combines score KD with query alignment. Best existing method because it gets both signals. Uses `distill_weight` + `align_weight`. |
| **distilcse_lite** | `-log(exp(sim(q_s, q_t) / tau) / sum_j exp(sim(q_s, q_t_j) / tau))` | InfoNCE on student-teacher query pairs: student query should be closest to its own teacher query vs all other teacher queries in the batch. Uses `relation_weight`. |
| **hpd** | `(1/B) * sum_i \|\|q_s_i - PCA(q_t_i)\|\|_2^2` | MSE alignment in PCA-compressed teacher space. Reduces dimensionality mismatch but loses information from discarded PCA components. Uses `align_weight`. |
| **BiMGA (ours)** | `(1/B) * sum_i sigmoid(m_i/tau) * (\|\|q_s_i - q_t_i\|\|_2 + \|\|d_s_i - d_t_i\|\|_2) + KL` | **Bidirectional** alignment (query + doc) weighted by teacher margin confidence. High-confidence examples get stronger alignment; uncertain examples are relaxed. Uses `align_weight` + `distill_weight`. |

#### Why embedding alignment outperforms score KD in practice

With `batch_size=32`, score KD sees only 32 documents per query — a narrow view of the ranking landscape. Embedding alignment provides a direct vector target for every example independently of batch composition. In our experiments, alignment alone (+0.053 MRR) nearly doubled the gain of score KD alone (+0.030 MRR).

## Implementation

BiMGA is implemented as a new method `bimga` in `training.py`. It uses the existing `align_weight` parameter to control bidirectional alignment strength and `distill_weight` for the score KD component. Recommended settings based on our experiments:

```bash
uv run mbpp-kd-two-phase \
  --distill-temperature 0.2 \
  --distill-weight 50 \
  --align-weight 10 \
  --methods bimga
```

## Experimental Context

Setup: TinyBERT-4L (14M params, 312d) distilled from MiniLM-L6-v2 (384d) on TACO code search (18,493 train, 1,000 test).

### Hyperparameter Findings

Two critical insights that informed BiMGA's design:

**1. Loss weight scaling is essential.** The KD score loss (`distill_kl`) is scaled by T^2 internally. With cosine similarities in [-1, 1] and `distill_temperature=0.2`, T^2=0.04 shrinks the raw KL from ~0.19 to ~0.007 — negligible vs the supervised loss (~0.3). Setting `distill_weight=50` compensates.

**2. Embedding alignment dominates score KD.** Query alignment (`align_weight=10`) gives +0.053 MRR, nearly double the gain from score KD alone (`distill_weight=100`, +0.030 MRR). This motivated extending alignment to the document side.

### Parameter Impact Ranking

| Rank | Parameter | Impact | Notes |
|:----:|-----------|--------|-------|
| 1 | `align_weight` | +0.053 MRR at aw=10 | Strongest single lever |
| 2 | `distill_weight` | +0.036 MRR at dw=100 | Needs high values due to T^2 scaling |
| 3 | `distill_temperature` | dt=0.2 optimal | For cosine similarities in [-1, 1] |
| 4 | `batch_size` | +0.012 MRR at bs=128 | Modest gains from more negatives |
| 5 | `lr` | No improvement | Default 2e-5 is fine |

## Results: BiMGA vs embed_distill Head-to-Head

All experiments: TinyBERT-4L student (14M params, 312d), MiniLM-L6-v2 teacher (384d), TACO dataset (18,493 train, 1,000 test). Control (supervised, no KD): MRR=0.1983.

### Direct Comparison

| Config | BiMGA | embed_distill | Winner | Margin |
|--------|------:|------:|--------|-------:|
| dw=50, aw=1 | 0.2253 (+0.027) | 0.2238 (+0.025) | BiMGA | +0.0015 |
| dw=50, aw=5 | 0.2408 (+0.042) | 0.2352 (+0.037) | BiMGA | +0.0056 |
| dw=50, aw=10 | 0.2527 (+0.054) | 0.2487 (+0.050) | BiMGA | +0.0040 |
| dw=100, aw=1 | 0.2320 (+0.034) | 0.2305 (+0.032) | BiMGA | +0.0015 |
| dw=100, aw=10 | 0.2473 (+0.049) | 0.2584 (+0.060) | embed_distill | -0.0111 |

BiMGA wins in **4 of 5 settings**. The advantage is most pronounced at moderate weights (aw=5-10, dw=50), where document alignment provides a complementary signal without overwhelming the loss.

At the highest combined weights (aw=10, dw=100), embed_distill overtakes BiMGA. The margin-guided weighting may suppress alignment on uncertain examples that would still benefit from uniform alignment at very high weight scales. This suggests an interaction between confidence weighting and loss magnitude that merits further investigation.

### Best Results Across All Methods

| Method | Best config | MRR | vs control |
|--------|------------|----:|----:|
| embed_distill | aw=10, dw=100 | **0.2584** | +0.0600 |
| **BiMGA** | aw=10, dw=50 | **0.2527** | +0.0544 |
| hpd | aw=10, dw=50 | 0.2431 | +0.0448 |
| adam_lite | dw=100 | 0.2342 | +0.0358 |
| score_distill | dw=100 | 0.2279 | +0.0296 |
| qed_align | aw=10, dw=50 | 0.2181 | +0.0197 |
| control_supervised | — | 0.1983 | — |

### Analysis

1. **Document alignment helps at moderate weights.** BiMGA consistently outperforms embed_distill when `align_weight <= 10` and `distill_weight <= 50`, confirming that the student's document encoder benefits from direct teacher supervision.

2. **Margin weighting has diminishing returns at high loss scales.** At aw=10, dw=100, the margin-guided sigmoid weights restrict alignment on low-confidence examples. With very high `align_weight`, even uncertain teacher embeddings provide useful signal — uniform alignment (embed_distill) wins. A future variant could anneal from margin-guided to uniform weighting as training progresses.

3. **Alignment methods dominate score-only methods.** The top 3 methods all use embedding alignment. Score-only methods (score_distill, adam_lite) plateau around +0.03-0.04 MRR regardless of `distill_weight`.

## References

[1] Srinivasan et al., "EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval," AAAI 2023. arXiv:2301.12005

[2] Wang & Hong, "Query Encoder Distillation via Embedding Alignment," SustaiNLP @ ACL 2023. arXiv:2306.11550

[3] MongoDB Research, "LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations," 2025. arXiv:2509.12539

[4] Yang et al., "CLIP-KD: An Empirical Study of CLIP Model Distillation," CVPR 2024.

[5] Zhan et al., "ADAM: Dense Retrieval Distillation with Adaptive Dark Examples," Findings of ACL 2024. arXiv:2212.10192

[6] Hofstatter et al., "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling," SIGIR 2021. arXiv:2104.06967
