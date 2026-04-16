# EmbedDistill: Paper Deconstruction & Implementation Verification

**Paper**: Kim et al. "EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval" (arXiv:2301.12005v2, Jul 2023)
**Implementation**: `mbpp_kd_suite/src/mbpp_kd_suite/` (method name: `embed_distill`)

---

## 1. Paper Summary (Section-by-Section)

### 1.1 Core Thesis (Sec. 1, Sec. 3)

Most IR distillation methods only match **teacher scores** (query-document relevance scores).
EmbedDistill goes further by directly aligning the **embedding spaces** of teacher and student.

Theoretical motivation (Theorem 3.1, Eq. 4): the teacher-student generalization gap is bounded by three controllable terms:
1. Uniform deviation of empirical distillation risk (reduced by standard distillation)
2. **Query embedding misalignment** R_Emb,Q — reduced by embedding matching (Eq. 7)
3. **Document embedding misalignment** R_Emb,D — reduced by inheriting teacher documents (Eq. 8) or document embedding matching

### 1.2 Architecture: Asymmetric Dual-Encoder (Sec. 4.1, Fig. 1a)

The paper proposes an **asymmetric DE configuration**:
- **Student query encoder**: small trainable model (e.g., BERT-mini, DistilBERT)
- **Document encoder**: **inherited from teacher** (frozen, not trained)
- The student only trains its query encoder; documents are encoded by the teacher offline

Key benefit: no increase in inference latency (only query encoding is online), while using the teacher's high-quality document index.

### 1.3 Loss Functions (Sec. 4.1, Sec. 5.2, Appendix A)

EmbedDistill uses three losses:

#### Loss 1: One-Hot Cross-Entropy (Eq. 9 in Appendix A)
Standard contrastive loss with in-batch negatives:
```
L_one_hot = CrossEntropy(s(q_i, d_j) / τ, labels=diagonal)
```
Where s(q,d) = <emb_q, emb_d> (dot product of L2-normalized embeddings).

#### Loss 2: Score-Based Distillation / KL Divergence (Eq. 11 in Appendix A)
Matches the teacher's score distribution over documents:
```
L_distill = KL(softmax(student_scores / τ) || softmax(teacher_scores / τ))
```
This is the standard score-matching distillation from prior work.

#### Loss 3: Query Embedding Matching (Eq. 7)
The **novel contribution** — directly aligns student and teacher query embeddings:
```
R_Emb,Q = (1/n) Σ ||emb_q^t - proj(emb_q^s)||
```
L2 distance between teacher and student query embeddings (after optional projection).

#### Combined Objective (Sec. 5.2)
```
L_total = L_one_hot + λ_distill * L_distill + λ_align * L_align
```
The paper says: "The two losses [distillation and embedding matching] are combined with weight of 1.0"

### 1.4 Projection Layer (Sec. 4.1, Eq. 7-8)

When teacher and student have different embedding dimensions, a **compatible projection layer** is used to map student embeddings to teacher space. The paper uses `proj()` to handle dimension mismatch.

### 1.5 Document Embedding Alignment (Eq. 8)

For the **symmetric** DE case, the paper also defines:
```
R_Emb,D = (1/n) Σ ||emb_d^t - proj(emb_d^s)||
```
But for the **asymmetric** case (recommended), document embeddings are inherited from the teacher, so R_Emb,D = 0 by construction. Only R_Emb,Q (query matching) is needed.

### 1.6 Evaluation Setup (Sec. 5.1-5.2)

- Benchmarks: NQ, MSMARCO, BEIR
- Teacher: AR2 (BERT-base, 110M params) or SentenceBERT-v5
- Students: DistilBERT (67.5M), BERT-mini (11.3M) — ~2/3 and ~1/10 of teacher
- [CLS]-pooling used for all student encoders
- No hard negatives from BM25 or other models
- Optimizer: AdamW (implied by standard practice, hyperparams in Appendix F.1)

### 1.7 Query Generation / Data Augmentation (Sec. 4.3)

The paper uses BART-base to generate synthetic queries, adding R_Emb,Q on unlabeled data to further align embedding spaces. This is **not** part of the core EmbedDistill loss — it's an orthogonal enhancement.

---

## 2. Implementation Mapping

### 2.1 Loss Functions

| Paper Component | Paper Equation | Implementation Location | Match? |
|---|---|---|---|
| One-hot cross-entropy | Eq. 9 (Appendix A) | `training.py:132-134` `one_hot_loss()` | **YES** |
| Score-based KL distillation | Eq. 11 (Appendix A) | `training.py:137-140` `distill_kl()` | **YES** |
| Query embedding matching | Eq. 7 | `training.py:143-144` `align_loss()` | **YES** |
| Document embedding matching | Eq. 8 | Not implemented separately | **CORRECT** — asymmetric config inherits teacher docs, making R_Emb,D = 0 |

#### Detailed Loss Comparison

**One-Hot Loss** (paper Eq. 9 vs `training.py:132-134`):
```python
# Implementation
def one_hot_loss(student_scores, temperature):
    labels = torch.arange(student_scores.size(0), device=student_scores.device)
    return F.cross_entropy(student_scores / temperature, labels)
```
- Paper: softmax cross-entropy with diagonal labels (y_{i,j}=1 iff i=j) → **EXACT MATCH**
- Temperature scaling applied before softmax → **EXACT MATCH**
- Uses in-batch negatives (all non-diagonal entries) → **EXACT MATCH**

**Distill KL** (paper Eq. 11 vs `training.py:137-140`):
```python
# Implementation
def distill_kl(student_scores, teacher_scores, temperature):
    student_log_probs = F.log_softmax(student_scores / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
```
- Paper Eq. 11: `Σ_j softmax(s^t_{i,j}) · log(softmax(s^s_{i,j}))` — this is the cross-entropy form.
  KL divergence = H(teacher, student) - H(teacher). Since H(teacher) is constant w.r.t. student, minimizing KL = minimizing cross-entropy.
- **MATCH**: The implementation uses `F.kl_div` which is mathematically equivalent for optimization (differs by a constant from Eq. 11)
- Temperature applied to both student and teacher → **EXACT MATCH**
- `reduction="batchmean"` averages correctly → **MATCH**

**Align Loss** (paper Eq. 7 vs `training.py:143-144`):
```python
# Implementation
def align_loss(student_query_emb, target_query_emb):
    return torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1).mean()
```
- Paper Eq. 7: `(1/n) Σ ||emb_q^t - proj(emb_q^s)||`
- `vector_norm` defaults to L2 norm → **EXACT MATCH**
- Mean over batch → **EXACT MATCH**
- The projection `proj()` is applied inside `StudentQueryEncoder.encode()` before this function is called → **CORRECT**

**Combined Loss** (`training.py:352-354` for embed_distill, weighted at `training.py:53-61`):
```python
# embed_distill branch
losses.distill_kl = distill_kl(student_scores, teacher_scores, cfg.temperature)
losses.align = align_loss(student_q, target_q)

# Total
total = one_hot + distill_weight * distill_kl + align_weight * align + ...
```
- Paper Sec. 5.2: "The two losses are combined with weight of 1.0"
- Default: `distill_weight=1.0`, `align_weight=1.0` → **EXACT MATCH**

### 2.2 Architecture

| Paper Component | Paper Reference | Implementation Location | Match? |
|---|---|---|---|
| Asymmetric DE | Sec. 4.1, Fig. 1a | `training.py:321-386` (student encodes queries only, teacher docs pre-cached) | **YES** |
| Frozen teacher | Sec. 4.1 | `experiment.py:61-63` (eval mode, requires_grad=False) | **YES** |
| Teacher pre-encoding | Sec. 4.1 | `experiment.py:67-107` `_encode_teacher_targets()` | **YES** |
| Student query encoder | Fig. 1a | `modeling.py:49-73` `StudentQueryEncoder` | **YES** |
| Projection layer | Eq. 7-8 | `modeling.py:59-63` (Linear, no bias, if dims differ) | **YES** |
| L2 normalization | Implicit in dot-product scoring | `modeling.py:73` `F.normalize(projected, p=2, dim=-1)` | **YES** |
| Pooling | Sec. 5.2: "[CLS]-pooling for all student encoders" | `modeling.py:95-104` (supports CLS and mean) | **PARTIAL** — see note |

**Pooling Note**: The paper uses CLS-pooling for BERT-based encoders. The implementation defaults to **mean pooling** for MiniLM models (which is correct for sentence-transformers MiniLM). The pooling strategy is inferred from model name via `infer_model_encoding_spec()`. This is a **correct adaptation** for the sentence-transformers ecosystem — using CLS on MiniLM would be wrong since MiniLM was trained with mean pooling.

### 2.3 Training Loop

| Paper Component | Paper Reference | Implementation Location | Match? |
|---|---|---|---|
| Query-only training batches | Sec. 4.1 (only query encoder trained) | `training.py:295-302` `make_query_dataloader` | **YES** |
| Student scores: student_q @ teacher_d.T | Implicit in asymmetric setup | `training.py:342` | **YES** |
| Teacher scores: teacher_q @ teacher_d.T | Implicit | `training.py:343` | **YES** |
| Optimizer: AdamW | Standard practice (Appendix F.1) | `training.py:546-549` | **YES** |
| Best checkpoint by validation | Standard practice | `training.py:624-628` (best val MRR) | **YES** |

**Score Construction Detail** (`training.py:330-343`):
```python
student_q = student_model.encode(tokenized_queries)     # [B, d]
target_q = targets.train_query[batch_indices]            # [B, d] (teacher)
target_d = targets.train_doc[batch_indices]              # [B, d] (teacher)
student_scores = student_q @ target_d.T                  # [B, B]
teacher_scores = target_q @ target_d.T                   # [B, B]
```
This exactly implements the asymmetric setup from the paper:
- Student queries are encoded live by the student model
- Teacher query and document embeddings are **pre-cached** and looked up by index
- The student's scores are computed against **teacher document embeddings** (asymmetric)
- The teacher's scores use **teacher query + teacher document** embeddings

### 2.4 Projection Layer Initialization

| Paper Component | Paper Reference | Implementation | Match? |
|---|---|---|---|
| Projection for dimension mismatch | Eq. 7 (`proj()`) | `modeling.py:59-63` (nn.Linear, no bias) | **YES** |
| Least-squares initialization | Not in paper | `training.py:76-129` | **EXTENSION** |

**Note**: The least-squares projection initialization is an **implementation extension** not described in the paper. The paper just uses `proj()` without specifying initialization. The default config has `projection_init="none"`, which means random initialization — matching the paper's implicit approach.

### 2.5 Evaluation

| Paper Component | Paper Reference | Implementation | Match? |
|---|---|---|---|
| Asymmetric eval (student q, teacher d) | Tables 1, 3 | `metrics.py` `evaluate_asymmetric()` | **YES** |
| Symmetric eval (student q, student d) | Tables 1, 3 | `metrics.py` `evaluate_symmetric_student()` | **YES** |
| MRR, Recall@K metrics | Sec. 5.1 | `metrics.py` `reciprocal_rank_metrics()` | **YES** |

### 2.6 Hyperparameters

| Parameter | Paper (Appendix F.1 / Sec. 5) | Implementation Default | Match? |
|---|---|---|---|
| Temperature τ | Temperature scaling mentioned, specific value not given in main text | 0.05 | **REASONABLE** |
| Distill weight | 1.0 (Sec. 5.2) | 1.0 | **YES** |
| Align weight | 1.0 (Sec. 5.2, except BERT-mini + query gen = 5.0) | 1.0 | **YES** |
| Optimizer | AdamW (standard) | AdamW | **YES** |
| Learning rate | Not specified in main text | 2e-5 | **REASONABLE** for BERT fine-tuning |
| Weight decay | Not specified | 1e-2 | **REASONABLE** |
| Batch size | Not specified in main text | 32 | **REASONABLE** |
| Epochs | Not specified in main text | 8 | **REASONABLE** |

---

## 3. What the Paper Describes That IS NOT Implemented

| Feature | Paper Section | Status | Impact |
|---|---|---|---|
| Query generation (BART) | Sec. 4.3 | **NOT IMPLEMENTED** | Orthogonal enhancement, not core EmbedDistill |
| CE to DE distillation with dual pooling | Sec. 4.2 | **NOT IMPLEMENTED** | Different setting (cross-encoder teacher) |
| Document embedding matching R_Emb,D (symmetric) | Eq. 8 | **NOT NEEDED** | Implementation uses asymmetric config where R_Emb,D = 0 |
| NQ/MSMARCO benchmarks | Sec. 5 | Different dataset (MBPP) | Expected — adapted for code search |
| BERT-base teacher / DistilBERT+BERT-mini students | Sec. 5.1 | MiniLM-L12 teacher / MiniLM-L6 student | Appropriate for code search scale |

None of these omissions affect the faithfulness of the core EmbedDistill method.

---

## 4. What the Implementation Adds Beyond the Paper

| Extension | Location | Description |
|---|---|---|
| Least-squares projection init | `training.py:76-129` | Solves for optimal linear map from student to teacher space before training |
| HPD (PCA projection) | `training.py:244-264` | Dimensionality reduction variant |
| Multiple alternative KD methods | `training.py:350-385` | `score_distill`, `qed_align`, `distilcse_lite`, `pair_distill`, `adam_lite`, `margin_mse`, `pointwise`, `hpd` |
| Diagnostic metrics | `training.py:433-516` | Query/doc alignment cosines, symmetric vs asymmetric gap |
| Model-aware encoding specs | `modeling.py:21-34` | Auto-detects BGE/E5 prefixes and pooling |

---

## 5. Verdict: Faithfulness Assessment

### Core EmbedDistill method: **FAITHFUL IMPLEMENTATION**

The three defining components from the paper are implemented exactly:

1. **One-hot contrastive loss** (Eq. 9) — cross-entropy with in-batch negatives, temperature-scaled ✓
2. **Score-based KL distillation** (Eq. 11) — KL divergence on softmax-normalized score distributions ✓
3. **Query embedding alignment** (Eq. 7) — L2 distance between teacher and student query embeddings ✓

The asymmetric architecture (frozen teacher documents, trainable student query encoder) matches Figure 1a exactly. The training loop correctly:
- Pre-caches all teacher embeddings (frozen)
- Trains only the student query encoder
- Scores student queries against teacher document embeddings
- Combines all three losses with weight 1.0

### Minor differences (all justified):

| Difference | Reason |
|---|---|
| Mean pooling instead of CLS | Correct for MiniLM (sentence-transformers uses mean pooling) |
| MBPP dataset instead of NQ/MSMARCO | Adapted for code search task |
| MiniLM models instead of BERT variants | Smaller scale appropriate for MBPP |
| No query generation | Orthogonal enhancement, not core method |
| KL divergence vs cross-entropy form | Mathematically equivalent for optimization (differ by constant) |

### Overall: The `embed_distill` method in the implementation is a **correct and faithful** reproduction of the EmbedDistill paper's core approach (Sec. 4.1 DE-to-DE distillation) adapted for the MBPP code search domain.
