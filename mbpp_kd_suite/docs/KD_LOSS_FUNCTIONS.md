# Knowledge Distillation Loss Functions

This document explains each loss function used in our 9 KD methods for code search retrieval. All methods share an asymmetric dual-encoder architecture: the **student trains only its query encoder** while document embeddings remain frozen from the teacher.

---

## Shared Foundation: One-Hot Contrastive Loss

Every method includes this base loss. It is a standard supervised contrastive retrieval objective.

```
L_CE = CrossEntropy( softmax( q_S · d_S^T / τ ), I )
```

where `q_S` is the student query embedding matrix (B×D), `d_S` is the student document embedding matrix (B×D), `τ = 0.05` is the temperature, and `I` is the identity (the diagonal entries are the correct query-document pairs).

**What it does:** Forces the student to score the correct document highest for each query within the batch. This is the standard in-batch negatives contrastive loss.

**Code:** `training.py:132-134`
```python
def one_hot_loss(student_scores, temperature):
    labels = torch.arange(student_scores.size(0), device=student_scores.device)
    return F.cross_entropy(student_scores / temperature, labels)
```

---

## Total Loss Formula

All losses are combined via weighted sum:

```
L_total = L_CE
        + α · L_distill_kl
        + γ · L_align
        + β · L_pairwise
        + δ · L_relation
        + α · L_dark_kl
```

Default weights: `α (distill_weight) = 1.0`, `γ (align_weight) = 1.0`, `β (pair_weight) = 1.0`, `δ (relation_weight) = 1.0`. Each method only activates its relevant loss terms (the rest stay at zero).

**Code:** `training.py:53-61`

---

## Method 1: score_distill

**Category:** Score-based

**Loss terms activated:** `L_CE + α · L_distill_kl`

### L_distill_kl — KL Divergence on Score Distributions

```
L_distill_kl = τ² · KL( log_softmax(S_student / τ)  ||  softmax(S_teacher / τ) )
```

where `S_student = q_S · d_T^T` and `S_teacher = q_T · d_T^T` are B×B score matrices (student queries against teacher docs, teacher queries against teacher docs).

**What it does:** The student learns to produce the same similarity distribution over all documents as the teacher. The temperature τ=0.05 makes the softmax very peaked, focusing on the hardest negatives. The τ² scaling preserves gradient magnitude across temperatures.

**Teacher signal:** Full score distribution over documents.

**Code:** `training.py:137-140`
```python
def distill_kl(student_scores, teacher_scores, temperature):
    student_log_probs = F.log_softmax(student_scores / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
```

---

## Method 2: embed_distill

**Category:** Score + Embedding
**Paper:** EmbedDistill (Kim et al., 2023)

**Loss terms activated:** `L_CE + α · L_distill_kl + γ · L_align`

Uses the same `L_distill_kl` as score_distill, plus:

### L_align — L2 Embedding Alignment

```
L_align = (1/B) · Σ_i || q_S_i - q_T_i ||_2
```

where `q_S_i` is the student query embedding and `q_T_i` is the teacher query embedding for query i.

**What it does:** Directly pulls each student query embedding toward the corresponding teacher query embedding in vector space. When dimensions differ (e.g. student=384, teacher=768), a learned linear projection maps the student embeddings up before computing the distance.

**Teacher signal:** Score distributions AND individual query embeddings.

**Code:** `training.py:143-144`
```python
def align_loss(student_query_emb, target_query_emb):
    return torch.linalg.vector_norm(student_query_emb - target_query_emb, dim=-1).mean()
```

---

## Method 3: qed_align

**Category:** Embedding-only
**Paper:** Query Encoder Distillation (SustainLP 2023)

**Loss terms activated:** `L_CE + γ · L_align`

Uses `L_align` (same as embed_distill) but **no score distillation**.

**What it does:** The simplest embedding-based approach. Just matches student query embeddings to teacher query embeddings via L2 distance, combined with the base contrastive loss.

**Teacher signal:** Query embeddings only (no scores).

---

## Method 4: distilcse_lite

**Category:** Relational embedding
**Paper:** DistilCSE (2021)

**Loss terms activated:** `L_CE + δ · L_relation`

### L_relation — Contrastive KD Loss (InfoNCE on Query Similarities)

```
L_relation = CrossEntropy( softmax( q_S · q_T^T / τ ), I )
```

where `q_S · q_T^T` is a B×B matrix of student-teacher query similarities.

**What it does:** Instead of directly aligning individual embeddings (L2), this preserves the *relational structure* between embeddings. Each student query should be most similar to its corresponding teacher query (diagonal) compared to all other teacher queries. This is effectively an InfoNCE contrastive loss in the cross-query similarity space.

**Key difference from qed_align:** qed_align minimizes absolute distance (||q_S - q_T||); distilcse_lite maximizes relative similarity (q_S_i should be closer to q_T_i than to q_T_j). Relational matching is more flexible — it doesn't force embeddings to be identical, just structurally consistent.

**Teacher signal:** Query embedding relationships (not individual positions).

**Code:** `training.py:147-153`
```python
def contrastive_kd_loss(student_query_emb, teacher_query_emb, temperature):
    relation_scores = student_query_emb @ teacher_query_emb.T
    return one_hot_loss(relation_scores, temperature)
```

---

## Method 5: pair_distill

**Category:** Score + Pairwise preference
**Paper:** PAIR DISTILL (EMNLP 2024)

**Loss terms activated:** `L_CE + α · L_distill_kl + β · L_pairwise`

Uses the same `L_distill_kl` as score_distill, plus:

### L_pairwise — Pairwise Preference Loss

```
preference_T = σ( (s_T_pos - s_T_neg) / τ )
logits_S    = (s_S_pos - s_S_neg) / τ
L_pairwise  = BCE_with_logits( logits_S, preference_T )
```

where for each query:
- The positive is the ground-truth document
- The negatives are the top-k documents by teacher score (excluding the positive) — these are the **hard negatives** the teacher identifies
- `σ` is the sigmoid function

**What it does:** Selects the k hardest negatives according to the teacher (documents the teacher scores highest that aren't the ground truth), then trains the student to rank the positive above each hard negative. The teacher's preference strength (how confident it is about the ranking) becomes the soft target.

**Key insight:** Regular score matching treats all negatives equally. This focuses the student on the boundary cases where discrimination matters most.

**Hyperparameters:** `pair_hard_negatives = 4` (number of hard negatives per query).

**Code:** `training.py:172-198`
```python
def pairwise_preference_loss(student_scores, teacher_scores, temperature, hard_negatives):
    # Select top-k teacher-scored negatives (excluding diagonal/positive)
    mask = torch.eye(batch_size, dtype=torch.bool, device=student_scores.device)
    teacher_negatives = teacher_scores.masked_fill(mask, -1e9)
    negative_indices = teacher_negatives.topk(negative_k, dim=-1).indices

    # Compute preference targets from teacher margins
    teacher_preference = torch.sigmoid((teacher_positive - teacher_negative) / temperature)
    student_logits = (student_positive - student_negative) / temperature
    return F.binary_cross_entropy_with_logits(student_logits, teacher_preference)
```

---

## Method 6: adam_lite

**Category:** Score + Synthetic negatives
**Paper:** ADAM — Adaptive Dark Examples (ACL Findings 2024)

**Loss terms activated:** `L_CE + α · L_distill_kl + α · L_dark_kl`

Uses the same `L_distill_kl` as score_distill, plus:

### L_dark_kl — Confidence-Weighted Dark Example KL

**Step 1 — Generate dark examples:**
```
d_dark = normalize( λ · d_pos + (1-λ) · d_hard_neg )
```
where `λ = 0.65` (dark_mix_ratio). Dark documents are interpolations between the positive document and teacher-identified hard negatives, then re-normalized.

**Step 2 — Score dark examples:**
```
S_dark_student = q_S · [d_pos ; d_dark]^T
S_dark_teacher = q_T · [d_pos ; d_dark]^T
```

**Step 3 — Confidence-weighted KL:**
```
confidence_i = σ( (s_T_pos_i - mean(s_T_dark_i)) / τ )
L_dark_kl = (1/B) · Σ_i confidence_i · KL( softmax(S_dark_student_i / τ) || softmax(S_dark_teacher_i / τ) )
```

**What it does:** Creates synthetic hard negatives by interpolating between positive and negative documents in embedding space. These "dark examples" sit near the decision boundary. The loss is weighted by teacher confidence — when the teacher is very sure about the ranking, the student should pay more attention.

**Key insight:** Instead of relying only on in-batch negatives (which may be easy), this manufactures adversarial examples right at the boundary where learning is most useful.

**Hyperparameters:** `dark_negatives = 4`, `dark_mix_ratio = 0.65`.

**Code:** `training.py:201-241`
```python
def adam_dark_example_loss(student_query_emb, teacher_query_emb, teacher_doc_embs,
                           teacher_scores, temperature, hard_negatives, dark_mix_ratio):
    # Interpolate positive and hard-negative docs to create dark examples
    dark_docs = F.normalize(
        dark_mix_ratio * positive_docs.unsqueeze(1) + (1 - dark_mix_ratio) * hard_negative_docs,
        p=2, dim=-1,
    )
    # Score dark examples with both student and teacher
    # Weight KL loss by teacher confidence (sigmoid of teacher margin)
    confidence = torch.sigmoid(teacher_margin / temperature)
    return (per_row_kl * confidence).mean(), float(confidence.mean().item())
```

---

## Method 7: hpd

**Category:** Compressed embedding
**Paper:** HPD — Homomorphic Projective Distillation (ACL Findings 2022)

**Loss terms activated:** `L_CE + γ · L_align`

Uses the same `L_align` loss as qed_align, but operates on **PCA-compressed embeddings**.

### Preprocessing — PCA Target Compression

Before training:
1. Concatenate all teacher query and document embeddings from the training set
2. Fit PCA to reduce to `hpd_dim = 128` dimensions
3. Project all teacher embeddings: `q_T_compressed = normalize( (q_T - μ) · W_PCA^T )`

During training, `L_align` is computed between the student query embedding and the PCA-compressed teacher query embedding.

**What it does:** Sidesteps the dimension mismatch problem by compressing the teacher's embedding space to a lower dimension via PCA before alignment. The PCA captures the most important variance directions in the teacher's representations.

**Key difference from qed_align:** qed_align aligns in the teacher's full space (possibly with a learned projection); hpd aligns in a PCA-compressed space, which may discard noise and focus on the most informative dimensions.

**Hyperparameters:** `hpd_dim = 128`.

**Code:** `training.py:244-266`
```python
def fit_hpd_targets(cfg, full_teacher_targets):
    fit_matrix = torch.cat([full_teacher_targets.train_query, full_teacher_targets.train_doc], dim=0)
    pca = PCA(n_components=cfg.hpd_dim, random_state=cfg.seed)
    pca.fit(fit_matrix.detach().cpu().numpy())
    # Project all teacher embeddings through PCA and normalize
```

---

## Method 8: margin_mse

**Category:** Pairwise score matching
**Paper:** Margin-MSE (SIGIR 2021)

**Loss terms activated:** `L_CE + β · L_pairwise`

### L_pairwise (margin_mse variant) — Margin MSE Loss

```
margin_S_ij = s_S_ii - s_S_ij    (student: positive score minus each negative score)
margin_T_ij = s_T_ii - s_T_ij    (teacher: same)
L_margin_mse = (1/N) · Σ_{i≠j} ( margin_S_ij - margin_T_ij )²
```

where `s_ii` is the positive (diagonal) score and `s_ij` (off-diagonal) is the negative score.

**What it does:** Instead of matching absolute scores (pointwise) or full distributions (KL), this matches the **relative ranking margins** between all document pairs. If the teacher thinks document A is 0.3 better than document B for a query, the student should too.

**Key insight:** Margins are more robust than absolute scores. Two models can have very different score scales but identical ranking behavior — margin matching captures the ranking geometry directly.

**Code:** `training.py:156-165`
```python
def margin_mse_loss(student_scores, teacher_scores):
    pos_s = student_scores.diag().unsqueeze(1)   # positive scores
    pos_t = teacher_scores.diag().unsqueeze(1)
    student_margin = pos_s - student_scores       # margin: positive minus each
    teacher_margin = pos_t - teacher_scores
    mask = 1.0 - torch.eye(batch_size, device=student_scores.device)  # exclude self
    return (mask * (student_margin - teacher_margin) ** 2).sum() / mask.sum()
```

---

## Method 9: pointwise

**Category:** Direct score matching

**Loss terms activated:** `L_CE + α · L_distill_kl` (where L_distill_kl slot is reused for pointwise MSE)

### Pointwise MSE Loss

```
L_pointwise = (1 / B²) · Σ_{i,j} ( s_S_ij - s_T_ij )²
```

**What it does:** The simplest distillation approach. Directly minimizes squared error between each individual student score and teacher score in the full B×B score matrix. No ranking structure, no pairwise comparisons, no distribution matching.

**Key limitation:** Sensitive to score scale differences between teacher and student. If the teacher operates at a different magnitude, MSE penalizes the offset rather than ranking errors.

**Code:** `training.py:168-169`
```python
def pointwise_loss(student_scores, teacher_scores):
    return F.mse_loss(student_scores, teacher_scores)
```

---

## Summary: Which Losses Each Method Uses

| Method | L_CE | L_distill_kl | L_align | L_pairwise | L_relation | L_dark_kl |
|--------|:----:|:------------:|:-------:|:----------:|:----------:|:---------:|
| score_distill | x | x | | | | |
| embed_distill | x | x | x | | | |
| qed_align | x | | x | | | |
| distilcse_lite | x | | | | x | |
| pair_distill | x | x | | x | | |
| adam_lite | x | x | | | | x |
| hpd | x | | x (PCA) | | | |
| margin_mse | x | | | x | | |
| pointwise | x | x (MSE) | | | | |

## Complexity Spectrum

From simplest to most complex:

1. **pointwise** — MSE on individual scores
2. **qed_align** — L2 on query embeddings
3. **score_distill** — KL on score distributions
4. **margin_mse** — MSE on pairwise score margins
5. **distilcse_lite** — InfoNCE on query-query similarities
6. **hpd** — L2 on PCA-compressed embeddings
7. **embed_distill** — KL + L2 alignment (two objectives)
8. **pair_distill** — KL + BCE on hard-negative preferences
9. **adam_lite** — KL + confidence-weighted synthetic negatives
