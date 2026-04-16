# S7 & S8 Analysis: Definitions, Methods, and Calculations

This document explains every metric, visualization, and computational procedure used in the S7 (Embedding Alignment) and S8 (Internal Alignment) analyses. It is written for readers who want to understand not just *what* the figures show, but *how* each number was produced and *why* each metric was chosen.

---

## Table of Contents

1. [Experimental Setup](#1-experimental-setup)
2. [S7: Embedding Alignment Analysis](#2-s7-embedding-alignment-analysis)
   - [2.1 What is Doc-Cosine?](#21-what-is-doc-cosine)
   - [2.2 How Doc-Cosine is Computed](#22-how-doc-cosine-is-computed)
   - [2.3 Figure (a): A1-A4 Ablation Histograms](#23-figure-a-a1-a4-ablation-histograms)
   - [2.4 Figure (b): Saturated Alignment Methods](#24-figure-b-saturated-alignment-methods)
   - [2.5 Why score_distill and hard_neg_pair Are Excluded](#25-why-score_distill-and-hard_neg_pair-are-excluded)
3. [S8: Internal Alignment Analysis](#3-s8-internal-alignment-analysis)
   - [3.1 What is KL Divergence (on attention)?](#31-what-is-kl-divergence-on-attention)
   - [3.2 How Attention KL is Computed](#32-how-attention-kl-is-computed)
   - [3.3 Figure (a): Overall Attention Similarity Bar Chart](#33-figure-a-overall-attention-similarity-bar-chart)
   - [3.4 Figure (b): Per-layer Attention Divergence Line Plot](#34-figure-b-per-layer-attention-divergence-line-plot)
   - [3.5 What is CKA?](#35-what-is-cka)
   - [3.6 How CKA is Computed](#36-how-cka-is-computed)
   - [3.7 Figure (c): Per-layer CKA Improvement over Control](#37-figure-c-per-layer-cka-improvement-over-control)
4. [Statistical Validation](#4-statistical-validation)
5. [Key Takeaways](#5-key-takeaways)
6. [Source Scripts](#6-source-scripts)

---

## 1. Experimental Setup

**Models under analysis** (6 student models, all TinyBERT-4L with 4 transformer layers, 12 attention heads):

| Label | Method | MRR | Training |
|-------|--------|:---:|----------|
| Control (Supervised) | Cross-entropy only, no KD | 0.205 | 39 epochs |
| Score Distill | CE + KL on score distributions | 0.301 | 132 epochs |
| Hard Neg Pairwise | CE + pairwise hard-negative loss | 0.302 | 82 epochs |
| Embed Distill | CE + KL + query-side L2 alignment | 0.303 | 69 epochs |
| BiMGA Uniform | CE + KL + bidirectional L2 (uniform weight) | 0.313 | 78 epochs |
| BiMGA (Full) | CE + KL + bidirectional L2 (margin-gated) | 0.325 | 159 epochs |

**Teacher:** all-MiniLM-L6-v2 (for embeddings) / all-MiniLM-L12-v2 (for attention analysis, which has 12 transformer layers matching student head count).

**Dataset:** TACO competitive programming (1,000 test query-code pairs).

All models were trained to convergence (early stopping, patience 15, up to 200 epochs).

---

## 2. S7: Embedding Alignment Analysis

### 2.1 What is Doc-Cosine?

**Doc-Cosine** (document alignment cosine) measures how close a student's document embeddings are to the teacher's document embeddings. It answers the question: *when the student encodes a code snippet, does it produce a vector that points in the same direction as the teacher's vector for that same snippet?*

Formally, for a single test code example `c_i`:

```
doc_cosine(i) = cosine_similarity(student_embed(c_i), teacher_embed(c_i))
              = (s_i . t_i) / (||s_i|| * ||t_i||)
```

where:
- `s_i` is the student's embedding of code example `i` (a 312-dimensional vector for TinyBERT-4L)
- `t_i` is the fine-tuned teacher's embedding of code example `i` (a 384-dimensional vector projected to the shared space)
- The cosine ranges from -1 (opposite directions) to +1 (identical directions)

**Why it matters:** In symmetric retrieval, the student must encode *both* the query and the code document. If the student's document embeddings diverge from the teacher's, the student's ranking quality degrades. Doc-cosine directly measures this risk.

### 2.2 How Doc-Cosine is Computed

The computation happens in `mbpp_kd_suite/src/mbpp_kd_suite/metrics.py`, function `doc_alignment_cosine_student_vs_target`:

1. **Encode all 1,000 test code snippets** through the student model (using the document encoding path with mean pooling over token embeddings)
2. **Load the precomputed teacher document embeddings** from the fine-tuned teacher checkpoint
3. **Compute per-example cosine similarity** using PyTorch's `F.cosine_similarity(student_d, target_doc_embs, dim=-1)` -- this gives a 1,000-element vector of cosine values
4. The histograms in S7 plot **all 1,000 individual cosine values** as a distribution, not just the mean

The per-example approach (plotting all 1,000 values) is critical because it reveals the *shape* of the distribution, not just its center. A method with mean 0.85 and tight spread is very different from one with mean 0.85 but a long left tail.

### 2.3 Figure (a): A1-A4 Ablation Histograms

**Purpose:** Isolate which component of BiMGA (bidirectional alignment vs. margin gating) is responsible for the doc-cosine improvement.

**The four ablation variants** (all trained at 30 epochs, dw=100, aw=10 to control for hyperparameter differences):

| Variant | Query Alignment | Doc Alignment | Margin Gate | What This Tests |
|---------|:-:|:-:|:-:|---|
| **A1** (embed_distill) | Yes (uniform) | No | No | Baseline: query-only alignment |
| **A2** (bimga_uniform) | Yes (uniform) | Yes (uniform) | No | Effect of adding document alignment |
| **A3** (bimga_query_only) | Yes (margin-weighted) | No | Yes | Effect of margin gating alone |
| **A4** (bimga_margin) | Yes (margin-weighted) | Yes (margin-weighted) | Yes | Full BiMGA |

**How to read the histogram:**
- X-axis: `cos(student_doc, teacher_doc)` for each of the 1,000 test codes
- Y-axis: count of test codes in each cosine bin
- Further right = student doc embedding closer to teacher = better

**Key contrasts:**
- **A2 vs A1** (+0.16 mean shift, from 0.65 to 0.81): Adding document-side alignment is the main driver
- **A3 vs A1** (negligible shift): Margin gating without document alignment does almost nothing for doc quality
- **A4 vs A2** (nearly identical): Margin gating on top of bidirectional alignment has minimal additional effect

**Conclusion:** Bidirectional alignment (aligning both query and document embeddings to the teacher) is the active ingredient. The margin gate is not the main contributor.

### 2.4 Figure (b): Saturated Alignment Methods

**Purpose:** Show the doc-cosine distributions after each method has been trained to full convergence (not just 30 epochs).

**Methods shown** (only the three methods that produce meaningful doc-cosine values):

| Method | Mean Doc-Cosine | % Above 0.9 | Std Dev | Epochs to Convergence |
|--------|:-:|:-:|:-:|:-:|
| embed_distill | 0.70 | 0% | - | 69 |
| bimga_uniform | 0.85 | 32% | - | 78 |
| bimga_margin | 0.88 | 47% | 0.07 | 159 |

**How to read:** The three overlapping histograms show that even after saturation, embed_distill cannot push any individual example above 0.9 cosine, while BiMGA variants push a large fraction of examples into very high alignment (> 0.9). BiMGA (Full/margin) has the tightest distribution and the highest mean.

### 2.5 Why score_distill and hard_neg_pair Are Excluded

These methods are not plotted in the S7 histograms because their loss functions are **rotation-invariant**: they operate on dot-product scores between query and document embeddings, not on the embeddings themselves. This means:

- The student can achieve a perfect score by learning *any* embedding space that preserves the relative ranking, even one that points in completely different directions from the teacher
- Their doc-cosine values are centered near zero (mean ~ 0, std ~ 0.05) -- essentially random alignment
- Plotting them on the same x-axis as the alignment methods would compress the interesting part of the plot

They are reported separately in the README tables (doc-cosine ~ 0.001 for score_distill, ~ 0.004 for hard_neg_pair).

---

## 3. S8: Internal Alignment Analysis

S7 showed that BiMGA produces better output embeddings. S8 asks: *is this just the output layer being pushed closer to the teacher (a trivial effect of the alignment loss), or does BiMGA reshape the model's internal representations?*

Two complementary metrics answer this: **KL divergence** (on attention patterns) and **CKA** (on hidden representations).

### 3.1 What is KL Divergence (on attention)?

**KL divergence** (Kullback-Leibler divergence) measures how different two probability distributions are. In this context, attention weights form a probability distribution: for each token position, the attention head produces a distribution over all other positions (the weights sum to 1 via softmax).

KL divergence between a student attention distribution `S` and a teacher attention distribution `T` is:

```
KL(S || T) = sum_j  S(j) * log(S(j) / T(j))
```

where `j` ranges over all token positions in the sequence.

**Properties:**
- KL >= 0 always (non-negative)
- KL = 0 means the distributions are identical
- Not symmetric: KL(S||T) != KL(T||S) in general
- Higher KL = student attends to tokens differently than the teacher

**What it tells us:** If a student layer has low KL from a teacher layer, the student is routing information (deciding "which tokens to pay attention to") in a similar way to the teacher. This is a measure of **processing strategy**, not just what information is encoded.

### 3.2 How Attention KL is Computed

The computation is in `attention_teacher_kl.py`, function `compute_teacher_student_kl`:

**Step 1: Extract attention matrices**

For each test query (1,000 total), run forward passes through both the teacher (12 transformer layers) and the student (4 transformer layers) to get raw attention weights:
- Teacher: 12 layers x 12 heads x seq_len x seq_len
- Student: 4 layers x 12 heads x seq_len x seq_len

Both models use the same tokenizer (BERT WordPiece) so the sequence lengths match exactly. Padding is masked out using the attention mask.

**Step 2: Compute per-example KL for every (teacher_layer, student_layer) pair**

For each test example `i`:
```python
for each teacher_layer ti (0..11):
    for each student_layer si (0..3):
        # Get attention matrices, masked to actual sequence length
        t_a = teacher_attn[ti][i, :, :seq_len, :seq_len]  # shape (12_heads, seq_len, seq_len)
        s_a = student_attn[si][i, :, :seq_len, :seq_len]  # shape (12_heads, seq_len, seq_len)

        # KL per head per query-position, averaged over heads and positions
        kl = sum(s_a * log(s_a / t_a), axis=-1).mean()  # scalar

        example_kl[ti, si] = kl
```

This produces a (12, 4) matrix per example: the KL of each student layer from each teacher layer.

**Step 3: Best-matching teacher layer**

Since the student has only 4 layers and the teacher has 12, we find the best match:
```python
mean_kl = per_example_kl.mean(axis=0)        # average over 1000 examples -> (12, 4)
best_kl = mean_kl.min(axis=0)                # best teacher match per student layer -> (4,)
best_teacher = mean_kl.argmin(axis=0)         # which teacher layer was closest -> (4,)
```

For each student layer, the "best-matching teacher layer" is the one with the lowest KL divergence. This accounts for the fact that a 4-layer student's layers don't map 1:1 to a 12-layer teacher's layers.

**Step 4: Overall scalar**

```python
overall_mean = best_kl[1:].mean()   # average over student transformer layers L1-L3
                                     # (skip L0 = embedding layer, no attention)
```

### 3.3 Figure (a): Overall Attention Similarity Bar Chart

**What it shows:** Horizontal bars ranking each model by its mean KL from teacher (lower = more teacher-like). Each bar is the overall_mean from Step 4 above.

**How to read:**
- X-axis: Mean KL from Teacher (averaged over best-matching teacher layers for student layers L1-L4)
- Y-axis: Models, sorted from lowest KL (most similar) to highest (most different)
- Each label includes the model's MRR for comparison

| Model | Mean KL | Interpretation |
|-------|---------|----------------|
| BiMGA (Full) | 1.626 | Most teacher-like attention patterns |
| BiMGA Uniform | 1.654 | Very close to Full |
| Embed Distill | 1.697 | Partial alignment helps |
| Score Distill | 1.905 | No embedding alignment -> divergent attention |
| Hard Neg Pairwise | 1.923 | Similar to score distill |
| Control (Supervised) | 2.071 | Most divergent from teacher |

**Why this ordering matters:** The hierarchy mirrors the alignment objective intensity. Methods that explicitly align embeddings (BiMGA, Embed) develop attention patterns that are more teacher-like. Methods that only match output scores (Score, HNP) develop different internal strategies. All confidence intervals are non-overlapping (bootstrap n=1,000).

### 3.4 Figure (b): Per-layer Attention Divergence Line Plot

**What it shows:** The best-matching KL value at each student transformer layer, plotted as a line per model.

**How to read:**
- X-axis: Student transformer layers L1 (tf-0) through L4 (tf-3/output)
- Y-axis: KL from Teacher (best-matching teacher layer for that student layer)
- Each line is one model

**Key pattern:**
- **Alignment methods (BiMGA, Embed):** Steep downward slope -- deeper layers become progressively more teacher-like. The alignment loss creates a gradient signal that flows backward through the network, making each successive layer more aligned.
- **Output-matching methods (Score, HNP, Control):** Relatively flat -- KL stays roughly constant across layers. These methods find a different internal route to their output, without being pushed to mirror the teacher's processing.

### 3.5 What is CKA?

**CKA** (Centered Kernel Alignment) measures whether two sets of representations organize examples the same way. Unlike cosine similarity (which compares individual vectors), CKA compares **the overall geometry** of how a layer represents a dataset.

**Intuition:** If I give layer A and layer B the same 1,000 examples, do they produce representations where the same examples cluster together? High CKA means yes -- the two layers encode similar relational structure, even if the actual vector dimensions differ.

**Formal definition (Linear CKA):**

Given:
- `X` = matrix of representations from layer A, shape (n_examples, dim_A), centered (mean subtracted)
- `Y` = matrix of representations from layer B, shape (n_examples, dim_B), centered

```
Linear CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F^2 * ||Y^T Y||_F^2)^0.5
```

where `||.||_F` is the Frobenius norm (square root of sum of squared entries).

**Properties:**
- CKA ranges from 0 to 1
- CKA = 1 means the two layers organize examples identically (up to linear transformation)
- CKA = 0 means completely unrelated geometry
- Invariant to orthogonal transformations and isotropic scaling
- Does NOT require the two layers to have the same dimensionality (student: 312d, teacher: 384d)

**What CKA tells us vs. what KL tells us:**
- **CKA:** Do the layers encode similar *information*? (Which examples are similar to which?)
- **KL:** Do the layers use similar *processing strategies*? (Which tokens attend to which?)

These are complementary views. A model could have high CKA (same information encoded) but different KL (different attention routing to get there), or vice versa.

### 3.6 How CKA is Computed

The computation is in `attention_analysis.py`, functions `linear_cka` and `compute_cka_matrix`:

**Step 1: Extract CLS representations at every layer**

For each student model and the teacher, run all 1,000 test queries through the model with `output_hidden_states=True`. Extract the [CLS] token's hidden state at each layer:

```python
outputs = model(**encoded, output_hidden_states=True)
# hidden_states is a tuple: (embedding_layer, tf_layer_0, tf_layer_1, ..., tf_layer_N)
cls_hidden = [h[:, 0, :]  for h in outputs.hidden_states]
# Result: list of (1000, hidden_dim) matrices, one per layer
```

- Student: 5 layers (embedding + 4 transformer layers), each producing (1000, 312) matrix
- Teacher: 13 layers (embedding + 12 transformer layers), each producing (1000, 384) matrix

**Step 2: Compute CKA for every (teacher_layer, student_layer) pair**

```python
def linear_cka(X, Y):
    X = X - X.mean(axis=0)          # center: subtract mean across examples
    Y = Y - Y.mean(axis=0)
    hsic_xy = ||X^T @ Y||_F^2       # cross-similarity
    hsic_xx = ||X^T @ X||_F^2       # self-similarity of X
    hsic_yy = ||Y^T @ Y||_F^2       # self-similarity of Y
    return hsic_xy / sqrt(hsic_xx * hsic_yy)

cka_matrix = zeros(13, 5)           # (teacher_layers, student_layers)
for i in range(13):                 # each teacher layer
    for j in range(5):              # each student layer
        cka_matrix[i, j] = linear_cka(teacher_reps[i], student_reps[j])
```

This produces a (13, 5) heatmap per student model.

**Step 3: Collapse to per-student-layer scalar**

For the per-layer gain plot, the CKA is averaged across all teacher layers:

```python
per_student_layer_cka = cka_matrix.mean(axis=0)  # average over teacher layers -> (5,)
```

This gives a single number per student layer: "on average, how similar is this student layer to the teacher's layers?"

### 3.7 Figure (c): Per-layer CKA Improvement over Control

**What it shows:** For each distillation method, the CKA gain at each student layer compared to the Control (supervised-only) baseline.

**How it is calculated:**

```python
# For each method:
gain[layer] = method_cka_per_layer[layer] - control_cka_per_layer[layer]
```

**How to read:**
- X-axis: Student layers L0 (embedding) through L4 (tf-3/output)
- Y-axis: CKA gain over Control (positive = more teacher-aligned than Control)
- Dashed line at 0 = same as Control
- Each line is one distillation method

**The critical finding -- where each method's gain is located:**

| Method | L2 (middle) | L3 (penultimate) | L4 (output) | Pattern |
|--------|:-:|:-:|:-:|---|
| Score Distill | +0.020 | +0.031 | +0.031 | Gains at output |
| Hard Neg Pairwise | +0.009 | +0.010 | +0.029 | Gains at output |
| Embed Distill | +0.030 | -0.016 | +0.011 | Mixed |
| BiMGA Uniform | +0.050 | +0.069 | +0.019 | Gains in middle |
| **BiMGA (Full)** | **+0.076** | **+0.134** | **+0.013** | **Gains in middle** |

**What this means:**
- **Score Distill and Hard Neg Pairwise** improve alignment mostly at the output layer. Their losses operate on output scores, so the gradient signal affects the output layer most. The internal layers learn a different route to a slightly better output.
- **BiMGA** shows the opposite pattern: its largest gains are at L2 and L3 (middle layers), while its output layer gain (+0.013) is the *smallest* of any method. This means BiMGA restructures how the model internally processes input, making the intermediate representations structurally mirror the teacher's. The output improvement is a consequence of better internal processing, not a direct push from the loss.

**Why this matters:** If BiMGA's advantage were only at the output layer, it would be a trivial consequence of the L2 alignment loss. The middle-layer finding is the non-trivial evidence: bidirectional alignment forces gradient signal to propagate backward through the shared backbone, reshaping intermediate representations.

---

## 4. Statistical Validation

### Bootstrap Confidence Intervals (n=1,000)

For the KL divergence analysis, bootstrap resampling was used to verify that the differences between methods are statistically significant:

1. Resample 1,000 test examples with replacement
2. Recompute the overall mean KL for each model on the bootstrap sample
3. Repeat 1,000 times
4. Report 95% confidence intervals from the 2.5th and 97.5th percentiles

**Result:** All method CIs are non-overlapping, confirming the hierarchy is not due to chance.

### CKA Bootstrap

For the CKA gain comparison:
- Mean CKA difference (BiMGA vs Control, deep layers L2-L4) = 0.076
- 95% CI: [0.064, 0.087]
- P(BiMGA > Control) = 100% (all 1,000 bootstrap samples showed BiMGA ahead)

### Random Baseline Sanity Check

To verify CKA is measuring genuine learned alignment (not just tokenizer artifacts):
- Real CKA at student L1: 0.57
- CKA with random projection baseline: 0.06
- Z-score: 457

This confirms the CKA values reflect actual learned structure, not coincidental similarity from shared tokenization.

---

## 5. Key Takeaways

1. **S7 answers "what":** BiMGA produces document embeddings that are far closer to the teacher (cosine 0.88) than any other method. The key ingredient is *bidirectional* alignment (query + document), not the margin gate.

2. **S8 answers "where":** BiMGA's advantage is concentrated in the **middle layers** (L2-L3), not the output layer. Both KL divergence and CKA confirm this from complementary angles:
   - KL: BiMGA's attention patterns become progressively more teacher-like with depth
   - CKA: BiMGA's representational geometry gains peak at middle layers

3. **The mechanistic story:** By aligning both query and document embeddings to the teacher, BiMGA creates gradient pressure on the shared backbone layers (which process both input types). This forces the middle layers to structurally mirror the teacher's representations, rather than just the final output. Output-matching methods (Score, HNP) take a shortcut: they improve the output embedding without changing how the model internally processes the input.

---

## 6. Source Scripts

| Script | What It Produces |
|--------|-----------------|
| `mbpp_kd_suite/src/mbpp_kd_suite/metrics.py` | `doc_alignment_cosine_student_vs_target()` -- computes per-example doc-cosine |
| `analysis/build_quant_package.py` | Replays all HF checkpoints, computes doc-cosine across 31 runs, generates scatter plots |
| `attention_analysis.py` | Main analysis: extracts attention matrices, computes CKA matrices, entropy, KL between students |
| `attention_analysis/attention_teacher_kl.py` | Teacher-student KL divergence computation with bootstrap |
| `attention_analysis/attention_poster_final.py` | Generates poster figures: CKA heatmaps + gain plot (Figure A), KL bar + per-layer (Figure B) |
| `attention_final_figures.py` | Publication-ready versions of CKA and KL figures |
| `attention_analysis/attention_probing.py` | Linear probing analysis (supplementary) |
