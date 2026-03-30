# Novel KD Approaches for Text-to-Code Retrieval

All four approaches build on BiMGA (bidirectional margin-guided alignment + score KD). Each targets a different bottleneck in the current pipeline.

## 1. Progressive Margin Annealing (`bimga_progressive`)

**Problem:** BiMGA's margin-guided weighting suppresses alignment on uncertain examples. Early in training, the student is bad at everything — even "uncertain" teacher examples provide useful signal. At high loss weights (aw=10, dw=100), this suppression causes embed_distill to beat BiMGA.

**Approach:** Anneal the margin temperature from high (uniform weights) to low (selective weights) over training epochs.

```
effective_tau = tau_max * (1 - progress) + tau_min * progress
weights = sigmoid(margin / effective_tau)
```

- Epoch 1: tau=2.0, weights ~ 0.5 for all examples (nearly uniform)
- Epoch 10: tau=0.1, weights polarized by teacher confidence

**Hypothesis:** Uniform alignment provides a strong learning signal early; selective alignment avoids copying teacher mistakes late. This should close the gap at high weights and potentially exceed both BiMGA and embed_distill.

**References:** CL-DRD [1] applies curriculum learning to KD data difficulty; PROD [2] progressively increases teacher capability. Our approach anneals the *loss weighting* schedule instead.

## 2. Corpus-Level Hard Negative Mining (`bimga_hardneg`)

**Problem:** With batch_size=32, each query sees 31 random negatives — 0.17% of the 18K training corpus. Most are trivially easy (sorting problem vs graph problem). The model never learns to discriminate genuinely confusable code snippets.

**Approach:** Pre-compute the full teacher similarity matrix (18K x 18K) to find the top-32 hardest negative documents per query. During training, expand the score matrix used for KL distillation with these hard negatives:

```
aug_student_scores = cat([in_batch_scores, hard_neg_scores], dim=1)  # (B, B+K)
aug_teacher_scores = cat([in_batch_teacher,  hard_neg_teacher], dim=1)
KL_loss = KL(softmax(aug_student) || softmax(aug_teacher))
```

The KL distillation now operates over B+K=64 documents instead of B=32, with the extra 32 being the hardest negatives from the full corpus. The alignment loss (BiMGA) stays unchanged on in-batch pairs.

**Hypothesis:** The teacher can discriminate these hard negatives but the student can't yet — this is exactly the knowledge gap where KD should focus. Hard negative mining typically gives +0.05-0.10 MRR in dense retrieval [3, 4].

**References:** ANCE [3] mines hard negatives from an ANN index; CoCoHaNeRe [4] uses a memory bank for code search. Our approach uses teacher embeddings (already precomputed) for mining, which is zero-cost and directly aligned with the KD objective.

## 3. Code-Aware Attention Pooling (`bimga_attn_pool`)

**Problem:** Mean pooling weighs all tokens equally. In code, tokens carry vastly different semantic weight — `return dp[n]` is far more informative than `import sys` or whitespace tokens. The 312-dim student embedding wastes capacity encoding boilerplate.

**Approach:** Replace mean pooling with a learned attention layer (313 extra parameters):

```python
weights = softmax(Linear(hidden_states))  # (B, L) — learned token importance
pooled = sum(hidden_states * weights, dim=1)  # (B, H) — weighted aggregation
```

The attention layer learns which tokens matter for retrieval during fine-tuning. The same BiMGA loss is used — only the pooling mechanism changes.

**Hypothesis:** The student's backbone (TinyBERT-4L, pretrained on English text) produces per-token representations. Mean pooling discards the information about which tokens are important for code understanding. A learned pooling that emphasizes function names, return values, and algorithmic keywords over boilerplate should produce better code embeddings with the same backbone capacity.

**References:** Sentence-BERT explored WeightedMeanPooling; CodeSage [5] uses identifier deobfuscation to force attention to semantically meaningful tokens during pretraining. Our approach adds a lightweight learned pooling during KD fine-tuning.

## 4. Semantic Bridge (`bimga_bridge`)

**Problem:** The student's 384-dim projected embedding must simultaneously encode: (a) shared text-code semantics (for cross-modal matching), and (b) modality-specific details (code syntax, NL grammar). These objectives compete for representational capacity.

**Approach:** Split the embedding into a shared subspace (first 192 dims) and a specific subspace (last 192 dims). Apply BiMGA alignment only on the shared subspace, plus a cross-correlation orthogonality regularizer:

```
L_align = BiMGA_weighted(student_shared, teacher_shared)
L_ortho = mean(cross_corr(shared, specific)^2)  # Barlow Twins-style
L_total = one_hot + dw * KL + aw * (L_align + 0.01 * L_ortho)
```

Retrieval uses the full embedding (shared + specific) for scoring. The orthogonality term encourages the two subspaces to encode different information.

**Hypothesis:** By concentrating alignment pressure on half the dimensions, the student can more precisely match the teacher's semantic structure in the shared space while using the specific space freely for modality-specific features that help discrimination. This is the most speculative approach.

**References:** Barlow Twins [6] uses cross-correlation for self-supervised learning; disentangled representations are studied in VQ-VAE and beta-VAE literature. Application to retrieval KD subspace decomposition is novel.

## Summary

| Approach | Targets | Change | Risk |
|----------|---------|--------|------|
| Progressive | Loss dynamics | ~10 lines in loss fn | Low |
| Hard Neg Mining | Training signal quality | Data pipeline + score matrix | Low |
| Attention Pooling | Embedding quality | New pooling module (313 params) | Medium |
| Semantic Bridge | Capacity allocation | Subspace split + regularizer | High |

## References

[1] Zeng et al., "Curriculum Learning for Dense Retrieval Distillation," SIGIR 2022.
[2] Wu et al., "Progressive Distillation for Dense Retrieval," arXiv 2022.
[3] Xiong et al., "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval," ICLR 2021.
[4] Li et al., "Effective Hard Negative Mining for Contrastive Learning-Based Code Search," ACM TOSEM 2024.
[5] Zhang et al., "CodeSage: Code Representation Learning At Scale," ICLR 2024.
[6] Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction," ICML 2021.
