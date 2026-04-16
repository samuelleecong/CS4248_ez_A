# KD Alternatives Note (PairDistill + ADAM) for MBPP Code Search

This note captures the discussion on alternatives to `EmbedDistill` and `DistilCSE` for your current project.

## 1) DistilCSE is not "just InfoNCE"

DistilCSE does use an InfoNCE-style contrastive objective, but the method change is broader:
- It distills **relative similarity behavior** (not only pointwise embedding matching).
- It keeps objective consistency across teacher training, KD, and student fine-tuning.
- It relies on many negatives (in-batch/queue style) to sharpen retrieval ranking.

Practical takeaway:
- If you only swap MSE to InfoNCE but ignore negatives/objective consistency, you do not get the full DistilCSE effect.

## 2) What pairwise reranking KD changes

PairDistill-style training teaches:
- for query `q`, whether candidate `d_i` should rank above `d_j`.

Typical form:
- Teacher preference: `p_ij = sigmoid((t_i - t_j) / tau)`
- Student preference: `q_ij = sigmoid((s_i - s_j) / tau)`
- Distill by minimizing CE/KL between `p_ij` and `q_ij`.

Why this differs from EmbedDistill/DistilCSE:
- EmbedDistill/DistilCSE are mostly embedding/contrastive consistency oriented.
- PairDistill explicitly transfers **fine-grained pair ordering**, useful for near-duplicate confusions.

## 3) Do we need a strong reranker?

Ideal answer: yes.
- Pairwise KD works best when pair labels come from a stronger reranker (often cross-encoder).

But you can still run a practical "lite" version without adding a new reranker:
1. Use your fine-tuned teacher bi-encoder scores as pseudo pair labels on top-k.
2. Use ground-truth positive vs mined hard negatives as pairwise labels.

This is lower quality than cross-encoder supervision, but still actionable for your timeline.

## 4) ADAM: how it differs

ADAM focuses on **better KD examples** rather than only changing loss form:
- builds moderate-relevance "dark examples" (harder/more informative than easy negatives),
- uses adaptive/self-paced distillation based on teacher confidence.

Why useful for your project:
- MBPP train split is small, so naive negatives can become too easy.
- ADAM-style negative construction often improves the teacher signal that reaches the student.

## 5) Recommended implementation order (for your current lanes)

Given your existing setup (UniXcoder teacher, MiniLM/GTE students):

1. Keep baseline KD (`L_align + L_contrastive`) as reference.
2. Add PairDistill-lite:
   - pairwise loss on `(positive, hard-negative)` from top-k.
   - no reranker required for first pass.
3. Add ADAM-lite:
   - dark negative generation + confidence-based weighting.
4. Compare `MRR`, `R@1/5/10`, latency, and memory on the same validation protocol.

## 6) Citations (primary sources)

- DistilCSE: Zhang, Y. et al. (2023). *DistilCSE: Bootstrapping Dense Retrieval with Contrastive Distillation*. arXiv:2112.05638.  
  https://arxiv.org/abs/2112.05638

- EmbedDistill: Pan, X. et al. (2023). *EmbedDistill: Geometric Knowledge Distillation for Information Retrieval*. arXiv:2301.12005.  
  https://arxiv.org/abs/2301.12005

- PairDistill: Huang, C.-W. and Chen, Y.-N. (2024). *PairDistill: Pairwise Relevance Distillation for Dense Retrieval*. EMNLP 2024.  
  https://aclanthology.org/2024.emnlp-main.1013/

- ADAM: Tao, C. et al. (2024). *ADAM: Dense Retrieval Distillation with Adaptive Dark Examples*. Findings ACL 2024.  
  https://aclanthology.org/2024.findings-acl.692/
