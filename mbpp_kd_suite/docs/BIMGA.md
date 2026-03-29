# BiMGA: Bidirectional Margin-Guided Alignment for Retrieval Knowledge Distillation

## Motivation

Knowledge distillation for bi-encoder retrieval models transfers ranking knowledge from a large teacher to a smaller student. Existing embedding-alignment methods (EmbedDistill [1], QED [2], LEAF [3]) align the student's **query encoder** to the teacher's query space, but leave the **document encoder** to learn only through contrastive loss. This creates an asymmetry: the query side receives direct teacher supervision, while the document side must independently discover a compatible embedding geometry.

We observe that embedding alignment is the single most impactful KD signal in our experiments (Section 4), yet half the student's parameters (the document encoder) receive none of it. BiMGA closes this gap.

## Method

BiMGA extends embedding alignment in two ways:

### 1. Bidirectional Alignment

We align both the student query and document embeddings to their teacher counterparts:

$$
\mathcal{L}_{\text{align}} = \frac{1}{B} \sum_{i=1}^{B} w_i \left( \| \mathbf{q}_i^s - \mathbf{q}_i^t \|_2 + \| \mathbf{d}_i^s - \mathbf{d}_i^t \|_2 \right)
$$

where $\mathbf{q}^s, \mathbf{d}^s$ are student embeddings and $\mathbf{q}^t, \mathbf{d}^t$ are finetuned teacher embeddings.

No existing retrieval KD method trains the student document encoder with direct embedding alignment. EmbedDistill [1] defines a document alignment loss (Eq. 8) but freezes the document encoder in practice. QED [2] and LEAF [3] are explicitly query-only. CLIP-KD [4] aligns both towers but in the vision-language domain, not text retrieval.

### 2. Margin-Guided Weighting

Not all examples benefit equally from alignment. We weight each example by the teacher's ranking confidence, measured as the margin between the positive score and the hardest in-batch negative:

$$
m_i = s^t(q_i, d_i^+) - \max_{j \neq i} s^t(q_i, d_j)
$$

$$
w_i = \sigma(m_i / \tau)
$$

where $\tau$ is the distillation temperature and $\sigma$ is the sigmoid function.

**Intuition:** When the teacher is confident (large margin), the alignment signal is reliable and should be enforced strongly. When the teacher is uncertain (small margin), forcing alignment risks copying the teacher's mistakes.

This differs from existing confidence-weighted approaches:
- ADAM [5] weights a **score-level KL loss** by teacher confidence, not an embedding alignment loss.
- Margin-MSE [6] uses margins as the **distillation target** (MSE between student and teacher margins), not as a **weight** on alignment.

### 3. Full Loss

BiMGA combines bidirectional margin-guided alignment with standard contrastive and score KD:

$$
\mathcal{L} = \mathcal{L}_{\text{one\_hot}} + \lambda_d \cdot \mathcal{L}_{\text{distill\_kl}} + \lambda_a \cdot \mathcal{L}_{\text{align}}^{\text{BiMGA}}
$$

where $\lambda_d$ (`distill_weight`) and $\lambda_a$ (`align_weight`) control the contribution of each component.

## Relation to Existing Methods

| Method | Query align | Doc align | Weighting | Score KD |
|--------|:-----------:|:---------:|-----------|:--------:|
| EmbedDistill [1] | L2 | frozen | uniform | KL |
| QED [2] | L2 | -- | uniform | -- |
| LEAF [3] | L2 | -- | uniform | -- |
| ADAM [5] | -- | -- | confidence on score KD | KL |
| Margin-MSE [6] | -- | -- | margin as target | MSE |
| CLIP-KD [4] | MSE | MSE | uniform | contrastive |
| **BiMGA (ours)** | **L2** | **L2** | **margin-guided** | **KL** |

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

Our hyperparameter sweep (TinyBERT-4L student, MiniLM-L6 teacher, TACO dataset) found that:

- Embedding alignment is the highest-impact KD signal (+0.053 MRR with `align_weight=10`)
- Score KD requires high `distill_weight` (50+) due to T^2 scaling on cosine similarities
- The document encoder receives no direct teacher supervision in any existing method

These observations directly motivated BiMGA's design.

## References

[1] Srinivasan et al., "EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval," AAAI 2023. arXiv:2301.12005

[2] Wang & Hong, "Query Encoder Distillation via Embedding Alignment," SustaiNLP @ ACL 2023. arXiv:2306.11550

[3] MongoDB Research, "LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations," 2025. arXiv:2509.12539

[4] Yang et al., "CLIP-KD: An Empirical Study of CLIP Model Distillation," CVPR 2024.

[5] Zhan et al., "ADAM: Dense Retrieval Distillation with Adaptive Dark Examples," Findings of ACL 2024. arXiv:2212.10192

[6] Hofstatter et al., "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling," SIGIR 2021. arXiv:2104.06967
