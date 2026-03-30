# Paper Implementations

This suite keeps one shared MBPP benchmark and implements lightweight variants of several embedding-oriented KD papers.

See also:

- `docs/BIMGA.md` for the full BiMGA method note
- `docs/BIMGA_POSITIONING.md` for the text-to-code motivation and symmetric vs asymmetric evaluation framing

## Included papers

### EmbedDistill

- Paper: `EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval`
- Local method: `embed_distill`
- MBPP mapping: inherited baseline with score KL plus query embedding alignment
- Gap: synthetic query generation and broader asymmetric ablations are not reproduced

### Query Encoder Distillation via Embedding Alignment

- Paper: `Query Encoder Distillation via Embedding Alignment`
- Local method: `qed_align`
- MBPP mapping: retrieval positive contrastive loss plus direct teacher-query embedding alignment
- Gap: implemented as a minimal alignment-first variant inside the same MBPP harness

### DistilCSE

- Paper: `DistilCSE: Effective Knowledge Distillation For Contrastive Sentence Embeddings`
- Local method: `distilcse_lite`
- MBPP mapping: retrieval contrastive loss plus CKD-style query-space InfoNCE between student queries and teacher queries
- Gap: no external unlabeled corpus and no memory bank queue

### PairDistill

- Paper: `PAIR DISTILL: Pairwise Relevance Distillation for Dense Retrieval`
- Local method: `hard_negative_pair_distill`
- MBPP mapping: standard score distillation plus pairwise preference loss against teacher-induced hard negatives
- Gap: uses the teacher bi-encoder as the pairwise supervisor instead of a pairwise reranker

### ADAM

- Paper: `ADAM: Dense Retrieval Distillation with Adaptive Dark Examples`
- Local method: `adam_lite`
- MBPP mapping: builds moderate-relevance dark examples by interpolating positive and hard-negative teacher document embeddings, then weights KL loss by teacher confidence
- Gap: dark examples are created in embedding space instead of token space

### HPD

- Paper: `Compressing Sentence Representation for Semantic Retrieval via Homomorphic Projective Distillation`
- Local method: `hpd`
- MBPP mapping: fits a PCA projector on teacher embeddings, distills the student into the compressed target space, and retrieves against projected teacher docs
- Gap: uses PCA-based compression as the practical stand-in for the projective distillation target

### BiMGA

- Paper status: repo method note rather than direct external paper reproduction
- Local method: `bimga`
- MBPP mapping: score KL plus bidirectional query/code embedding alignment, weighted by teacher margin confidence
- Gap: positioned as a symmetric text-to-code extension of embedding-alignment KD rather than an exact reproduction of a prior asymmetric retrieval paper
