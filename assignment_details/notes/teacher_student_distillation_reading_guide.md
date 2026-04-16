# Teacher-Student Distillation Reading Guide (MBPP Text-to-Code Retrieval)

This guide is tailored to your IU setup:
- Teacher: `unixcoder-base` (fine-tuned retriever)
- Student: small embedding model (e.g., MiniLM / GTE-small)
- Task: text-to-code retrieval on MBPP

## Downloaded papers (local)
- `assignment_details/papers/01_program_synthesis_with_llms_2108.07732.pdf`
- `assignment_details/papers/02_codebert_2002.08155.pdf`
- `assignment_details/papers/03_unixcoder_2203.03850.pdf`
- `assignment_details/papers/04_sentence_bert_1908.10084.pdf`
- `assignment_details/papers/05_distilcse_2112.05638.pdf`
- `assignment_details/papers/06_embeddistill_2301.12005.pdf`
- `assignment_details/papers/07_codesearchnet_1909.09436.pdf`
- `assignment_details/papers/08_pairdistill_2024.emnlp-main.1013.pdf`
- `assignment_details/papers/09_adam_2024.findings-acl.692.pdf`

## What to read first (distillation-focused)

### 1) DistilCSE (`05_distilcse_2112.05638.pdf`) — sentence embedding KD blueprint
Read these sections first:
- `2. Background`
- `3. DistilCSE`
- `4. Experiments`
- `A. More Details on Distillation`

What to extract:
- Their combined objective for embedding distillation (teacher-student alignment + contrastive behavior).
- How they avoid student collapse while preserving semantic structure.
- Hyperparameter sensitivity and practical training knobs from experiments/appendix.

Why it matters for you:
- This is the closest direct template for compressing an embedding retriever while retaining retrieval quality.

### 2) EmbedDistill (`06_embeddistill_2301.12005.pdf`) — ranking-aware IR distillation
Read these sections:
- `2. Preliminary and Motivation`
- `3. Embedding Distillation with Query Generation`
- `3.1 Coarse-grained Distillation via Regression`
- `3.2 Fine-grained Distillation via Ranking`
- `4. Experiments`

What to extract:
- Two complementary KD signals:
  - Coarse-grained embedding regression/alignment.
  - Fine-grained ranking consistency (listwise / ranking-oriented signal).
- Their query-generation idea for broadening transfer signal when labeled data is limited.
- Which ablations show ranking loss actually helps beyond pure embedding MSE.

Why it matters for you:
- Your metric is ranking-based (MRR/Recall@K), so ranking-aware KD is directly relevant.

## Then read model/task context papers

### 3) UniXcoder (`03_unixcoder_2203.03850.pdf`) — teacher representation assumptions
Prioritize:
- `3. UniXcoder`
- `3.1 Cross-modal Generation`
- `3.2 Contrastive Learning for Better Representations`
- `4. Experiment`

What to extract:
- Why UniXcoder embeddings are strong for code-text alignment.
- Which representation properties are worth preserving in student distillation.

### 4) CodeBERT (`02_codebert_2002.08155.pdf`) — retrieval baseline context
Prioritize:
- `2. CodeBERT`
- `3. Experiments` (code search parts)

What to extract:
- Typical code search setup, metrics, and pretraining assumptions.

### 5) Sentence-BERT (`04_sentence_bert_1908.10084.pdf`) — bi-encoder foundation
Prioritize:
- `3. Method`
- `5. Evaluation`

What to extract:
- Why siamese/bi-encoder architecture is efficient for retrieval.
- Similarity-space training principles that DistilCSE builds on.

### 6) CodeSearchNet (`07_codesearchnet_1909.09436.pdf`) — benchmark framing
Prioritize:
- `Task/benchmark setup`
- `Metrics`

What to extract:
- Standard retrieval evaluation framing and common failure modes for code search.

## Lower priority for distillation design

### 7) Program Synthesis with LLMs (`01_program_synthesis_with_llms_2108.07732.pdf`)
Use as background for MBPP/generation context, not core distillation method design.

## Distillation approach options you can realistically implement

### Option A: DistilCSE-style embedding KD (simpler)
- Student loss = embedding alignment + contrastive loss.
- Pros: easiest to implement on your current pipeline.
- Risk: may improve global alignment but miss ranking nuances.

### Option B: EmbedDistill-style ranking-aware KD (stronger for retrieval)
- Add ranking consistency/listwise signal on top of embedding alignment.
- Pros: usually better for MRR/Recall@K retention.
- Risk: more tuning complexity.

### Option C: Hybrid (recommended)
- Start with Option A as stable baseline.
- Add ranking-aware term (Option B) and compare deltas via ablations.

## Minimal ablation plan (for IU + final report)
- `A1`: Contrastive only (no teacher KD)
- `A2`: Contrastive + embedding alignment (DistilCSE-style)
- `A3`: A2 + ranking consistency (EmbedDistill-style)
- Compare MRR, R@1/5/10, plus latency and model size.

## Follow-up note on newer alternatives
- See `assignment_details/notes/kd_alternatives_pairdistill_adam.md` for:
  - why DistilCSE is more than "just InfoNCE",
  - how PairDistill pairwise supervision works,
  - when a reranker is needed (and fallback without one),
  - ADAM-style dark-example KD for small-data settings.
