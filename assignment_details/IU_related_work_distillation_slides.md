# IU Related Work Slides (Paste-Ready)

Use this as 2-3 slides in your Intermediate Update deck.
Theme: distilling large retrievers into smaller embedding models for efficient text-to-code retrieval.

---

## Slide RW1 - Related Work Landscape

**Related Work: Distillation for Efficient Retrieval**

- **Contrastive embedding distillation** improves small embedding models by matching teacher geometry, not just logits.
- **Retrieval-aware KD** uses ranking-specific signals (listwise/pairwise/in-batch) to preserve ranking quality after compression.
- **Code retrieval studies** show distillation can reduce inference cost while maintaining code-search performance.

**Representative papers**
- DistilCSE (CKD with InfoNCE) [1]
- SimTDE (simple single-stage distillation) [2]
- EmbedDistill (embedding matching + asymmetric student) [3]
- TCT-ColBERT / in-batch KD [4,5]
- RocketQAv2 and PairDistill (listwise/pairwise distillation) [6,7]
- SPENCER (code retrieval distillation) [8]

---

## Slide RW2 - What Prior Methods Actually Do

**Method Comparison (for our design choices)**

| Paper | Core distillation idea | Key training signal | Reported trade-off |
|---|---|---|---|
| DistilCSE [1] | 2-stage KD (unlabeled KD -> contrastive finetune) | InfoNCE-based CKD | Little/no drop with small students |
| SimTDE [2] | Compact student + shallow encoder + projection | Token-embedding + sentence-embedding losses | Strong retention with large size/latency reduction |
| EmbedDistill [3] | Align teacher/student embedding geometry | Score KD + embedding matching + query generation | 95-97% teacher quality at ~1/10 query encoder size |
| TCT-ColBERT [4] | Distill ColBERT MaxSim -> dot-product retriever | Tight teacher-student coupling | Faster ANN retrieval with modest quality loss |
| In-batch KD [5] | Use all query-doc combinations in minibatch | KL over in-batch distributions | Stronger signals at manageable training cost |
| RocketQAv2 [6] | Joint retriever-reranker learning | Dynamic listwise distillation + hybrid hard negatives | Better retrieval on MS MARCO / NQ |
| PairDistill [7] | Distill relative pairwise preferences | Pointwise KD + pairwise KD + iterative refresh | Better top-rank quality than pointwise-only KD |
| SPENCER [8] | Code retrieval framework with distillation | Distilled dual-encoder + TA selection + rerank stage | >98% retained performance, 70% faster query encoder |

---

## Slide RW3 - How We Position Our Method

**Gap + Our Positioning**

- Most KD papers focus on general IR or sentence similarity, not MBPP-style text-to-code retrieval.
- Code retrieval papers confirm distillation works, but few evaluate compact embedding students on MBPP-like educational code tasks.

**Our approach (grounded in prior work)**
- Start with a **code-aware teacher** (UniXcoder/CodeBERT) [9,10].
- Distill to small student via:
  - **Embedding alignment** (from EmbedDistill [3])
  - **Contrastive/ranking consistency** (from DistilCSE [1], TCT/In-batch KD [4,5], PairDistill [7])
  - **Hard negatives + iterative refresh** (from RocketQAv2/PairDistill [6,7])
- Evaluate both **quality and efficiency**:
  - MRR, Recall@1/5/10
  - latency, throughput, memory
  - quality-retention ratio (`MRR_student / MRR_teacher`)

**One-line takeaway**
- Prior work supports that combining embedding alignment + ranking-aware KD is the most defensible path to get near-teacher retrieval quality with much faster inference.

---

## References (for slide footer or final references section)

[1] DistilCSE: https://arxiv.org/abs/2112.05638  
[2] SimTDE (SIGIR 2023): https://assets.amazon.science/30/2c/6912e75b450ba016b4168a2a436d/simtde-simple-transformer-distillation-for-sentence-embeddings.pdf  
[3] EmbedDistill: https://arxiv.org/abs/2301.12005  
[4] Distilling Dense Representations with Tightly-Coupled Teachers: https://arxiv.org/abs/2010.11386  
[5] In-Batch Negatives for KD with Tightly-Coupled Teachers: https://aclanthology.org/2021.repl4nlp-1.17.pdf  
[6] RocketQAv2: https://arxiv.org/abs/2110.07367  
[7] PairDistill (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.1013.pdf  
[8] SPENCER (code retrieval distillation): https://arxiv.org/abs/2508.00546  
[9] CodeBERT: https://arxiv.org/abs/2002.08155  
[10] UniXcoder: https://arxiv.org/abs/2203.03850
