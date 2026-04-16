# KD Training Pipeline — Poster Diagram

```mermaid
flowchart TD
    DATA["<b>MBPP Dataset</b><br/>query-code pairs"]

    T["<b>Teacher Encoder</b><br/>all-MiniLM-L6-v2 | frozen"]
    S["<b>Student Encoder</b><br/>TinyBERT-4L | trainable"]

    DATA --> T & S

    TE["$$\textbf{Teacher Embeddings} \\ \mathbf{T}_q \;,\; \mathbf{T}_d$$"]
    SE["$$\textbf{Student Embeddings} \\ \mathbf{S}_q \;,\; \mathbf{S}_d$$"]

    T -->|precompute| TE
    S --> SE

    COMPUTE["$$\textbf{Scoring + Pairing} \\ \mathbf{S}_{sc} = \mathbf{S}_q \cdot \mathbf{T}_d^\top \quad \mathbf{T}_{sc} = \mathbf{T}_q \cdot \mathbf{T}_d^\top$$"]

    TE --> COMPUTE
    SE --> COMPUTE

    subgraph LOSS ["Multi-Objective KD Loss"]
        direction LR
        L1["$$\mathcal{L}_{CE} \\ \text{Contrastive} \\ \text{Retrieval}$$"]
        L2["$$\mathcal{L}_{KL} \\ \text{Score} \\ \text{Distillation}$$"]
        L3["$$\mathcal{L}_{align} \\ \text{Embedding} \\ \text{Alignment}$$"]
    end

    COMPUTE --> L1
    COMPUTE --> L2
    COMPUTE --> L3

    TOTAL["$$\mathcal{L} \;=\; \mathcal{L}_{CE} \;+\; \lambda_d\,\mathcal{L}_{KL} \;+\; \lambda_a\,\mathcal{L}_{align}$$"]
    L1 --> TOTAL
    L2 --> TOTAL
    L3 --> TOTAL

    BP["<b>Backprop</b> -- update student weights only"]
    TOTAL --> BP

    EVAL["<b>Evaluate</b> -- MRR | R@k | nDCG"]
    BP --> EVAL

    %% Styles
    classDef frozen fill:#4a90d9,stroke:#2a5a8a,color:#fff
    classDef train fill:#e8744f,stroke:#b35530,color:#fff
    classDef loss fill:#f5c542,stroke:#c9a035,color:#333
    classDef data fill:#6ab070,stroke:#3d7a46,color:#fff
    classDef compute fill:#b8a9c9,stroke:#7d6b94,color:#333
    classDef eval fill:#7ec8c8,stroke:#4a9696,color:#333
    classDef total fill:#d96030,stroke:#a04020,color:#fff

    class DATA data
    class T,TE frozen
    class S,SE train
    class COMPUTE compute
    class L1,L2,L3 loss
    class TOTAL total
    class BP train
    class EVAL eval

    style LOSS fill:#fdf5dc,stroke:#c9a035,stroke-width:2px
```

## Overview

We distill retrieval knowledge from a frozen all-MiniLM-L6-v2 teacher into a TinyBERT-4L student on TACO query-code pairs. Training optimizes a multi-objective loss combining contrastive cross-entropy (L_CE) for hard ranking, KL divergence (L_KL) for soft score alignment, and an embedding alignment term (L_align) that directly minimizes the representational gap between student and teacher. Each KD variant activates a different subset of these objectives: supervised training uses L_CE alone, score distillation adds L_KL, and our proposed BiMGA further introduces bidirectional margin-guided alignment over both query and code embeddings, weighted by the teacher's confidence margin via a sigmoid gate. Only student weights are updated; evaluation uses MRR, Recall@k, and nDCG.

## Method Variants

| Method | Active Losses |
|---|---|
| supervised | $\mathcal{L}_{CE}$ only |
| score_distill | $\mathcal{L}_{CE} + \mathcal{L}_{KL}$ |
| embed_distill | $\mathcal{L}_{CE} + \mathcal{L}_{KL} + \mathcal{L}_{align}$ (query only) |
| **BiMGA** | $\mathcal{L}_{CE} + \mathcal{L}_{KL} + \mathcal{L}_{align}$ (query + code, margin-weighted) |
