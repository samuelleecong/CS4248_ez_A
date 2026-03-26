# Student-Teacher Mermaid Notes

This note explains the current comparison you have been running most often:

- `supervised_student`: normal student finetuning, no KD
- `embed_distill`: student trained with dataset retrieval loss plus teacher imitation

Paper mapping:

- `embed_distill` is the local approximation of `EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval`
- `supervised_student` is not from a paper; it is the fair no-KD baseline

## Full End-to-End Chart

```mermaid
flowchart TD
    START[Run CLI<br/>uv run mbpp-kd-suite] --> CFG[Parse TrainConfig<br/>teacher, student, dataset, methods,<br/>epochs, eval_mode, weights]
    CFG --> DATA[Load retrieval dataset]
    DATA --> SPLITS[Build train / validation / test splits<br/>query i matches code i]

    CFG --> DEVICE[Pick device and runtime settings]

    SPLITS --> TLOAD[Load teacher model and tokenizer]
    TLOAD --> TFREEZE[Freeze teacher weights]
    TFREEZE --> TEQ[Encode all train/val/test queries]
    TFREEZE --> TED[Encode all train/val/test code snippets]
    TEQ --> TT[Teacher target tensors<br/>train_query, train_doc,<br/>val_query, val_doc,<br/>test_query, test_doc]
    TED --> TT

    TT --> TARGETS[Build per-method target spaces<br/>full teacher space for most methods,<br/>PCA-compressed space for HPD]

    SPLITS --> DB1[direct_big_teacher]
    TT --> DB1A[Score teacher_query x teacher_doc]
    DB1 --> DB1A

    SPLITS --> DB2[direct_small_student]
    DB2 --> DB2A[Load raw student checkpoint]
    DB2A --> DB2B[Encode student queries and code]
    DB2B --> DB2C[Score student_query x student_code]

    SPLITS --> LOOP
    TARGETS --> LOOP
    CFG --> LOOP

    LOOP[For each requested method] --> INIT[Load student tokenizer and student encoder]
    INIT --> PROJ[Optional projection head<br/>student hidden size to target hidden size]
    PROJ --> PINIT[Optional least-squares projection init]

    PINIT --> BRANCH{Which method?}

    BRANCH -->|supervised_student| SUP1[Use paired query-code dataloader]
    SUP1 --> SUP2[Student encodes queries]
    SUP1 --> SUP3[Student encodes code]
    SUP2 --> SUP4[student_scores = student_q dot student_d^T]
    SUP3 --> SUP4
    SUP4 --> SUP5[Loss = one_hot retrieval loss only]

    BRANCH -->|all KD methods| KD1[Use query-only dataloader<br/>batch indices + queries]
    KD1 --> KD2[Student encodes queries only]
    KD2 --> KD3[Lookup frozen teacher targets by batch index]
    KD3 --> KD4[target_q and target_d]
    KD3 --> KD5[full_teacher_q and full_teacher_d]
    KD2 --> KD6[student_scores = student_q dot target_d^T]
    KD4 --> KD6
    KD4 --> KD7[teacher_scores = target_q dot target_d^T]
    KD6 --> KD8[Base retrieval loss<br/>one_hot on student_scores]

    KD7 --> S1
    KD6 --> S1
    S1[score_distill<br/>add KL student_scores vs teacher_scores]

    KD7 --> E1
    KD6 --> E1
    KD2 --> E1
    KD4 --> E1
    E1[embed_distill<br/>add KL on scores<br/>plus align student_q to target_q]

    KD2 --> Q1
    KD4 --> Q1
    Q1[qed_align<br/>add query embedding alignment only]

    KD2 --> D1
    KD4 --> D1
    D1[distilcse_lite<br/>add contrastive relation loss<br/>between student_q and target_q]

    KD6 --> P1
    KD7 --> P1
    P1[hard_negative_pair_distill<br/>score KL + pairwise prefs<br/>on teacher top-k negatives only]

    KD2 --> A1
    KD5 --> A1
    A1[adam_lite<br/>add KL<br/>plus dark-example loss built from<br/>teacher positive and hard-negative docs]

    KD2 --> H1
    KD4 --> H1
    H1[hpd<br/>align student queries to compressed<br/>teacher target space]

    SUP5 --> OPT[Backprop and optimizer step]
    S1 --> OPT
    E1 --> OPT
    Q1 --> OPT
    D1 --> OPT
    P1 --> OPT
    A1 --> OPT
    H1 --> OPT

    OPT --> EPOCH[Repeat batches for each epoch]
    EPOCH --> VAL[Evaluate validation MRR]
    VAL --> VMODE{eval_mode}
    VMODE -->|symmetric| VSYM[Encode student queries and code<br/>score student_query x student_code]
    VMODE -->|asymmetric| VASYM[Encode student queries only<br/>score student_query x fixed target_doc]
    VSYM --> KEEP[Keep best checkpoint by validation MRR]
    VASYM --> KEEP

    KEEP --> FINAL[Load best checkpoint]
    FINAL --> TEST[Evaluate on train / validation / test]
    TEST --> DIAG[Optional diagnostics<br/>symmetric vs asymmetric gap,<br/>query alignment, doc alignment]
    DIAG --> SAVE[Write history.json, metrics.json,<br/>results_summary.json, diagnostics_summary.json]

    SAVE --> COMPARE[Compare rows]
    DB1A --> COMPARE
    DB2C --> COMPARE
    COMPARE --> OUT[Main interpretation<br/>compare KD methods against<br/>supervised_student, not direct_small_student]
```

## High-Level Run Flow

```mermaid
flowchart TD
    A[MBPP or TACO dataset] --> B[Build paired retrieval splits<br/>query i matches code i]

    T[Teacher checkpoint] --> T1[Freeze teacher weights]
    T1 --> T2[Encode all train/val/test queries]
    T1 --> T3[Encode all train/val/test code snippets]
    T2 --> TT[Teacher target tensors]
    T3 --> TT

    S[Student checkpoint] --> ZS[Zero-shot student baseline<br/>direct_small_student]
    TT --> ZT[Zero-shot teacher baseline<br/>direct_big_teacher]

    B --> SB[supervised_student training]
    S --> SB
    SB --> SB1[Pair loader:<br/>query batch + code batch]
    SB1 --> SB2[Student encodes queries and code]
    SB2 --> SB3[Scores = student_q dot student_d^T]
    SB3 --> SB4[Loss = one_hot retrieval loss only]

    B --> KD[embed_distill training]
    S --> KD
    TT --> KD
    KD --> KD1[Query-only loader:<br/>batch indices + query batch]
    KD1 --> KD2[Student encodes queries only]
    KD2 --> KD3[Lookup teacher target_q and target_d<br/>for the same batch indices]
    KD3 --> KD4[Student scores = student_q dot target_d^T]
    KD3 --> KD5[Teacher scores = target_q dot target_d^T]
    KD4 --> KD6[one_hot retrieval loss]
    KD5 --> KD7[KL distillation loss]
    KD2 --> KD8[Align student_q to target_q]
    KD6 --> KD9[Final loss]
    KD7 --> KD9
    KD8 --> KD9

    SB4 --> SEL[Select best checkpoint by validation MRR]
    KD9 --> SEL
    SEL --> EV[Default fair evaluation:<br/>student_query x student_code]
```

## What `embed_distill` Actually Optimizes

```mermaid
flowchart LR
    Q[Batch of training queries] --> SQ[Student query encoder]
    SQ --> SQE[student_q<br/>shape B x d]

    IDX[Batch indices] --> LOOKUP[Lookup frozen teacher targets]
    LOOKUP --> TQ[target_q<br/>shape B x d]
    LOOKUP --> TD[target_d<br/>shape B x d]

    SQE --> SS[student_scores = student_q dot target_d^T<br/>shape B x B]
    TD --> SS
    TQ --> TS[teacher_scores = target_q dot target_d^T<br/>shape B x B]
    TD --> TS

    SS --> HOT[one_hot retrieval loss]
    SS --> KL[KL against teacher score distribution]
    TS --> KL
    SQE --> AL[alignment loss to target_q]
    TQ --> AL

    HOT --> FINAL[total loss]
    KL --> FINAL
    AL --> FINAL
```

## The Key Comparison

```mermaid
flowchart TD
    A[direct_small_student] --> A1[Zero-shot student only]
    B[supervised_student] --> B1[Student trained on query-code retrieval]
    C[embed_distill] --> C1[Student trained on retrieval plus teacher imitation]

    A1 --> D[Not a fair KD baseline by itself]
    B1 --> E[Fair baseline for KD]
    C1 --> F[Should be compared against supervised_student]
```

## Read This Alongside The Code

- Teacher is frozen and precomputed in [`src/mbpp_kd_suite/experiment.py`](../src/mbpp_kd_suite/experiment.py).
- Student model and optional projection live in [`src/mbpp_kd_suite/modeling.py`](../src/mbpp_kd_suite/modeling.py) and [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py).
- `supervised_student` uses paired query-code training in [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py).
- KD methods use query-only batches and frozen teacher targets in [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py).
- `embed_distill` specifically adds KL plus query alignment in [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py).
- The paper mapping is recorded in [PAPER_IMPLEMENTATIONS.md](/Users/samuellee/BME/CS4248/proj_1/CS4248_ez_A/mbpp_kd_suite/docs/PAPER_IMPLEMENTATIONS.md#L7).
