# EmbedDistill Mermaid Diagram

This is the repo-specific `embed_distill` flow.

- Local method: `embed_distill`
- Paper inspiration: `EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval`
- Important repo detail: this suite uses a lightweight MBPP/TACO mapping, not a full paper reproduction

## End-to-End `embed_distill` Flow

```mermaid
flowchart TD
    A[Dataset split<br/>train / val / test<br/>query i matches code i] --> B[Load frozen teacher]
    A --> C[Load trainable student]

    B --> D[Encode all teacher queries]
    B --> E[Encode all teacher code snippets]
    D --> F[Teacher target cache<br/>train_query, val_query, test_query]
    E --> G[Teacher doc cache<br/>train_doc, val_doc, test_doc]

    C --> H[Student encoder<br/>backbone plus optional projection]
    F --> I[Training loop]
    G --> I
    H --> I

    I --> J[Query-only minibatch<br/>batch indices + raw queries]
    J --> K[Tokenize student queries]
    K --> L[Student encodes queries]
    L --> M[student_q]

    J --> N[Use batch indices to lookup<br/>teacher query embeddings]
    J --> O[Use batch indices to lookup<br/>teacher doc embeddings]
    N --> P[target_q]
    O --> Q[target_d]

    M --> R[student_scores = student_q dot target_d^T]
    Q --> R
    P --> S[teacher_scores = target_q dot target_d^T]
    Q --> S

    R --> T[one_hot retrieval loss<br/>diagonal pair should rank highest]
    R --> U[KL distillation input]
    S --> U[teacher score distribution]
    M --> V[alignment input]
    P --> V[target query embeddings]

    U --> W[distill_kl loss]
    V --> X[align_loss]

    T --> Y[final loss]
    W --> Y
    X --> Y

    Y --> Z[Backprop through student only]
    Z --> AA[Update student weights]
    AA --> AB[Repeat for all batches and epochs]

    AB --> AC[Validation checkpoint selection]
    AC --> AD{eval_mode}
    AD -->|symmetric| AE[Encode student queries and student code<br/>score student_query x student_code]
    AD -->|asymmetric| AF[Encode student queries only<br/>score student_query x frozen teacher_doc]

    AE --> AG[Save best checkpoint metrics]
    AF --> AG
    AG --> AH[Compare against baselines]
    AH --> AI[Fair baseline is supervised_student<br/>not direct_small_student]
```

## How To Read It

- The teacher is only a source of frozen targets.
- The student is the only model updated by gradient descent.
- During `embed_distill` training, the student usually encodes only queries for the minibatch.
- The training loss is:
  - dataset retrieval loss on `student_scores`
  - plus KL to teacher score distributions
  - plus direct query embedding alignment
- Fair reporting should compare `embed_distill` to `supervised_student`, because both are trained students.

## Code Pointers

- Teacher freeze and precompute: [`src/mbpp_kd_suite/experiment.py`](../src/mbpp_kd_suite/experiment.py)
- Student model and projection: [`src/mbpp_kd_suite/modeling.py`](../src/mbpp_kd_suite/modeling.py) and [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py)
- KD query-only training path: [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py)
- `embed_distill` loss branch: [`src/mbpp_kd_suite/training.py`](../src/mbpp_kd_suite/training.py)
- Symmetric vs asymmetric evaluation: [`src/mbpp_kd_suite/metrics.py`](../src/mbpp_kd_suite/metrics.py)
