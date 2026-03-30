# BiMGA Positioning: Text-to-Code Asymmetry and Evaluation

This note summarizes the conceptual framing behind `bimga` and explains why symmetric evaluation matters more in our text-to-code setting than in the asymmetric retrieval setups used by many prior KD papers.

## Short Version

BiMGA is motivated by a deployment mismatch.

Most embedding-distillation papers for retrieval assume an **asymmetric deployment**:

- the large teacher encoder still handles documents
- the student only needs to encode queries
- evaluation is effectively `student(query) x teacher(doc)`

That makes query-only alignment a reasonable design choice.

Our setting is different. In MBPP and code search more broadly, we want a **small self-contained bi-encoder** that can encode both natural-language queries and code snippets at inference time. Once evaluation becomes `student(query) x student(code)`, the document/code tower is no longer protected by the teacher. Query-only alignment is therefore incomplete.

BiMGA extends retrieval KD to this setting by:

- aligning both query and code embeddings to teacher targets
- weighting that alignment by teacher confidence using an in-batch margin

## Core Argument

The main point is not only that prior work is "text-text" while ours is "text-code." The more important distinction is that prior work is usually designed for **asymmetric retrieval deployment**, while our project targets a **symmetric student retriever**.

In standard dense retrieval distillation:

- the student is trained to produce query embeddings compatible with a frozen teacher document space
- the teacher document encoder can be reused offline
- document alignment is unnecessary because the teacher already owns the document side

In text-to-code retrieval for this project:

- the query side is natural language but the document side is source code
- the student is evaluated as a standalone model that must encode both sides
- the code/document encoder therefore needs direct supervision too

That changes the KD problem. If the student code tower only learns through the supervised retrieval loss, it receives an indirect ranking signal but no explicit geometric target in the teacher space. BiMGA adds that missing signal.

## Why Text-to-Code Makes This More Interesting

Text-to-code retrieval is asymmetric at the data level even when the model architecture is symmetric.

The two sides differ in several ways:

- natural-language queries express intent, constraints, and behavior
- code snippets express structure, syntax, API usage, and execution logic
- code carries stronger compositional and lexical regularities than ordinary text
- matching often depends on behavioral equivalence rather than surface similarity

Because of this, transferring the teacher's code-space geometry matters. Prior asymmetric KD results can look strong partly because the harder side of the retrieval problem, the document/code representation, is still handled by the teacher. Under symmetric evaluation, that advantage disappears, so code-side alignment becomes a more meaningful contribution.

## Asymmetric vs Symmetric Evaluation

We use the following distinction.

### Asymmetric evaluation

The student encodes only queries:

```text
score(query, code) = student_query(query) dot teacher_code(code)
```

This tests whether the student can enter the teacher's code space well enough to retrieve against frozen teacher document embeddings.

### Symmetric evaluation

The student encodes both queries and code:

```text
score(query, code) = student_query(query) dot student_code(code)
```

This tests the real quality of the standalone distilled retriever.

### Why symmetric evaluation is the right primary metric here

For this repository, symmetric evaluation is the more faithful metric because:

- deployment targets a teacher-free student model
- code corpora may be dynamic, local, or updated frequently
- the student code tower directly determines final MRR and Recall@k
- asymmetric evaluation can hide weaknesses in the student's code embeddings

That is why the suite now treats symmetric evaluation as the default comparison mode.

## How BiMGA Fits

BiMGA can be described as a bidirectional extension of embedding-alignment KD for a symmetric text-to-code bi-encoder.

Instead of aligning only student queries, BiMGA aligns both:

- student query embeddings to teacher query embeddings
- student code embeddings to teacher code embeddings

This reflects the actual inference path used by the student model.

## BiMGA Loss Framing

BiMGA combines three signals:

1. `L_one_hot`
   Standard supervised retrieval loss on positive pairs with in-batch negatives.
2. `L_distill_kl`
   Score-level distillation that matches the teacher's similarity distribution.
3. `L_bimga_align`
   Embedding-level alignment on both query and code representations.

The total loss is:

```text
L_total = L_one_hot + dw * L_distill_kl + aw * L_bimga_align
```

where:

- `dw` is `distill_weight`
- `aw` is `align_weight`

The alignment term is:

```text
L_bimga_align = (1/B) * sum_i w_i * (||q_s_i - q_t_i||_2 + ||d_s_i - d_t_i||_2)
```

where:

- `q_s_i, d_s_i` are student query and code embeddings
- `q_t_i, d_t_i` are teacher query and code embeddings
- `w_i` is a teacher-confidence weight

## Margin-Guided Weighting

BiMGA does not trust all teacher examples equally. It uses the teacher's in-batch ranking confidence:

```text
m_i = s_t(q_i, d_i+) - max_{j != i} s_t(q_i, d_j)
w_i = sigmoid(m_i / tau)
```

Intuition:

- if the teacher gives the positive code a large margin over the hardest negative, the example is reliable and alignment should be strong
- if the teacher is uncertain, forcing exact alignment may copy noise or errors

So the novelty is not only bidirectional alignment. It is **bidirectional alignment weighted by teacher confidence**.

## Suggested Report Wording

Use this if you want a concise project description:

> BiMGA extends embedding-alignment knowledge distillation from the standard asymmetric retrieval setting to a symmetric text-to-code bi-encoder. Prior methods mainly align student queries because the teacher document encoder is reused at inference time. In our setting, however, the student must encode both natural-language queries and code snippets, so the code tower also requires direct teacher supervision. BiMGA addresses this with bidirectional alignment on query and code embeddings, combined with margin-guided weighting so that high-confidence teacher examples receive stronger alignment than uncertain ones.

Use this if you want a concise evaluation description:

> We distinguish asymmetric evaluation, which scores `student(query) x teacher(code)`, from symmetric evaluation, which scores `student(query) x student(code)`. For text-to-code retrieval, symmetric evaluation is the more meaningful primary metric because it measures the actual quality of the standalone distilled retriever rather than the student's compatibility with a teacher-owned code space.

## Takeaway

The key contribution is best framed as:

- adapting retrieval KD from an asymmetric teacher-doc setting to a symmetric student bi-encoder
- showing why that distinction matters more in text-to-code retrieval
- introducing a bidirectional, margin-guided alignment loss that supervises both towers rather than only the query side
