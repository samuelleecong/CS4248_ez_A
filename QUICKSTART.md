# MBPP Project Quickstart

> **Read time**: ~15 minutes. After this you'll understand both tasks, know what's expected, and be ready to run code.

**What we're building**: Two systems using the [MBPP dataset](https://huggingface.co/datasets/google-research-datasets/mbpp) (974 Python problems with descriptions, solutions, and tests).

| Task | Input | Output | Primary Metric |
|------|-------|--------|----------------|
| **Code Generation** | "Write a function to find the sum of a list" | `def sum_list(lst): return sum(lst)` | pass@k |
| **Code Search** | "Write a function to find the sum of a list" | Ranked list of code snippets | MRR |

---

## The Dataset

Each MBPP problem has three fields:
- **text**: Natural language description
- **code**: Python solution
- **test_list**: 3 assert-style test cases

| Split | Count | Purpose |
|-------|-------|---------|
| Prompt (1–10) | 10 | Few-shot examples |
| Test (11–510) | 500 | Final evaluation |
| Validation (511–600) | 90 | Tuning hyperparameters |
| Train (601–974) | 374 | Fine-tuning |

374 training examples is small. That's why efficient methods (LoRA) and data augmentation matter.

---

## Key Concepts

| Concept | One-liner | Why it matters |
|---------|-----------|----------------|
| **Fine-tuning** | Continue training a pre-trained model on your data | Adapts general knowledge to MBPP's format |
| **LoRA** | Freeze the model, add tiny trainable matrices (~0.06% of params) | Makes fine-tuning possible on a single GPU |
| **QLoRA** | LoRA + 4-bit quantization | Fits a 7B model into 5GB VRAM |
| **Tokenizer** | Converts text/code to numbers | Each model has its own; always load the matching one |
| **Embeddings** | Fixed-size vector representation of text/code | Enable similarity-based search |
| **Contrastive learning** | Train by pulling similar pairs close, pushing dissimilar apart | Core technique for code search |

---

## Task 1: Code Generation

### The idea

A code LLM already knows Python. We fine-tune it on MBPP so it learns our instruction format and produces clean, test-passing functions.

```
                    ┌──────────────────┐
"Find the sum..." ──▶│  Fine-tuned LLM   │──▶ def sum_list(lst): return sum(lst)
                    └──────────────────┘
```

### Pipeline (pseudocode)

```
1. Load model        →  CodeLlama-7B-Instruct (or see model table below)
2. Attach LoRA       →  freeze model, add ~4M trainable params
3. Format MBPP       →  "### Instruction:\n{text}\n### Solution:\n{code}"
4. Train (SFTTrainer)→  3 epochs, lr=2e-4, batch=4×4
5. Generate          →  model.generate(prompt, temperature=0.2)
6. Evaluate          →  run generated code against test cases → pass@k
```

> **Full runnable code**: [ML_PIPELINE.md](./ML_PIPELINE.md) walks through every stage with complete code blocks.

### Which model?

| GPU VRAM | Model | Notes |
|----------|-------|-------|
| < 8 GB | [CodeT5-small](https://huggingface.co/Salesforce/codet5-small) (60M) | CPU-friendly, lower quality |
| 8–16 GB | [CodeLlama-7B-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) + QLoRA | **Start here** — best documented |
| 16–24 GB | [StarCoder2-7B](https://huggingface.co/bigcode/starcoder2-7b) + LoRA | Strong alternative |

### Evaluation: pass@k

Generate **k** code samples per problem. If **any** sample passes all test cases, the problem counts as solved. Report pass@1, pass@5, pass@10.

**Baselines** (from literature):
| Model | MBPP pass@1 |
|-------|-------------|
| CodeT5-large (770M) | ~40% |
| CodeLlama-7B | ~48% |
| StarCoder2-15B | ~50–55% |

Supplementary metrics: **CodeBLEU** (AST-aware), **ChrF** (character-level F-score). Do NOT use BLEU for code.

---

## Task 2: Code Search

### The idea

Encode queries and code into the same vector space. At search time, find the code whose vector is closest to the query vector.

```
"Find the sum..."  ──▶ [Encoder A] ──▶  query_vec  (768-dim)
                                              ↕ cosine similarity
"def sum_list(...)  ──▶ [Encoder B] ──▶  code_vec   (768-dim)
```

This is called a **dual-encoder** architecture.

### Pipeline (pseudocode)

```
1. Load model         →  UniXcoder-base (or see model table below)
2. Create pairs       →  (query=text, positive=code) from MBPP
3. Train (contrastive)→  MultipleNegativesRankingLoss pushes matching
                          pairs together, uses other batch items as negatives
4. Encode corpus      →  embed all code snippets once
5. Search             →  embed query → cosine similarity → top-k
6. Evaluate           →  MRR, Recall@1/5/10
```

> **Full runnable code**: See [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md) Sections 2.3–2.7.

### Which model?

| Model | Size | Notes |
|-------|------|-------|
| [UniXcoder-base](https://huggingface.co/microsoft/unixcoder-base) | 125M | **Start here** — best for code search |
| [CodeBERT-base](https://huggingface.co/microsoft/codebert-base) | 125M | Well-documented, foundational |
| [CodeT5p-110m-embedding](https://huggingface.co/Salesforce/codet5p-110m-embedding) | 110M | Purpose-built for embeddings |

### Evaluation: MRR and Recall@k

For each query, rank all code snippets by similarity. **MRR** = average of 1/rank of the correct match. **Recall@k** = fraction of correct matches in top-k. Report MRR, Recall@1, Recall@5, Recall@10.

**Expected performance**:
| Approach | MRR |
|----------|-----|
| TF-IDF baseline | 0.10–0.20 |
| Pre-trained code model (no fine-tuning) | 0.60–0.70 |
| Fine-tuned code model | 0.75+ |

---

## Connecting the Two Tasks: RAG

Your code search model (Task 2) can boost code generation (Task 1):

```
New problem ──▶ Search for similar solved problems ──▶ Include as context ──▶ Generate
```

This is **Retrieval-Augmented Generation**. +5–15% pass@1, no retraining needed.

---

## Quick Zero-Shot Baseline

Before fine-tuning anything, test a frontier model out of the box to set a baseline:

```bash
ollama pull qwen2.5-coder:7b-instruct
```

Then prompt it with MBPP problems and measure pass@k. This gives you a ceiling to compare against.

**Top open models (Feb 2026)**: Qwen2.5-Coder-7B (Apache 2.0), DeepSeek-Coder-6.7B (MIT), Yi-Coder-9B (Apache 2.0). See [MBPP_TUTORIAL.md Part 3](./MBPP_TUTORIAL.md#part-3-frontier-models-to-test-20242026) for the full landscape.

---

## What To Do First

### If your team is doing Task 1 (Code Generation):
1. Read this doc (15 min)
2. Skim the [MBPP paper abstract](https://arxiv.org/abs/2108.07732) (10 min)
3. Walk through [ML_PIPELINE.md](./ML_PIPELINE.md) and run the code (1–2 hrs)
4. Evaluate with pass@k

### If your team is doing Task 2 (Code Search):
1. Read this doc (15 min)
2. Skim the [CodeBERT paper abstract](https://arxiv.org/abs/2002.08155) (10 min)
3. Run Sections 2.3–2.7 in [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md) (1–2 hrs)
4. Evaluate with MRR/Recall

### If your team is doing both:
1. Start with Task 2 (faster to prototype, simpler pipeline)
2. Then Task 1 (more compute-intensive, builds on Task 2 understanding)
3. Connect them with RAG

---

## Where To Go Deeper

| Want to... | Read |
|------------|------|
| See full runnable code for the pipeline | [ML_PIPELINE.md](./ML_PIPELINE.md) |
| See full code for both tasks + model tables | [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md) Parts 1–3 |
| Explore advanced techniques (DPO, RLEF, synthetic data) | [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md) Part 4 |
| Look up a term | [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md) Appendix A (Glossary) |
| Find papers to read | [MBPP_TUTORIAL.md](./MBPP_TUTORIAL.md) Appendix B |
