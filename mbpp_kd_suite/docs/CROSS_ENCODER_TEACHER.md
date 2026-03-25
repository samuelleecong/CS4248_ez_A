# Cross-Encoder Teacher

This note describes how the cross-encoder teacher is currently trained, what was added to improve it, and how to compare it against the bi-encoder teacher baselines already used in the suite.

## Goal

The purpose of this workflow is to train a cross-encoder reranker strongly enough that it becomes a credible teacher for later bi-encoder distillation experiments.

That means the immediate target is not just "train a cross-encoder", but:

- improve its retrieval quality on MBPP or TACO
- compare it directly against a strong bi-encoder teacher
- save enough run metadata that later KD experiments can reuse the teacher cleanly

## Current Entry Point

Run:

```bash
uv run mbpp-kd-cross-teacher --help
```

The command is implemented in:

- `src/mbpp_kd_suite/cross_encoder_teacher.py`

## Measures Added

### 1. Hard Negative Mining

The original version used only random negatives. That usually makes the task too easy and does not teach the reranker how to separate near-miss code snippets.

The improved version can mine hard negatives from a configurable bi-encoder:

- `--negative-strategy hard`
- `--negative-strategy mixed`
- `--negative-miner-model sentence-transformers/all-mpnet-base-v2`
- `--hard-negative-pool-size 20`

Hard negatives are mined on the train split by encoding all train queries and code snippets with the bi-encoder miner, then taking the top non-matching code candidates for each query.

### 2. Mixed Negatives

Pure hard-negative training can over-specialize early, especially on MBPP where the train split is small.

The `mixed` strategy therefore combines:

- mostly hard negatives
- the remaining negatives filled with random negatives

This keeps the decision boundary sharper than random-only training while remaining more stable than hard-only training.

### 3. Ranking-Aware Objective

The original version optimized independent pair classification with binary cross-entropy.

That is useful, but rerankers are usually better served by a ranking-style objective. The improved version supports:

- `bce`
- `group_softmax`
- `combined`

`group_softmax` builds each training item as:

- one query
- one positive code
- `k` negatives

and learns to place the positive at rank 1 inside that group.

`combined` adds a small pairwise BCE term on top of the grouped ranking loss. This tends to preserve score calibration while still training for ranking.

### 4. Stronger Default Training Budget

The improved config raises the default training strength slightly:

- `negatives_per_query`: `6`
- `max_length`: `512`
- default objective: `combined`

This gives the model more difficult groups and reduces truncation for longer code snippets.

### 5. Built-In Baseline Comparison

The cross-encoder command now optionally evaluates a bi-encoder baseline teacher first:

- `--baseline-bi-encoder-model sentence-transformers/all-mpnet-base-v2`

The run summary then records:

- zero-shot cross-encoder vs bi-encoder baseline
- fine-tuned cross-encoder vs bi-encoder baseline

This makes it easy to answer the exact question:

"Did the trained cross-encoder beat the teacher bi-encoder?"

## Recommended MBPP Run

For a serious first comparison, use a code-aware base model plus a strong bi-encoder miner:

```bash
uv run mbpp-kd-cross-teacher \
  --model-name microsoft/codebert-base \
  --dataset-name google-research-datasets/mbpp \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --negative-strategy mixed \
  --negative-miner-model sentence-transformers/all-mpnet-base-v2 \
  --hard-negative-pool-size 20 \
  --negatives-per-query 6 \
  --train-objective combined \
  --pair-bce-weight 0.25 \
  --baseline-bi-encoder-model sentence-transformers/all-mpnet-base-v2 \
  --output-dir teachers/codebert_cross_vs_mpnet
```

## Smoke Run

Use this first to validate the pipeline on Colab or a smaller machine:

```bash
uv run mbpp-kd-cross-teacher \
  --model-name microsoft/codebert-base \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --negative-strategy mixed \
  --negatives-per-query 4 \
  --hard-negative-pool-size 10 \
  --max-train-queries 128 \
  --max-eval-queries 100 \
  --output-dir teachers/cross_encoder_smoke
```

## Outputs

Each run writes a timestamped directory containing:

- `config.json`
- `history.json`
- `negative_sampling.json`
- `results_summary.json`
- `model/` when saving is enabled

The most important fields in `results_summary.json` are:

- `baseline_bi_encoder`
- `zero_shot`
- `best_finetuned`
- `comparisons`
- `improvement`

## How To Read The Result

If:

- `comparisons.finetuned_vs_baseline.test_mrr_delta > 0`

then the fine-tuned cross-encoder beat the chosen bi-encoder baseline on test MRR.

If it is still negative, the next things to try are usually:

1. switch to a more code-aware base cross-encoder backbone
2. increase the hard-negative pool size
3. increase `negatives_per_query`
4. keep `combined` loss but tune `pair_bce_weight`
5. train longer only if validation MRR is still climbing

## Why This Matters For Later KD

If the cross-encoder teacher becomes stronger than the bi-encoder teacher, it becomes a better candidate to supervise:

- pairwise preferences for `pair_distill`
- harder negative selection for `adam_lite`
- future reranker-to-bi-encoder distillation variants

That is why this workflow is teacher-first: it isolates the teacher quality question before folding the reranker into the KD loop.

## Fine-Tuned MPNet In This Repo

The repo already contains a fine-tuning pipeline for `sentence-transformers/all-mpnet-base-v2` in:

- `experiments/kai/scripts/run_mbpp_experiments.py`

Relevant configuration found there:

- model candidate includes `sentence-transformers/all-mpnet-base-v2`
- sweep configs:
  - `sweep_e1_b16_lr2e5`
  - `sweep_e2_b16_lr2e5`
  - `sweep_e1_b32_lr1e5`
- final standard training:
  - train split: `train + validation + prompt`
  - objective: `MultipleNegativesRankingLoss`
- optional harder second stage:
  - hard-negative triplet fine-tuning from the final standard checkpoint

Expected fine-tuned checkpoint locations from that pipeline:

- `experiments/kai/results/<run_id>/checkpoints/final_standard_mnr/sentence-transformers__all-mpnet-base-v2`
- `experiments/kai/results/<run_id>/checkpoints/final_hardneg_triplet/sentence-transformers__all-mpnet-base-v2`

No fine-tuned MPNet checkpoint is checked into this repo right now, so the fair comparison command expects you to point it at your own local fine-tuned checkpoint path.

## Fair Comparison Command

To compare a cross-encoder and a bi-encoder fairly, use:

```bash
uv run mbpp-kd-compare-cross-vs-bi \
  --cross-encoder-model mixedbread-ai/mxbai-rerank-base-v1 \
  --bi-encoder-model sentence-transformers/all-mpnet-base-v2 \
  --protocol heldout_test \
  --rerank-top-k 10,25,50 \
  --output-dir ../experiments/immanuel_tim/cross_vs_mpnet_pipeline
```

This command evaluates both models on exactly the same candidate pool:

- `heldout_test`: MBPP test queries against MBPP test code only
- `full_corpus`: train + validation + test code pool, using the same aligned query/code ordering used by the Kai experiments

The cross-encoder scores every query-code pair directly.
The bi-encoder encodes all queries and all code snippets, then ranks by cosine-style inner product on normalized embeddings.

That keeps the quality comparison fair, even though the cross-encoder is much slower.

The comparison command also reports a hybrid pipeline:

- retrieve candidates with the bi-encoder
- rerank only the bi-encoder top-k candidates with the cross-encoder

This is the right tool for testing whether a `bi-encoder + reranker` architecture improves over a standalone bi-encoder on MBPP.
