# Paper Experiment Results

Knowledge Distillation for Text-to-Code Retrieval (CS4248, April 2026)

## Setup

- **Student**: TinyBERT-4L (14M params, 312d) — `huawei-noah/TinyBERT_General_4L_312D`
- **Teacher**: MiniLM-L6-v2 (22M params, 384d) — `sentence-transformers/all-MiniLM-L6-v2`
- **Dataset**: TACO code search — `BEE-spoke-data/TACO-hf` (18K train, 1K val, 1K test)
- **Training**: Two-phase (phase 1: supervised fine-tune both models, phase 2: KD from fine-tuned teacher)
- **Evaluation**: Symmetric (student encodes both queries and documents)
- **Epochs**: 30 (phase 2), early stopping patience=5

## Experiment Sets

### Set 1: Core Hyperparameter Sweep (bs=32, seed=42)

Sweep `distill_weight` and `align_weight`/`pair_weight` for all 4 KD methods.

| Run | Method | dw | aw/pw | Test MRR | R@1 | R@10 | Asym MRR | Doc Cosine |
|-----|--------|:--:|:-----:|:--------:|:---:|:----:|:--------:|:----------:|
| s1_control_bs32 | control (no KD) | - | - | 0.1983 | 0.133 | 0.329 | - | - |
| s1_score_dw25 | score_distill | 25 | - | 0.2253 | 0.148 | 0.378 | 0.0062 | 0.0014 |
| s1_score_dw50 | score_distill | 50 | - | 0.2459 | 0.167 | 0.412 | 0.0075 | 0.0030 |
| s1_score_dw100 | score_distill | 100 | - | 0.2664 | 0.184 | 0.421 | 0.0066 | 0.0021 |
| s1_embed_dw50_aw1 | embed_distill | 50 | aw=1 | 0.2564 | 0.171 | 0.412 | 0.2309 | 0.5333 |
| s1_embed_dw50_aw5 | embed_distill | 50 | aw=5 | 0.2791 | 0.194 | 0.433 | 0.2784 | 0.6172 |
| s1_embed_dw50_aw10 | embed_distill | 50 | aw=10 | 0.2775 | 0.194 | 0.438 | 0.2854 | 0.6446 |
| s1_embed_dw100_aw10 | embed_distill | 100 | aw=10 | 0.2818 | 0.200 | 0.442 | 0.2868 | 0.6365 |
| s1_hnp_dw50_pw1 | hard_neg_pair | 50 | pw=1 | 0.2519 | 0.173 | 0.414 | 0.0070 | 0.0032 |
| s1_hnp_dw50_pw5 | hard_neg_pair | 50 | pw=5 | 0.2580 | 0.175 | 0.421 | 0.0066 | 0.0040 |
| s1_hnp_dw50_pw10 | hard_neg_pair | 50 | pw=10 | 0.2616 | 0.176 | 0.428 | 0.0068 | 0.0046 |
| s1_hnp_dw100_pw10 | hard_neg_pair | 100 | pw=10 | 0.2683 | 0.180 | 0.438 | 0.0068 | 0.0041 |
| s1_bimga_dw50_aw1 | bimga | 50 | aw=1 | 0.2612 | 0.172 | 0.423 | 0.2527 | 0.6965 |
| s1_bimga_dw50_aw5 | bimga | 50 | aw=5 | 0.2885 | 0.202 | 0.450 | 0.2850 | 0.7857 |
| s1_bimga_dw50_aw10 | bimga | 50 | aw=10 | 0.2973 | 0.211 | 0.465 | 0.2992 | 0.8120 |
| s1_bimga_dw100_aw10 | bimga | 100 | aw=10 | 0.2936 | 0.206 | 0.462 | 0.2989 | 0.8088 |

### Set 2: Batch Size Ablation (bs=64, seed=42)

Best config per method re-run at batch_size=64. Uses same phase 1 checkpoint (trained at bs=32).

| Run | Method | Test MRR | R@1 | R@10 | vs bs=32 |
|-----|--------|:--------:|:---:|:----:|:--------:|
| s2_control_bs64 | control | 0.2040 | 0.135 | 0.344 | +0.006 |
| s2_score_dw100_bs64 | score_distill | 0.2514 | 0.168 | 0.408 | -0.015 |
| s2_embed_dw100_aw10_bs64 | embed_distill | 0.2708 | 0.187 | 0.426 | -0.011 |
| s2_hnp_dw100_pw10_bs64 | hard_neg_pair | 0.2619 | 0.177 | 0.423 | -0.006 |
| s2_bimga_dw100_aw10_bs64 | bimga | 0.2876 | 0.203 | 0.449 | -0.006 |

### Set 3: BiMGA Ablation (A1-A4)

Isolates document alignment vs margin weighting. All at dw=100, aw=10.

| # | Variant | Q align | D align | Margin | Test MRR | Doc Cosine |
|:-:|---------|:-------:|:-------:|:------:|:--------:|:----------:|
| A1 | embed_distill (s1_embed_dw100_aw10) | uniform | - | - | 0.2818 | 0.6365 |
| A2 | bimga_uniform (s3_A2) | uniform | uniform | - | 0.2978 | 0.8183 |
| A3 | bimga_query_only (s3_A3) | margin | - | yes | 0.2789 | 0.6269 |
| A4 | bimga full (s1_bimga_dw50_aw10) | margin | margin | yes | 0.2973 | 0.8120 |

Differences: A2 vs A1 = +0.016 MRR, A3 vs A1 = -0.003 MRR, A4 vs A2 = -0.001 MRR.

### Set 4: Multi-Seed (seeds 42, 123, 456)

| Method | seed=42 | seed=123 | seed=456 | Mean +/- Std |
|--------|:-------:|:--------:|:--------:|:------------:|
| score_distill (dw=100) | 0.2664 | 0.2488 | 0.2600 | 0.258 +/- 0.009 |
| embed_distill (dw=100,aw=10) | 0.2818 | 0.2851 | 0.2680 | 0.278 +/- 0.009 |
| hard_neg_pair (dw=100,pw=10) | 0.2683 | 0.2629 | 0.2571 | 0.263 +/- 0.006 |
| bimga (dw=100,aw=10) | 0.2936 | 0.3041 | 0.3047 | 0.301 +/- 0.006 |

### Set 5: Second Model Pair — MiniLM-L6 student, mpnet-base-v2 teacher (seed=42)

Validates that BiMGA generalises beyond the TinyBERT/MiniLM pair.

| Run | Method | Test MRR | vs Control |
|-----|--------|:--------:|:----------:|
| s5_control | control | 0.346 | — |
| s5_score_dw100 | score_distill | 0.378 | +0.031 |
| s5_embed_dw100_aw10 | embed_distill | 0.378 | +0.032 |
| s5_bimga_dw100_aw10 | bimga | 0.403 | +0.057 |

### Set 7: Saturation Runs (70 epochs, patience=7, bs=32, seed=42)

Best config per method from Set 1, trained for 70 epochs. Resumed from Set 1 checkpoints.

| Run | Method | 30ep MRR | 70ep MRR | Gain | Stopped |
|-----|--------|:--------:|:--------:|:----:|:-------:|
| s7_control_bs32 | control | 0.198 | 0.205 | +0.007 | ep 39 |
| s7_score_dw100 | score_distill | 0.266 | 0.286 | +0.020 | ep 70 |
| s7_embed_dw100_aw10 | embed_distill | 0.282 | 0.303 | +0.021 | ep 69 |
| s7_hnp_dw100_pw10 | hard_neg_pair | 0.268 | 0.309 | +0.041 | ep 70 |
| s7_bimga_dw50_aw10 | bimga | 0.297 | 0.315 | +0.018 | ep 70 |
| s7_A2_bimga_uniform | bimga_uniform | 0.298 | 0.314 | +0.016 | ep 70 |

### Set 8: Extended Saturation (120 epochs, patience=10)

Unsaturated models from Set 7 continued to 120 epochs.

| Run | Method | 70ep MRR | 120ep MRR | Stopped |
|-----|--------|:--------:|:---------:|:-------:|
| s8_A2_bimga_uniform | bimga_uniform | 0.314 | 0.313 | ep 78 |
| s8_hnp_dw100_pw10 | hard_neg_pair | 0.309 | 0.302 | ep 82 |
| s8_score_dw100 | score_distill | 0.286 | 0.301 | ep 120 |
| s8_bimga_dw50_aw10 | bimga (dw=50) | 0.315 | 0.316 | ep 120 |

### Set 9: Deep Saturation (200 epochs, patience=15)

Remaining unsaturated models from Set 8 continued to 200 epochs.

| Run | Method | 120ep MRR | Final MRR | Stopped |
|-----|--------|:---------:|:---------:|:-------:|
| s9_score_dw100 | score_distill | 0.301 | 0.301 | ep 132 |
| s9_bimga_dw50_aw10 | bimga (dw=50) | 0.316 | 0.325 | ep 159 |

### Set 10: BiMGA at dw=100 (200 epochs, patience=15)

New BiMGA run at dw=100/aw=10 trained from scratch to saturation.

| Run | Method | Config | Final MRR | R@1 | R@10 | Stopped |
|-----|--------|--------|:---------:|:---:|:----:|:-------:|
| s10_bimga_dw100_aw10 | bimga | dw=100, aw=10 | 0.325 | 0.241 | 0.486 | ep 159 |

### Final Saturated Models

Each method's best fully-saturated checkpoint, used for analysis. All trained at bs=32, seed=42.

| # | Run | Method | MRR | R@1 | R@10 | Epochs | Asym MRR | Doc Cosine |
|:-:|-----|--------|:---:|:---:|:----:|:------:|:--------:|:----------:|
| 1 | s7_control_bs32 | control | 0.205 | 0.143 | 0.331 | 39 | — | — |
| 2 | s7_embed_dw100_aw10 | embed_distill | 0.303 | 0.218 | 0.461 | 69 | 0.310 | 0.679 |
| 3 | s8_A2_bimga_uniform | bimga_uniform | 0.313 | 0.232 | 0.469 | 78 | 0.316 | 0.856 |
| 4 | s8_hnp_dw100_pw10 | hard_neg_pair | 0.302 | 0.221 | 0.461 | 82 | 0.007 | 0.001 |
| 5 | s9_score_dw100 | score_distill | 0.301 | 0.215 | 0.466 | 132 | 0.006 | -0.000 |
| 6 | s10_bimga_dw100_aw10 | **bimga** | **0.325** | **0.241** | **0.486** | 159 | 0.321 | 0.881 |

## Symmetric vs Asymmetric Evaluation

Every model was evaluated in both symmetric mode (student encodes both sides) and asymmetric mode (student queries, teacher docs). Best config per method shown.

| Method | Sym MRR | Asym MRR | Gap | Doc Cosine |
|--------|:-------:|:--------:|:---:|:----------:|
| score_distill | 0.266 | 0.007 | +0.260 | 0.002 |
| hard_neg_pair | 0.268 | 0.007 | +0.262 | 0.004 |
| embed_distill | 0.282 | 0.287 | -0.005 | 0.636 |
| bimga_query_only (A3) | 0.279 | 0.286 | -0.007 | 0.627 |
| bimga | 0.297 | 0.299 | -0.002 | 0.812 |
| bimga_uniform (A2) | 0.298 | 0.302 | -0.005 | 0.818 |

Note: "Doc Cosine" = average cosine similarity between student doc embeddings and teacher doc embeddings on the test set. "Asym MRR" uses teacher doc embeddings at inference instead of student doc embeddings.

## Caveats

- **Sets 1-4 not saturated**: All KD methods ran the full 30 epochs without early stopping triggering. Sets 7-10 continued training to saturation (early stopping triggered for all methods).
- **Early stopping**: Control stopped at epoch 39; embed_distill at 69; bimga_uniform at 78; hard_neg_pair at 82; score_distill at 132; bimga at 159. See "Final Saturated Models" table for definitive results.
- **Batch size comparison**: The phase 1 checkpoint was trained at bs=32. Set 2 (bs=64) starts from this same checkpoint, so the bs=64 control was not trained from scratch at bs=64 — it tests batch size sensitivity during phase 2 KD training only.
- **Ablation config mismatch**: A1 and A4 in the ablation table use their respective best configs from Set 1. A1 (embed_distill) is at dw=100/aw=10; A4 (bimga) is at dw=50/aw=10 (its best). A2 and A3 were run at dw=100/aw=10 to match A1.
- **Multi-seed Set 4 uses dw=100**: Set 4 runs use dw=100 (the best for embed_distill, score_distill, hard_neg_pair), but bimga's best config in Set 1 was dw=50/aw=10 (0.2973) not dw=100/aw=10 (0.2936). The multi-seed bimga runs use dw=100 for consistency across methods.

## File Structure

```
experiments/
  README.md                     # This file
  run_sweep.py                  # Sweep script — run or extend experiments
  artifacts/                    # All experiment outputs
    results_summary.json        # All results in one file
    s1_bimga_dw50_aw10/         # One directory per run
      run_config.json           # Hyperparameters for this run
      metrics.json              # Full eval metrics (train/val/test + diagnostics)
      history.json              # Per-epoch training history (val MRR, losses)
    s1_embed_dw100_aw10/
      ...
```

## Running Your Own Experiments

The `run_sweep.py` script lets you reproduce or extend the experiments. Run it from the `mbpp_kd_suite` directory:

```bash
cd <repo>/mbpp_kd_suite

# Reproduce all Sets 1-4 (skips existing results):
uv run python ../submission/experiments/run_sweep.py --resume

# Run only specific sets:
uv run python ../submission/experiments/run_sweep.py --sets 1,2 --resume

# Run Set 5 with a different model pair:
uv run python ../submission/experiments/run_sweep.py --sets 5 \
  --checkpoint path/to/pair2/phase1/checkpoint.pt

# Add custom runs: edit build_all_runs() in run_sweep.py,
# uncomment the custom section, then:
uv run python ../submission/experiments/run_sweep.py --sets custom --resume
```

New results are written to `artifacts/` alongside existing ones. The `--resume` flag skips successful runs and retries failed ones.

To add a new experiment, edit `build_all_runs()` in `run_sweep.py`:
```python
# Example: test a new distill_weight
custom = []
custom.append(RunConfig(name="custom_bimga_dw200", method="bimga", distill_weight=200, align_weight=10))
sets["set6_custom"] = custom
```

### Reading results_summary.json

```python
import json
with open("artifacts/results_summary.json") as f:
    data = json.load(f)

# Get test MRR for a run
print(data["s1_bimga_dw50_aw10"]["test"]["MRR"])  # 0.2973

# Get asymmetric MRR (teacher docs at inference)
print(data["s1_bimga_dw50_aw10"]["diagnostics"]["asymmetric_test"]["MRR"])  # 0.2992

# Get document alignment quality
print(data["s1_bimga_dw50_aw10"]["diagnostics"]["doc_alignment_cosine_test_student_vs_target"])  # 0.812
```

### Reading per-run metrics.json

Same structure as the entry in results_summary.json, but for a single run.

### Reading history.json

List of per-epoch dictionaries. Use to plot training curves:
```python
import json
with open("artifacts/s1_bimga_dw50_aw10/history.json") as f:
    hist = json.load(f)
# hist is a list of dicts, one per epoch
# Each has: validation_MRR, train_loss, etc.
val_mrrs = [ep["validation_MRR"] for ep in hist]
```

## HuggingFace Model IDs

All trained models are uploaded to the `cs4248-nlp` HuggingFace organization.

**Pattern**: `cs4248-nlp/paper-{run_name}-tinybert-general-4l-312d-taco-hf-20260402-015143` (with underscores replaced by hyphens)

**Final saturated models (Sets 7-10):**

| Run Name | HuggingFace Repo |
|----------|-----------------|
| s7_control_bs32 | `cs4248-nlp/paper-s7-control-bs32-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s7_embed_dw100_aw10 | `cs4248-nlp/paper-s7-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s8_A2_bimga_uniform | `cs4248-nlp/paper-s8-a2-bimga-uniform-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s8_hnp_dw100_pw10 | `cs4248-nlp/paper-s8-hnp-dw100-pw10-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s9_score_dw100 | `cs4248-nlp/paper-s9-score-dw100-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s10_bimga_dw100_aw10 | `cs4248-nlp/paper-s10-bimga-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143` |

**Earlier models (Sets 1-3):**

| Run Name | HuggingFace Repo |
|----------|-----------------|
| s1_control_bs32 | `cs4248-nlp/paper-s1-control-bs32-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s1_score_dw100 | `cs4248-nlp/paper-s1-score-dw100-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s1_embed_dw100_aw10 | `cs4248-nlp/paper-s1-embed-dw100-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s1_hnp_dw100_pw10 | `cs4248-nlp/paper-s1-hnp-dw100-pw10-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s1_bimga_dw50_aw10 | `cs4248-nlp/paper-s1-bimga-dw50-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s3_A2_bimga_uniform | `cs4248-nlp/paper-s3-a2-bimga-uniform-tinybert-general-4l-312d-taco-hf-20260402-015143` |
| s3_A3_bimga_query_only | `cs4248-nlp/paper-s3-a3-bimga-query-only-tinybert-general-4l-312d-taco-hf-20260402-015143` |

### Loading a model

```python
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch

repo = "cs4248-nlp/paper-s1-bimga-dw50-aw10-tinybert-general-4l-312d-taco-hf-20260402-015143"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModel.from_pretrained(repo)

# Load projection layer (312d backbone -> 384d teacher space)
proj = torch.nn.Linear(312, 384, bias=False)
proj.load_state_dict(torch.load(hf_hub_download(repo, "projection.pt"), map_location="cpu", weights_only=True))

def encode(text, max_length=160):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(emb.size()).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(proj(pooled), p=2, dim=-1)

query_emb = encode("find the maximum element in a list")
# query_emb shape: (1, 384) -- L2-normalized, ready for cosine similarity
```

Note: the control model (`s1_control_bs32`) has no projection layer. Load it with just `AutoModel` + mean pooling.

## Methods Reference

| Method | Loss Components | What It Transfers |
|--------|----------------|-------------------|
| **control** | L_supervised only | Contrastive learning, no teacher signal |
| **score_distill** | L_sup + dw * KL(student_scores \|\| teacher_scores) | Relative ranking from teacher |
| **embed_distill** | L_sup + dw * KL + aw * L2(student_q, teacher_q) | Rankings + query embedding geometry |
| **hard_neg_pair** | L_sup + dw * KL + pw * BCE(pos-neg margins) | Rankings + pairwise discrimination on hard negatives |
| **bimga** | L_sup + dw * KL + aw * sigma(margin/tau) * (L2_q + L2_d) | Rankings + bidirectional embedding geometry, margin-weighted |
| **bimga_uniform** (A2) | L_sup + dw * KL + aw * (L2_q + L2_d) | Same as bimga but uniform weights (no margin) |
| **bimga_query_only** (A3) | L_sup + dw * KL + aw * sigma(margin/tau) * L2_q | Same as bimga but query-only (no doc alignment) |

## Key Hyperparameters

| Parameter | Description | Values Tested |
|-----------|-------------|---------------|
| `distill_weight` (dw) | Weight on KL divergence loss | 25, 50, 100 |
| `align_weight` (aw) | Weight on embedding alignment loss | 1, 5, 10 |
| `pair_weight` (pw) | Weight on pairwise preference loss | 1, 5, 10 |
| `distill_temperature` | Softmax temperature for KD | 0.2 (fixed) |
| `batch_size` | Training batch size | 32, 64 |
| `seed` | Random seed | 42, 123, 456 |
| `epochs` | Phase 2 training epochs | 30 (Sets 1-4), 70 (Set 7), 120 (Set 8), 200 (Sets 9-10) |
| `lr` | Learning rate (AdamW) | 2e-5 (fixed) |
