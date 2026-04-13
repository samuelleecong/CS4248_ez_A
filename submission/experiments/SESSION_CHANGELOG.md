# Session Changelog

All changes made during the April 2, 2026 working session.

## Code Changes

### New Files Created

| File | Purpose |
|------|---------|
| `mbpp_kd_suite/run_full_sweep.py` | Main experiment sweep script. Runs Sets 1-8 from a phase 1 checkpoint. Supports `--sets`, `--resume`, `--checkpoint` flags. |
| `mbpp_kd_suite/run_merge_and_route.py` | Model merging (weight averaging) and cluster-based routing experiments. |
| `mbpp_kd_suite/run_resume_training.py` | Resume training from saved checkpoints (model + optimizer state). |
| `submission/experiments/README.md` | Documentation for teammates: results tables, file structure, HuggingFace model IDs, loading code, how to run experiments. |
| `submission/experiments/run_sweep.py` | Self-contained sweep script for teammates. Outputs to `artifacts/` subdirectory. |
| `slides/create_slides.js` | PPTX presentation generator (14 slides). |
| `slides/week_progress_and_plan.md` | Markdown version of presentation. |

### Modified Files

| File | Change |
|------|--------|
| `mbpp_kd_suite/src/mbpp_kd_suite/training.py` | Added `bimga_uniform_loss()` (A2), `bimga_query_only_loss()` (A3), method dispatch branches, training checkpoint saving (model + optimizer state). |
| `mbpp_kd_suite/src/mbpp_kd_suite/constants.py` | Added `bimga_uniform` and `bimga_query_only` to `KD_METHOD_ORDER`. |
| `mbpp_kd_suite/src/mbpp_kd_suite/upload_models.py` | Fixed unicode encoding for Windows, flattened HuggingFace upload structure (backbone/tokenizer files at repo root instead of subfolders). |
| `.gitignore` | Added exception for `submission/experiments/artifacts/`, excluded model files and tensorboard logs. |

### New Loss Functions (Ablation Variants)

**`bimga_uniform` (A2)**: Bidirectional alignment WITHOUT margin weighting.
```python
# Uniform weight on both query and doc alignment
loss = (||q_s - q_t||_2 + ||d_s - d_t||_2).mean()
```

**`bimga_query_only` (A3)**: Margin-weighted query-only alignment, no doc alignment.
```python
# Margin-weighted query alignment only
weights = sigmoid(margin / temperature)
loss = (weights * ||q_s - q_t||_2).mean()
```

## Experiments Run

### Set 1: Core Hyperparameter Sweep (16 runs, bs=32, seed=42, 30 epochs)
- control + score_distill(dw=25,50,100) + embed_distill(dw×aw=4 combos) + hard_neg_pair(dw×pw=4 combos) + bimga(dw×aw=4 combos)

### Set 2: Batch Size Ablation (5 runs, bs=64, seed=42, 30 epochs)
- Best config per method at batch_size=64

### Set 3: BiMGA Ablation A2+A3 (2 runs, bs=32, seed=42, 30 epochs)
- bimga_uniform (A2) and bimga_query_only (A3) at dw=100, aw=10

### Set 4: Multi-Seed (8 runs, bs=32, seeds 123+456, 30 epochs)
- 4 methods × 2 additional seeds at best configs

### Set 6: Higher Parameter Exploration (5 runs, bs=32, seed=42, 30 epochs)
- bimga at dw=50/aw=20, dw=200/aw=10, dw=100/aw=20
- embed_distill at dw=100/aw=20
- score_distill at dw=200

### Set 7: Saturation Runs (6 runs, bs=32, seed=42, 70 epochs)
- Best config per method trained for 70 epochs with patience=7

### Set 8: Extended Saturation (started, 120 epochs, patience=10)
- 4 unsaturated models from Set 7

### Model Merging (25 combinations)
- 5 model pairs × 5 alpha ratios (0.0, 0.25, 0.5, 0.75, 1.0)

### Cluster-Based Routing (4 configurations)
- k = 4, 8, 12, 16 clusters
- Hard routing, soft routing, oracle upper bound

## Key Results

### Best Methods (30 epochs, Set 1)

| Method | Config | Test MRR |
|--------|--------|:--------:|
| bimga | dw=50, aw=10 | 0.2973 |
| bimga_uniform (A2) | dw=100, aw=10 | 0.2978 |
| embed_distill | dw=100, aw=10 | 0.2818 |
| hard_neg_pair | dw=100, pw=10 | 0.2683 |
| score_distill | dw=100 | 0.2664 |
| control | — | 0.1983 |

### Ablation (A1-A4)

| Variant | Q align | D align | Margin | MRR | Delta vs A1 |
|---------|:-------:|:-------:|:------:|:---:|:-----------:|
| A1 (embed_distill) | uniform | - | - | 0.2818 | — |
| A2 (bimga_uniform) | uniform | uniform | - | 0.2978 | +0.016 |
| A3 (bimga_query_only) | margin | - | yes | 0.2789 | -0.003 |
| A4 (bimga full) | margin | margin | yes | 0.2973 | +0.016 |

### Symmetric vs Asymmetric

| Method | Sym MRR | Asym MRR | Gap | Doc Cosine |
|--------|:-------:|:--------:|:---:|:----------:|
| score_distill | 0.266 | 0.007 | +0.260 | 0.002 |
| hard_neg_pair | 0.268 | 0.007 | +0.262 | 0.004 |
| embed_distill | 0.282 | 0.287 | -0.005 | 0.636 |
| bimga | 0.297 | 0.299 | -0.002 | 0.812 |
| bimga_uniform | 0.298 | 0.302 | -0.005 | 0.818 |

### 70-Epoch Results (Set 7)

| Method | 30ep MRR | 70ep MRR | Gain | Stopped |
|--------|:--------:|:--------:|:----:|:-------:|
| control | 0.198 | 0.205 | +0.007 | ep 39 |
| score_distill | 0.266 | 0.286 | +0.020 | ep 70 |
| embed_distill | 0.282 | 0.303 | +0.021 | ep 69 |
| hard_neg_pair | 0.268 | 0.309 | +0.041 | ep 70 |
| bimga | 0.297 | 0.315 | +0.018 | ep 70 |
| bimga_uniform | 0.314 | 0.314 | +0.016 | ep 70 |

### Model Merging
- Best merge: bimga(75%)+hard_neg_pair(25%) = 0.2980 (+0.0007 vs bimga alone)
- Merging does not meaningfully improve over single models.

### Cluster-Based Routing (30-epoch models)

| Strategy | k=4 | k=8 | k=12 | k=16 |
|----------|:---:|:---:|:----:|:----:|
| Best single (bimga_uniform) | 0.298 | 0.298 | 0.298 | 0.298 |
| Hard routing | 0.298 | 0.295 | 0.284 | 0.287 |
| Soft routing | **0.310** | **0.310** | **0.308** | **0.309** |
| Oracle | 0.381 | 0.381 | 0.381 | 0.381 |

### Set 5: Second Model Pair (MiniLM-L6 → mpnet-base-v2)

| Method | MRR | vs Control |
|--------|:---:|:----------:|
| control | 0.346 | — |
| score_distill | 0.378 | +0.031 |
| embed_distill | 0.378 | +0.032 |
| bimga | 0.403 | +0.057 |

### Higher Parameters (Set 6)
- bimga and embed_distill saturated at dw=50-100, aw=10. Higher values do not help.
- score_distill still benefits from dw=200 (0.280 vs 0.266 at dw=100).

## HuggingFace Uploads

All models uploaded to `cs4248-nlp` organization with flat file structure:
- Pattern: `cs4248-nlp/paper-{run_name}-{student}-{dataset}-{timestamp}`
- Files at root: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `projection.pt`
- Loading: `AutoModel.from_pretrained(repo)` + `AutoTokenizer.from_pretrained(repo)` works directly

## Presentation (slides/)

14-slide PPTX covering:
1. Title
2. Results landscape
3. 4 methods + why
4-5. Loss functions
6. Symmetric gap motivation
7-10. Experiment plan (detailed, prescriptive)
11. 6 analysis tasks
12. How pieces fit together (router + narrative)
13. Work distribution
14. Next steps
