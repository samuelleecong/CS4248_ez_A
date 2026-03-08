# MBPP Retrieval Experiments (Kai)

This folder contains the reproducible experiment pipeline and notebook for MBPP text-to-code retrieval (Task 2).

## Structure

```
experiments/kai/
├── notebooks/
│   └── mbpp.ipynb                   # interactive development notebook
├── mbpp.ipynb                       # compatibility symlink -> notebooks/mbpp.ipynb
├── scripts/
│   ├── run_mbpp_experiments.py      # full experiment matrix runner (source of truth)
│   ├── run_kd_experiments.py        # standalone KD runner (explicit student + teacher)
│   ├── export_teacher_targets.py    # export teacher embeddings for KD
│   ├── plot_mbpp_results.py         # plot generator from run artifacts
│   └── run_all.sh                   # end-to-end helper script
├── artifacts/
│   └── kd_targets/                  # exported teacher embeddings (.npz + _meta.json)
└── results/
    └── <run_id>/                    # one directory per run (see results/README.md)
```

---

## Experiment Pipeline

Each run executes up to six stages in order. Later stages depend on results from earlier ones.

| Stage | Flag required | What it does |
|---|---|---|
| 1. Baselines | always | Random and TF-IDF retrieval |
| 2. Pretrained | always | Zero-shot dense retrieval across all candidate models |
| 3. Sweep | `--full-matrix` | Hyperparameter sweep (MNR loss) on train+prompt → eval on validation |
| 4. Final standard | `--full-matrix` | Best sweep config retrained on train+validation+prompt |
| 5. Hard negatives | `--full-matrix` | Triplet fine-tuning from the standard checkpoint |
| 6. KD | `--teacher-targets` | Knowledge-distillation fine-tuning (see KD section below) |

Every stage appends rows to `metrics/metrics_all.csv`. Checkpoints go under `checkpoints/`. The run can be interrupted and resumed at any point with `--resume`.

---

## Quickstart

```bash
# Smoke test (schema check, no training)
python experiments/kai/scripts/run_mbpp_experiments.py --fast-smoke

# Pretrained benchmarks only
python experiments/kai/scripts/run_mbpp_experiments.py --run-id pretrained_only

# Full training matrix
python experiments/kai/scripts/run_mbpp_experiments.py \
  --run-id mbpp_full_matrix \
  --full-matrix \
  --finetune-all-pretrained \
  --device auto \
  --seed 42

# Resume an interrupted run
python experiments/kai/scripts/run_mbpp_experiments.py \
  --run-id mbpp_full_matrix \
  --full-matrix \
  --finetune-all-pretrained \
  --resume
```

Generate plots from any completed run:

```bash
python experiments/kai/scripts/plot_mbpp_results.py \
  --run-dir experiments/kai/results/mbpp_full_matrix
```

---

## Knowledge Distillation (KD)

KD trains a student encoder to mimic a stronger teacher by matching their per-batch similarity distributions, combined with the standard MNR retrieval objective.

**Loss per batch:**
```
total = alpha × KL( softmax(student_sims / T) ‖ softmax(teacher_sims / T) )
      + (1 − alpha) × MNR_CrossEntropy(student_sims)
```

- `alpha` controls the trade-off (0 = pure MNR, 1 = pure distillation, default 0.5)
- `T` softens the teacher distribution so the student learns from near-misses too (default 4.0)
- Similarity matrices are compared, not raw vectors — teacher and student can have different embedding dimensions

There are two ways to run KD depending on what you need:

| | `run_kd_experiments.py` | `run_mbpp_experiments.py --teacher-targets` |
|---|---|---|
| Student model | explicit `--student` flag | auto-selected (best pretrained benchmark) |
| Scope | KD only (zero-shot → KD) | full pipeline (baselines → pretrained → sweep → hardneg → KD) |
| Comparison | KD vs student zero-shot | KD vs standard MNR fine-tune |
| Use when | you know which student you want | you want a complete experiment matrix |

---

### Step 1 — Export teacher targets (required for both)

Run once and reuse across experiments. Choose a model **stronger than your student**.

```bash
python experiments/kai/scripts/export_teacher_targets.py \
  --teacher sentence-transformers/all-mpnet-base-v2 \
  --split all \
  --output-dir experiments/kai/artifacts \
  --batch-size 32 \
  --device auto
```

Output:
```
experiments/kai/artifacts/kd_targets/
  sentence-transformers__all-mpnet-base-v2_all.npz   ← task_ids, query_emb, code_emb
  sentence-transformers__all-mpnet-base-v2_all_meta.json
```

| Flag | Default | Description |
|---|---|---|
| `--teacher` | required | HuggingFace model name (must be cached locally) |
| `--split` | `all` | `train`, `validation`, `test`, `prompt`, or `all` (974 examples) |
| `--output-dir` | `experiments/kai/artifacts` | Root dir; targets go under `<dir>/kd_targets/` |
| `--batch-size` | 32 | Encoding batch size |
| `--device` | `auto` | `auto` picks CUDA > MPS > CPU |
| `--force` | off | Overwrite existing `.npz` |

---

### Option A — Standalone KD (`run_kd_experiments.py`)

Explicitly specify both student and teacher. Runs zero-shot eval → KD training → eval → comparison.

```bash
# Single config (use CLI flags to tune hyperparams)
python experiments/kai/scripts/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets experiments/kai/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm \
  --epochs 2 --alpha 0.5 --temperature 4.0

# Sweep 4 configs (alpha × epochs)
python experiments/kai/scripts/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets experiments/kai/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm_sweep \
  --full-matrix

# Quick schema check (zero-shot eval only, no training)
python experiments/kai/scripts/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets experiments/kai/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --fast-smoke

# Resume an interrupted run
python experiments/kai/scripts/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets experiments/kai/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm --resume
```

All CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--student` | required | Student model name (cached locally) |
| `--teacher-targets` | required | Path to `.npz` from export step |
| `--run-id` | timestamp | Run identifier |
| `--output-dir` | `experiments/kai/results` | Output root |
| `--device` | `auto` | `auto`, `cpu`, or `mps` |
| `--seed` | 42 | |
| `--epochs` | 1 | |
| `--batch-size` | 16 | |
| `--lr` | 2e-5 | |
| `--warmup-ratio` | 0.10 | |
| `--weight-decay` | 0.01 | |
| `--max-grad-norm` | 1.0 | |
| `--alpha` | 0.5 | KD loss weight (0 = pure MNR, 1 = pure distillation) |
| `--temperature` | 4.0 | Softmax temperature |
| `--full-matrix` | off | Sweep 4 configs instead of using CLI hyperparams |
| `--resume` | off | Skip already-completed steps |
| `--fast-smoke` | off | Zero-shot eval only, no training |

Comparison written: `kd_vs_student_zero_shot` (always available).

---

### Option B — Integrated KD (`run_mbpp_experiments.py --teacher-targets`)

KD runs as stage 6 of the full pipeline, after baselines → pretrained → sweep → hardneg.
The student is auto-selected as whichever pretrained model had the highest MRR.

```bash
# Pretrained benchmarks + 1 KD config (student auto-selected)
python experiments/kai/scripts/run_mbpp_experiments.py \
  --run-id mbpp_kd \
  --teacher-targets experiments/kai/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --device auto

# Full pipeline + 4 KD configs
python experiments/kai/scripts/run_mbpp_experiments.py \
  --run-id mbpp_kd_full \
  --teacher-targets experiments/kai/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --full-matrix \
  --device auto
```

Comparison written: `kd_vs_final_standard` (requires `--full-matrix` since standard fine-tune must run first).

---

### KD sweep configs

Both scripts use the same four configs when `--full-matrix` is passed:

| config_id | epochs | alpha | temperature |
|---|---|---|---|
| `kd_e1_b16_lr2e5_a05_t4` | 1 | 0.5 | 4.0 |
| `kd_e2_b16_lr2e5_a05_t4` | 2 | 0.5 | 4.0 |
| `kd_e1_b16_lr2e5_a03_t4` | 1 | 0.3 | 4.0 |
| `kd_e1_b16_lr2e5_a07_t4` | 1 | 0.7 | 4.0 |

---

### Inspect KD results

```bash
python -c "
import pandas as pd
r = 'experiments/kai/results/<run_id>'
m = pd.read_csv(f'{r}/metrics/metrics_all.csv')
print(m[m.method=='kd'][['protocol','config_id','mrr','recall@10','status']])
c = pd.read_csv(f'{r}/metrics/comparisons.csv')
print(c[['comparison','metric','base_value','compare_value','delta','ci_low','ci_high']])
"
```

---

## Run Artifacts

Each run produces:

```
results/<run_id>/
├── checkpoints/
│   ├── sweep/                   # sweep MNR checkpoints
│   ├── final_standard_mnr/      # standard fine-tune checkpoint
│   ├── final_hardneg_triplet/   # hard-negative checkpoint
│   └── kd_mnr/                  # KD checkpoints (one per config)
├── logs/
│   └── failures.log
├── metadata/
│   └── run_metadata.json        # full config snapshot
├── metrics/
│   ├── metrics_all.csv          # one row per (method, stage, protocol, config)
│   ├── training_stats.csv       # per-epoch loss for all trained models
│   └── comparisons.csv          # delta + bootstrap CI between key method pairs
├── plots/                       # generated by plot_mbpp_results.py
├── ranks/                       # raw rank arrays (.npy) for bootstrap CI
└── reports/
    ├── summary.md
    └── summary.txt
```

### Key columns in `metrics_all.csv`

| Column | Values |
|---|---|
| `method` | `baseline`, `pretrained`, `finetune`, `kd` |
| `stage` | `baseline`, `pretrained`, `sweep_mnr`, `final_standard`, `final_hardneg`, `kd_mnr` |
| `technique` | `random`, `tfidf`, `zero_shot`, `mnr`, `hard_negative_triplet`, `kd_kl` |
| `protocol` | `heldout_test` (primary), `full_corpus` (diagnostic), `tune_validation`, `train_only` |
| `status` | `success`, `failed` |

---

## Tips

- **Export once, reuse everywhere.** The `.npz` is deterministic — the same teacher + split always produces the same file.
- **Teacher must be different from (and stronger than) the student.** Same model = no knowledge transfer.
- **Use `--split all` for export.** KD training draws from train+validation+prompt (464 examples); `--split train` misses validation and prompt pairs.
- **`--fast-smoke` skips KD training** in both scripts. It is designed for CI schema checks only.
- **`--resume` is safe to re-run.** Already-completed steps (identified by their step key in `metrics_all.csv`) are skipped.
- **CSV schemas are compatible.** `run_kd_experiments.py` and `run_mbpp_experiments.py` write the same column layout, so results from both can be concatenated and plotted together.
