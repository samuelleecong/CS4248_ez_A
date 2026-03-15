# Knowledge Distillation — MBPP Code Search (Task 2)

Trains a student `SentenceTransformer` to mimic a stronger teacher, combined
with the standard MNR retrieval objective. Two KD loss modes are supported via
`--kd-loss`.

**Listwise (default) — KL divergence on full similarity matrix:**
```
total = alpha × KL( softmax(student_sims / T) ‖ softmax(teacher_sims / T) )
      + (1 − alpha) × MNR_CrossEntropy(student_sims)
```

**Pairwise — Margin MSE (Hofstätter et al., 2021):**
```
total = alpha × mean( (student_margin_ij − teacher_margin_ij)² )
      + (1 − alpha) × MNR_CrossEntropy(student_sims)

where margin_ij = sim(query_i, pos_i) − sim(query_i, neg_j)  for j ≠ i
```

**PairDistill — binary KL on pairwise preferences (Huang & Chen, EMNLP 2024):**
```
P(pos ≻ neg | q) = softmax([sim(q, pos), sim(q, neg)])[0]

total = alpha × mean( KL( P_teacher(pos ≻ neg | q_i) ‖ P_student(pos ≻ neg | q_i) ) )
      + (1 − alpha) × MNR_CrossEntropy(student_sims)
```

Instead of absolute relevance scores, the teacher signal is a *relative preference* — which document wins a head-to-head comparison. This produces more reliable ranking signals for similarly-scored documents.

- `alpha` — trade-off between distillation and task loss (0 = pure MNR, 1 = pure KD, default 0.5)
- `T` — softmax temperature for listwise mode; higher values spread probability across near-misses (default 4.0)
- Similarity matrices are compared, not raw vectors — teacher and student can have **different embedding dimensions**

---

## Structure

```
knowledge-distillation/
├── export_teacher_targets.py   # encode all MBPP pairs with a teacher model → .npz
├── run_kd_experiments.py       # standalone KD runner (explicit student + teacher)
├── artifacts/
│   └── kd_targets/             # exported .npz files + _meta.json sidecars (git-ignored)
└── results/                    # run output (git-ignored)
    └── <run_id>/
        ├── checkpoints/kd_mnr/ # trained student checkpoints
        ├── logs/failures.log
        ├── metadata/run_metadata.json
        ├── metrics/
        │   ├── metrics_all.csv     # one row per (method, stage, protocol, config)
        │   ├── training_stats.csv  # per-epoch loss
        │   └── comparisons.csv     # kd_vs_student_zero_shot with bootstrap CI
        ├── ranks/              # raw rank arrays (.npy) for bootstrap CI
        └── reports/
            ├── summary.md
            └── summary.txt
```

---

## Quickstart

### Step 1 — Export teacher targets

Run once and reuse. Choose a model **stronger than your student**.

```bash
python knowledge-distillation/export_teacher_targets.py \
  --teacher sentence-transformers/all-mpnet-base-v2 \
  --split all \
  --device auto
```

Output goes to `knowledge-distillation/artifacts/kd_targets/`.

| Flag | Default | Description |
|---|---|---|
| `--teacher` | required | HuggingFace model name (must be cached locally) |
| `--split` | `all` | `train`, `validation`, `test`, `prompt`, or `all` (974 examples) |
| `--output-dir` | `knowledge-distillation/artifacts` | Root dir; targets saved under `<dir>/kd_targets/` |
| `--batch-size` | 32 | Encoding batch size |
| `--device` | `auto` | `auto` picks CUDA > MPS > CPU |
| `--force` | off | Overwrite existing `.npz` |

### Step 2 — Run KD

```bash
# Single config — listwise KD (default)
python knowledge-distillation/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets knowledge-distillation/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm \
  --epochs 2 --alpha 0.5 --temperature 4.0

# Single config — pairwise KD (Margin MSE)
python knowledge-distillation/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets knowledge-distillation/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm_pairwise \
  --epochs 2 --alpha 0.5 --kd-loss pairwise

# Single config — PairDistill (binary KL on pairwise preferences)
python knowledge-distillation/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets knowledge-distillation/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm_pairdistil \
  --epochs 2 --alpha 0.5 --kd-loss pairdistil

# Sweep 4 configs (alpha × epochs)
python knowledge-distillation/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets knowledge-distillation/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm_sweep \
  --full-matrix

# Quick schema check — zero-shot eval only, no training
python knowledge-distillation/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets knowledge-distillation/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --fast-smoke

# Resume an interrupted run
python knowledge-distillation/run_kd_experiments.py \
  --student sentence-transformers/all-MiniLM-L6-v2 \
  --teacher-targets knowledge-distillation/artifacts/kd_targets/sentence-transformers__all-mpnet-base-v2_all.npz \
  --run-id kd_minilm --resume
```

---

## All CLI flags (`run_kd_experiments.py`)

| Flag | Default | Description |
|---|---|---|
| `--student` | required | Student model name (cached locally) |
| `--teacher-targets` | required | Path to `.npz` from export step |
| `--run-id` | timestamp | Run identifier |
| `--output-dir` | `knowledge-distillation/results` | Output root relative to project root |
| `--device` | `auto` | `auto`, `cpu`, or `mps` |
| `--seed` | 42 | |
| `--epochs` | 1 | |
| `--batch-size` | 16 | |
| `--lr` | 2e-5 | |
| `--warmup-ratio` | 0.10 | |
| `--weight-decay` | 0.01 | |
| `--max-grad-norm` | 1.0 | |
| `--alpha` | 0.5 | KD loss weight (0 = pure MNR, 1 = pure distillation) |
| `--temperature` | 4.0 | Softmax temperature for similarity distributions (listwise only) |
| `--kd-loss` | `listwise` | `listwise` (KL on full sim matrix), `pairwise` (Margin MSE), or `pairdistil` (binary KL on pairwise preferences) |
| `--full-matrix` | off | Sweep 4 configs instead of using CLI hyperparams |
| `--resume` | off | Skip already-completed steps |
| `--fast-smoke` | off | Zero-shot eval only, no training |

---

## Sweep configs (`--full-matrix`)

| config_id | epochs | alpha | temperature |
|---|---|---|---|
| `kd_e1_b16_lr2e5_a05_t4` | 1 | 0.5 | 4.0 |
| `kd_e2_b16_lr2e5_a05_t4` | 2 | 0.5 | 4.0 |
| `kd_e1_b16_lr2e5_a03_t4` | 1 | 0.3 | 4.0 |
| `kd_e1_b16_lr2e5_a07_t4` | 1 | 0.7 | 4.0 |

---

## Inspect results

```bash
python -c "
import pandas as pd
r = 'knowledge-distillation/results/<run_id>'
m = pd.read_csv(f'{r}/metrics/metrics_all.csv')
print(m[m.method=='kd'][['protocol','config_id','mrr','recall@10','status']])
c = pd.read_csv(f'{r}/metrics/comparisons.csv')
print(c[['comparison','metric','base_value','compare_value','delta','ci_low','ci_high']])
"
```

Comparison written: **`kd_vs_student_zero_shot`** — bootstrap CI delta between the best KD config and the student's zero-shot baseline.

---

## Key columns in `metrics_all.csv`

| Column | Values |
|---|---|
| `method` | `pretrained` (zero-shot), `kd` |
| `stage` | `pretrained`, `kd_mnr` |
| `technique` | `zero_shot`, `kd_kl` (listwise), `kd_margin_mse` (pairwise), `kd_pairdistil` (pairdistil) |
| `protocol` | `heldout_test` (primary), `full_corpus` (diagnostic) |
| `status` | `success`, `failed` |

Output is CSV-compatible with `experiments/kai/scripts/run_mbpp_experiments.py` so results from both can be merged and plotted together.

---

## Tips

- **Export once, reuse everywhere.** The `.npz` is deterministic — same teacher + split always produces the same file.
- **Teacher must be stronger than the student.** Same model = no knowledge transfer.
- **Use `--split all` for export.** KD trains on train+validation+prompt (464 examples); `--split train` misses validation and prompt pairs.
- **`--fast-smoke` skips training.** Use it for import/schema checks only.
- **`--resume` is safe to re-run.** Already-completed steps are identified by their key in `metrics_all.csv` and skipped.
