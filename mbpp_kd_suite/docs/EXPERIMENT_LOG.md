# Experiment Log

This log records runs executed in `mbpp_kd_suite/`.

Historical note: on March 15, 2026, the older top-level run buckets were moved under `artifacts/legacy/`.
If an entry below mentions a path like `smoke_runs/<timestamp>/...` or `fair_compare_mbpp/<timestamp>/...`, the current on-disk path is `artifacts/legacy/<that-bucket>/<timestamp>/...`.

## Planned comparison

Baseline comparisons use the same MBPP protocol as the earlier internal baseline runs:

- `direct_big_teacher`
- `direct_small_student`
- `score_distill`
- `embed_distill`
- selected paper-inspired methods from:
  - `qed_align`
  - `distilcse_lite`
  - `hard_negative_pair_distill` (logged in older tables as `pair_distill`)
  - `all_pairs_distill`
  - `adam_lite`
  - `hpd`

## Pending runs

## Run: `20260312_230307` (`smoke_runs/`, 1 epoch)

### Command

```bash
cd mbpp_kd_suite
./.venv/bin/mbpp-kd-suite --methods score_distill --epochs 1 --batch-size 16 --eval-batch-size 32 --output-dir smoke_runs --skip-diagnostics
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.7860 | 0.6980 | 0.8920 | 0.9400 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| score_distill | 0.7144 | 0.6140 | 0.8460 | 0.9040 |

Artifacts:
- `smoke_runs/20260312_230307/results_summary.json`

## Run: `20260312_230346` (`quick_compare/`, 1 epoch)

### Command

```bash
cd mbpp_kd_suite
./.venv/bin/mbpp-kd-suite --methods score_distill,embed_distill,qed_align,distilcse_lite,hard_negative_pair_distill,adam_lite,hpd --epochs 1 --batch-size 16 --eval-batch-size 32 --output-dir quick_compare --skip-diagnostics
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.7860 | 0.6980 | 0.8920 | 0.9400 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| score_distill | 0.7144 | 0.6140 | 0.8460 | 0.9040 |
| embed_distill | 0.7122 | 0.6000 | 0.8560 | 0.9060 |
| qed_align | 0.7083 | 0.6000 | 0.8480 | 0.9140 |
| distilcse_lite | 0.7119 | 0.6040 | 0.8540 | 0.9140 |
| pair_distill | 0.6990 | 0.5820 | 0.8480 | 0.8980 |
| adam_lite | 0.7017 | 0.5920 | 0.8380 | 0.8920 |
| hpd | 0.0471 | 0.0080 | 0.0640 | 0.1140 |

### First-pass read

- The direct small baseline remains stronger than all 1-epoch distillation variants on MBPP test retrieval.
- Among the newly added paper-inspired methods, `distilcse_lite` is the strongest first pass by MRR.
- `hpd` is not competitive in the current retrieval adaptation and needs redesign before further comparison.

Artifacts:
- `quick_compare/20260312_230346/results_summary.json`
- `quick_compare/20260312_230346/config.json`

## Run: `20260312_230741` (`longer_compare/`, 3 epochs)

### Command

```bash
cd mbpp_kd_suite
./.venv/bin/mbpp-kd-suite --methods score_distill,embed_distill,qed_align,distilcse_lite,hard_negative_pair_distill,adam_lite,hpd --epochs 3 --batch-size 16 --eval-batch-size 32 --output-dir longer_compare --skip-diagnostics
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.7860 | 0.6980 | 0.8920 | 0.9400 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| score_distill | 0.7257 | 0.6140 | 0.8700 | 0.9200 |
| embed_distill | 0.7375 | 0.6300 | 0.8660 | 0.9300 |
| qed_align | 0.7359 | 0.6260 | 0.8700 | 0.9260 |
| distilcse_lite | 0.7245 | 0.6180 | 0.8620 | 0.9300 |
| pair_distill | 0.7163 | 0.6000 | 0.8660 | 0.9180 |
| adam_lite | 0.7269 | 0.6180 | 0.8740 | 0.9200 |
| hpd | 0.1924 | 0.0920 | 0.2740 | 0.3980 |

### First-pass read

- Additional training helped almost every method relative to the 1-epoch run.
- `embed_distill` is now the strongest distilled method by MRR, with `qed_align` very close behind.
- The direct small baseline still leads all distilled methods on MBPP test retrieval.
- `hpd` improved from its 1-epoch failure mode but remains far from competitive.

Artifacts:
- `longer_compare/20260312_230741/results_summary.json`
- `longer_compare/20260312_230741/config.json`

## Teacher Trials

### `sentence-transformers/multi-qa-mpnet-base-dot-v1`

Command shape:

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/transformers HF_DATASETS_CACHE=.hf_cache/datasets ./.venv/bin/python <direct-baseline-script>
```

Test metrics:

| Teacher | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| current `all-MiniLM-L12-v2` | 0.7860 | 0.6980 | 0.8920 | 0.9400 |
| `multi-qa-mpnet-base-dot-v1` | 0.7715 | 0.6720 | 0.8960 | 0.9400 |

Read:

- This is not an upgrade over the current teacher on MBPP.

Artifacts:
- `teacher_trials/multi_qa_mpnet_base_dot_v1/direct_baseline_metrics.json`

### `sentence-transformers/all-mpnet-base-v2`

Test metrics:

| Teacher | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| current `all-MiniLM-L12-v2` | 0.7860 | 0.6980 | 0.8920 | 0.9400 |
| `all-mpnet-base-v2` | 0.8229 | 0.7480 | 0.9240 | 0.9640 |

Read:

- `all-mpnet-base-v2` is a clear improvement over the current MiniLM teacher.
- This is the strongest generic embedding teacher tried so far that fits the current pipeline without custom shims.

Artifacts:
- `teacher_trials/all_mpnet_base_v2/direct_baseline_metrics.json`

## Run: `20260312_233935` (`mpnet_teacher_compare/`, 3 epochs)

### Command

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/transformers HF_DATASETS_CACHE=.hf_cache/datasets ./.venv/bin/mbpp-kd-suite --teacher-model sentence-transformers/all-mpnet-base-v2 --methods score_distill,embed_distill,qed_align,adam_lite --epochs 3 --batch-size 16 --eval-batch-size 32 --output-dir mpnet_teacher_compare --skip-diagnostics
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.8229 | 0.7480 | 0.9240 | 0.9640 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| score_distill | 0.2476 | 0.1340 | 0.3480 | 0.4800 |
| embed_distill | 0.2610 | 0.1320 | 0.3840 | 0.5400 |
| qed_align | 0.2513 | 0.1280 | 0.3720 | 0.5340 |
| adam_lite | 0.2367 | 0.1180 | 0.3480 | 0.4760 |

### Read

- The stronger MPNet teacher clearly improves the direct-teacher baseline.
- Under the current asymmetric KD setup, distillation becomes much worse rather than better.
- This strongly suggests the current losses and inference coupling are not robust to a larger teacher-space shift.

Artifacts:
- `mpnet_teacher_compare/20260312_233935/results_summary.json`
- `mpnet_teacher_compare/20260312_233935/config.json`

## Run: `20260313_222015` (`mpnet_projection_init_compare/`, 3 epochs)

### Command

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/hub HF_DATASETS_CACHE=.hf_cache/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/mbpp-kd-suite --teacher-model sentence-transformers/all-mpnet-base-v2 --methods embed_distill --epochs 3 --batch-size 16 --eval-batch-size 32 --output-dir mpnet_projection_init_compare --projection-init least_squares_both
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.8229 | 0.7480 | 0.9240 | 0.9640 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| embed_distill | 0.7137 | 0.6020 | 0.8660 | 0.9200 |

### Read

- The `MiniLM -> MPNet` collapse was not a hard architecture incompatibility.
- A large part of the failure came from randomly initializing the student's `384 -> 768` projection head and trying to learn it from only 374 MBPP training pairs.
- Initializing that projection with least-squares alignment on train queries and codes raised `embed_distill` from `0.2610` to `0.7137` test MRR.
- The method still trails the direct small baseline, but the catastrophic failure mode is resolved.

Artifacts:
- `mpnet_projection_init_compare/20260313_222015/results_summary.json`
- `mpnet_projection_init_compare/20260313_222015/embed_distill/metrics.json`

## Run: `20260313_223010` (`mpnet_projection_longer/`, 8 epochs)

### Command

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/hub HF_DATASETS_CACHE=.hf_cache/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/mbpp-kd-suite --teacher-model sentence-transformers/all-mpnet-base-v2 --methods embed_distill --epochs 8 --batch-size 16 --eval-batch-size 32 --output-dir mpnet_projection_longer --projection-init least_squares_both
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.8229 | 0.7480 | 0.9240 | 0.9640 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| embed_distill | 0.6940 | 0.5700 | 0.8480 | 0.9180 |

### Read

- Longer training did not help the fixed `MiniLM -> MPNet` setup.
- The best 3-epoch run (`MRR=0.7137`) outperformed the 8-epoch run (`MRR=0.6940`), so the method is now overfitting rather than undertraining.
- This shifts the remaining bottleneck away from optimization and toward the asymmetric student-query vs teacher-doc setup.

Artifacts:
- `mpnet_projection_longer/20260313_223010/results_summary.json`
- `mpnet_projection_longer/20260313_223010/embed_distill/metrics.json`

## Run: `20260313_223554` (`mpnet_symmetric_eval/`, 3 epochs, symmetric selection)

### Command

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/hub HF_DATASETS_CACHE=.hf_cache/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/mbpp-kd-suite --teacher-model sentence-transformers/all-mpnet-base-v2 --methods embed_distill --epochs 3 --batch-size 16 --eval-batch-size 32 --output-dir mpnet_symmetric_eval --projection-init least_squares_both --eval-mode symmetric
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.8229 | 0.7480 | 0.9240 | 0.9640 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| embed_distill | 0.7385 | 0.6320 | 0.8780 | 0.9220 |

### Read

- This run makes symmetric retrieval the primary validation and test metric instead of treating it as a diagnostic only.
- The symmetric `embed_distill` result is `0.7385` test MRR, versus `0.6878` for the same checkpoint under asymmetric evaluation.
- That confirms the student is materially better when allowed to own both query and code embeddings, but it still trails the direct small baseline.

Artifacts:
- `mpnet_symmetric_eval/20260313_223554/results_summary.json`
- `mpnet_symmetric_eval/20260313_223554/embed_distill/metrics.json`

## Run: `20260314_124824` (`taco_smoke_runs/`, TACO, 1 epoch)

### Command

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/hub HF_DATASETS_CACHE=.hf_cache/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/mbpp-kd-suite --teacher-model /Users/samuellee/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/936af83a2ecce5fe87a09109ff5cbcefe073173a --dataset-name BEE-spoke-data/TACO-hf --methods embed_distill --epochs 1 --batch-size 16 --eval-batch-size 32 --output-dir taco_smoke_runs --skip-diagnostics
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.1424 | 0.1040 | 0.1710 | 0.2090 |
| direct_small_student | 0.1336 | 0.0930 | 0.1650 | 0.2050 |
| embed_distill | 0.1603 | 0.1080 | 0.2010 | 0.2590 |

### Read

- The TACO adapter is viable as a larger MBPP-like retrieval task, with `18,493` train, `1,000` validation, and `1,000` test examples after filtering and the validation holdout.
- On this larger dataset, even a 1-epoch `embed_distill` run improved over both the direct teacher and the direct small baseline.
- This is the first result in the suite where the inherited distillation objective clearly beats the direct small baseline without needing a stronger teacher.
- The run still used the suite's default asymmetric evaluation mode, so future TACO comparisons should keep that fixed or switch the whole comparison set to symmetric evaluation.

### Follow-up Attempt

- I also started a TACO follow-up with `all-mpnet-base-v2` plus `--projection-init least_squares_both`, but stopped it before completion.
- On CPU, the dominant cost is teacher-side precomputation over the full TACO train corpus, and the MPNet teacher remained too slow to finish in a reasonable iteration loop.
- Increasing `--eval-batch-size` from `32` to `64` did not materially improve wall-clock time on this machine.

Artifacts:
- `taco_smoke_runs/20260314_124824/results_summary.json`
- `taco_smoke_runs/20260314_124824/embed_distill/metrics.json`

## Run: `20260314_140409` (`fair_compare_mbpp/`, MBPP, 1 epoch, symmetric fair comparison)

### Command

```bash
cd mbpp_kd_suite
HF_HOME=.hf_cache TRANSFORMERS_CACHE=.hf_cache/hub HF_DATASETS_CACHE=.hf_cache/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run mbpp-kd-suite --teacher-model /Users/samuellee/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/936af83a2ecce5fe87a09109ff5cbcefe073173a --student-model /Users/samuellee/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --methods supervised_student,embed_distill --epochs 1 --batch-size 16 --eval-batch-size 32 --eval-mode symmetric --output-dir fair_compare_mbpp --skip-diagnostics
```

### Test Metrics

| Model | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| direct_big_teacher | 0.7860 | 0.6980 | 0.8920 | 0.9400 |
| direct_small_student | 0.7717 | 0.6820 | 0.8840 | 0.9220 |
| supervised_student | 0.7926 | 0.7020 | 0.9000 | 0.9400 |
| embed_distill | 0.7806 | 0.6920 | 0.8920 | 0.9240 |

### Read

- This is the first fair MBPP comparison in the suite where the student baseline is trained under the same budget as the KD method.
- Under that comparison, `embed_distill` no longer beats the student baseline: it trails `supervised_student` by `0.0120` MRR.
- The old zero-shot small-student baseline was therefore understating how strong normal student fine-tuning already is on MBPP.

Artifacts:
- `fair_compare_mbpp/20260314_140409/results_summary.json`
- `fair_compare_mbpp/20260314_140409/supervised_student/metrics.json`
- `fair_compare_mbpp/20260314_140409/embed_distill/metrics.json`

Recommended first run:

```bash
cd mbpp_kd_suite
uv run mbpp-kd-suite --epochs 3 --batch-size 32 --eval-batch-size 64
```
