# KD Investigation Report

Last updated: 2026-03-26

Companion manifests:
- `docs/kd_experiment_inventory.csv`
- `docs/kd_checkpoint_manifest.csv`
- `docs/kd_dataset_stats.csv`
- `docs/hf_publication_report.csv`
- `docs/hf_publication_report.md`
- `docs/kd_mbpp_teacher_supervised_embed_per_query.csv`
- `docs/kd_mbpp_teacher_supervised_embed_bucket_summary.csv`
- `docs/kd_codesearchnet_teacher_supervised_posthoc_topk_per_query.csv`
- `docs/kd_codesearchnet_teacher_supervised_posthoc_topk_bucket_summary.csv`

## 1. Objective

This report investigates why knowledge distillation (KD) is not improving the smaller retriever in the current code-search pipeline, and narrows the next KD path to a weak-student setup with real headroom.

The report is evidence-driven. Every experimental claim below is tied to either:
- evaluator outputs under `eval/runs/...`,
- saved checkpoint metrics under `artifacts/...`, or
- direct code inspection of the current suite.

Older historical results from `docs/EXPERIMENT_LOG.md` are kept as context, but they are explicitly marked `doc-only` when the original recoverable artifact is no longer present.

## 2. Current Implementation

The current implementation has three relevant layers.

First, the training harness lives under `src/mbpp_kd_suite/`. The main experiment entrypoint is `src/mbpp_kd_suite/experiment.py`, and the KD losses are implemented in `src/mbpp_kd_suite/training.py`. The current suite supports `supervised_student`, `score_distill`, `embed_distill`, `qed_align`, `distilcse_lite`, `pair_distill`, `adam_lite`, and `hpd`.

Second, the suite now supports both `MBPP` and `CodeSearchNet Python` as retrieval datasets. `src/mbpp_kd_suite/data.py` can load local CodeSearchNet JSONL data or fall back to remote loading, and the config surface now supports dataset path overrides and sample caps.

Third, reportable evaluation is decoupled from training. The evaluator under `eval/` is the standard scoring path for this report. It supports Hugging Face models, local Hugging Face checkpoints, suite-style student checkpoints, and shared Hugging Face repos with per-checkpoint subfolders. The evaluation protocol used here is exact paired retrieval with `MRR`, `Recall@1/5/10`, `nDCG@10`, `MAP@10`, runtime, and peak memory.

Fourth, reportable checkpoint publication is now standardized. The curated shared repo is `cs4248-nlp/cs4248-model-weights`, and only the four core CodeSearchNet checkpoints from the earlier investigation phase were uploaded. The upload layout is nested by dataset, teacher, student, method, seed, and checkpoint tag. The local publication record lives in `docs/hf_publication_report.csv` and `docs/hf_publication_report.md`.

## 3. Verified Experiments and Results

### 3.1 Audit Notes

One audit caveat matters before reading the tables:

- An early raw `all-MiniLM-L12-v2` MBPP row with `MRR=0.8923` exists in `eval/runs`. That row was produced before the MBPP adapter was fixed to prefer the official Hugging Face split-parquet data. It used the older unsplit local JSONL path and is not comparable to the official 500-example MBPP test protocol.
- The official MBPP raw-teacher reference for this report is the later evaluator run with `MRR=0.7860` on the corrected 500-example test split.

### 3.2 Dataset Audit

The current loader-backed dataset statistics are:

| Dataset | Split | Count | Avg Query Tokens | Avg Code Tokens | Avg Query Token Overlap Rate | Avg Query-Code Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| MBPP | train | 374 | 14.59 | 23.53 | 0.0693 | 0.0407 |
| MBPP | validation | 90 | 14.36 | 25.34 | 0.0607 | 0.0327 |
| MBPP | test | 500 | 14.44 | 24.59 | 0.0627 | 0.0369 |
| CodeSearchNet | train | 412178 | 37.71 | 100.03 | 0.9984 | 0.4553 |
| CodeSearchNet | validation | 23107 | 43.19 | 109.03 | 0.9982 | 0.4751 |
| CodeSearchNet | test | 22176 | 39.15 | 100.88 | 0.9969 | 0.4537 |

Interpretation:

CodeSearchNet is much longer than MBPP, but it is also far easier lexically. The query tokens almost entirely appear in the code/document side. MBPP has very low query-code lexical overlap, so matching there is more semantic and less docstring-like. This directly matches the observed difficulty gap in the evaluator results.

### 3.3 Core Baselines

#### Checkpoint-verified, official evaluator runs

| Evidence | Dataset | Model | Method | MRR | R@1 | R@5 | R@10 | nDCG@10 | MAP@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| checkpoint-verified | CodeSearchNet test | `sentence-transformers/all-MiniLM-L12-v2` | raw | 0.8956 | 0.8449 | 0.9577 | 0.9718 | 0.9138 | 0.8946 |
| checkpoint-verified | CodeSearchNet test | `sentence-transformers/all-MiniLM-L6-v2` | raw | 0.8773 | 0.8221 | 0.9447 | 0.9635 | 0.8977 | 0.8761 |
| checkpoint-verified | CodeSearchNet test | `nreimers/BERT-Tiny_L-2_H-128_A-2` | raw | 0.4472 | 0.3866 | 0.5120 | 0.5542 | 0.4685 | 0.4413 |
| checkpoint-verified | MBPP test | `sentence-transformers/all-MiniLM-L12-v2` | raw | 0.7860 | 0.6980 | 0.8920 | 0.9400 | 0.8216 | 0.7833 |
| checkpoint-verified | MBPP test | `sentence-transformers/all-MiniLM-L6-v2` | raw | 0.7717 | 0.6820 | 0.8840 | 0.9220 | 0.8060 | 0.7682 |
| checkpoint-verified | MBPP test | `nreimers/BERT-Tiny_L-2_H-128_A-2` | raw | 0.1001 | 0.0560 | 0.1240 | 0.1720 | 0.1063 | 0.0863 |

Immediate conclusion:

- The strong-student raw gap on CodeSearchNet is already tiny: `0.8956 - 0.8773 = 0.0183` MRR.
- The weak-student raw gap on CodeSearchNet is large: `0.8956 - 0.4472 = 0.4484` MRR.

That is the core reason the next KD path should target a weak student rather than another strong student.

### 3.4 Historical MBPP KD Results

#### Doc-only context from `docs/EXPERIMENT_LOG.md`

| Evidence | Dataset | Teacher | Student | Method | MRR | Notes |
|---|---|---|---|---|---:|---|
| doc-only | MBPP test | `all-MiniLM-L12-v2` | `all-MiniLM-L6-v2` | `supervised_student` | 0.7926 | fair 1-epoch MBPP comparison |
| doc-only | MBPP test | `all-MiniLM-L12-v2` | `all-MiniLM-L6-v2` | `embed_distill` | 0.7806 | fair 1-epoch MBPP comparison |
| doc-only | MBPP test | `all-MiniLM-L12-v2` | `all-MiniLM-L6-v2` | `embed_distill` | 0.7375 | strongest 3-epoch KD variant in historical MBPP runs |
| doc-only | MBPP test | `all-mpnet-base-v2` | `all-MiniLM-L6-v2` | `embed_distill` | 0.7137 | least-squares projection init fixed collapse but still trailed direct student |
| doc-only | MBPP test | `all-mpnet-base-v2` | `all-MiniLM-L6-v2` | `embed_distill` | 0.6940 | longer 8-epoch run overfit |

These doc-only MBPP runs consistently tell the same story: KD does not beat the fair supervised student baseline under the current retrieval setup.

## 4. What Failed and Why

There are three distinct failure modes.

### 4.1 Strong-student KD has very little room to improve on CodeSearchNet

The checkpoint-verified raw teacher/student gap on full CodeSearchNet test is only `0.0183` MRR (`0.8956` vs `0.8773`). That means a strong small bi-encoder is already near the teacher before KD is even applied.

The partially completed strong-student diagnosis also produced a saved `supervised_student` checkpoint. That checkpoint was evaluator-scored on the full test sets:

| Evidence | Dataset | Teacher | Student | Method | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---|---:|---:|---:|---:|
| checkpoint-verified | CodeSearchNet test | `all-MiniLM-L12-v2` | `all-MiniLM-L6-v2` | `supervised_student` | 0.9706 | 0.9486 | 0.9956 | 0.9979 |
| checkpoint-verified | MBPP test | `all-MiniLM-L12-v2` | `all-MiniLM-L6-v2` | `supervised_student` | 0.7243 | 0.6220 | 0.8540 | 0.8960 |

This is enough to support the diagnosis: a strong student can already saturate CodeSearchNet after standard supervised fine-tuning. Under that regime, KD is unlikely to show large retrieval gains, and the more defensible report story is efficiency or matching teacher quality, not absolute improvement.

### 4.2 Weak-student KD is currently optimizing the wrong thing

The weak-student pilot used:
- teacher: `sentence-transformers/all-MiniLM-L12-v2`
- student: `nreimers/BERT-Tiny_L-2_H-128_A-2`
- training slice: 5000 train / 1000 validation / 1000 test
- methods: `supervised_student`, `embed_distill`, `score_distill`

The saved pilot metrics on the 1000-example internal test slice were:

| Evidence | Dataset | Student | Method | MRR | R@1 | R@5 | R@10 | nDCG@10 | MAP@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny raw | direct baseline | 0.6399 | 0.5590 | 0.7280 | 0.7690 | 0.6666 | 0.6336 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny supervised | `supervised_student` | 0.8993 | 0.8520 | 0.9570 | 0.9730 | 0.9168 | 0.8983 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny KD | `score_distill` | 0.5998 | 0.5180 | 0.6900 | 0.7430 | 0.6294 | 0.5932 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny KD | `embed_distill` | 0.5940 | 0.5080 | 0.6940 | 0.7470 | 0.6254 | 0.5867 |

The full evaluator runs are more important. They show what survives outside the sampled slice:

| Evidence | Dataset | Student | Method | MRR | R@1 | R@5 | R@10 | nDCG@10 | MAP@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| checkpoint-verified | CodeSearchNet test | tiny raw | raw | 0.4472 | 0.3866 | 0.5120 | 0.5542 | 0.4685 | 0.4413 |
| checkpoint-verified | CodeSearchNet test | tiny supervised | `supervised_student` | 0.8495 | 0.7951 | 0.9138 | 0.9353 | 0.8690 | 0.8484 |
| checkpoint-verified | CodeSearchNet test | tiny KD | `score_distill` | 0.3917 | 0.3347 | 0.4519 | 0.4936 | 0.4130 | 0.3856 |
| checkpoint-verified | CodeSearchNet test | tiny KD | `embed_distill` | 0.3941 | 0.3347 | 0.4572 | 0.5015 | 0.4161 | 0.3880 |

This is the clearest result in the investigation.

The supervised tiny student closes almost the entire teacher gap on CodeSearchNet:
- raw teacher gap to raw tiny: `0.4484`
- raw teacher gap to supervised tiny: `0.0461`
- gap reduction from plain supervised fine-tuning: `0.4023` MRR

The KD variants do the opposite. They do not just fail to beat the supervised tiny student; they underperform the raw tiny student on the full evaluator.

That strongly suggests the current KD objective is pulling the tiny student toward the teacher space in a way that damages retrieval structure instead of helping it.

### 4.3 KD also damages cross-dataset transfer

The MBPP transfer numbers make the same point.

| Evidence | Dataset | Student | Method | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---:|---:|---:|---:|
| checkpoint-verified | MBPP test | tiny raw | raw | 0.1001 | 0.0560 | 0.1240 | 0.1720 |
| checkpoint-verified | MBPP test | tiny supervised | `supervised_student` | 0.1621 | 0.1080 | 0.1960 | 0.2460 |
| checkpoint-verified | MBPP test | tiny KD | `score_distill` | 0.0872 | 0.0480 | 0.1040 | 0.1460 |
| checkpoint-verified | MBPP test | tiny KD | `embed_distill` | 0.0680 | 0.0300 | 0.0900 | 0.1380 |

Interpretation:

- plain CodeSearchNet fine-tuning helps the tiny student transfer to MBPP (`0.1001 -> 0.1621`)
- both KD variants transfer worse than the raw tiny model

So KD is not only failing in-domain. It is also making the weak student less robust out of domain.

## 5. Strong-Student Diagnosis on CodeSearchNet

The current verified strong-student diagnosis is:

1. The raw teacher/student gap is already very small on full CodeSearchNet (`0.8956` vs `0.8773`).
2. A strong student fine-tuned under the current pipeline reaches `0.9706` MRR on full CodeSearchNet test and `0.7243` on MBPP transfer.
3. This makes CodeSearchNet a poor place to expect visible KD gains for a strong student, because supervised fine-tuning alone already saturates the retrieval task.

What is still missing:

- a checkpoint-verified strong-student KD run on CodeSearchNet under the same exact pipeline

That missing artifact is now an evidence gap, not a conceptual blocker. The report can already justify the next decision without it: KD should not be evaluated mainly on a strong student if the goal is visible improvement.

## 6. Weak-Student KD Plan and Results

### 6.1 Current verdict

The weak-student direction is the correct next KD path, because the raw gap is large enough to justify distillation. However, the current KD losses are not helping that weak student.

The verified outcome is:
- `supervised_student` is the correct baseline to beat
- `score_distill` and `embed_distill` both underperform that baseline badly
- on the full evaluator, both KD variants are even below the raw tiny model

### 6.2 Revised hybrid-KD pilot (student loss primary, KD secondary)

To test whether the failure was caused by KD overpowering the retrieval objective, a second weak-student pilot was run with the supervised pair loss kept as the primary objective and KD added only as a regularizer.

Pilot setup:
- dataset: `CodeSearchNet Python`
- caps: `5000 train / 1000 validation / 1000 test`
- teacher: `sentence-transformers/all-MiniLM-L12-v2`
- student: `nreimers/BERT-Tiny_L-2_H-128_A-2`
- methods:
  - `supervised_student`
  - `supervised_score_distill`
  - `supervised_embed_distill`
  - `supervised_embed_distill_warmup`
  - `supervised_embed_distill_projinit`
- epochs: `3`
- checkpoint strategy: `all_epochs_then_prune`
- checkpoint selection: best validation `MRR`

The validation-gated pilot result is still negative:

| Evidence | Dataset | Student | Method | Best Val MRR | Best Epoch | Test MRR | Test R@1 | Test R@5 | Test R@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny supervised | `supervised_student` | 0.9037 | 3 | 0.9098 | 0.8650 | 0.9640 | 0.9760 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny hybrid KD | `supervised_score_distill` | 0.8983 | 3 | 0.9024 | 0.8540 | 0.9610 | 0.9740 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny hybrid KD | `supervised_embed_distill` | 0.8934 | 3 | 0.9001 | 0.8520 | 0.9620 | 0.9750 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny hybrid KD | `supervised_embed_distill_warmup` | 0.8816 | 2 | 0.8932 | 0.8440 | 0.9550 | 0.9670 |
| checkpoint-verified | CodeSearchNet pilot test (1000) | tiny hybrid KD | `supervised_embed_distill_projinit` | 0.8587 | 3 | 0.8628 | 0.8080 | 0.9320 | 0.9530 |

Interpretation:

- Keeping supervised retrieval primary prevents the catastrophic collapse seen in the earlier pure-KD tiny runs.
- But none of the hybrid variants beats the plain supervised tiny baseline on validation or test.
- `supervised_score_distill` is the strongest hybrid, but it still trails the supervised baseline by `0.0054` validation MRR and `0.0074` test MRR.

Because no hybrid method beat the pilot baseline, the planned full CodeSearchNet confirmatory run was not advanced. That was an intentional gate, not an omission: scaling a method that already lost the pilot wastes compute and does not change the report decision.

### 6.3 Posthoc KD rescue attempt

The final rescue attempt followed a stricter rule: do not retrain the tiny student from scratch, and do not let KD define the whole objective. Instead, warm-start from the saved supervised tiny checkpoint and continue training for two additional epochs with KD as a light secondary regularizer.

Fixed setup:
- warm-start checkpoint: `artifacts/codesearchnet_tiny_hybrid_pilot/20260326_155721/supervised_student/best-val`
- dataset: `CodeSearchNet Python`
- caps: `5000 train / 1000 validation / 1000 test`
- teacher: `sentence-transformers/all-MiniLM-L12-v2`
- student: `nreimers/BERT-Tiny_L-2_H-128_A-2`
- checkpoint strategy: `all_epochs_then_prune`
- distill weight: `0.10`
- alignment, pairwise, relation weights: `0`

Methods:
- `posthoc_score_distill`: supervised retrieval loss plus full in-batch teacher score KL
- `posthoc_teacher_topk_distill`: supervised retrieval loss plus teacher score KL restricted to the gold positive and the teacher’s top-3 in-batch negatives

The gate for success was:
- validation `MRR` improvement of at least `+0.003` over the frozen supervised baseline
- no regression on pilot test `MRR`

#### Decoupled evaluator results on the capped pilot split

| Evidence | Dataset | Method | Validation MRR | Test MRR | Test R@1 | Test R@5 | Test R@10 | Test nDCG@10 | Test MAP@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| checkpoint-verified | CodeSearchNet pilot (1000) | `supervised_student` | 0.9037 | 0.9098 | 0.8650 | 0.9640 | 0.9760 | 0.9255 | 0.9088 |
| checkpoint-verified | CodeSearchNet pilot (1000) | `posthoc_score_distill` | 0.9205 | 0.9225 | 0.8830 | 0.9690 | 0.9830 | 0.9368 | 0.9217 |
| checkpoint-verified | CodeSearchNet pilot (1000) | `posthoc_teacher_topk_distill` | 0.9269 | 0.9249 | 0.8870 | 0.9710 | 0.9870 | 0.9398 | 0.9244 |

This is the first KD result in the suite that clearly improves the weak student on the same in-domain pilot protocol after the student has already learned a usable retrieval geometry. The top-k posthoc variant is the winner on the pilot gate.

#### MBPP transfer check for the pilot winner

The rescue attempt still fails the transfer gate.

| Evidence | Dataset | Method | MRR | R@1 | R@5 | R@10 |
|---|---|---|---:|---:|---:|---:|
| checkpoint-verified | MBPP test | `supervised_student` | 0.1677 | 0.1040 | 0.2100 | 0.2820 |
| checkpoint-verified | MBPP test | `posthoc_teacher_topk_distill` | 0.1492 | 0.0940 | 0.1860 | 0.2580 |

The acceptance threshold was “no more than `0.005` below the supervised baseline on MBPP MRR.” The observed regression is `0.0185`, so the method does not qualify as a usable rescue.

Interpretation:
- posthoc KD can improve easy in-domain CodeSearchNet ranking for the tiny student
- but the improvement does not transfer to MBPP
- therefore KD is still not worth adopting as the final retrieval recipe for this project

### 6.4 Why this likely happens

The code and results together suggest four concrete issues.

First, the original KD targets were too teacher-space oriented. In the weak-student pilot, both pure KD methods collapse on the full evaluator. That pattern fits a student that learns to imitate the teacher representation locally without preserving the retrieval geometry needed for broader generalization.

Second, the current KD objective competes with the main retrieval objective unless the student has already learned a good geometry. The posthoc results show that KD helps only after supervised training, not during the main learning phase.

Third, even after warm-starting, the teacher signal still appears overly tuned to the in-domain CodeSearchNet ranking structure. The posthoc top-k method improves the capped CodeSearchNet pilot but regresses on MBPP, which is exactly the opposite of what a generally useful code-retrieval distillation signal should do.

Fourth, small-sample internal evaluation can still be misleading if used alone. The posthoc winner is real on the 1000-example CodeSearchNet pilot, but the MBPP transfer check shows that a local gain on the easy dataset is not enough to justify adopting the method.

### 6.5 Query-level evidence

The MBPP per-query analysis used:
- teacher: raw `all-MiniLM-L12-v2`
- baseline: tiny `supervised_student`
- KD: tiny `embed_distill`

Bucket summary on the official 500-example MBPP test split:

| Bucket | Count | Avg Teacher Rank | Avg Supervised Rank | Avg KD Rank |
|---|---:|---:|---:|---:|
| `kd_helps` | 127 | 6.77 | 184.35 | 112.10 |
| `kd_hurts` | 355 | 3.77 | 83.60 | 198.43 |
| `student_rank1` | 54 | 1.13 | 1.00 | 34.02 |
| `teacher_wins_kd_not_transfer` | 314 | 2.35 | 94.47 | 217.44 |

This remains the clearest diagnosis of why the original KD losses failed: KD hurts on most MBPP queries, often destroys already-correct rank-1 predictions, and frequently fails to transfer teacher advantages even when the teacher is clearly better.

The final posthoc CodeSearchNet comparison used:
- teacher: raw `all-MiniLM-L12-v2`
- baseline: tiny `supervised_student`
- KD: `posthoc_teacher_topk_distill`

Bucket summary on the capped 1000-example CodeSearchNet pilot test split:

| Bucket | Count | Avg Teacher Rank | Avg Supervised Rank | Avg Posthoc KD Rank |
|---|---:|---:|---:|---:|
| `kd_helps` | 77 | 8.42 | 17.94 | 7.71 |
| `kd_hurts` | 24 | 3.54 | 2.17 | 3.75 |
| `student_rank1` | 865 | 2.18 | 1.00 | 1.02 |
| `teacher_wins_kd_not_transfer` | 38 | 1.26 | 2.68 | 3.26 |

That comparison is much healthier than the original MBPP KD failure:
- on the easy in-domain slice, KD helps more queries than it hurts
- when the baseline is already rank 1, posthoc KD usually preserves that advantage
- but there is still a persistent set of teacher-better examples where the KD signal does not transfer correctly

Representative `kd_hurts` examples on the CodeSearchNet pilot include:
- “Detect operating system.”
- “Download ONE PART of the course.”
- “Loads a tab-delimited file into a database table”

Representative `kd_helps` examples include:
- “Downloads Dailymotion videos by URL.”
- “video page”
- “wrapper”

So the final picture is consistent. KD can help the weak student when used late and locally on CodeSearchNet, but it is still not robust enough to improve the final project objective across datasets.

## 7. Cross-Encoder Result as a Negative Finding

The current verified cross-encoder result on MBPP is:

| Evidence | Dataset | Model | Method | MRR | R@1 | R@5 | R@10 |
|---|---|---|---|---:|---:|---:|---:|
| checkpoint-verified | MBPP test | `cross-encoder/ms-marco-TinyBERT-L2` | exhaustive cross-encoder scoring | 0.5156 | 0.4100 | 0.6340 | 0.6960 |
| checkpoint-verified | MBPP test | `sentence-transformers/all-MiniLM-L6-v2` | bi-encoder raw | 0.7717 | 0.6820 | 0.8840 | 0.9220 |

So the tested cross-encoder is both slower and worse than the bi-encoder baseline on MBPP.

There is also a practical scaling issue. Exhaustive CodeSearchNet cross-encoder scoring would require scoring roughly `22,176 x 22,176` query-code pairs for the full test set, which is not viable under the current compute budget. For this project, cross-encoders should be documented as a negative result rather than treated as the main retrieval path.

## 8. Synthetic Data Status

From direct repo inspection, there is currently no standardized synthetic-data generation module inside `mbpp_kd_suite/`, no saved synthetic-data checkpoints, and no checkpoint-verified synthetic-data experiment in the current manifests.

So the current project status is:
- synthetic data remains a possible future augmentation direction
- it is not part of the current verified KD evidence base
- no claim in this report depends on synthetic data

## 9. Recommended Next KD Directions

The project should now stop broad KD experimentation and pivot to analysis and presentation. The one last posthoc rescue attempt was informative, but it still does not produce a final recipe that is better than the supervised tiny baseline on the combined project criteria.

### 9.1 Keep the tiny student as the analysis target

Do not spend any more project time trying to show KD gains on a strong student. The strong student already saturates CodeSearchNet under plain supervised fine-tuning.

### 9.2 Do not add another KD training branch inside the project scope

The current evidence does not justify another training sweep. The project now has three checkpoint-verified conclusions:
- pure KD on the weak student fails badly
- hybrid during-training KD avoids collapse but still loses to plain supervised training
- posthoc KD can improve in-domain CodeSearchNet pilot ranking, but regresses on MBPP transfer

That is enough evidence for the final report. Another KD branch would most likely create more noise than insight.

If future work is needed beyond this project, the only defensible next ideas are:
- two-stage post-supervised KD with a very short finetuning phase
- hard-negative or rank-only distillation restricted to teacher-better cases

Those should be framed as future work, not current action items.

### 9.3 Keep per-epoch checkpoint selection, but prune aggressively

The current suite now supports per-epoch checkpoint capture with best-validation materialization and pruning. The revised hybrid pilot used `all_epochs_then_prune`, which kept:
- `best-val/`
- `history.csv`
- `history.json`
- `checkpoint_index.json`

while removing the temporary epoch directories after the winner was known.

This should remain the standard for reportable runs. It gives checkpoint-verified best-model selection without permanently bloating the local repo.

### 9.4 Keep Hugging Face publication curated

Only selected reportable checkpoints should be uploaded to Hugging Face. The current curated upload set is already in place under `cs4248-nlp/cs4248-model-weights`, and it intentionally contains only four core CodeSearchNet checkpoints:
- strong-student supervised reference
- tiny `supervised_student`
- tiny `score_distill`
- tiny `embed_distill`

The repo layout should stay nested and explicit:
- `checkpoints/codesearchnet/python/teacher_.../student_.../method/seed42/best-val/`

The revised hybrid and posthoc pilot checkpoints were not uploaded, because neither branch produced a final report-worthy replacement for the supervised tiny baseline.

### 9.5 Keep the report distinction explicit

The final paper/report should separate three claims:
- `strong student`: CodeSearchNet is already easy enough that supervised fine-tuning saturates a strong small retriever
- `weak student`: there is real room for KD, but the current KD objectives fail or overfit to the easy in-domain dataset
- `negative result`: cross-encoder reranking is not competitive for this scope and dataset setup

## Bottom Line

The investigation result is not “KD never helps.”

The verified result is more precise:

- On CodeSearchNet, strong students already have too little headroom for KD to show convincing gains.
- On a weak student, pure KD and hybrid during-training KD hurt or fail to improve retrieval quality.
- A posthoc top-k KD regularizer can improve the easy in-domain CodeSearchNet pilot after supervised training, but it still hurts MBPP transfer.
- So KD is not worth adopting as the final project retrieval recipe. The final project story should be a well-supported negative result with analysis, not another round of training.
