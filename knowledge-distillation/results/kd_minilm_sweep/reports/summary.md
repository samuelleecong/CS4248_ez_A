# MBPP KD Results: `kd_minilm_sweep`

## Run Info
- Student : `sentence-transformers/all-MiniLM-L6-v2`
- Teacher : `sentence-transformers/all-mpnet-base-v2`
- Device  : `mps`
- Seed    : `42`
- Started : 2026-03-15T10:24:51.138263+00:00
- Finished: 2026-03-15T10:25:45.654158+00:00

## Held-out Test Ranking
| method | stage | technique | model_name | config_id | mrr | recall@1 | recall@5 | recall@10 | recall@20 | map@10 | ndcg@10 | latency_ms_per_query | throughput_qps | model_params_m | model_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e2_b16_lr2e5_a05_t4 | 0.807475 | 0.726000 | 0.912000 | 0.946000 | 0.976000 | 0.804861 | 0.839471 | 0.497 | 2011.4 | 22.713 | 90.9 |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e1_b16_lr2e5_a05_t4 | 0.804034 | 0.722000 | 0.906000 | 0.944000 | 0.976000 | 0.801187 | 0.836167 | 0.369 | 2709.7 | 22.713 | 90.9 |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e1_b16_lr2e5_a07_t4 | 0.800418 | 0.712000 | 0.904000 | 0.948000 | 0.972000 | 0.797998 | 0.834715 | 0.433 | 2310.5 | 22.713 | 90.9 |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e1_b16_lr2e5_a03_t4 | 0.800119 | 0.716000 | 0.900000 | 0.942000 | 0.974000 | 0.797163 | 0.832625 | 0.367 | 2726.2 | 22.713 | 90.9 |
| pretrained | pretrained | zero_shot | sentence-transformers/all-MiniLM-L6-v2 | zero_shot | 0.771397 | 0.682000 | 0.884000 | 0.922000 | 0.954000 | 0.767905 | 0.805753 | 0.379 | 2637.5 | 22.713 | 90.9 |

## Full-corpus Ranking
| method | stage | technique | model_name | config_id | mrr | recall@1 | recall@5 | recall@10 | recall@20 | map@10 | ndcg@10 | latency_ms_per_query | throughput_qps | model_params_m | model_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e2_b16_lr2e5_a05_t4 | 0.740464 | 0.625257 | 0.896304 | 0.940452 | 0.969199 | 0.737595 | 0.787399 |  |  |  |  |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e1_b16_lr2e5_a03_t4 | 0.731738 | 0.614990 | 0.878850 | 0.930185 | 0.968172 | 0.728452 | 0.777905 |  |  |  |  |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e1_b16_lr2e5_a05_t4 | 0.730490 | 0.614990 | 0.878850 | 0.931211 | 0.969199 | 0.727090 | 0.777033 |  |  |  |  |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e1_b16_lr2e5_a07_t4 | 0.725338 | 0.603696 | 0.878850 | 0.933265 | 0.965092 | 0.722192 | 0.773894 |  |  |  |  |
| pretrained | pretrained | zero_shot | sentence-transformers/all-MiniLM-L6-v2 | zero_shot | 0.696345 | 0.577002 | 0.848049 | 0.909651 | 0.939425 | 0.692732 | 0.745635 |  |  |  |  |

## Comparisons
| run_id | timestamp | comparison | protocol | metric | base_method | base_stage | base_model | compare_method | compare_stage | compare_model | base_value | compare_value | delta | ci_low | ci_high | n_bootstrap | quality_retention | speedup | size_ratio | teacher_mrr | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kd_minilm_sweep | 2026-03-15T10:25:45.615701+00:00 | kd_vs_student_zero_shot | heldout_test | mrr | pretrained | pretrained | sentence-transformers/all-MiniLM-L6-v2 | kd | kd_mnr | sentence-transformers/all-MiniLM-L6-v2 | 0.771397 | 0.807475 | 0.036079 | 0.020387 | 0.052914 | 2000 | 0.9820774562455026 | 3.661972 | 4.820411 | 0.822211 | success | bootstrap_delta_mean=0.036576 |
| kd_minilm_sweep | 2026-03-15T10:25:45.632934+00:00 | kd_vs_student_zero_shot | heldout_test | recall@1 | pretrained | pretrained | sentence-transformers/all-MiniLM-L6-v2 | kd | kd_mnr | sentence-transformers/all-MiniLM-L6-v2 | 0.682000 | 0.726000 | 0.044000 | 0.018000 | 0.072000 | 2000 |  | 3.661972 | 4.820411 | 0.822211 | success | bootstrap_delta_mean=0.044807 |
| kd_minilm_sweep | 2026-03-15T10:25:45.650395+00:00 | kd_vs_student_zero_shot | heldout_test | recall@10 | pretrained | pretrained | sentence-transformers/all-MiniLM-L6-v2 | kd | kd_mnr | sentence-transformers/all-MiniLM-L6-v2 | 0.922000 | 0.946000 | 0.024000 | 0.010000 | 0.040000 | 2000 |  | 3.661972 | 4.820411 | 0.822211 | success | bootstrap_delta_mean=0.023892 |