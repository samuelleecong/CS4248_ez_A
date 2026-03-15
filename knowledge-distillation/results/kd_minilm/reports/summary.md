# MBPP KD Results: `kd_minilm`

## Run Info
- Student : `sentence-transformers/all-MiniLM-L6-v2`
- Teacher : `sentence-transformers/all-mpnet-base-v2`
- Device  : `mps`
- Seed    : `42`
- Started : 2026-03-08T18:02:27.186944+00:00
- Finished: 2026-03-08T18:02:49.629742+00:00

## Held-out Test Ranking
| method | stage | technique | model_name | config_id | mrr | recall@1 | recall@5 | recall@10 | recall@20 | map@10 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e2_b16_lr2e5_a0_t4 | 0.800628 | 0.716000 | 0.910000 | 0.942000 | 0.974000 | 0.797731 | 0.833215 |
| pretrained | pretrained | zero_shot | sentence-transformers/all-MiniLM-L6-v2 | zero_shot | 0.771397 | 0.682000 | 0.884000 | 0.922000 | 0.954000 | 0.767905 | 0.805753 |

## Full-corpus Ranking
| method | stage | technique | model_name | config_id | mrr | recall@1 | recall@5 | recall@10 | recall@20 | map@10 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kd | kd_mnr | kd_kl | sentence-transformers/all-MiniLM-L6-v2 | kd_e2_b16_lr2e5_a0_t4 | 0.739049 | 0.627310 | 0.881930 | 0.942505 | 0.968172 | 0.736439 | 0.786674 |
| pretrained | pretrained | zero_shot | sentence-transformers/all-MiniLM-L6-v2 | zero_shot | 0.696345 | 0.577002 | 0.848049 | 0.909651 | 0.939425 | 0.692732 | 0.745635 |

## Comparisons
| run_id | timestamp | comparison | protocol | metric | base_method | base_stage | base_model | compare_method | compare_stage | compare_model | base_value | compare_value | delta | ci_low | ci_high | n_bootstrap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kd_minilm | 2026-03-08T18:02:49.608327+00:00 | kd_vs_student_zero_shot | heldout_test | mrr | pretrained | pretrained | sentence-transformers/all-MiniLM-L6-v2 | kd | kd_mnr | sentence-transformers/all-MiniLM-L6-v2 | 0.771397 | 0.800628 | 0.029232 | 0.012094 | 0.046941 | 2000 | success | bootstrap_delta_mean=0.029633 |
| kd_minilm | 2026-03-08T18:02:49.625728+00:00 | kd_vs_student_zero_shot | heldout_test | recall@10 | pretrained | pretrained | sentence-transformers/all-MiniLM-L6-v2 | kd | kd_mnr | sentence-transformers/all-MiniLM-L6-v2 | 0.922000 | 0.942000 | 0.020000 | 0.006000 | 0.036000 | 2000 | success | bootstrap_delta_mean=0.019871 |