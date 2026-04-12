# Eval Run Summary

- Timestamp: `2026-03-26 17:09:33`
- Dataset: `codesearchnet`
- Split: `validation`
- Model source: `local`
- Model label: `posthoc_score_distill__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_posthoc_pilot/20260326_170318/posthoc_score_distill/best-val`
- Query count: `1000`
- Code count: `1000`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.920549 |
| MedianRank | 1.000000 |
| Recall@1 | 0.872000 |
| MAP@1 | 0.872000 |
| nDCG@1 | 0.872000 |
| Recall@5 | 0.980000 |
| MAP@5 | 0.919117 |
| nDCG@5 | 0.934656 |
| Recall@10 | 0.985000 |
| MAP@10 | 0.919783 |
| nDCG@10 | 0.936271 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.221648 | 519.33 |
| dataset_load | 35.713090 | 1299.36 |
| query_encode | 1.034852 | 912.20 |
| code_encode | 0.720251 | 802.83 |
| similarity_retrieval | 0.000385 | 805.77 |
| metric_aggregation | 0.034499 | 806.08 |
| total_eval | 37.826796 | 1299.36 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
