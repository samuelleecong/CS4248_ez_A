# Eval Run Summary

- Timestamp: `2026-03-26 17:08:09`
- Dataset: `codesearchnet`
- Split: `validation`
- Model source: `local`
- Model label: `supervised_student__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_tiny_hybrid_pilot/20260326_155721/supervised_student/best-val`
- Query count: `1000`
- Code count: `1000`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.903743 |
| MedianRank | 1.000000 |
| Recall@1 | 0.846000 |
| MAP@1 | 0.846000 |
| nDCG@1 | 0.846000 |
| Recall@5 | 0.971000 |
| MAP@5 | 0.901233 |
| nDCG@5 | 0.919060 |
| Recall@10 | 0.983000 |
| MAP@10 | 0.902925 |
| nDCG@10 | 0.923030 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.213308 | 521.28 |
| dataset_load | 36.248694 | 1326.89 |
| query_encode | 0.958150 | 937.39 |
| code_encode | 0.795336 | 843.50 |
| similarity_retrieval | 0.000549 | 847.61 |
| metric_aggregation | 0.034683 | 847.84 |
| total_eval | 38.349725 | 1326.89 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
