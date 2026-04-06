# Eval Run Summary

- Timestamp: `2026-03-26 17:10:15`
- Dataset: `codesearchnet`
- Split: `test`
- Model source: `local`
- Model label: `posthoc_score_distill__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_posthoc_pilot/20260326_170318/posthoc_score_distill/best-val`
- Query count: `1000`
- Code count: `1000`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.922462 |
| MedianRank | 1.000000 |
| Recall@1 | 0.883000 |
| MAP@1 | 0.883000 |
| nDCG@1 | 0.883000 |
| Recall@5 | 0.969000 |
| MAP@5 | 0.919950 |
| nDCG@5 | 0.932430 |
| Recall@10 | 0.983000 |
| MAP@10 | 0.921661 |
| nDCG@10 | 0.936800 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.225662 | 516.31 |
| dataset_load | 34.524495 | 1381.70 |
| query_encode | 1.155623 | 992.41 |
| code_encode | 0.699722 | 920.97 |
| similarity_retrieval | 0.000595 | 899.88 |
| metric_aggregation | 0.035236 | 900.27 |
| total_eval | 36.749439 | 1381.70 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
