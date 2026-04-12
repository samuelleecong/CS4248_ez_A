# Eval Run Summary

- Timestamp: `2026-03-26 17:11:56`
- Dataset: `mbpp`
- Split: `test`
- Model source: `local`
- Model label: `supervised_student__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_tiny_hybrid_pilot/20260326_155721/supervised_student/best-val`
- Query count: `500`
- Code count: `500`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.167683 |
| MedianRank | 43.500000 |
| Recall@1 | 0.104000 |
| MAP@1 | 0.104000 |
| nDCG@1 | 0.104000 |
| Recall@5 | 0.210000 |
| MAP@5 | 0.143433 |
| nDCG@5 | 0.160000 |
| Recall@10 | 0.282000 |
| MAP@10 | 0.152860 |
| nDCG@10 | 0.183106 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.231661 | 520.03 |
| dataset_load | 0.875221 | 524.62 |
| query_encode | 0.970985 | 709.39 |
| code_encode | 1.062997 | 915.55 |
| similarity_retrieval | 0.000236 | 915.72 |
| metric_aggregation | 0.008522 | 915.83 |
| total_eval | 3.149941 | 915.83 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
