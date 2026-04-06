# Eval Run Summary

- Timestamp: `2026-03-26 17:08:51`
- Dataset: `codesearchnet`
- Split: `test`
- Model source: `local`
- Model label: `supervised_student__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_tiny_hybrid_pilot/20260326_155721/supervised_student/best-val`
- Query count: `1000`
- Code count: `1000`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.909777 |
| MedianRank | 1.000000 |
| Recall@1 | 0.865000 |
| MAP@1 | 0.865000 |
| nDCG@1 | 0.865000 |
| Recall@5 | 0.964000 |
| MAP@5 | 0.907083 |
| nDCG@5 | 0.921559 |
| Recall@10 | 0.976000 |
| MAP@10 | 0.908795 |
| nDCG@10 | 0.925549 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.226358 | 513.81 |
| dataset_load | 35.341357 | 1312.59 |
| query_encode | 1.189532 | 925.05 |
| code_encode | 0.680930 | 706.25 |
| similarity_retrieval | 0.000601 | 710.31 |
| metric_aggregation | 0.034662 | 710.64 |
| total_eval | 37.577613 | 1312.59 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
