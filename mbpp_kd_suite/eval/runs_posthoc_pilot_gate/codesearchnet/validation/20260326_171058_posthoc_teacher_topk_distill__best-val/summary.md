# Eval Run Summary

- Timestamp: `2026-03-26 17:10:58`
- Dataset: `codesearchnet`
- Split: `validation`
- Model source: `local`
- Model label: `posthoc_teacher_topk_distill__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_posthoc_pilot/20260326_170318/posthoc_teacher_topk_distill/best-val`
- Query count: `1000`
- Code count: `1000`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.926923 |
| MedianRank | 1.000000 |
| Recall@1 | 0.881000 |
| MAP@1 | 0.881000 |
| nDCG@1 | 0.881000 |
| Recall@5 | 0.982000 |
| MAP@5 | 0.925433 |
| nDCG@5 | 0.939902 |
| Recall@10 | 0.990000 |
| MAP@10 | 0.926555 |
| nDCG@10 | 0.942543 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.241644 | 519.06 |
| dataset_load | 36.222108 | 1297.70 |
| query_encode | 0.905516 | 956.92 |
| code_encode | 0.785986 | 975.53 |
| similarity_retrieval | 0.000416 | 975.86 |
| metric_aggregation | 0.034087 | 976.33 |
| total_eval | 38.284831 | 1297.70 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
