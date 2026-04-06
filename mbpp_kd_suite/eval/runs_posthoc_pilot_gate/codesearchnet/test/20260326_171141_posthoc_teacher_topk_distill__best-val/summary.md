# Eval Run Summary

- Timestamp: `2026-03-26 17:11:41`
- Dataset: `codesearchnet`
- Split: `test`
- Model source: `local`
- Model label: `posthoc_teacher_topk_distill__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_posthoc_pilot/20260326_170318/posthoc_teacher_topk_distill/best-val`
- Query count: `1000`
- Code count: `1000`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.924937 |
| MedianRank | 1.000000 |
| Recall@1 | 0.887000 |
| MAP@1 | 0.887000 |
| nDCG@1 | 0.887000 |
| Recall@5 | 0.971000 |
| MAP@5 | 0.922283 |
| nDCG@5 | 0.934636 |
| Recall@10 | 0.987000 |
| MAP@10 | 0.924381 |
| nDCG@10 | 0.939773 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.364706 | 516.98 |
| dataset_load | 35.569636 | 1316.67 |
| query_encode | 1.385618 | 930.88 |
| code_encode | 0.751560 | 807.94 |
| similarity_retrieval | 0.000609 | 790.03 |
| metric_aggregation | 0.035166 | 791.89 |
| total_eval | 38.203005 | 1316.67 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
