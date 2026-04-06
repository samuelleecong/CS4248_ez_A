# Eval Run Summary

- Timestamp: `2026-03-26 17:12:03`
- Dataset: `mbpp`
- Split: `test`
- Model source: `local`
- Model label: `posthoc_teacher_topk_distill__best-val`
- Model path/name: `/Users/kaichen/Desktop/School/Y4S2/CS4248 Natural Language Processing/CODE4248/Project/CS4248_ez_A/mbpp_kd_suite/artifacts/codesearchnet_posthoc_pilot/20260326_170318/posthoc_teacher_topk_distill/best-val`
- Query count: `500`
- Code count: `500`

## Metrics

| Metric | Value |
| --- | ---: |
| MRR | 0.149169 |
| MedianRank | 47.500000 |
| Recall@1 | 0.094000 |
| MAP@1 | 0.094000 |
| nDCG@1 | 0.094000 |
| Recall@5 | 0.186000 |
| MAP@5 | 0.124067 |
| nDCG@5 | 0.139251 |
| Recall@10 | 0.258000 |
| MAP@10 | 0.133661 |
| nDCG@10 | 0.162530 |

## Profiling

| Stage | Seconds | Peak Memory (MB) |
| --- | ---: | ---: |
| model_load | 0.211335 | 515.48 |
| dataset_load | 0.801234 | 520.62 |
| query_encode | 0.827387 | 702.81 |
| code_encode | 1.064869 | 908.89 |
| similarity_retrieval | 0.000180 | 909.06 |
| metric_aggregation | 0.008446 | 909.17 |
| total_eval | 2.913752 | 909.17 |

## Files

- `metrics.csv`: flat machine-readable summary
- `profiling.csv`: per-stage timing and memory
- `metrics.json`: nested full result payload
- `config.json`: resolved evaluation config
