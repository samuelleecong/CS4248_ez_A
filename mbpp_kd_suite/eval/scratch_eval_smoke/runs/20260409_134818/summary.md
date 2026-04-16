# OOD And Robustness Evaluation

- Models: `1`
- Task selection: `all`
- Split: `test`
- Split seed: `9`

## Metrics

| Model | Task | Dataset | Tier | MRR | R@1 | R@5 | R@10 | Mean Rank | dMRR vs clean | dR@10 vs clean |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| org/mock-model | mbpp_ood | mbpp_sanitized_ood | clean | 0.7500 | 0.5000 | 1.0000 | 1.0000 | 1.50 | 0.0000 | 0.0000 |
| org/mock-model | taco_robustness | taco_local | clean | 0.5208 | 0.2500 | 1.0000 | 1.0000 | 2.50 | 0.0000 | 0.0000 |
| org/mock-model | taco_robustness | taco_local | keyword_swap_type | 0.5208 | 0.2500 | 1.0000 | 1.0000 | 2.50 | 0.0000 | 0.0000 |

## Files

- `metrics.csv`: per-model, per-task, per-tier summary metrics
- `per_query_results.csv`: original/perturbed queries with paired ranks
- `selected_ids.json`: MBPP OOD split record IDs
- `summary.json`: machine-readable run summary
