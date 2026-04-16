# Analysis Summary

- Models evaluated: `1`
- Tasks: `all`
- Split: `test`

## OOD Snapshot

- `org/mock-model` MBPP MRR=0.7500, R@10=1.0000

## Keyword Probe Snapshot

- `org/mock-model` `keyword_swap_type` dMRR=0.0000, dR@10=0.0000

## Representative Cases

- `keyword_failure` `org/mock-model` record `test:0`: clean_rank=4 perturbed_rank=4
- `keyword_success` `org/mock-model` record `test:3`: clean_rank=1 perturbed_rank=1
- `mbpp_ood_success` `org/mock-model` record `7`: clean_rank=1 perturbed_rank=1
- `mbpp_ood_failure` `org/mock-model` record `9`: clean_rank=2 perturbed_rank=2
