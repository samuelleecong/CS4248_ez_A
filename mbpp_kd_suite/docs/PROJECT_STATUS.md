# Project Status

Last updated: 2026-03-14

## Scope Completed

- Created a reusable `uv` benchmark project at `mbpp_kd_suite/`.
- Preserved the MBPP baseline evaluation protocol used in the earlier internal runs.
- Added a paper registry and local paper download workflow.
- Added a trained `supervised_student` baseline so KD methods can be compared against a student fine-tuned under the same budget.
- Implemented multiple paper-inspired KD variants on one harness:
  - `supervised_student`
  - `score_distill`
  - `embed_distill`
  - `qed_align`
  - `distilcse_lite`
  - `hard_negative_pair_distill`
  - `all_pairs_distill`
  - `adam_lite`
  - `hpd`

## Current Position

- The suite is designed for controlled comparisons, not exact paper reproduction.
- `score_distill` and `embed_distill` are the direct inherited baselines.
- The suite now defaults to symmetric evaluation so trained-student comparisons use `student query x student code`, not the older query-only asymmetric path.
- The added methods are intentionally lightweight MBPP-compatible variants so they can be compared on the same query-to-code retrieval setup.
- In the 3-epoch comparison, the direct small baseline (`MRR=0.7717`) still remained stronger than every distillation variant.
- In the first fair MBPP smoke run with the same 1-epoch budget, `supervised_student` reached `MRR=0.7926` while `embed_distill` reached `MRR=0.7806`, so KD did not beat normal student fine-tuning there.
- Among the current distillation variants, `embed_distill` is the strongest (`MRR=0.7375`), with `qed_align` (`MRR=0.7359`) next and `adam_lite` (`MRR=0.7269`) slightly behind.
- `score_distill`, `embed_distill`, `qed_align`, `distilcse_lite`, `hard_negative_pair_distill` (formerly `pair_distill`), and `adam_lite` all improved over their 1-epoch runs.
- Switching to the stronger generic teacher `all-mpnet-base-v2` raises the direct teacher baseline to `MRR=0.8229`.
- The initial `MiniLM -> MPNet` KD collapse was largely caused by a randomly initialized student projection head (`384 -> 768`) on a very small MBPP train split.
- Adding least-squares teacher-space initialization for that projection recovers `embed_distill` to `MRR=0.7137` on test, up from `0.2610`.
- Training that fixed setup longer to 8 epochs lowers `embed_distill` to `MRR=0.6940`, which suggests overfitting rather than undertraining.
- This means the main issue is not raw architecture incompatibility; it is cross-space initialization plus the brittleness of the current asymmetric student-query vs teacher-doc coupling.
- The current `hpd` adaptation underperformed badly on MBPP and should be treated as experimental only.
- TACO support is now wired into the same harness through `BEE-spoke-data/TACO-hf`, yielding `18,493` train examples plus `1,000` validation and `1,000` test after filtering and the validation split.
- On TACO, a 1-epoch `embed_distill` run with the inherited MiniLM teacher/student setup reached `MRR=0.1603`, beating both `direct_big_teacher` (`0.1424`) and `direct_small_student` (`0.1336`).
- That older TACO result should now be treated as provisional until it is rerun against `supervised_student` under symmetric evaluation.
- A stronger MPNet teacher is available locally, but full TACO MPNet runs are currently impractical on CPU because teacher-side corpus precomputation dominates runtime.

## Remaining Risks

- `PairDistill` and `ADAM` are implemented without a separate cross-encoder reranker; they rely on the teacher bi-encoder to generate pairwise or dark-example supervision.
- `DistilCSE` is approximated as a query-space CKD objective plus the MBPP retrieval contrastive loss; the original two-stage unlabeled-then-labeled training recipe is not reproduced exactly.
- `HPD` is implemented as a PCA-compressed teacher target space for retrieval, which captures the embedding compression spirit but not every modeling detail from the paper.

## Next Steps

1. Run full multi-seed comparisons and update `docs/EXPERIMENT_LOG.md`.
2. Add teacher-embedding caching if you want practical repeated TACO runs with stronger teachers such as `all-mpnet-base-v2`.
3. Re-run the strongest methods with `all-mpnet-base-v2` and `--projection-init least_squares_both` once caching or faster hardware is available.
4. Re-run TACO with `supervised_student` under symmetric evaluation so the larger-dataset comparison matches the new MBPP protocol.
5. Add a true cross-encoder teacher if you want closer `PairDistill` and `ADAM` reproductions.
